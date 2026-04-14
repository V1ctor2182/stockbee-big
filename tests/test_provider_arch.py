# tests/test_provider_arch.py
"""Provider 架构基础设施测试。
指令如下
# PYTHONPATH=src .venv/bin/pytest tests/test_provider_arch.py -v
测试对象：CacheManager、Registry、ConfigLoader、BaseProvider
不测试具体 Provider 实现（还没有），只测框架行为。
"""

import os
import time
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stockbee.providers.base import BaseProvider, ProviderConfig, ProviderEnv
from stockbee.providers.cache import CacheManager, CacheStats, L1MemoryCache, L2ParquetCache
from stockbee.providers.config import (
    _substitute_env_vars,
    _process_values,
    load_provider_configs,
    load_env_config,
)
from stockbee.providers.interfaces import CalendarProvider, MarketDataProvider
from stockbee.providers.registry import ProviderRegistry


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def tmp_cache_dir(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def cache_manager(tmp_cache_dir):
    return CacheManager(cache_dir=tmp_cache_dir, l1_max_size=4, l1_default_ttl=3600)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": [date(2024, 1, 2), date(2024, 1, 3)],
        "close": [150.0, 152.0],
    })


@pytest.fixture
def backtest_yaml(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    yaml_content = """\
CalendarProvider:
  implementation: FakeCalendarProvider
  calendar_path: data/calendars/

MarketDataProvider:
  implementation: FakeMarketData
  data_path: ${DATA_DIR:-data}/ohlcv/
  fallback: FakeMarketDataFallback
"""
    (config_dir / "providers-backtest.yaml").write_text(yaml_content, encoding="utf-8")
    return config_dir


# =========================================================================
# 用于测试的假 Provider 实现
# =========================================================================

class FakeCalendarProvider(CalendarProvider):
    def _do_initialize(self):
        self.calendar_path = self._config.params.get("calendar_path", "")
        self._initialized_count = getattr(self, "_initialized_count", 0) + 1

    def get_trading_days(self, start, end, exchange="NYSE"):
        return [date(2024, 1, 2), date(2024, 1, 3)]

    def is_trading_day(self, d, exchange="NYSE"):
        return d.weekday() < 5

    def next_trading_day(self, d, exchange="NYSE"):
        return date(2024, 1, 3)

    def prev_trading_day(self, d, exchange="NYSE"):
        return date(2024, 1, 2)


class FakeMarketData(MarketDataProvider):
    def _do_initialize(self):
        self.data_path = self._config.params.get("data_path", "")

    def get_daily_bars(self, tickers, start, end, fields=None):
        return pd.DataFrame()

    def get_latest_price(self, tickers):
        return {"AAPL": 150.0}


class FailingProvider(CalendarProvider):
    def _do_initialize(self):
        raise ConnectionError("模拟连接失败")

    def get_trading_days(self, start, end, exchange="NYSE"):
        return []

    def is_trading_day(self, d, exchange="NYSE"):
        return False

    def next_trading_day(self, d, exchange="NYSE"):
        return d

    def prev_trading_day(self, d, exchange="NYSE"):
        return d


class FailingShutdownProvider(CalendarProvider):
    """shutdown 时必定抛异常，用于测试 shutdown_all 的容错。"""

    def _do_initialize(self):
        pass

    def _do_shutdown(self):
        raise RuntimeError("shutdown exploded")

    def get_trading_days(self, start, end, exchange="NYSE"):
        return []

    def is_trading_day(self, d, exchange="NYSE"):
        return False

    def next_trading_day(self, d, exchange="NYSE"):
        return d

    def prev_trading_day(self, d, exchange="NYSE"):
        return d


class TrackingShutdownProvider(CalendarProvider):
    """记录 _do_shutdown 是否被调用。"""
    shutdown_called = False

    def _do_initialize(self):
        TrackingShutdownProvider.shutdown_called = False

    def _do_shutdown(self):
        TrackingShutdownProvider.shutdown_called = True

    def get_trading_days(self, start, end, exchange="NYSE"):
        return []

    def is_trading_day(self, d, exchange="NYSE"):
        return False

    def next_trading_day(self, d, exchange="NYSE"):
        return d

    def prev_trading_day(self, d, exchange="NYSE"):
        return d


# =========================================================================
# L1 内存缓存测试
# =========================================================================

class TestL1MemoryCache:

    # 普通测试：set 后 get 返回正确值
    def test_basic_get_set(self):
        cache = L1MemoryCache(max_size=10)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    # 边界：访问不存在的 key 返回 None 而非抛异常
    def test_get_nonexistent_returns_none(self):
        cache = L1MemoryCache()
        assert cache.get("missing") is None

    # 边界：条目超过 TTL 后 get() 应返回 None 并从 store 中清除
    def test_ttl_expiration(self):
        cache = L1MemoryCache(default_ttl=0.01)
        cache.set("k1", "v1")
        time.sleep(0.02)
        assert cache.get("k1") is None

    # 边界：set() 传入的 ttl 参数应覆盖构造函数的 default_ttl
    def test_explicit_ttl_overrides_default(self):
        cache = L1MemoryCache(default_ttl=3600)
        cache.set("k1", "v1", ttl=0.01)
        time.sleep(0.02)
        assert cache.get("k1") is None

    # 边界：TTL=0 表示永不过期
    def test_never_expire_when_ttl_zero(self):
        cache = L1MemoryCache(default_ttl=0)
        cache.set("k1", "v1")
        assert cache.get("k1") == "v1"

    # 边界：容量满时，最早插入且未被访问的条目（LRU）被驱逐
    def test_lru_eviction(self):
        cache = L1MemoryCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"

    # 边界：get() 会将条目移至队尾（最近使用），改变淘汰顺序
    def test_lru_access_refreshes_order(self):
        cache = L1MemoryCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.get("k1")            # k1 移至队尾
        cache.set("k3", "v3")      # 淘汰 k2（此时最久未用）
        assert cache.get("k1") == "v1"
        assert cache.get("k2") is None
        assert cache.get("k3") == "v3"

    # 边界：同 key 覆写应原地更新值，不增加 size
    def test_overwrite_existing_key(self):
        cache = L1MemoryCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k1", "v2")
        assert cache.get("k1") == "v2"
        assert cache.size == 1

    # 边界：invalidate 后该 key 应不可访问
    def test_invalidate(self):
        cache = L1MemoryCache()
        cache.set("k1", "v1")
        cache.invalidate("k1")
        assert cache.get("k1") is None

    # 边界：invalidate 不存在的 key 不应抛异常
    def test_invalidate_nonexistent_key_no_error(self):
        cache = L1MemoryCache()
        cache.invalidate("nonexistent")

    # 边界：clear 后 size 归零，所有 key 不可访问
    def test_clear(self):
        cache = L1MemoryCache()
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.size == 0
        assert cache.get("k1") is None

    # 边界 Bug 3 回归：缓存满时，应优先驱逐已过期条目，而非有效的 LRU 条目
    def test_expired_entry_evicted_before_valid_lru(self):
        cache = L1MemoryCache(max_size=2)
        cache.set("k1", "v1", ttl=0.01)  # 即将过期
        cache.set("k2", "v2", ttl=3600)  # 长期有效
        time.sleep(0.02)                  # k1 已过期
        cache.set("k3", "v3")             # 触发淘汰：应淘汰 k1，保留 k2
        assert cache.get("k2") == "v2"    # 有效热数据未被驱逐
        assert cache.get("k3") == "v3"

    # 边界：size 直接返回 len(_store)，过期条目在 get() 前不会被清除
    def test_size_includes_logically_expired_entries(self):
        cache = L1MemoryCache(default_ttl=0.01)
        cache.set("k1", "v1")
        time.sleep(0.02)
        assert cache.size == 1            # 条目仍在 store 中，只是逻辑过期
        cache.get("k1")                   # get() 触发清除
        assert cache.size == 0

    # 边界：0、False、""、[] 均不是 None，L1 应能正确存取这些 falsy 值
    def test_falsy_non_none_values_stored_and_retrieved(self):
        cache = L1MemoryCache()
        for key, val in [("int", 0), ("bool", False), ("str", ""), ("list", [])]:
            cache.set(key, val)
            assert cache.get(key) == val

    # 边界：覆写已有 key 后，该 key 应移至 LRU 队列末尾（最近使用）
    def test_update_extends_entry_to_tail(self):
        cache = L1MemoryCache(max_size=2)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k1", "v1_new")  # 覆写 k1，k1 移至末尾
        cache.set("k3", "v3")      # 淘汰 k2（此时最久未用）
        assert cache.get("k1") == "v1_new"
        assert cache.get("k2") is None
        assert cache.get("k3") == "v3"


# =========================================================================
# L2 Parquet 缓存测试
# =========================================================================

class TestL2ParquetCache:

    # 普通测试：DataFrame set 后 get 返回完全相同的 DataFrame
    def test_basic_get_set(self, tmp_cache_dir, sample_df):
        cache = L2ParquetCache(tmp_cache_dir)
        cache.set("k1", sample_df)
        result = cache.get("k1")
        assert result is not None
        pd.testing.assert_frame_equal(result, sample_df)

    # 边界：访问不存在的 key 返回 None
    def test_get_nonexistent_returns_none(self, tmp_cache_dir):
        cache = L2ParquetCache(tmp_cache_dir)
        assert cache.get("missing") is None

    # 边界：非 DataFrame 类型传入 set() 应被静默忽略，不写入磁盘
    def test_ignores_non_dataframe(self, tmp_cache_dir):
        cache = L2ParquetCache(tmp_cache_dir)
        cache.set("k1", {"not": "a dataframe"})
        assert cache.get("k1") is None

    # 边界：invalidate 后对应的 parquet 文件应被删除
    def test_invalidate(self, tmp_cache_dir, sample_df):
        cache = L2ParquetCache(tmp_cache_dir)
        cache.set("k1", sample_df)
        cache.invalidate("k1")
        assert cache.get("k1") is None

    # 边界：clear 删除目录下所有 parquet 文件
    def test_clear(self, tmp_cache_dir, sample_df):
        cache = L2ParquetCache(tmp_cache_dir)
        cache.set("k1", sample_df)
        cache.set("k2", sample_df)
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    # 边界：构造函数应自动创建不存在的多级缓存目录
    def test_creates_dir_if_not_exists(self, tmp_path):
        new_dir = tmp_path / "deep" / "nested" / "cache"
        cache = L2ParquetCache(new_dir)
        assert new_dir.exists()

    # 边界：空 DataFrame（0 行）也应能正常 round-trip
    def test_empty_dataframe_roundtrip(self, tmp_cache_dir):
        cache = L2ParquetCache(tmp_cache_dir)
        empty_df = pd.DataFrame({"close": pd.Series([], dtype=float)})
        cache.set("k1", empty_df)
        result = cache.get("k1")
        assert result is not None
        assert len(result) == 0
        assert list(result.columns) == ["close"]

    # 边界：磁盘文件损坏时，get() 应返回 None 而非抛出异常
    def test_corrupted_file_returns_none(self, tmp_cache_dir):
        cache = L2ParquetCache(tmp_cache_dir)
        path = cache._key_to_path("k1")
        path.write_bytes(b"this is not a valid parquet file")
        assert cache.get("k1") is None

    # 边界：对同一 key set 两次，第二次写入的值应覆盖第一次
    def test_overwrite_existing_key(self, tmp_cache_dir, sample_df):
        cache = L2ParquetCache(tmp_cache_dir)
        cache.set("k1", sample_df)
        new_df = pd.DataFrame({"close": [999.0]})
        cache.set("k1", new_df)
        result = cache.get("k1")
        assert result["close"].iloc[0] == 999.0

    # 边界：不同 key 应映射到不同的 parquet 文件，互不干扰
    def test_different_keys_use_different_files(self, tmp_cache_dir):
        cache = L2ParquetCache(tmp_cache_dir)
        df_a = pd.DataFrame({"v": [1]})
        df_b = pd.DataFrame({"v": [2]})
        cache.set("key_a", df_a)
        cache.set("key_b", df_b)
        assert cache.get("key_a")["v"].iloc[0] == 1
        assert cache.get("key_b")["v"].iloc[0] == 2

    # Regression: cache filename must be derived via SHA-256 (not MD5).
    # This locks in the hash choice so a future revert to MD5 breaks loudly.
    def test_key_filename_uses_sha256(self, tmp_cache_dir):
        import hashlib
        cache = L2ParquetCache(tmp_cache_dir)
        path = cache._key_to_path("some-cache-key")
        expected = hashlib.sha256(b"some-cache-key").hexdigest()
        assert path.stem == expected
        # SHA-256 hex is 64 chars; MD5 would be 32.
        assert len(path.stem) == 64


# =========================================================================
# CacheManager 三层联动测试
# =========================================================================

class TestCacheManager:

    # 普通测试：set 写入后 get 命中 L1，stats 正确计数
    def test_l1_hit(self, cache_manager):
        cache_manager.set("k1", "v1")
        result = cache_manager.get("k1")
        assert result == "v1"
        assert cache_manager.stats.l1_hits == 1

    # 边界：L1 miss 但 L2 有数据时，应命中 L2 并回填 L1
    def test_l2_hit_and_backfill_l1(self, cache_manager, sample_df):
        cache_manager._l2.set("k1", sample_df)
        result = cache_manager.get("k1")
        assert result is not None
        assert cache_manager.stats.l2_hits == 1
        assert cache_manager._l1.get("k1") is not None

    # 边界：L1、L2 均 miss 时，调用 fetcher (L3) 并回填 L1
    def test_l3_fetcher_called_on_miss(self, cache_manager):
        fetcher_called = False
        def fetcher():
            nonlocal fetcher_called
            fetcher_called = True
            return "from_api"
        result = cache_manager.get("k1", fetcher=fetcher)
        assert result == "from_api"
        assert fetcher_called
        assert cache_manager.stats.l3_hits == 1
        assert cache_manager._l1.get("k1") == "from_api"

    # 边界：L1 命中后不应调用 fetcher
    def test_l3_fetcher_not_called_on_l1_hit(self, cache_manager):
        cache_manager.set("k1", "cached")
        fetcher_called = False
        def fetcher():
            nonlocal fetcher_called
            fetcher_called = True
            return "from_api"
        result = cache_manager.get("k1", fetcher=fetcher)
        assert result == "cached"
        assert not fetcher_called

    # 边界：fetcher 抛异常时，应返回 None 并计入 miss
    def test_l3_fetcher_failure_counts_as_miss(self, cache_manager):
        def bad_fetcher():
            raise RuntimeError("API down")
        result = cache_manager.get("k1", fetcher=bad_fetcher)
        assert result is None
        assert cache_manager.stats.misses == 1

    # 边界：fetcher 返回 None 时，应计入 miss 而非 L3 hit
    def test_l3_fetcher_returns_none_counts_as_miss(self, cache_manager):
        result = cache_manager.get("k1", fetcher=lambda: None)
        assert result is None
        assert cache_manager.stats.misses == 1

    # 边界：无 fetcher 且全部 miss 时，返回 None 并计入 miss
    def test_no_fetcher_and_miss(self, cache_manager):
        result = cache_manager.get("nonexistent")
        assert result is None
        assert cache_manager.stats.misses == 1

    # 边界：L3 fetcher 返回 DataFrame 时，应同时回填 L2
    def test_l3_dataframe_backfills_l2(self, cache_manager, sample_df):
        result = cache_manager.get("k1", fetcher=lambda: sample_df)
        assert result is not None
        l2_result = cache_manager._l2.get("k1")
        assert l2_result is not None

    # 边界：L3 fetcher 返回非 DataFrame 时，不应写入 L2（L2 只存 DataFrame）
    def test_l3_non_dataframe_does_not_backfill_l2(self, cache_manager):
        cache_manager.get("k1", fetcher=lambda: [1, 2, 3])
        l2_result = cache_manager._l2.get("k1")
        assert l2_result is None

    # 边界：invalidate 应同时清除 L1 和 L2
    def test_invalidate_clears_both_layers(self, cache_manager, sample_df):
        cache_manager.set("k1", sample_df)
        cache_manager.invalidate("k1")
        assert cache_manager._l1.get("k1") is None
        assert cache_manager._l2.get("k1") is None

    # 边界：clear 应清空所有缓存并重置统计计数
    def test_clear_resets_everything(self, cache_manager):
        cache_manager.set("k1", "v1")
        cache_manager.set("k2", "v2")
        cache_manager.get("k1")
        cache_manager.clear()
        assert cache_manager._l1.size == 0
        assert cache_manager.stats.total_requests == 0

    # 边界：report() 应返回包含命中率在内的摘要字典
    def test_report(self, cache_manager):
        cache_manager.set("k1", "v1")
        cache_manager.get("k1")
        cache_manager.get("missing")
        report = cache_manager.report()
        assert report["l1_hits"] == 1
        assert report["misses"] == 1
        assert "hit_rate" in report

    # 边界：0 和 False 不是 None，CacheManager 应将其视为 L1 命中而非穿透到 L2/L3
    def test_falsy_non_none_value_counts_as_l1_hit(self, cache_manager):
        cache_manager._l1.set("zero", 0)
        cache_manager._l1.set("false", False)
        assert cache_manager.get("zero") == 0
        assert cache_manager.stats.l1_hits == 1
        assert cache_manager.get("false") is False
        assert cache_manager.stats.l1_hits == 2

    # 边界：L2 命中后，即使提供了 fetcher，也不应调用 L3
    def test_l2_hit_skips_l3_fetcher(self, cache_manager, sample_df):
        cache_manager._l2.set("k1", sample_df)
        fetcher_called = False
        def fetcher():
            nonlocal fetcher_called
            fetcher_called = True
            return "from_api"
        cache_manager.get("k1", fetcher=fetcher)
        assert not fetcher_called
        assert cache_manager.stats.l2_hits == 1

    # 边界：set() 对已存在的 key 应同时更新 L1 和 L2
    def test_set_updates_existing_key_in_both_layers(self, cache_manager, sample_df):
        cache_manager.set("k1", sample_df)
        new_df = pd.DataFrame({"close": [999.0]})
        cache_manager.set("k1", new_df)
        l1_result = cache_manager._l1.get("k1")
        assert l1_result["close"].iloc[0] == 999.0
        l2_result = cache_manager._l2.get("k1")
        assert l2_result["close"].iloc[0] == 999.0

    # 边界：无任何请求时，hit_rate 应为 0.0 而非除零错误
    def test_hit_rate_zero_when_no_requests(self, cache_manager):
        assert cache_manager.stats.hit_rate == 0.0

    # 边界：多次 get() 后，各层命中计数应正确累加
    def test_stats_accumulate_across_multiple_gets(self, cache_manager, sample_df):
        cache_manager.set("l1key", "v1")
        cache_manager._l2.set("l2key", sample_df)
        cache_manager.get("l1key")
        cache_manager.get("l2key")
        cache_manager.get("missing1")
        cache_manager.get("missing2")
        assert cache_manager.stats.l1_hits == 1
        assert cache_manager.stats.l2_hits == 1
        assert cache_manager.stats.misses == 2
        assert cache_manager.stats.total_requests == 4


# =========================================================================
# ConfigLoader 测试
# =========================================================================

class TestEnvVarSubstitution:

    # 普通测试：环境变量已设置时，${VAR} 被替换为对应值
    def test_with_env_var_set(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "/custom/path")
        result = _substitute_env_vars("${MY_VAR}/data")
        assert result == "/custom/path/data"

    # 边界：环境变量未设置但有默认值时，使用 :- 后的默认值
    def test_with_default_value(self):
        os.environ.pop("MY_MISSING_VAR", None)
        result = _substitute_env_vars("${MY_MISSING_VAR:-fallback}/data")
        assert result == "fallback/data"

    # 边界：环境变量未设置且无默认值时，保留原始 ${VAR} 文本
    def test_env_var_not_set_no_default(self):
        os.environ.pop("TOTALLY_MISSING", None)
        result = _substitute_env_vars("${TOTALLY_MISSING}/data")
        assert result == "${TOTALLY_MISSING}/data"

    # 边界：一个字符串中包含多个环境变量，应全部替换
    def test_multiple_vars_in_one_string(self, monkeypatch):
        monkeypatch.setenv("A", "hello")
        monkeypatch.setenv("B", "world")
        result = _substitute_env_vars("${A}-${B}")
        assert result == "hello-world"

    # 边界：不含 ${} 的普通字符串应原样返回
    def test_no_vars(self):
        result = _substitute_env_vars("plain string")
        assert result == "plain string"


class TestProcessValues:

    # 普通测试：递归替换嵌套 dict 和 list 中的环境变量
    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("X", "replaced")
        data = {"a": {"b": "${X}"}, "c": [1, "${X}"]}
        result = _process_values(data)
        assert result["a"]["b"] == "replaced"
        assert result["c"][1] == "replaced"
        assert result["c"][0] == 1

    # 边界：非字符串类型（int、bool、None）应原样透传，不做替换
    def test_non_string_passthrough(self):
        assert _process_values(42) == 42
        assert _process_values(True) is True
        assert _process_values(None) is None


class TestLoadProviderConfigs:

    # 普通测试：加载 YAML 文件并解析为 ProviderConfig 字典
    def test_loads_backtest_yaml(self, backtest_yaml):
        configs = load_provider_configs(backtest_yaml / "providers-backtest.yaml")
        assert "CalendarProvider" in configs
        assert "MarketDataProvider" in configs
        assert len(configs) == 2
        cal = configs["CalendarProvider"]
        assert cal.implementation == "FakeCalendarProvider"
        assert cal.params["calendar_path"] == "data/calendars/"
        assert cal.fallback is None
        market = configs["MarketDataProvider"]
        assert market.implementation == "FakeMarketData"
        assert market.params["data_path"] == "data/ohlcv/"
        assert market.fallback == "FakeMarketDataFallback"

    # 边界：YAML 中的 ${ENV_VAR:-default} 应被实际环境变量值替换
    def test_env_var_substitution_in_yaml(self, backtest_yaml, monkeypatch):
        monkeypatch.setenv("DATA_DIR", "/custom")
        configs = load_provider_configs(backtest_yaml / "providers-backtest.yaml")
        assert configs["MarketDataProvider"].params["data_path"] == "/custom/ohlcv/"

    # 边界：文件不存在时应抛 FileNotFoundError
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_provider_configs("nonexistent.yaml")

    # 边界：YAML 顶层不是 dict（如纯字符串）时应抛 ValueError
    def test_invalid_yaml_format(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("just a string", encoding="utf-8")
        with pytest.raises(ValueError, match="expected dict"):
            load_provider_configs(bad_yaml)

    # 边界：空文件 → yaml.safe_load 返回 None → 应触发 ValueError 而非 AttributeError
    def test_empty_yaml_raises_value_error(self, tmp_path):
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="expected dict"):
            load_provider_configs(empty_yaml)

    # 边界 Bug 1 回归：pop() 后 params 中不应残留 implementation 或 fallback 键
    def test_implementation_and_fallback_not_in_params(self, backtest_yaml):
        configs = load_provider_configs(backtest_yaml / "providers-backtest.yaml")
        assert "implementation" not in configs["CalendarProvider"].params
        assert "fallback" not in configs["CalendarProvider"].params
        assert "implementation" not in configs["MarketDataProvider"].params
        assert "fallback" not in configs["MarketDataProvider"].params

    # 边界 Bug 1 回归：同一文件加载两次，结果应完全一致（shallow copy 阻止了原始 dict 被 pop 污染）
    def test_loading_same_file_twice_is_idempotent(self, backtest_yaml):
        path = backtest_yaml / "providers-backtest.yaml"
        first = load_provider_configs(path)
        second = load_provider_configs(path)
        assert first["CalendarProvider"].implementation == second["CalendarProvider"].implementation
        assert first["MarketDataProvider"].fallback == second["MarketDataProvider"].fallback

    # 边界：缺少 implementation 字段的条目应被跳过，不影响其他条目
    def test_entry_missing_implementation_is_skipped(self, tmp_path):
        yaml_content = "NoImplProvider:\n  some_param: value\n\nGoodProvider:\n  implementation: Foo\n"
        yaml_file = tmp_path / "providers.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        configs = load_provider_configs(yaml_file)
        assert "NoImplProvider" not in configs
        assert "GoodProvider" in configs

    # 边界：顶层值不是 dict 的条目（如纯字符串）应被静默跳过
    def test_top_level_non_dict_entry_is_skipped(self, tmp_path):
        yaml_content = "ValidProvider:\n  implementation: Bar\nscalar_entry: just_a_string\n"
        yaml_file = tmp_path / "providers.yaml"
        yaml_file.write_text(yaml_content, encoding="utf-8")
        configs = load_provider_configs(yaml_file)
        assert "scalar_entry" not in configs
        assert "ValidProvider" in configs


class TestLoadEnvConfig:

    # 普通测试：按环境名找到 providers-{env}.yaml 并加载
    def test_loads_backtest_env(self, backtest_yaml):
        configs = load_env_config(config_dir=backtest_yaml, env=ProviderEnv.BACKTEST)
        assert "CalendarProvider" in configs

    # 边界：环境专属文件不存在时，应 fallback 到 providers.yaml
    def test_fallback_to_default(self, tmp_path):
        config_dir = tmp_path / "config"
        config_dir.mkdir()
        yaml_content = "CalendarProvider:\n  implementation: DefaultCal\n"
        (config_dir / "providers.yaml").write_text(yaml_content, encoding="utf-8")
        configs = load_env_config(config_dir=config_dir, env=ProviderEnv.BACKTEST)
        assert configs["CalendarProvider"].implementation == "DefaultCal"

    # 边界：环境专属和默认文件都不存在时，应抛 FileNotFoundError
    def test_no_config_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_env_config(config_dir=tmp_path / "empty", env=ProviderEnv.BACKTEST)


# =========================================================================
# BaseProvider 生命周期测试
# =========================================================================

class TestBaseProvider:

    # 普通测试：initialize() 将 is_initialized 设为 True
    def test_initialize_sets_flag(self):
        provider = FakeCalendarProvider()
        provider.cache = CacheManager()
        assert not provider.is_initialized
        provider.initialize()
        assert provider.is_initialized

    # 边界：连续调用 initialize() 两次，_do_initialize 只应执行一次（幂等）
    def test_initialize_idempotent(self):
        config = ProviderConfig(implementation="FakeCalendarProvider")
        provider = FakeCalendarProvider(config)
        provider.cache = CacheManager()
        provider.initialize()
        provider.initialize()
        assert provider._initialized_count == 1

    # 边界：shutdown 后 is_initialized 应重置为 False
    def test_shutdown(self):
        provider = FakeCalendarProvider()
        provider.cache = CacheManager()
        provider.initialize()
        provider.shutdown()
        assert not provider.is_initialized

    # 边界：未 initialize 就调用 shutdown 应为无操作，不抛异常
    def test_shutdown_without_initialize_is_noop(self):
        provider = FakeCalendarProvider()
        provider.shutdown()

    # 边界：health_check 在未初始化时返回 False，初始化后返回 True
    def test_health_check(self):
        provider = FakeCalendarProvider()
        provider.cache = CacheManager()
        assert not provider.health_check()
        provider.initialize()
        assert provider.health_check()

    # 边界：config.params 中的参数应在 _do_initialize 中可访问
    def test_config_params_accessible(self):
        config = ProviderConfig(
            implementation="FakeCalendarProvider",
            params={"calendar_path": "/test/path"},
        )
        provider = FakeCalendarProvider(config)
        provider.cache = CacheManager()
        provider.initialize()
        assert provider.calendar_path == "/test/path"

    # 边界：__repr__ 应包含初始化状态文本
    def test_repr(self):
        provider = FakeCalendarProvider()
        assert "not initialized" in repr(provider)
        provider.cache = CacheManager()
        provider.initialize()
        assert "initialized" in repr(provider)

    # 边界：config=None 时应自动生成以类名为 implementation 的默认配置
    def test_default_config_when_none(self):
        provider = FakeCalendarProvider()
        assert provider._config.implementation == "FakeCalendarProvider"

    # 边界：_do_initialize 抛出异常时，is_initialized 应保持 False
    def test_do_initialize_failure_leaves_provider_uninitialized(self):
        provider = FailingProvider()
        with pytest.raises(ConnectionError):
            provider.initialize()
        assert not provider.is_initialized

    # 边界：shutdown 后可以再次 initialize，完整生命周期可重复
    def test_reinitialize_after_shutdown(self):
        provider = FakeCalendarProvider()
        provider.cache = CacheManager()
        provider.initialize()
        provider.shutdown()
        assert not provider.is_initialized
        provider.initialize()
        assert provider.is_initialized

    # 边界：shutdown() 应实际调用子类的 _do_shutdown 钩子
    def test_do_shutdown_is_called(self):
        provider = TrackingShutdownProvider()
        provider.cache = CacheManager()
        provider.initialize()
        provider.shutdown()
        assert TrackingShutdownProvider.shutdown_called

    # 边界：连续调用 shutdown() 多次不应报错，第二次为 no-op
    def test_multiple_shutdowns_are_noop(self):
        provider = FakeCalendarProvider()
        provider.cache = CacheManager()
        provider.initialize()
        provider.shutdown()
        provider.shutdown()
        assert not provider.is_initialized


# =========================================================================
# Registry 测试
# =========================================================================

class TestProviderRegistry:

    # 普通测试：register 类 → create 实例 → get 获取实例
    def test_register_and_get(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        config = ProviderConfig(implementation="FakeCalendarProvider")
        registry.create("CalendarProvider", config)
        provider = registry.get("CalendarProvider")
        assert isinstance(provider, FakeCalendarProvider)

    # 边界：注册非 BaseProvider 子类时应抛 TypeError
    def test_register_non_provider_raises(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        with pytest.raises(TypeError):
            registry.register("NotAProvider", dict)

    # 边界：create 时 implementation 名未注册应抛 KeyError
    def test_create_unknown_implementation_raises(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        config = ProviderConfig(implementation="NonExistent")
        with pytest.raises(KeyError, match="Unknown provider implementation"):
            registry.create("SomeProvider", config)

    # 边界：get 不存在的 provider_name 应抛 KeyError
    def test_get_nonexistent_raises(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        with pytest.raises(KeyError, match="Provider not found"):
            registry.get("Missing")

    # 边界：get_or_none 不存在时返回 None 而非抛异常
    def test_get_or_none(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        assert registry.get_or_none("Missing") is None

    # 边界：create 时应自动将 cache_manager 注入到实例
    def test_cache_injected_on_create(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        config = ProviderConfig(implementation="FakeCalendarProvider")
        provider = registry.create("CalendarProvider", config)
        assert provider.cache is cache_manager

    # 边界：create_from_config 批量创建多个 provider
    def test_create_from_config(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        registry.register("FakeMarketData", FakeMarketData)
        configs = {
            "CalendarProvider": ProviderConfig(implementation="FakeCalendarProvider"),
            "MarketDataProvider": ProviderConfig(implementation="FakeMarketData"),
        }
        registry.create_from_config(configs)
        assert len(registry.active_providers) == 2

    # 边界：create_from_config 中有未注册的 implementation 时应抛 KeyError
    def test_create_from_config_fails_on_missing(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        configs = {
            "CalendarProvider": ProviderConfig(implementation="FakeCalendarProvider"),
        }
        with pytest.raises(KeyError):
            registry.create_from_config(configs)

    # 边界：initialize_all 应初始化所有已创建的实例
    def test_initialize_all(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        registry.register("FakeMarketData", FakeMarketData)
        configs = {
            "CalendarProvider": ProviderConfig(implementation="FakeCalendarProvider"),
            "MarketDataProvider": ProviderConfig(implementation="FakeMarketData"),
        }
        registry.create_from_config(configs)
        registry.initialize_all()
        assert registry.get("CalendarProvider").is_initialized
        assert registry.get("MarketDataProvider").is_initialized

    # 边界：initialize_all 中某个 provider 初始化失败时，异常应向上传播
    def test_initialize_all_propagates_failure(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FailingProvider", FailingProvider)
        config = ProviderConfig(implementation="FailingProvider")
        registry.create("CalendarProvider", config)
        with pytest.raises(ConnectionError):
            registry.initialize_all()

    # 边界：shutdown_all 应关闭所有已初始化的实例
    def test_shutdown_all(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        config = ProviderConfig(implementation="FakeCalendarProvider")
        registry.create("CalendarProvider", config)
        registry.initialize_all()
        registry.shutdown_all()
        assert not registry.get("CalendarProvider").is_initialized

    # 边界：health_check_all 在初始化前全部 False，初始化后全部 True
    def test_health_check_all(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        registry.register("FakeMarketData", FakeMarketData)
        configs = {
            "CalendarProvider": ProviderConfig(implementation="FakeCalendarProvider"),
            "MarketDataProvider": ProviderConfig(implementation="FakeMarketData"),
        }
        registry.create_from_config(configs)
        health = registry.health_check_all()
        assert not any(health.values())
        registry.initialize_all()
        health = registry.health_check_all()
        assert all(health.values())

    # 边界：registered_classes 和 active_providers 属性应分别反映注册的类和已创建的实例
    def test_registered_classes_and_active_providers(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        assert "FakeCalendarProvider" in registry.registered_classes
        assert len(registry.active_providers) == 0
        config = ProviderConfig(implementation="FakeCalendarProvider")
        registry.create("CalendarProvider", config)
        assert "CalendarProvider" in registry.active_providers

    # 边界：未传入 cache_manager 时，Registry 应自动创建默认实例
    def test_default_cache_manager_created(self):
        registry = ProviderRegistry()
        assert registry.cache is not None

    # 边界：同名 register 两次，后一次应覆盖前一次
    def test_register_same_name_overwrites(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("MyProvider", FakeCalendarProvider)
        registry.register("MyProvider", FakeMarketData)
        assert registry._classes["MyProvider"] is FakeMarketData

    # 边界：空配置字典传入 create_from_config 不应报错
    def test_create_from_config_empty_dict_is_noop(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.create_from_config({})
        assert len(registry.active_providers) == 0

    # 边界：无已创建实例时调用 initialize_all 不应报错
    def test_initialize_all_empty_registry_is_noop(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.initialize_all()

    # 边界：某个 provider 的 shutdown 抛异常时，shutdown_all 应继续处理其余 provider
    def test_shutdown_all_swallows_individual_errors(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FailingShutdownProvider", FailingShutdownProvider)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        registry.create("Failing", ProviderConfig(implementation="FailingShutdownProvider"))
        registry.create("Calendar", ProviderConfig(implementation="FakeCalendarProvider"))
        registry.initialize_all()
        registry.shutdown_all()
        assert not registry.get("Calendar").is_initialized

    # 边界：同一 provider_name 调用 create 两次，应以最新实例为准
    def test_create_replaces_existing_instance(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        config = ProviderConfig(implementation="FakeCalendarProvider")
        first = registry.create("CalendarProvider", config)
        second = registry.create("CalendarProvider", config)
        assert registry.get("CalendarProvider") is second
        assert registry.get("CalendarProvider") is not first

    # 边界：__repr__ 应包含 classes 和 instances 计数
    def test_repr(self, cache_manager):
        registry = ProviderRegistry(cache_manager=cache_manager)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        r = repr(registry)
        assert "classes=1" in r
        assert "instances=0" in r


# =========================================================================
# 端到端集成测试
# =========================================================================

class TestEndToEnd:

    # 普通测试：YAML 加载 → Registry 批量创建 → 初始化 → 调用 → 关闭 完整生命周期
    def test_full_flow_from_yaml(self, backtest_yaml, tmp_cache_dir):
        configs = load_env_config(config_dir=backtest_yaml, env=ProviderEnv.BACKTEST)
        cache = CacheManager(cache_dir=tmp_cache_dir)
        registry = ProviderRegistry(cache_manager=cache)
        registry.register("FakeCalendarProvider", FakeCalendarProvider)
        registry.register("FakeMarketData", FakeMarketData)
        registry.create_from_config(configs)
        registry.initialize_all()

        calendar = registry.get("CalendarProvider")
        days = calendar.get_trading_days(date(2024, 1, 1), date(2024, 3, 31))
        assert len(days) > 0

        market = registry.get("MarketDataProvider")
        prices = market.get_latest_price(["AAPL"])
        assert "AAPL" in prices

        registry.shutdown_all()
        assert not calendar.is_initialized
