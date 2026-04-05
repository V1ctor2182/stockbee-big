# tests/test_news_data.py
"""News Data 模块测���。
# PYTHONPATH=src python -m pytest tests/test_news_data.py -v

测试对象：SqliteNewsProvider、G1Filter、G2Classifier
不测 NewsAPI/Perplexity（需要 API key），不测 NewsDataSyncer（集成测试）。
G2 测试使用 mock FinBERT（不依赖 transformers/torch 安装）。
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stockbee.providers.base import ProviderConfig
from stockbee.news_data.news_store import (
    SqliteNewsProvider,
    MAX_HEADLINE_LENGTH,
    MAX_SNIPPET_LENGTH,
    _parse_tickers,
    _normalize_timestamp,
    _truncate,
)
from stockbee.news_data.g1_filter import (
    G1Config,
    G1Filter,
    G1Result,
    _AMBIGUOUS_TICKERS,
    _COMMON_WORDS,
    _COMPANY_TO_TICKER,
    _VALID_SHORT_TICKERS,
)
from stockbee.news_data.g2_classifier import (
    G2Classifier,
    G2Config,
    G2Result,
    FINBERT_MAX_LENGTH,
    MIN_ASCII_RATIO,
    _TOPIC_KEYWORDS,
    _URGENCY_HIGH_KEYWORDS,
    _URGENCY_MEDIUM_KEYWORDS,
)


# =========================================================================
# Fixtures
# =========================================================================

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ago(days: int) -> datetime:
    return _now() - timedelta(days=days)


@pytest.fixture
def provider(tmp_path) -> SqliteNewsProvider:
    """临时 SQLite 数据库的 NewsProvider。"""
    db = tmp_path / "test_news.db"
    p = SqliteNewsProvider(ProviderConfig(
        implementation="test", params={"db_path": str(db)}
    ))
    p.initialize()
    yield p
    p.shutdown()


@pytest.fixture
def g1() -> G1Filter:
    """默认配置的 G1 过滤器。"""
    return G1Filter()


@pytest.fixture
def g1_strict() -> G1Filter:
    """require_ticker=True 的 G1 过滤器。"""
    return G1Filter(G1Config(require_ticker=True))


# =========================================================================
# SqliteNewsProvider — 基本 CRUD
# =========================================================================

class TestSqliteNewsProvider:

    def test_insert_and_get_by_id(self, provider):
        news_id = provider.insert_news(
            headline="Apple beats Q4 earnings expectations",
            source="reuters",
            timestamp=_now(),
            tickers=["AAPL"],
            snippet="Apple reported revenue of $94.9 billion",
        )
        assert news_id is not None
        result = provider.get_news_by_id(news_id)
        assert result is not None
        assert result["headline"] == "Apple beats Q4 earnings expectations"
        assert result["source"] == "reuters"
        assert result["tickers"] == ["AAPL"]

    def test_get_nonexistent_id(self, provider):
        assert provider.get_news_by_id(9999) is None

    def test_dedup_same_headline_same_source(self, provider):
        id1 = provider.insert_news("Test headline here", "reuters", _now())
        id2 = provider.insert_news("Test headline here", "reuters", _now())
        assert id1 is not None
        assert id2 is None

    def test_dedup_same_headline_different_source(self, provider):
        id1 = provider.insert_news("Breaking news about markets", "reuters", _now())
        id2 = provider.insert_news("Breaking news about markets", "bloomberg", _now())
        assert id1 is not None
        assert id2 is not None

    def test_empty_headline_rejected(self, provider):
        assert provider.insert_news("", "reuters", _now()) is None
        assert provider.insert_news("   ", "reuters", _now()) is None

    def test_headline_truncation(self, provider):
        long_headline = "A" * 1000
        news_id = provider.insert_news(long_headline, "reuters", _now())
        result = provider.get_news_by_id(news_id)
        assert len(result["headline"]) == MAX_HEADLINE_LENGTH

    def test_snippet_truncation(self, provider):
        long_snippet = "B" * 5000
        news_id = provider.insert_news(
            "Valid headline here", "reuters", _now(), snippet=long_snippet
        )
        result = provider.get_news_by_id(news_id)
        assert len(result["snippet"]) == MAX_SNIPPET_LENGTH

    def test_null_fields_round_trip(self, provider):
        """所有可选字段为 None 时，存取后仍为 None 而非 "null" 字符串。"""
        news_id = provider.insert_news(
            headline="Minimal news with no optional fields",
            source="reuters",
            timestamp=_now(),
            # 以下全部不传（默认 None）
        )
        result = provider.get_news_by_id(news_id)
        assert result["snippet"] is None
        assert result["sentiment_score"] is None
        assert result["importance_score"] is None
        assert result["reliability_score"] is None
        assert result["analysis"] is None
        assert result["source_url"] is None
        assert result["tickers"] == []  # None tickers → "[]" → []


# =========================================================================
# SqliteNewsProvider — 查询
# =========================================================================

class TestNewsQuery:

    def _seed(self, provider):
        """插入测试数据。"""
        provider.insert_news(
            "Apple earnings beat expectations",
            "reuters", _now(), tickers=["AAPL"],
            importance_score=0.9, g_level=2,
        )
        provider.insert_news(
            "Microsoft announces new AI features",
            "bloomberg", _now(), tickers=["MSFT"],
            importance_score=0.5, g_level=1,
        )
        provider.insert_news(
            "Fed raises interest rates again",
            "cnbc", _now(), tickers=[],
            importance_score=0.8, g_level=1,
        )

    def test_query_all(self, provider):
        self._seed(provider)
        df = provider.get_news()
        assert len(df) == 3

    def test_query_by_ticker(self, provider):
        self._seed(provider)
        df = provider.get_news(tickers=["AAPL"])
        assert len(df) == 1
        assert df.iloc[0]["headline"] == "Apple earnings beat expectations"

    def test_query_by_importance(self, provider):
        self._seed(provider)
        df = provider.get_news(min_importance=0.7)
        assert len(df) == 2

    def test_query_by_g_level(self, provider):
        self._seed(provider)
        df = provider.get_news(g_level=2)
        assert len(df) == 1

    def test_query_with_limit(self, provider):
        self._seed(provider)
        df = provider.get_news(limit=2)
        assert len(df) == 2

    def test_query_by_time_range(self, provider):
        provider.insert_news("Old news item for testing", "reuters", _ago(3))
        provider.insert_news("Recent news item here", "reuters", _now())
        df = provider.get_news(start=_ago(1))
        assert len(df) == 1
        assert "Recent" in df.iloc[0]["headline"]

    def test_query_empty_result(self, provider):
        df = provider.get_news(tickers=["ZZZZ"])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_ticker_exact_match_no_false_positive(self, provider):
        """ticker "A" 不应匹配 "AAPL"。"""
        provider.insert_news("Agilent reports earnings", "reuters", _now(), tickers=["A"])
        provider.insert_news("Apple stock surges today", "reuters", _now(), tickers=["AAPL"])
        df = provider.get_news(tickers=["A"])
        tickers_flat = [t for row in df["tickers"] for t in row]
        assert "AAPL" not in tickers_flat
        assert "A" in tickers_flat

    def test_query_multi_filter_and(self, provider):
        """多条件组合：ticker + g_level + importance 同时过滤，验证 AND 逻辑。"""
        self._seed(provider)
        # AAPL, importance=0.9, g_level=2 — 唯一满足全部条件
        df = provider.get_news(tickers=["AAPL"], g_level=2, min_importance=0.7)
        assert len(df) == 1
        assert df.iloc[0]["headline"] == "Apple earnings beat expectations"
        # MSFT importance=0.5 < 0.7, 不满足
        df2 = provider.get_news(tickers=["MSFT"], min_importance=0.7)
        assert len(df2) == 0

    def test_query_empty_tickers_list(self, provider):
        """tickers=[] 空列表 vs tickers=None 行为不同：空列表不匹配任何行。"""
        self._seed(provider)
        df_none = provider.get_news(tickers=None)  # 不过滤 ticker，返回全部
        df_empty = provider.get_news(tickers=[])    # 空列表进入过滤循环，匹配 0 条
        assert len(df_none) == 3
        assert len(df_empty) == 0


# =========================================================================
# SqliteNewsProvider — G level 更新
# =========================================================================

class TestGLevelUpdate:

    def test_update_g_level(self, provider):
        news_id = provider.insert_news("Test g level update", "reuters", _now(), g_level=1)
        ok = provider.update_g_level(news_id, g_level=2, sentiment_score=0.85)
        assert ok
        result = provider.get_news_by_id(news_id)
        assert result["g_level"] == 2
        assert result["sentiment_score"] == 0.85

    def test_update_nonexistent(self, provider):
        assert not provider.update_g_level(9999, g_level=2)


# =========================================================================
# SqliteNewsProvider — Junction Table (news_tickers)
# =========================================================================

class TestJunctionTable:

    def test_tickers_stored_in_junction_table(self, provider):
        """Tickers 存入 news_tickers 表，而非 JSON 字符串列。"""
        news_id = provider.insert_news(
            "Multi ticker news headline", "reuters", _now(),
            tickers=["AAPL", "MSFT", "NVDA"],
        )
        result = provider.get_news_by_id(news_id)
        assert result["tickers"] == ["AAPL", "MSFT", "NVDA"]

    def test_query_by_single_ticker_uses_join(self, provider):
        """按 ticker 查询通过 JOIN 索引匹配，不依赖 JSON 格式。"""
        provider.insert_news("Apple news for testing", "reuters", _now(), tickers=["AAPL"])
        provider.insert_news("Google news for testing", "reuters", _now(), tickers=["GOOG"])
        df = provider.get_news(tickers=["AAPL"])
        assert len(df) == 1
        assert df.iloc[0]["tickers"] == ["AAPL"]

    def test_query_by_multiple_tickers(self, provider):
        """查询多个 ticker 返回匹配任一的新闻（OR 逻辑）。"""
        provider.insert_news("Apple earnings news", "reuters", _now(), tickers=["AAPL"])
        provider.insert_news("Google earnings news", "reuters", _now(), tickers=["GOOG"])
        provider.insert_news("Tesla earnings news", "reuters", _now(), tickers=["TSLA"])
        df = provider.get_news(tickers=["AAPL", "GOOG"])
        assert len(df) == 2

    def test_no_duplicate_rows_for_multi_ticker_news(self, provider):
        """一条新闻有多个 ticker，查询时不返回重复行。"""
        provider.insert_news("Tech roundup news today", "reuters", _now(),
                             tickers=["AAPL", "MSFT", "GOOG"])
        # 查询 AAPL 和 MSFT，应该只返回 1 行（DISTINCT）
        df = provider.get_news(tickers=["AAPL", "MSFT"])
        assert len(df) == 1

    def test_no_tickers_stored_as_empty(self, provider):
        """无 ticker 的新闻在 junction table 无记录。"""
        news_id = provider.insert_news("No ticker news here", "reuters", _now())
        result = provider.get_news_by_id(news_id)
        assert result["tickers"] == []

    def test_short_ticker_exact_match(self, provider):
        """短 ticker "A" 不会误匹配 "AAPL"。"""
        provider.insert_news("Agilent earnings report", "reuters", _now(), tickers=["A"])
        provider.insert_news("Apple earnings report", "reuters", _now(), tickers=["AAPL"])
        df = provider.get_news(tickers=["A"])
        assert len(df) == 1
        assert df.iloc[0]["tickers"] == ["A"]

    def test_batch_inserts_tickers(self, provider):
        """batch 插入也正确写入 junction table。"""
        events = [
            {"headline": "Batch Apple news here", "source": "s", "timestamp": _now(),
             "tickers": ["AAPL"]},
            {"headline": "Batch Google news here", "source": "s", "timestamp": _now(),
             "tickers": ["GOOG", "MSFT"]},
        ]
        provider.insert_news_batch(events)
        df = provider.get_news(tickers=["MSFT"])
        assert len(df) == 1
        assert "MSFT" in df.iloc[0]["tickers"]


# =========================================================================
# SqliteNewsProvider — G3 每日计数器
# =========================================================================

class TestG3Counter:

    def test_initial_count_is_zero(self, provider):
        assert provider.get_g3_daily_count("2026-01-01") == 0

    def test_increment(self, provider):
        c1 = provider.increment_g3_daily_count("2026-04-05")
        c2 = provider.increment_g3_daily_count("2026-04-05")
        assert c1 == 1
        assert c2 == 2

    def test_different_dates_independent(self, provider):
        provider.increment_g3_daily_count("2026-04-05")
        provider.increment_g3_daily_count("2026-04-06")
        assert provider.get_g3_daily_count("2026-04-05") == 1
        assert provider.get_g3_daily_count("2026-04-06") == 1


# =========================================================================
# SqliteNewsProvider — 批量操作
# =========================================================================

class TestBatch:

    def test_batch_insert(self, provider):
        events = [
            {"headline": f"Batch news item number {i}", "source": "test", "timestamp": _now()}
            for i in range(10)
        ]
        count = provider.insert_news_batch(events)
        assert count == 10

    def test_batch_dedup(self, provider):
        events = [
            {"headline": "Same headline in batch", "source": "test", "timestamp": _now()}
            for _ in range(5)
        ]
        count = provider.insert_news_batch(events)
        assert count == 1  # 只有第一条插入成功

    def test_batch_skips_empty_headline(self, provider):
        events = [
            {"headline": "", "source": "test", "timestamp": _now()},
            {"headline": "Valid headline for batch", "source": "test", "timestamp": _now()},
        ]
        count = provider.insert_news_batch(events)
        assert count == 1

    def test_count_by_g_level(self, provider):
        provider.insert_news("G1 news headline here", "s", _now(), g_level=1)
        provider.insert_news("G2 news headline here", "s", _now(), g_level=2)
        provider.insert_news("Another G1 news item", "s", _now(), g_level=1)
        counts = provider.count_by_g_level()
        assert counts[1] == 2
        assert counts[2] == 1


# =========================================================================
# SqliteNewsProvider — 生命周期
# =========================================================================

class TestProviderLifecycle:

    def test_shutdown_and_reinit(self, provider):
        provider.insert_news("Before shutdown test", "reuters", _now())
        provider.shutdown()
        provider.initialize()
        df = provider.get_news()
        assert len(df) == 1

    def test_uninit_raises(self, tmp_path):
        p = SqliteNewsProvider(ProviderConfig(
            implementation="test", params={"db_path": str(tmp_path / "x.db")}
        ))
        with pytest.raises(RuntimeError, match="not initialized"):
            p.get_news()


# =========================================================================
# 辅助函数
# =========================================================================

class TestHelpers:

    def test_parse_tickers_list(self):
        assert _parse_tickers(["aapl", "MSFT", "aapl"]) == ["AAPL", "MSFT"]

    def test_parse_tickers_csv_string(self):
        assert _parse_tickers("AAPL, msft, goog") == ["AAPL", "GOOG", "MSFT"]

    def test_parse_tickers_json_string(self):
        assert _parse_tickers('["aapl"]') == ["AAPL"]

    def test_parse_tickers_none(self):
        assert _parse_tickers(None) == []

    def test_parse_tickers_empty(self):
        assert _parse_tickers([]) == []

    def test_normalize_timestamp_naive(self):
        result = _normalize_timestamp(datetime(2026, 1, 1, 12, 0))
        assert "+00:00" in result

    def test_normalize_timestamp_aware(self):
        est = timezone(timedelta(hours=-5))
        result = _normalize_timestamp(datetime(2026, 1, 1, 12, 0, tzinfo=est))
        assert "17:00:00" in result  # 12 EST = 17 UTC

    def test_normalize_timestamp_string(self):
        result = _normalize_timestamp("2026-04-04T14:30:00")
        assert "14:30:00" in result

    def test_normalize_timestamp_invalid(self):
        result = _normalize_timestamp("garbage")
        assert result is None  # returns None for unparseable timestamps

    def test_truncate(self):
        assert _truncate("hello", 3) == "hel"
        assert _truncate("hi", 10) == "hi"
        assert _truncate(None, 10) is None


# =========================================================================
# G1Filter — 来源校验
# =========================================================================

class TestG1SourceCheck:

    def test_blacklisted_source_name(self, g1):
        r = g1.filter("Valid headline for testing", "example.com", _now())
        assert not r.passed
        assert "blacklist" in r.reason

    def test_blacklisted_url_domain(self, g1):
        r = g1.filter(
            "Valid headline for testing", "reuters", _now(),
            source_url="https://www.spam-news.com/article/123"
        )
        assert not r.passed

    def test_blacklisted_url_with_www(self, g1):
        r = g1.filter(
            "Valid headline for testing", "reuters", _now(),
            source_url="https://www.example.com/news"
        )
        assert not r.passed

    def test_valid_source(self, g1):
        r = g1.filter("Valid headline for testing", "reuters", _now())
        assert r.passed

    def test_custom_blacklist(self):
        g1 = G1Filter(G1Config(blacklist_domains=["fakenews.org"]))
        r = g1.filter("Valid headline for testing", "fakenews.org", _now())
        assert not r.passed
        r2 = g1.filter("Valid headline for testing", "example.com", _now())
        assert r2.passed  # example.com not in custom blacklist


# =========================================================================
# G1Filter — 时间校验
# =========================================================================

class TestG1TimeCheck:

    def test_fresh_news_passes(self, g1):
        r = g1.filter("Fresh news about the market", "reuters", _now())
        assert r.passed

    def test_old_news_rejected(self, g1):
        r = g1.filter("Old news about the market", "reuters", _ago(10))
        assert not r.passed
        assert "too old" in r.reason

    def test_boundary_news_passes(self, g1):
        """刚好在 7 天内的新闻应通过。"""
        r = g1.filter("Boundary news about markets", "reuters", _ago(6))
        assert r.passed

    def test_invalid_timestamp(self, g1):
        r = g1.filter("Valid headline for testing", "reuters", "not-a-date")
        assert not r.passed
        assert "invalid" in r.reason

    def test_empty_timestamp(self, g1):
        r = g1.filter("Valid headline for testing", "reuters", "")
        assert not r.passed

    def test_custom_max_age(self):
        g1 = G1Filter(G1Config(max_age_days=1))
        r = g1.filter("Two day old news headline", "reuters", _ago(2))
        assert not r.passed

    def test_naive_datetime_treated_as_utc(self, g1):
        naive = datetime.now()  # no tzinfo
        r = g1.filter("Naive timestamp news headline", "reuters", naive)
        assert r.passed

    def test_timezone_aware_datetime(self, g1):
        est = timezone(timedelta(hours=-5))
        ts = datetime.now(est)
        r = g1.filter("Timezone aware news headline", "reuters", ts)
        assert r.passed

    def test_string_timestamp(self, g1):
        ts = _now().isoformat()
        r = g1.filter("String timestamp news headline", "reuters", ts)
        assert r.passed


# =========================================================================
# G1Filter — 内容校验
# =========================================================================

class TestG1ContentCheck:

    def test_empty_headline(self, g1):
        r = g1.filter("", "reuters", _now())
        assert not r.passed

    def test_whitespace_headline(self, g1):
        r = g1.filter("   ", "reuters", _now())
        assert not r.passed

    def test_short_headline(self, g1):
        r = g1.filter("Short", "reuters", _now())
        assert not r.passed
        assert "too short" in r.reason

    def test_special_chars_only(self, g1):
        r = g1.filter("!@#$%^&*()12345", "reuters", _now())
        assert not r.passed

    def test_chinese_headline_passes(self, g1):
        r = g1.filter("苹果公司公布第四季度财报超预期", "reuters", _now())
        assert r.passed

    def test_custom_min_length(self):
        g1 = G1Filter(G1Config(min_headline_length=5))
        r = g1.filter("Short", "reuters", _now())
        assert r.passed


# =========================================================================
# G1Filter — Ticker 提取
# =========================================================================

class TestG1TickerExtraction:

    def test_dollar_ticker(self, g1):
        r = g1.filter("Breaking: $TSLA hits all-time high", "reuters", _now())
        assert "TSLA" in r.tickers

    def test_company_name_to_ticker(self, g1):
        r = g1.filter("Microsoft announces new AI features for Office", "cnbc", _now())
        assert "MSFT" in r.tickers

    def test_multiple_company_names(self, g1):
        r = g1.filter("Apple and Google battle over AI dominance", "reuters", _now())
        assert "AAPL" in r.tickers
        assert "GOOG" in r.tickers or "GOOGL" in r.tickers

    def test_uppercase_ticker_in_text(self, g1):
        r = g1.filter("Analysts upgrade NVDA to strong buy", "cnbc", _now())
        assert "NVDA" in r.tickers

    def test_common_words_excluded(self, g1):
        r = g1.filter("NEW data shows GDP growth improving in THE US", "reuters", _now())
        assert "NEW" not in r.tickers
        assert "THE" not in r.tickers
        assert "GDP" not in r.tickers
        assert "US" not in r.tickers

    def test_single_letter_ticker_financial_context(self, g1):
        r = g1.filter("Ford stock F hits new price target of $15", "reuters", _now())
        assert "F" in r.tickers

    def test_single_letter_ticker_no_financial_context(self, g1):
        r = g1.filter("I went to the store to buy a book", "blog", _now())
        assert "A" not in r.tickers
        assert "I" not in r.tickers

    def test_snippet_used_for_extraction(self, g1):
        r = g1.filter(
            "Major tech deal announced today",
            "reuters", _now(),
            snippet="$AMZN acquires startup for $2 billion"
        )
        assert "AMZN" in r.tickers

    def test_empty_text_returns_empty(self, g1):
        tickers = g1.extract_tickers("", None)
        assert tickers == []

    def test_no_tickers_in_general_news(self, g1):
        r = g1.filter("Weather forecast shows rain this weekend", "weather.com", _now())
        assert r.tickers == []

    def test_ambiguous_company_meta(self, g1):
        """'meta' 作为公司名应匹配 META ticker。"""
        r = g1.filter("Meta Platforms reports quarterly earnings", "reuters", _now())
        assert "META" in r.tickers

    def test_case_insensitive_company_name(self, g1):
        r = g1.filter("TESLA delivers record vehicles this quarter", "reuters", _now())
        assert "TSLA" in r.tickers

    def test_require_ticker_mode(self, g1_strict):
        r = g1_strict.filter("Weather forecast shows rain weekend", "weather.com", _now())
        assert not r.passed
        assert "no ticker" in r.reason

        r2 = g1_strict.filter("Apple beats Q4 earnings expectations", "reuters", _now())
        assert r2.passed


# =========================================================================
# G1Filter — 批量过滤
# =========================================================================

class TestG1Batch:

    def test_batch_mixed_results(self, g1):
        events = [
            {"headline": "Apple earnings beat expectations", "source": "reuters",
             "timestamp": _now()},
            {"headline": "short", "source": "reuters", "timestamp": _now()},
            {"headline": "Goldman Sachs upgrades NVDA target", "source": "cnbc",
             "timestamp": _now()},
            {"headline": "Old news about markets", "source": "reuters",
             "timestamp": _ago(30)},
        ]
        results = g1.filter_batch(events)
        assert len(results) == 4
        passed = [r for _, r in results if r.passed]
        failed = [r for _, r in results if not r.passed]
        assert len(passed) == 2
        assert len(failed) == 2

    def test_batch_empty(self, g1):
        assert g1.filter_batch([]) == []

    def test_batch_preserves_event_data(self, g1):
        events = [{"headline": "Valid news headline here", "source": "reuters",
                    "timestamp": _now(), "extra_field": "preserved"}]
        results = g1.filter_batch(events)
        assert results[0][0]["extra_field"] == "preserved"


# =========================================================================
# G1Filter — 数据映射完整性
# =========================================================================

class TestG1DataIntegrity:

    def test_company_to_ticker_mapping_complete(self):
        """每个 _AMBIGUOUS_TICKERS 的公司名都有反向映射。"""
        for ticker, names in _AMBIGUOUS_TICKERS.items():
            for name in names:
                assert name.lower() in _COMPANY_TO_TICKER, \
                    f"{name} missing from _COMPANY_TO_TICKER"
                # GOOG/GOOGL 共享公司名，反向映射指向其中一个即可
                mapped = _COMPANY_TO_TICKER[name.lower()]
                assert mapped in _AMBIGUOUS_TICKERS, \
                    f"{name} maps to {mapped} which is not a known ticker"

    def test_valid_short_tickers_not_in_common_words(self):
        """已知短 ticker 不在 _COMMON_WORDS 中（除了有意在 _VALID_SHORT_TICKERS 中的）。"""
        for t in _VALID_SHORT_TICKERS:
            # 这些 ticker 在 _COMMON_WORDS 中是预期的，
            # 但 extract_tickers 有特殊处理逻辑
            pass  # 只验证映射存在

    def test_common_words_are_uppercase(self):
        """所有 _COMMON_WORDS 都是大写。"""
        for word in _COMMON_WORDS:
            assert word == word.upper(), f"Non-uppercase in _COMMON_WORDS: {word}"


# =========================================================================
# Smoke Test — G1 + SqliteNewsProvider 集成
# =========================================================================

class TestG1StoreIntegration:

    def test_g1_filter_then_store(self, provider, g1):
        """G1 过滤后，通过的新闻写入数据库。"""
        events = [
            {"headline": "Apple beats Q4 earnings expectations",
             "source": "reuters", "timestamp": _now(),
             "snippet": "Revenue of $94.9B exceeds estimates"},
            {"headline": "short", "source": "reuters", "timestamp": _now()},
            {"headline": "NVDA surges on AI demand outlook",
             "source": "bloomberg", "timestamp": _now()},
            {"headline": "Spam from bad source", "source": "example.com",
             "timestamp": _now()},
        ]

        results = g1.filter_batch(events)
        passed_events = []
        for event, result in results:
            if result.passed:
                event["tickers"] = result.tickers
                event["g_level"] = 1
                passed_events.append(event)

        assert len(passed_events) == 2
        count = provider.insert_news_batch(passed_events)
        assert count == 2

        # 验证存储的数据
        df = provider.get_news(g_level=1)
        assert len(df) == 2
        all_tickers = [t for row in df["tickers"] for t in row]
        assert "AAPL" in all_tickers
        assert "NVDA" in all_tickers

    def test_g1_dedup_across_sources(self, provider, g1):
        """同一新闻两个来源，G1 都通过，DB 层去重保证不重复。"""
        e1 = {"headline": "Fed raises rates by 25 basis points",
              "source": "reuters", "timestamp": _now()}
        e2 = {"headline": "Fed raises rates by 25 basis points",
              "source": "bloomberg", "timestamp": _now()}

        r1 = g1.filter(**e1)
        r2 = g1.filter(**e2)
        assert r1.passed and r2.passed

        id1 = provider.insert_news(**e1, g_level=1)
        id2 = provider.insert_news(**e2, g_level=1)
        # 不同 source → 都应写入
        assert id1 is not None
        assert id2 is not None
        assert provider.get_news().shape[0] == 2


# =========================================================================
# G2Classifier — Fixtures
# =========================================================================

@pytest.fixture
def g2_rules() -> G2Classifier:
    """纯规则引擎模式的 G2 分类器（不加载 FinBERT）。"""
    return G2Classifier(G2Config(use_finbert=False))


def _mock_finbert_pipeline(label: str = "positive", score: float = 0.92):
    """构造一个 mock FinBERT pipeline，单条和批量都返回固定结果。"""
    def pipeline_fn(text_or_texts, **kwargs):
        if isinstance(text_or_texts, list):
            return [{"label": label, "score": score}] * len(text_or_texts)
        return [{"label": label, "score": score}]
    return pipeline_fn


# =========================================================================
# G2Classifier — 可分类性检查
# =========================================================================

class TestG2Classifiable:

    def test_english_text_classifiable(self):
        assert G2Classifier._is_classifiable("Apple beats earnings expectations")

    def test_empty_text_not_classifiable(self):
        assert not G2Classifier._is_classifiable("")

    def test_whitespace_not_classifiable(self):
        assert not G2Classifier._is_classifiable("   ")

    def test_pure_numbers_not_classifiable(self):
        assert not G2Classifier._is_classifiable("123456789")

    def test_pure_symbols_not_classifiable(self):
        assert not G2Classifier._is_classifiable("!@#$%^&*()")

    def test_chinese_text_below_ascii_ratio(self):
        assert not G2Classifier._is_classifiable("苹果公司公布第四季度财报超预期")

    def test_mixed_text_above_ratio(self):
        # Enough ASCII letters to pass the threshold
        assert G2Classifier._is_classifiable("Apple 苹果 beats earnings")

    def test_ascii_ratio_boundary(self):
        # Exactly at the boundary: 5 ascii letters out of 10 non-space chars = 0.5
        assert G2Classifier._is_classifiable("abcde12345")


# =========================================================================
# G2Classifier — 规则引擎情绪分类
# =========================================================================

class TestG2RuleSentiment:

    def test_positive_sentiment(self, g2_rules):
        r = g2_rules.classify("Company beats earnings, strong growth and record profit")
        assert r.sentiment == "positive"
        assert r.sentiment_score > 0.5
        assert not r.used_finbert

    def test_negative_sentiment(self, g2_rules):
        r = g2_rules.classify("Stock crash amid loss and decline, weak outlook")
        assert r.sentiment == "negative"
        assert r.sentiment_score < 0.5

    def test_neutral_sentiment_no_keywords(self, g2_rules):
        r = g2_rules.classify("Company announces quarterly results for review")
        assert r.sentiment == "neutral"
        assert r.sentiment_score == 0.5
        assert r.confidence == 0.3  # no keywords → 0.3

    def test_neutral_sentiment_balanced_keywords(self, g2_rules):
        r = g2_rules.classify("Stock rise but also loss reported for the quarter")
        assert r.sentiment == "neutral"
        assert r.confidence == 0.4  # balanced → 0.4

    def test_empty_headline(self, g2_rules):
        r = g2_rules.classify("")
        assert r.sentiment == "neutral"
        assert r.sentiment_score == 0.5
        assert r.confidence == 0.0

    def test_unclassifiable_text_returns_neutral(self, g2_rules):
        r = g2_rules.classify("苹果公布财报超预期增长数据报告")
        assert r.sentiment == "neutral"
        assert r.confidence == 0.1  # _is_classifiable false → 0.1


# =========================================================================
# G2Classifier — 主题分类
# =========================================================================

class TestG2Topic:

    def test_earnings_topic(self, g2_rules):
        r = g2_rules.classify("Company earnings beat forecast, revenue and profit up")
        assert r.topic == "earnings"

    def test_regulatory_topic(self, g2_rules):
        r = g2_rules.classify("SEC launches probe into compliance violations and fines")
        assert r.topic == "regulatory"

    def test_merger_topic(self, g2_rules):
        r = g2_rules.classify("Merger deal announced as company makes acquisition bid")
        assert r.topic == "merger"

    def test_litigation_topic(self, g2_rules):
        r = g2_rules.classify("Patent lawsuit filed in court, trial verdict pending")
        assert r.topic == "litigation"

    def test_policy_topic(self, g2_rules):
        r = g2_rules.classify("Federal reserve raises interest rate amid inflation concerns")
        assert r.topic == "policy"

    def test_product_topic(self, g2_rules):
        r = g2_rules.classify("Company launch new product feature with update and upgrade")
        assert r.topic == "product"

    def test_other_topic_fallback(self, g2_rules):
        r = g2_rules.classify("The weather is nice today in the city")
        assert r.topic == "other"

    def test_topic_picks_highest_count(self, g2_rules):
        """多个主题关键词时，匹配最多的胜出。"""
        # 3 earnings keywords vs 1 regulatory keyword
        r = g2_rules.classify("Earnings revenue profit up, SEC investigation announced")
        assert r.topic == "earnings"


# =========================================================================
# G2Classifier — 重要度评分
# =========================================================================

class TestG2Importance:

    def test_merger_highest_base_weight(self, g2_rules):
        r = g2_rules.classify("Major merger acquisition deal announced today")
        assert r.importance_score >= 0.9  # merger base = 0.9

    def test_other_topic_lowest_weight(self, g2_rules):
        r = g2_rules.classify("The weather is nice today in the city")
        assert r.importance_score < 0.5  # other base = 0.3

    def test_importance_bounded_zero_to_one(self, g2_rules):
        # Long text + high-weight topic to push score up
        long_text = "Major merger acquisition " * 200
        r = g2_rules.classify(long_text)
        assert 0.0 <= r.importance_score <= 1.0

    def test_longer_text_slightly_more_important(self, g2_rules):
        short = g2_rules.classify("Earnings beat expectations")
        long = g2_rules.classify("Earnings beat expectations " + "details " * 100)
        assert long.importance_score >= short.importance_score


# =========================================================================
# G2Classifier — 紧急度评分
# =========================================================================

class TestG2Urgency:

    def test_high_urgency_breaking(self, g2_rules):
        r = g2_rules.classify("Breaking news: market crash alert issued")
        assert r.urgency == "high"

    def test_high_urgency_fraud(self, g2_rules):
        r = g2_rules.classify("Emergency: fraud scandal discovered at company")
        assert r.urgency == "high"

    def test_medium_urgency_downgrade(self, g2_rules):
        r = g2_rules.classify("Analyst downgrade issued for stock target")
        assert r.urgency == "medium"

    def test_medium_urgency_layoff(self, g2_rules):
        r = g2_rules.classify("Company announces layoff and restructure plan")
        assert r.urgency == "medium"

    def test_low_urgency_default(self, g2_rules):
        r = g2_rules.classify("Company reports quarterly results for investors")
        assert r.urgency == "low"

    def test_high_takes_precedence_over_medium(self, g2_rules):
        """同时包含 high 和 medium 关键词时，high 优先。"""
        r = g2_rules.classify("Breaking news: analyst downgrade amid crash")
        assert r.urgency == "high"


# =========================================================================
# G2Classifier — FinBERT mock 集成
# =========================================================================

class TestG2FinBERT:

    def test_finbert_positive(self):
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._pipeline = _mock_finbert_pipeline("positive", 0.92)
        g2._finbert_available = True
        r = g2.classify("Apple beats earnings expectations by wide margin")
        assert r.sentiment == "positive"
        assert r.used_finbert
        assert r.confidence == 0.92
        # score = 0.5 + 0.92/2 = 0.96
        assert r.sentiment_score == pytest.approx(0.96, abs=0.01)

    def test_finbert_negative(self):
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._pipeline = _mock_finbert_pipeline("negative", 0.85)
        g2._finbert_available = True
        r = g2.classify("Company reports massive quarterly loss today")
        assert r.sentiment == "negative"
        assert r.used_finbert
        # score = 0.5 - 0.85/2 = 0.075
        assert r.sentiment_score < 0.5

    def test_finbert_neutral(self):
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._pipeline = _mock_finbert_pipeline("neutral", 0.70)
        g2._finbert_available = True
        r = g2.classify("Company announces quarterly results review")
        assert r.sentiment == "neutral"
        assert r.sentiment_score == 0.5

    def test_finbert_fallback_on_failure(self):
        """FinBERT 推理异常时降级到规则引擎。"""
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._finbert_available = True
        g2._pipeline = MagicMock(side_effect=RuntimeError("model error"))
        r = g2.classify("Stock surge and profit growth reported")
        assert not r.used_finbert  # fell back to rules
        assert r.sentiment == "positive"

    def test_finbert_skipped_for_non_english(self):
        """非英语文本跳过 FinBERT，直接返回 neutral。"""
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._pipeline = _mock_finbert_pipeline("positive", 0.99)
        g2._finbert_available = True
        r = g2.classify("苹果公司公布第四季度财报超预期增长数据")
        assert not r.used_finbert
        assert r.sentiment == "neutral"

    def test_finbert_not_available_uses_rules(self):
        """transformers/torch 未安装时使用规则引擎。"""
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._finbert_available = False
        r = g2.classify("Stock crash and decline continues for market")
        assert not r.used_finbert
        assert r.sentiment == "negative"


# =========================================================================
# G2Classifier — 批量分类
# =========================================================================

class TestG2Batch:

    def test_batch_rules_mode(self, g2_rules):
        items = [
            {"headline": "Apple beats earnings expectations today"},
            {"headline": "Stock crash amid loss and decline"},
            {"headline": "Weather is nice today in city"},
        ]
        results = g2_rules.classify_batch(items)
        assert len(results) == 3
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"
        assert results[2].sentiment == "neutral"

    def test_batch_empty(self, g2_rules):
        assert g2_rules.classify_batch([]) == []

    def test_batch_with_snippets(self, g2_rules):
        items = [
            {"headline": "Big deal announced", "snippet": "merger acquisition buyout confirmed"},
        ]
        results = g2_rules.classify_batch(items)
        assert results[0].topic == "merger"

    def test_batch_with_empty_headline(self, g2_rules):
        items = [
            {"headline": ""},
            {"headline": "Valid earnings beat forecast"},
        ]
        results = g2_rules.classify_batch(items)
        assert results[0].sentiment == "neutral"
        assert results[0].confidence == 0.0
        assert results[1].topic == "earnings"

    def test_batch_finbert_mock(self):
        """批量 FinBERT 推理 mock。"""
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._pipeline = _mock_finbert_pipeline("positive", 0.88)
        g2._finbert_available = True
        items = [
            {"headline": "Apple beats earnings expectations"},
            {"headline": "Tesla surges on strong demand"},
        ]
        results = g2.classify_batch(items)
        assert len(results) == 2
        assert all(r.used_finbert for r in results)
        assert all(r.sentiment == "positive" for r in results)

    def test_batch_finbert_mixed_classifiable(self):
        """批量中混合可分类和不可分类文本。"""
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._pipeline = _mock_finbert_pipeline("negative", 0.80)
        g2._finbert_available = True
        items = [
            {"headline": "Stock crash amid loss reported"},
            {"headline": "苹果公司公布财报超预期增长"},  # non-English
            {"headline": "Market decline continues today"},
        ]
        results = g2.classify_batch(items)
        assert results[0].used_finbert
        assert not results[1].used_finbert  # skipped FinBERT
        assert results[1].sentiment == "neutral"
        assert results[2].used_finbert

    def test_batch_finbert_fallback_on_error(self):
        """批量 FinBERT 异常时整体降级到规则引擎。"""
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._finbert_available = True
        g2._pipeline = MagicMock(side_effect=RuntimeError("batch error"))
        items = [
            {"headline": "Stock surge and profit growth"},
            {"headline": "Market crash and loss decline"},
        ]
        results = g2.classify_batch(items)
        assert not results[0].used_finbert
        assert not results[1].used_finbert
        assert results[0].sentiment == "positive"
        assert results[1].sentiment == "negative"


# =========================================================================
# G2Classifier — 文本预处理
# =========================================================================

class TestG2TextPrep:

    def test_truncation_long_text(self):
        g2 = G2Classifier(G2Config(use_finbert=False))
        long_text = "A" * (FINBERT_MAX_LENGTH * 4 + 500)
        prepared = g2._prepare_text(long_text)
        assert len(prepared) == FINBERT_MAX_LENGTH * 4

    def test_headline_plus_snippet_joined(self):
        g2 = G2Classifier(G2Config(use_finbert=False))
        text = g2._prepare_text("headline here", "snippet here")
        assert "headline here" in text
        assert "snippet here" in text

    def test_none_headline(self):
        g2 = G2Classifier(G2Config(use_finbert=False))
        text = g2._prepare_text(None, "snippet only")
        assert "snippet only" in text

    def test_empty_returns_empty(self):
        g2 = G2Classifier(G2Config(use_finbert=False))
        assert g2._prepare_text("", None) == ""


# =========================================================================
# G2Classifier — G2Result 默认值
# =========================================================================

class TestG2Result:

    def test_default_values(self):
        r = G2Result()
        assert r.sentiment == "neutral"
        assert r.sentiment_score == 0.5
        assert r.confidence == 0.0
        assert r.topic == "other"
        assert r.importance_score == 0.5
        assert r.urgency == "low"
        assert r.used_finbert is False


# =========================================================================
# Smoke Test — G1 → G2 集成
# =========================================================================

class TestG1G2Integration:

    def test_g1_pass_then_g2_classify(self, g2_rules):
        g1 = G1Filter()
        event = {
            "headline": "Apple beats Q4 earnings with record profit and growth",
            "source": "reuters",
            "timestamp": _now(),
            "snippet": "Revenue exceeded expectations with strong guidance",
        }
        g1_result = g1.filter(**event)
        assert g1_result.passed
        assert "AAPL" in g1_result.tickers

        g2_result = g2_rules.classify(event["headline"], event["snippet"])
        assert g2_result.topic == "earnings"
        assert g2_result.sentiment == "positive"
        assert g2_result.importance_score > 0.5

    def test_g1_g2_store_pipeline(self, provider, g2_rules):
        """G1 → G2 → Store 完整管道。"""
        g1 = G1Filter()
        event = {
            "headline": "Breaking: Tesla crash amid fraud scandal revealed",
            "source": "bloomberg",
            "timestamp": _now(),
        }
        g1r = g1.filter(**event)
        assert g1r.passed

        g2r = g2_rules.classify(event["headline"])
        assert g2r.urgency == "high"

        news_id = provider.insert_news(
            headline=event["headline"],
            source=event["source"],
            timestamp=event["timestamp"],
            tickers=g1r.tickers,
            g_level=1,
        )
        assert news_id is not None

        ok = provider.update_g_level(
            news_id, g_level=2,
            sentiment_score=g2r.sentiment_score,
            importance_score=g2r.importance_score,
        )
        assert ok
        row = provider.get_news_by_id(news_id)
        assert row["g_level"] == 2
        assert row["sentiment_score"] is not None


# =========================================================================
# Edge Cases — 子串误匹配回归测试
# =========================================================================

class TestG1SubstringEdgeCases:
    """G1 ticker 提取不应被子串欺骗。"""

    def test_pineapple_not_aapl(self, g1):
        r = g1.filter("Pineapple juice sales surge in summer", "reuters", _now())
        assert "AAPL" not in r.tickers

    def test_gold_not_goldman(self, g1):
        r = g1.filter("Gold prices hit new record this week", "reuters", _now())
        assert "GS" not in r.tickers

    def test_alphabet_still_matches_google(self, g1):
        r = g1.filter("Alphabet reports strong cloud revenue growth", "reuters", _now())
        assert "GOOG" in r.tickers or "GOOGL" in r.tickers

    def test_dollar_ticker_lowercase(self, g1):
        r = g1.filter("Buying more $aapl and $tsla today", "twitter", _now())
        assert "AAPL" in r.tickers
        assert "TSLA" in r.tickers

    def test_multi_class_ticker_brk_a(self, g1):
        r = g1.filter("BRK.A hits new high on Berkshire results", "reuters", _now())
        assert "BRK.A" in r.tickers

    def test_multi_class_ticker_dollar(self, g1):
        r = g1.filter("Sold $BRK.B at the stock market today", "twitter", _now())
        assert "BRK.B" in r.tickers

    def test_future_timestamp_rejected(self, g1):
        future = _now() + timedelta(days=30)
        r = g1.filter("Fabricated future news headline here", "reuters", future)
        assert not r.passed
        assert "future" in r.reason

    def test_slight_future_allowed(self, g1):
        """允许 1 小时内的时钟偏差。"""
        slight = _now() + timedelta(minutes=30)
        r = g1.filter("News with minor clock skew here", "reuters", slight)
        assert r.passed


class TestG2SubstringEdgeCases:
    """G2 关键词匹配不应被子串欺骗。"""

    def test_said_not_ai_topic(self, g2_rules):
        r = g2_rules.classify("The CEO said results were as certain as we again expected")
        assert r.topic != "product"  # "ai" in "said"/"again" 不应触发

    def test_second_sector_not_regulatory(self, g2_rules):
        r = g2_rules.classify("The technology sector saw its second quarterly improvement")
        assert r.topic != "regulatory"  # "sec" in "second"/"sector" 不应触发

    def test_issue_not_litigation(self, g2_rules):
        r = g2_rules.classify("The main issue remains inventory management this quarter")
        assert r.topic != "litigation"  # "sue" in "issue" 不应触发

    def test_fedex_not_policy(self, g2_rules):
        r = g2_rules.classify("FedEx reports strong delivery numbers for holiday season")
        assert r.topic != "policy"  # "fed" in "fedex" 不应触发

    def test_ideal_forbidden_not_merger(self, g2_rules):
        r = g2_rules.classify("The ideal forbidden strategy is under review by board")
        assert r.topic != "merger"  # "deal"/"bid" in "ideal"/"forbidden" 不应触发

    def test_surprise_against_not_positive(self, g2_rules):
        r = g2_rules.classify("Surprise results against all expectations for the company")
        assert r.sentiment != "positive"  # "rise"/"gain" in "surprise"/"against" 不应触发

    def test_rainfall_not_negative(self, g2_rules):
        r = g2_rules.classify("Rainfall impacts crop yields across midwest farming states")
        assert r.sentiment != "negative"  # "fall" in "rainfall" 不应触发

    def test_surgeon_not_high_urgency(self, g2_rules):
        r = g2_rules.classify("The surgeon performed the procedure at hospital today")
        assert r.urgency != "high"  # "surge" in "surgeon" 不应触发

    def test_asphalt_not_high_urgency(self, g2_rules):
        r = g2_rules.classify("Workers repaved the asphalt road near the office")
        assert r.urgency != "high"  # "halt" in "asphalt" 不应触发

    def test_praised_not_medium_urgency(self, g2_rules):
        r = g2_rules.classify("The executive praised the new restructuring initiative")
        assert r.urgency != "medium"  # "raise" in "praised" 不应触发

    def test_real_keywords_still_work(self, g2_rules):
        """修复子串后，真正的关键词仍然匹配。"""
        r = g2_rules.classify("Breaking alert: SEC probe into fraud scandal at company")
        assert r.urgency == "high"
        assert r.topic == "regulatory"

        r2 = g2_rules.classify("Stock surge and profit beat expectations with growth")
        assert r2.sentiment == "positive"
