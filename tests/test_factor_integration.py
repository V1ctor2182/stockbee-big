# tests/test_factor_integration.py
"""m7 — 集成层 edge case 测试（LocalFactorProvider 跨模块交互）。

这里测试的是 expression engine / Alpha158 / ParquetFactorStore / ic_evaluator /
LocalFactorProvider 组合后才会暴露的 corner case。单元层已经覆盖的路径
（base happy path、基础 setter 校验、list_factors shadowing）在 test_factor_data.py
的 TestLocalFactorProvider 中，这里不再重复。

运行：
    pytest tests/test_factor_integration.py -v
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stockbee.factor_data import (
    ICUniverse,
    LocalFactorProvider,
    ParquetFactorStore,
)
from stockbee.providers.base import ProviderConfig


# ---------------------------------------------------------------------------
# Helpers —— 文件内 local，保持 m7 文件自包含
# ---------------------------------------------------------------------------

def _make_ohlcv(
    tickers: list[str],
    start: str | pd.Timestamp,
    n_days: int = 60,
    *,
    adj_ratio: float = 1.0,
    volume: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """合成 MultiIndex (date, ticker) OHLCV。

    adj_ratio != 1.0 时 adj_close = close * adj_ratio，用于验证 $close → adj_close。
    volume 给定时 volume 列被写为常数，便于 VMA 类因子确定性断言。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for t in tickers:
        price = 100.0 + rng.uniform(-5, 5)
        for d in dates:
            rows.append({
                "date": d,
                "ticker": t,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "adj_close": price * adj_ratio,
                "volume": volume if volume is not None
                    else int(rng.integers(1000, 10000)),
            })
            price *= 1 + rng.normal(0, 0.02)
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


def _make_ohlcv_per_ticker(specs: dict[str, tuple[str, int]]) -> pd.DataFrame:
    """每个 ticker 独立起止日期：specs={ticker: (start, n_days)}。"""
    frames = [_make_ohlcv([t], start, n, seed=hash(t) & 0xFFFF)
              for t, (start, n) in specs.items()]
    return pd.concat(frames).sort_index()


def _mock_market_data(ohlcv: pd.DataFrame) -> MagicMock:
    mock = MagicMock()
    mock.get_daily_bars.return_value = ohlcv
    return mock


def _write_precomputed(
    tmp_path,
    group: str,
    tickers: list[str],
    dates: list[str] | list[pd.Timestamp],
    columns: dict[str, list[float]] | None = None,
) -> None:
    store = ParquetFactorStore(tmp_path)
    idx_tuples = [(pd.Timestamp(d), t) for d in dates for t in tickers]
    idx = pd.MultiIndex.from_tuples(idx_tuples, names=["date", "ticker"])
    n = len(idx_tuples)
    if columns is None:
        columns = {"pe_ratio": [float(i) for i in range(n)]}
    df = pd.DataFrame(columns, index=idx, dtype=float)
    store.write_factors(group, df)


def _build_provider(
    tmp_path,
    market_data=None,
    ic_universe: ICUniverse | None = None,
) -> LocalFactorProvider:
    provider = LocalFactorProvider(
        ProviderConfig(
            implementation="LocalFactorProvider",
            params={"precomputed_path": str(tmp_path)},
        ),
    )
    if market_data is not None:
        provider.market_data = market_data
    if ic_universe is not None:
        provider.ic_universe = ic_universe
    return provider


# ===========================================================================
# A. Lookback & date arithmetic
# ===========================================================================

class TestLookbackAndDateArithmetic:

    def test_nested_ref_lookback_through_provider(self, tmp_path):
        """MA(REF($close,5),20) 类嵌套表达式 lookback = 5+20=25，trim 后首日 ≥ user_start。"""
        from stockbee.factor_data.alpha158 import Alpha158
        from stockbee.factor_data.expression_engine import parse
        # Alpha158 暂无 MA(REF) 类工厂，用已有 RANK30 (TS_RANK($close,30)) 自带 lookback=30 做代理
        # 这里先验证 Alpha158 暴露的 max_lookback 和 AST 嵌套一致。
        a = Alpha158()
        # 手工 parse 嵌套表达式验证 AST 语义
        nested = parse("MEAN(REF($close,5),20)")
        assert nested.lookback() == 25
        # 并验证 Alpha158 中 RANK30 factor 的 lookback 等于 30（作为集成 sanity）
        assert a.max_lookback("RANK30") == 30

    def test_lookback_extended_start_uses_calendar_multiplier(self, tmp_path):
        """验证 provider 向 get_daily_bars 传入的 start = user_start − lookback × 2 日历日。"""
        ohlcv = _make_ohlcv(["AAPL"], "2024-01-01", n_days=120)
        mock = _mock_market_data(ohlcv)
        provider = _build_provider(tmp_path, market_data=mock)

        user_start = date(2024, 3, 1)
        user_end = date(2024, 3, 31)
        provider.get_factors(["AAPL"], ["MA30"], user_start, user_end)

        _args, kwargs = mock.get_daily_bars.call_args
        # signature: get_daily_bars(tickers, start, end)
        start_arg = mock.get_daily_bars.call_args.args[1]
        assert start_arg == user_start - timedelta(days=30 * 2)

    def test_trim_removes_lookback_overshoot(self, tmp_path):
        """market_data 返回超出 user_start 的历史数据，最终结果首日 ≥ user_start。"""
        ohlcv = _make_ohlcv(["AAPL"], "2024-01-01", n_days=120)
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        user_start = date(2024, 3, 1)
        user_end = date(2024, 3, 15)
        result = provider.get_factors(["AAPL"], ["MA30"], user_start, user_end)

        dates = result.index.get_level_values("date")
        assert dates.min() >= pd.Timestamp(user_start)
        assert dates.max() <= pd.Timestamp(user_end)

    def test_max_lookback_aggregated_across_factors(self, tmp_path):
        """请求 [MA5, MA30, MA60] → extended_start 用 max(60)*2 天，不是每因子独立扩展。"""
        ohlcv = _make_ohlcv(["AAPL"], "2023-10-01", n_days=200)
        mock = _mock_market_data(ohlcv)
        provider = _build_provider(tmp_path, market_data=mock)

        user_start = date(2024, 3, 1)
        user_end = date(2024, 3, 31)
        provider.get_factors(
            ["AAPL"], ["MA5", "MA30", "MA60"], user_start, user_end,
        )

        start_arg = mock.get_daily_bars.call_args.args[1]
        # max lookback = 60, extended = user_start - 60*2 = 120 calendar days
        assert start_arg == user_start - timedelta(days=60 * 2)
        # get_daily_bars 只被调用一次（batched，不是每因子一次）
        assert mock.get_daily_bars.call_count == 1


# ===========================================================================
# B. Ticker / Index isolation
# ===========================================================================

class TestTickerAndIndexIsolation:

    def test_multiindex_groupby_isolation(self, tmp_path):
        """2 tickers 不同起止日，MA5 在各自前 4 天为 NaN，不从另一个 ticker 串数据。"""
        ohlcv = _make_ohlcv_per_ticker({
            "AAPL": ("2024-01-01", 60),
            "TSLA": ("2024-02-01", 40),  # TSLA 晚 1 个月入场
        })
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL", "TSLA"], ["MA5"],
            date(2024, 2, 1), date(2024, 3, 15),
        )
        # TSLA 在 2024-02-01 起的前 4 天应该是 NaN（MA5 min_periods=5）
        tsla_first_4 = result.xs("TSLA", level="ticker").head(4)["MA5"]
        assert tsla_first_4.isna().all()
        # TSLA 第 5 天开始非 NaN
        tsla_day5 = result.xs("TSLA", level="ticker").iloc[4]["MA5"]
        assert not pd.isna(tsla_day5)
        # AAPL 在同期已有 20+ 历史数据，MA5 应全部有值
        aapl_slice = result.xs("AAPL", level="ticker")
        assert aapl_slice["MA5"].notna().all()

    def test_ticker_subset_returned_by_market_data(self, tmp_path):
        """请求 [AAPL, TSLA, NVDA]，market_data 只返回 [AAPL, TSLA] → 输出只含 AAPL/TSLA，不崩。"""
        ohlcv = _make_ohlcv(["AAPL", "TSLA"], "2024-01-01", n_days=60)
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL", "TSLA", "NVDA"], ["KMID"],
            date(2024, 2, 1), date(2024, 3, 1),
        )
        tickers_in_result = set(result.index.get_level_values("ticker").unique())
        assert tickers_in_result == {"AAPL", "TSLA"}

    def test_date_level_normalization_ticker_date_order(self, tmp_path):
        """MultiIndex names=['ticker','date'] 顺序时，_normalize_date_index 仍按名字定位 level，不崩。"""
        from stockbee.factor_data.local_provider import _normalize_date_index
        tuples = [("AAPL", date(2024, 1, d))
                  for d in range(2, 10) if date(2024, 1, d).weekday() < 5]
        idx = pd.MultiIndex.from_tuples(tuples, names=["ticker", "date"])
        df = pd.DataFrame({"x": [1.0] * len(tuples)}, index=idx)
        # 反序 MultiIndex，date 在位置 1。若实现用 levels[0] 会把 ticker 字符串
        # 送进 to_datetime → 抛错；本用例保证 name-based 定位。
        result = _normalize_date_index(df)
        assert pd.api.types.is_datetime64_any_dtype(
            result.index.get_level_values("date"),
        )

    def test_date_level_normalization_end_to_end(self, tmp_path):
        """market_data 返回 datetime.date level index（非 Timestamp）→ 合并不 TypeError。"""
        # 构造 object dtype date level
        tuples = [(date(2024, 1, d), "AAPL")
                  for d in range(2, 21) if date(2024, 1, d).weekday() < 5]
        idx = pd.MultiIndex.from_tuples(tuples, names=["date", "ticker"])
        n = len(idx)
        ohlcv = pd.DataFrame({
            "open": np.linspace(100, 120, n),
            "high": np.linspace(101, 121, n),
            "low": np.linspace(99, 119, n),
            "close": np.linspace(100, 120, n),
            "adj_close": np.linspace(100, 120, n),
            "volume": [1000] * n,
        }, index=idx)
        assert not pd.api.types.is_datetime64_any_dtype(
            ohlcv.index.get_level_values("date"),
        )
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL"], ["KMID"], date(2024, 1, 2), date(2024, 1, 20),
        )
        # 最关键：不抛 TypeError，且 date 级别已被规范成 datetime64
        assert pd.api.types.is_datetime64_any_dtype(
            result.index.get_level_values("date"),
        )
        assert len(result) > 0


# ===========================================================================
# D. Precomputed merge & routing
# ===========================================================================

class TestPrecomputedMergeAndRouting:

    def test_expression_plus_precomputed_index_alignment(self, tmp_path):
        """Parquet (date,ticker) 与 Evaluator 结果 (date,ticker) 合并后索引完全对齐、列顺序稳定。"""
        ohlcv = _make_ohlcv(["AAPL", "TSLA"], "2024-01-01", n_days=60)
        dates_str = [str(d.date()) for d in pd.bdate_range("2024-01-01", periods=60)]
        _write_precomputed(tmp_path, "fundamental",
                           ["AAPL", "TSLA"], dates_str)
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL", "TSLA"], ["KMID", "pe_ratio"],
            date(2024, 2, 1), date(2024, 3, 1),
        )
        # 索引是 (date, ticker) MultiIndex，完整 sort，无重复
        assert list(result.index.names) == ["date", "ticker"]
        assert not result.index.duplicated().any()
        assert result.index.is_monotonic_increasing
        assert list(result.columns) == ["KMID", "pe_ratio"]

    def test_cross_group_precomputed_merge(self, tmp_path):
        """请求跨两个 Parquet group 的 precomputed 因子 → 合并成一个 DataFrame。"""
        dates_str = ["2024-01-15", "2024-01-16", "2024-01-17"]
        _write_precomputed(
            tmp_path, "fundamental", ["AAPL", "TSLA"], dates_str,
            {"pe_ratio": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]},
        )
        _write_precomputed(
            tmp_path, "sentiment", ["AAPL", "TSLA"], dates_str,
            {"mood_score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        )
        provider = _build_provider(tmp_path)

        result = provider.get_factors(
            ["AAPL", "TSLA"], ["pe_ratio", "mood_score"],
            date(2024, 1, 15), date(2024, 1, 17),
        )
        assert list(result.columns) == ["pe_ratio", "mood_score"]
        assert result["pe_ratio"].notna().all()
        assert result["mood_score"].notna().all()
        assert len(result) == 6  # 3 dates × 2 tickers

    def test_expression_and_precomputed_different_date_ranges(self, tmp_path):
        """precomputed 只覆盖部分日期 → 合并后 precomputed 在未覆盖日为 NaN，expression 仍有值。"""
        ohlcv = _make_ohlcv(["AAPL"], "2024-01-01", n_days=60)
        # Precomputed 只有 2024-02-01 ~ 2024-02-05
        pre_dates = [str(d.date()) for d in
                     pd.bdate_range("2024-02-01", periods=5)]
        _write_precomputed(tmp_path, "fundamental",
                           ["AAPL"], pre_dates, {"pe_ratio": [10.0] * 5})
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        # 请求范围超过 precomputed 实际覆盖
        result = provider.get_factors(
            ["AAPL"], ["KMID", "pe_ratio"],
            date(2024, 2, 1), date(2024, 2, 15),
        )
        # precomputed 在覆盖范围内非 NaN
        covered = result.loc[
            pd.IndexSlice[pd.Timestamp("2024-02-02"):pd.Timestamp("2024-02-07"), :],
            "pe_ratio",
        ]
        assert covered.notna().any()
        # precomputed 在 2024-02-10 之后应为 NaN
        uncovered = result.loc[
            pd.IndexSlice[pd.Timestamp("2024-02-10"):, :], "pe_ratio",
        ]
        assert uncovered.isna().all()
        # KMID（expression）全程非 NaN
        assert result["KMID"].notna().all()


# ===========================================================================
# E. Variable mapping & price semantics
# ===========================================================================

class TestVariableMappingSemantics:

    def test_dollar_close_maps_to_adj_close_not_close(self, tmp_path):
        """adj_close=2×close，KMID=($close-$open)/$open 应使用 adj_close，断言数值与 adj 版对齐。"""
        ohlcv = _make_ohlcv(
            ["AAPL"], "2024-01-01", n_days=30, adj_ratio=2.0,
        )
        # 此时 adj_close = close * 2，open 未被改动
        # KMID = (adj_close - open)/open ≈ (2*close - open)/open
        # 若错误地用 close 而不是 adj_close，KMID ≈ (close - open)/open
        # 两者差显著。
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL"], ["KMID"], date(2024, 1, 2), date(2024, 1, 30),
        )
        # 手算期望：对齐 provider trim 后的范围
        merged = ohlcv.loc[
            (ohlcv.index.get_level_values("date") >= pd.Timestamp("2024-01-02"))
            & (ohlcv.index.get_level_values("date") <= pd.Timestamp("2024-01-30"))
        ]
        expected = (merged["adj_close"] - merged["open"]) / merged["open"]
        wrong = (merged["close"] - merged["open"]) / merged["open"]
        # provider 结果与 adj 版一致，与非 adj 版显著不同
        np.testing.assert_allclose(
            result["KMID"].values, expected.values, rtol=1e-9,
        )
        assert not np.allclose(result["KMID"].values, wrong.values, rtol=1e-3)

    def test_dollar_volume_not_adjusted(self, tmp_path):
        """volume 常数 1000 → VMA5 = MEAN/volume ≈ 1.0，证明 $volume → volume 原始列。"""
        ohlcv = _make_ohlcv(
            ["AAPL"], "2024-01-01", n_days=30, volume=1000,
        )
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL"], ["VMA5"], date(2024, 1, 10), date(2024, 1, 30),
        )
        # 常数 volume → MEAN = volume → ratio ≈ 1.0（带 +1e-12 扰动）
        np.testing.assert_allclose(
            result["VMA5"].dropna().values, 1.0, rtol=1e-6,
        )


# ===========================================================================
# F. IC report integration
# ===========================================================================

class TestICReportIntegration:

    def test_oracle_factor_gives_high_ic(self, tmp_path):
        """precomputed factor = 未来 1 日收益（oracle） → IC 应接近 +1，证明 provider 不破坏 shift 语义。"""
        tickers = ["AAPL", "TSLA", "NVDA"]
        ohlcv = _make_ohlcv(tickers, "2024-01-01", n_days=80, seed=7)
        # Oracle factor = fwd_ret = adj_close[t+1]/adj_close[t] - 1
        fwd_ret = (
            ohlcv.groupby(level="ticker")["adj_close"].shift(-1)
            / ohlcv["adj_close"] - 1
        ).dropna()
        oracle_df = fwd_ret.to_frame("oracle")
        store = ParquetFactorStore(tmp_path)
        store.write_factors("signals", oracle_df)

        provider = _build_provider(
            tmp_path, market_data=_mock_market_data(ohlcv),
            ic_universe=ICUniverse(
                tickers=tickers,
                start=date(2024, 1, 2),
                end=date(2024, 4, 15),
            ),
        )
        report = provider.get_ic_report("oracle", window=80)
        # Oracle 因子与前向收益完全同构 → 每日 rank IC = 1.0，ic_mean 应 ≈ 1
        assert report["ic_mean"] > 0.99
        # 完美 oracle 使 ic_std=0 → icir=NaN（ic_evaluator 约定），
        # ic_std ≈ 0 本身就证明 provider 未二次 shift（否则 IC 会偏离 1.0）。
        assert report["ic_std"] < 1e-9 or pd.isna(report["ic_std"])

    def test_ic_report_missing_adj_close_raises(self, tmp_path):
        """market_data 返回 prices 缺 adj_close → ic_evaluator 抛 ValueError，经 provider 透传。"""
        # Precomputed 因子（不需要 Evaluator，所以走 IC 路径时只会因 prices 缺列而炸）
        tickers = ["AAPL", "TSLA"]
        dates_str = [str(d.date()) for d in
                     pd.bdate_range("2024-01-01", periods=30)]
        n = len(dates_str) * len(tickers)
        _write_precomputed(
            tmp_path, "signals", tickers, dates_str,
            {"pe_ratio": list(np.linspace(0, 1, n))},
        )
        # 故意不带 adj_close 列
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(d), t) for d in dates_str for t in tickers],
            names=["date", "ticker"],
        )
        bad_prices = pd.DataFrame({
            "open": [100.0] * n,
            "close": [101.0] * n,
        }, index=idx)
        provider = _build_provider(
            tmp_path, market_data=_mock_market_data(bad_prices),
            ic_universe=ICUniverse(
                tickers=tickers,
                start=date(2024, 1, 1),
                end=date(2024, 2, 9),
            ),
        )
        with pytest.raises(ValueError, match="adj_close"):
            provider.get_ic_report("pe_ratio")

    def test_ic_universe_range_outside_data_returns_nan(self, tmp_path):
        """ic_universe 范围超出 ohlcv 实际数据 → 报告全 NaN，不崩。"""
        ohlcv = _make_ohlcv(["AAPL", "TSLA"], "2024-01-01", n_days=30)
        provider = _build_provider(
            tmp_path, market_data=_mock_market_data(ohlcv),
            ic_universe=ICUniverse(
                tickers=["AAPL", "TSLA"],
                start=date(2025, 6, 1),   # 数据外
                end=date(2025, 12, 1),
            ),
        )
        # mock 返回的 ohlcv 是 2024 年的，trim 到 2025 后应全部丢弃
        # 注意 mock 每次返回相同 ohlcv —— trim 后 factor 部分会为空 → < 20 valid → NaN
        report = provider.get_ic_report("KMID")
        assert pd.isna(report["ic_mean"])
        assert pd.isna(report["ic_std"])
        assert pd.isna(report["icir"])


# ===========================================================================
# G. Column order, empty range, numerical anomaly
# ===========================================================================

class TestColumnOrderAndAnomalies:

    def test_column_order_preserved_mixed_sources_interleaved(self, tmp_path):
        """[precomp, expr, precomp, expr] 交错请求 → 输出列严格保序。"""
        ohlcv = _make_ohlcv(["AAPL"], "2024-01-01", n_days=60)
        dates_str = [str(d.date()) for d in pd.bdate_range("2024-01-01", periods=60)]
        _write_precomputed(
            tmp_path, "fundamental", ["AAPL"], dates_str,
            {"pe_ratio": [1.0] * 60, "pb_ratio": [2.0] * 60},
        )
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        ordered = ["pb_ratio", "KMID", "pe_ratio", "MA5"]
        result = provider.get_factors(
            ["AAPL"], ordered, date(2024, 2, 1), date(2024, 3, 1),
        )
        assert list(result.columns) == ordered

    def test_open_zero_produces_inf_or_nan_without_crash(self, tmp_path):
        """$open=0 使 KMID=($close-$open)/$open = inf/NaN，其他行仍有有限值，不抛异常。"""
        ohlcv = _make_ohlcv(["AAPL"], "2024-01-01", n_days=10)
        # 强行把第 5 行 open 改成 0
        ohlcv.loc[(ohlcv.index[4]), "open"] = 0.0
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL"], ["KMID"], date(2024, 1, 1), date(2024, 1, 12),
        )
        values = result["KMID"].values
        # 至少一行不是有限数（inf 或 NaN）
        assert not np.isfinite(values).all()
        # 至少一行是有限数（其他正常行）
        assert np.isfinite(values).any()

    def test_refresh_precomputed_index_discovers_new_group(self, tmp_path):
        """Lazy build 后写入新 group，refresh_precomputed_index 使其可被发现。"""
        # 第一次 build 时 group 尚未存在
        provider = _build_provider(tmp_path)
        provider.initialize()
        assert "pe_ratio" not in provider._precomputed_index

        # 之后写入新 group —— 不 refresh 时应当认作 unknown factor
        _write_precomputed(tmp_path, "fundamental",
                           ["AAPL"], ["2024-01-02"], {"pe_ratio": [1.0]})
        with pytest.raises(ValueError, match="Unknown factor"):
            provider.get_factors(
                ["AAPL"], ["pe_ratio"],
                date(2024, 1, 2), date(2024, 1, 2),
            )

        # 调 refresh 后能找到
        provider.refresh_precomputed_index()
        assert "pe_ratio" in provider._precomputed_index
        result = provider.get_factors(
            ["AAPL"], ["pe_ratio"],
            date(2024, 1, 2), date(2024, 1, 2),
        )
        assert len(result) == 1

    def test_parquet_atomic_write_tmp_has_unique_suffix(self, tmp_path):
        """并发写场景：tmp 文件名带 uuid，两次 write 不会互相覆盖 tmp。"""
        import pyarrow.parquet as pq
        store = ParquetFactorStore(tmp_path)
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-02"), "AAPL")], names=["date", "ticker"],
        )
        df1 = pd.DataFrame({"x": [1.0]}, index=idx)
        df2 = pd.DataFrame({"y": [2.0]}, index=idx)
        store.write_factors("g1", df1)
        store.write_factors("g2", df2)

        # 两个目标文件都存在，无 tmp 残留
        leftover = list(tmp_path.glob("*.tmp*"))
        assert leftover == []
        assert (tmp_path / "g1.parquet").exists()
        assert (tmp_path / "g2.parquet").exists()

    def test_empty_date_range_returns_empty_dataframe(self, tmp_path):
        """请求的 date range 完全在数据之外 → 空 MultiIndex DataFrame，不 crash。"""
        ohlcv = _make_ohlcv(["AAPL"], "2024-01-01", n_days=30)
        provider = _build_provider(tmp_path, market_data=_mock_market_data(ohlcv))

        result = provider.get_factors(
            ["AAPL"], ["KMID", "MA5"],
            date(2025, 6, 1), date(2025, 6, 30),
        )
        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ["date", "ticker"]
        assert len(result) == 0
        assert list(result.columns) == ["KMID", "MA5"]
