# tests/test_macro_data.py
"""Macro Data 模块测试。
# PYTHONPATH=src .venv/bin/python -m pytest tests/test_macro_data.py -v

测试对象：ParquetMacroProvider、indicators 定义、Z-score 计算、point-in-time 对齐。
不测 FredMacroProvider（需要 API key），不测 MacroDataSyncer（集成测试）。
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from stockbee.providers.base import ProviderConfig
from stockbee.macro_data.indicators import (
    ALL_CODES, FRED_INDICATORS, INDICATOR_GROUPS, FredIndicator,
)
from stockbee.macro_data.parquet_macro import (
    ParquetMacroProvider, Z_SCORE_CLIP, Z_SCORE_WINDOW,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def macro_data() -> pd.DataFrame:
    """生成 300 天模拟宏观数据（覆盖 Z-score 窗口需求）。"""
    dates = pd.bdate_range("2025-01-02", periods=300)
    rng = np.random.default_rng(42)
    data = {}

    # 模拟各频率指标
    # 日频指标：每天有值
    for code in ["DFF", "T10Y2Y", "DGS10", "VIXCLS", "DEXUSEU",
                 "DCOILWTICO", "GOLDAMGBD228NLBM", "BAMLH0A0HY2"]:
        data[code] = rng.normal(2.0, 0.5, 300)

    # 月频指标：每月一个值，其余 NaN
    for code in ["PAYEMS", "UNRATE", "CPIAUCSL", "PPIFIS",
                 "INDPRO", "EXHYLD", "M2SL"]:
        vals = np.full(300, np.nan)
        # 每 ~21 个交易日放一个值
        for i in range(0, 300, 21):
            vals[i] = rng.normal(100.0, 5.0)
        data[code] = vals

    # 周频指标：每 5 天一个值
    for code in ["ICSA"]:
        vals = np.full(300, np.nan)
        for i in range(0, 300, 5):
            vals[i] = rng.normal(200000, 10000)
        data[code] = vals

    # 季频指标：每 ~63 天一个值
    for code in ["GDP"]:
        vals = np.full(300, np.nan)
        for i in range(0, 300, 63):
            vals[i] = rng.normal(25000, 500)
        data[code] = vals

    return pd.DataFrame(data, index=dates)


@pytest.fixture
def macro_provider(tmp_path) -> ParquetMacroProvider:
    config = ProviderConfig(
        implementation="ParquetMacroProvider",
        params={
            "data_path": str(tmp_path / "macro"),
            "index_db": str(tmp_path / "macro_index.db"),
        },
    )
    provider = ParquetMacroProvider(config)
    provider.initialize()
    return provider


# =========================================================================
# Indicator Definition Tests
# =========================================================================

class TestIndicators:

    def test_indicators_defined(self):
        # Tech Design 说 19 个，实际 FRED 可映射 17 个（2 个为派生特征）
        assert len(FRED_INDICATORS) == 17
        assert len(ALL_CODES) == 17

    def test_8_dimensions(self):
        assert len(INDICATOR_GROUPS) == 8
        expected = {"利率", "就业", "物价", "增长", "信用", "流动性", "汇率", "商品"}
        assert set(INDICATOR_GROUPS.keys()) == expected

    def test_dimension_counts(self):
        assert len(INDICATOR_GROUPS["利率"]) == 3
        assert len(INDICATOR_GROUPS["就业"]) == 3
        assert len(INDICATOR_GROUPS["物价"]) == 2
        assert len(INDICATOR_GROUPS["增长"]) == 2
        assert len(INDICATOR_GROUPS["信用"]) == 2
        assert len(INDICATOR_GROUPS["流动性"]) == 2
        assert len(INDICATOR_GROUPS["汇率"]) == 1
        assert len(INDICATOR_GROUPS["商品"]) == 2

    def test_indicator_dataclass(self):
        dff = FRED_INDICATORS["DFF"]
        assert dff.code == "DFF"
        assert dff.dimension == "利率"
        assert dff.frequency == "daily"

    def test_all_codes_match_keys(self):
        assert ALL_CODES == list(FRED_INDICATORS.keys())


# =========================================================================
# ParquetMacroProvider Tests
# =========================================================================

class TestParquetMacroProvider:

    def test_write_and_read(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        result = macro_provider.get_macro_indicators(["DFF", "VIXCLS"])
        assert not result.empty
        assert "DFF" in result.columns
        assert "VIXCLS" in result.columns

    def test_read_empty(self, macro_provider):
        result = macro_provider.get_macro_indicators()
        assert result.empty

    def test_read_all_indicators(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        result = macro_provider.get_macro_indicators()
        assert len(result.columns) == 17

    def test_date_filtering(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        start = date(2025, 6, 1)
        end = date(2025, 7, 1)
        result = macro_provider.get_macro_indicators(["DFF"], start=start, end=end)
        assert not result.empty
        assert result.index.min().date() >= start
        # Point-in-time: end 使用 end-1
        assert result.index.max().date() <= end - timedelta(days=1)

    def test_point_in_time_alignment(self, macro_provider, macro_data):
        """end=T 时只返回 T-1 及之前的数据。"""
        macro_provider.write_indicators(macro_data)

        end = date(2025, 6, 15)
        pit_boundary = end - timedelta(days=1)
        result = macro_provider.get_macro_indicators(["DFF"], end=end)
        assert result.index.max().date() <= pit_boundary

    def test_forward_fill(self, macro_provider, macro_data):
        """不同频率的指标应 forward-fill 缺失日期。"""
        macro_provider.write_indicators(macro_data)

        result = macro_provider.get_macro_indicators(["GDP"])  # 季频
        # forward-fill 后不应有 NaN（除了最早几天可能没有之前的值）
        non_null = result["GDP"].dropna()
        assert len(non_null) > 200  # 300 天中大部分应该被 fill

    def test_z_score_computation(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        z_scores = macro_provider.get_latest_z_scores(["DFF", "VIXCLS"])
        assert "DFF" in z_scores
        assert "VIXCLS" in z_scores
        # Z-score 应该在 [-3, 3] 范围内
        for val in z_scores.values():
            assert -Z_SCORE_CLIP <= val <= Z_SCORE_CLIP

    def test_z_score_index_persistence(self, macro_provider, macro_data):
        """Z-score 写入 SQLite 后应该能直接读回。"""
        macro_provider.write_indicators(macro_data)

        # 第一次计算并写入
        z1 = macro_provider.get_latest_z_scores(["DFF"])
        assert "DFF" in z1

        # 第二次应该从 SQLite 索引读取
        z2 = macro_provider.get_latest_z_scores(["DFF"])
        assert z2["DFF"] == z1["DFF"]

    def test_rebuild_z_score_index(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        count = macro_provider.rebuild_z_score_index()
        # 日频指标应该全部有 Z-score，月/季频可能 min_periods 不够
        assert count >= 8  # 至少 8 个日频指标

    def test_incremental_write(self, macro_provider, macro_data):
        """增量写入应 merge 不覆盖已有数据。"""
        macro_provider.write_indicators(macro_data)

        # 写入更新数据
        new_dates = pd.bdate_range("2026-03-01", periods=10)
        new_data = pd.DataFrame(
            {"DFF": np.linspace(5.0, 5.5, 10)},
            index=new_dates,
        )
        macro_provider.write_indicators(new_data)

        result = macro_provider.get_macro_indicators(["DFF"])
        assert len(result) > 300  # 原始 300 + 新增 10

    def test_indicator_date_range(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        dr = macro_provider.indicator_date_range()
        assert dr is not None
        assert dr[0] == date(2025, 1, 2)

    def test_nonexistent_indicator(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        result = macro_provider.get_macro_indicators(["NONEXIST"])
        assert result.empty

    def test_subset_indicators(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)

        result = macro_provider.get_macro_indicators(["DFF", "GDP", "VIXCLS"])
        assert set(result.columns) == {"DFF", "GDP", "VIXCLS"}

    def test_health_check(self, macro_provider):
        assert macro_provider.health_check() is True

    def test_shutdown_and_reinit(self, macro_provider, macro_data):
        macro_provider.write_indicators(macro_data)
        macro_provider.shutdown()
        assert not macro_provider.is_initialized

        macro_provider.initialize()
        result = macro_provider.get_macro_indicators(["DFF"])
        assert not result.empty


# =========================================================================
# Z-score Edge Cases
# =========================================================================

class TestZScoreEdgeCases:

    def test_z_score_with_constant_series(self, macro_provider):
        """常数序列（std=0）应该返回 NaN 而非 inf。"""
        dates = pd.bdate_range("2025-01-02", periods=300)
        data = pd.DataFrame({"DFF": [2.5] * 300}, index=dates)
        macro_provider.write_indicators(data)

        z_scores = macro_provider.get_latest_z_scores(["DFF"])
        # 常数序列的 Z-score 应该不在结果中（NaN 被过滤）
        # 或者如果包含则不是 inf
        if "DFF" in z_scores:
            assert not np.isinf(z_scores["DFF"])

    def test_z_score_clip_extreme(self, macro_provider):
        """极端值应该被 clip 到 [-3, 3]。"""
        dates = pd.bdate_range("2025-01-02", periods=300)
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1, 300)
        # 最后一个值设为极端值
        values[-1] = 100.0
        data = pd.DataFrame({"DFF": values}, index=dates)
        macro_provider.write_indicators(data)

        z_scores = macro_provider.get_latest_z_scores(["DFF"])
        assert z_scores["DFF"] == Z_SCORE_CLIP  # 应该被 clip 到 3.0

    def test_z_score_insufficient_data(self, macro_provider):
        """数据不足 min_periods 时不应返回 Z-score。"""
        dates = pd.bdate_range("2025-01-02", periods=10)
        data = pd.DataFrame({"DFF": [2.0] * 10}, index=dates)
        macro_provider.write_indicators(data)

        z_scores = macro_provider.get_latest_z_scores(["DFF"])
        # 10 天数据不足 min_periods=20，不应有 Z-score
        assert "DFF" not in z_scores
