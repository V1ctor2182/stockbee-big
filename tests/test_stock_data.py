# tests/test_stock_data.py
"""Stock Data 模块测试。
# PYTHONPATH=src .venv/bin/pytest tests/test_stock_data.py -v

测试对象：ParquetMarketData、SqliteUniverseProvider、UniverseFunnel
不测 AlpacaMarketData（需要 API key），不测 StockDataSyncer（集成测试）。
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stockbee.providers.base import ProviderConfig
from stockbee.stock_data.parquet_store import ParquetMarketData, OHLCV_FIELDS
from stockbee.stock_data.universe_store import SqliteUniverseProvider, VALID_LEVELS
from stockbee.stock_data.funnel import FunnelConfig, UniverseFunnel


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def ohlcv_data() -> pd.DataFrame:
    """生成 5 天 OHLCV 测试数据。"""
    dates = pd.date_range("2026-01-05", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [105.0, 106.0, 107.0, 108.0, 109.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [104.0, 105.0, 106.0, 107.0, 108.0],
            "volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            "adj_close": [104.0, 105.0, 106.0, 107.0, 108.0],
        },
        index=dates,
    )


@pytest.fixture
def parquet_provider(tmp_path) -> ParquetMarketData:
    config = ProviderConfig(
        implementation="ParquetMarketData",
        params={"data_path": str(tmp_path / "ohlcv")},
    )
    provider = ParquetMarketData(config)
    provider.initialize()
    return provider


@pytest.fixture
def universe_provider(tmp_path) -> SqliteUniverseProvider:
    config = ProviderConfig(
        implementation="SqliteUniverseProvider",
        params={"db_path": str(tmp_path / "universe.db")},
    )
    provider = SqliteUniverseProvider(config)
    provider.initialize()
    return provider


@pytest.fixture
def sample_universe() -> pd.DataFrame:
    """生成模拟 broad_all 股票池。"""
    tickers = [f"STOCK{i:03d}" for i in range(100)]
    return pd.DataFrame({
        "ticker": tickers,
        "sector": ["Tech"] * 30 + ["Finance"] * 30 + ["Health"] * 20 + ["Energy"] * 20,
        "market_cap": [1e9 - i * 5e6 for i in range(100)],
        "avg_volume": [5e6 - i * 2e4 for i in range(100)],
        "avg_dollar_volume": [2e7 - i * 1e5 for i in range(100)],
        "short_able": [True] * 80 + [False] * 20,
    })


# =========================================================================
# ParquetMarketData Tests
# =========================================================================

class TestParquetMarketData:

    def test_write_and_read(self, parquet_provider, ohlcv_data):
        parquet_provider.write_ticker("AAPL", ohlcv_data)

        result = parquet_provider.get_daily_bars(
            ["AAPL"], date(2026, 1, 5), date(2026, 1, 9)
        )
        assert not result.empty
        assert len(result) == 5
        assert "close" in result.columns

    def test_read_nonexistent_ticker(self, parquet_provider):
        result = parquet_provider.get_daily_bars(
            ["NONEXIST"], date(2026, 1, 1), date(2026, 12, 31)
        )
        assert result.empty

    def test_date_filtering(self, parquet_provider, ohlcv_data):
        parquet_provider.write_ticker("MSFT", ohlcv_data)

        result = parquet_provider.get_daily_bars(
            ["MSFT"], date(2026, 1, 7), date(2026, 1, 8)
        )
        assert len(result) == 2

    def test_field_filtering(self, parquet_provider, ohlcv_data):
        parquet_provider.write_ticker("GOOG", ohlcv_data)

        result = parquet_provider.get_daily_bars(
            ["GOOG"], date(2026, 1, 5), date(2026, 1, 9), fields=["close", "volume"]
        )
        assert "close" in result.columns
        assert "volume" in result.columns
        assert "open" not in result.columns

    def test_multiple_tickers(self, parquet_provider, ohlcv_data):
        parquet_provider.write_ticker("AAPL", ohlcv_data)
        ohlcv2 = ohlcv_data.copy()
        ohlcv2["close"] = [200.0, 201.0, 202.0, 203.0, 204.0]
        parquet_provider.write_ticker("MSFT", ohlcv2)

        result = parquet_provider.get_daily_bars(
            ["AAPL", "MSFT"], date(2026, 1, 5), date(2026, 1, 9)
        )
        assert len(result) == 10

    def test_get_latest_price(self, parquet_provider, ohlcv_data):
        parquet_provider.write_ticker("AAPL", ohlcv_data)

        prices = parquet_provider.get_latest_price(["AAPL", "NONEXIST"])
        assert "AAPL" in prices
        assert prices["AAPL"] == 108.0
        assert "NONEXIST" not in prices

    def test_list_tickers(self, parquet_provider, ohlcv_data):
        parquet_provider.write_ticker("AAPL", ohlcv_data)
        parquet_provider.write_ticker("MSFT", ohlcv_data)

        tickers = parquet_provider.list_tickers()
        assert set(tickers) == {"AAPL", "MSFT"}

    def test_ticker_date_range(self, parquet_provider, ohlcv_data):
        parquet_provider.write_ticker("AAPL", ohlcv_data)

        dr = parquet_provider.ticker_date_range("AAPL")
        assert dr is not None
        assert dr[0] == date(2026, 1, 5)
        assert dr[1] == date(2026, 1, 9)

    def test_incremental_write(self, parquet_provider, ohlcv_data):
        """测试增量写入（追加新数据，去重）。"""
        parquet_provider.write_ticker("AAPL", ohlcv_data)

        new_dates = pd.date_range("2026-01-08", periods=5, freq="B")
        new_data = pd.DataFrame(
            {
                "open": [110.0, 111.0, 112.0, 113.0, 114.0],
                "high": [115.0, 116.0, 117.0, 118.0, 119.0],
                "low": [109.0, 110.0, 111.0, 112.0, 113.0],
                "close": [114.0, 115.0, 116.0, 117.0, 118.0],
                "volume": [1500000, 1600000, 1700000, 1800000, 1900000],
                "adj_close": [114.0, 115.0, 116.0, 117.0, 118.0],
            },
            index=new_dates,
        )
        parquet_provider.write_ticker("AAPL", new_data)

        result = parquet_provider.get_daily_bars(
            ["AAPL"], date(2026, 1, 5), date(2026, 1, 16)
        )
        # 5 原始 + 3 新增（2 重叠用新值覆盖）= 8
        assert len(result) == 8

    def test_uppercase_ticker(self, parquet_provider, ohlcv_data):
        """ticker 应统一为大写。"""
        parquet_provider.write_ticker("aapl", ohlcv_data)

        result = parquet_provider.get_daily_bars(
            ["AAPL"], date(2026, 1, 5), date(2026, 1, 9)
        )
        assert not result.empty


# =========================================================================
# SqliteUniverseProvider Tests
# =========================================================================

class TestSqliteUniverseProvider:

    def test_upsert_and_get(self, universe_provider, sample_universe):
        count = universe_provider.upsert_members("broad_all", sample_universe)
        assert count == 100

        result = universe_provider.get_universe("broad_all")
        assert len(result) == 100
        assert "ticker" in result.columns

    def test_invalid_level(self, universe_provider):
        with pytest.raises(ValueError, match="Invalid level"):
            universe_provider.get_universe("invalid_level")

    def test_empty_universe(self, universe_provider):
        result = universe_provider.get_universe("u100")
        assert result.empty

    def test_member_removal_tracking(self, universe_provider):
        """测试成员移除时的历史追踪。"""
        initial = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "sector": ["Tech", "Tech", "Tech"],
            "market_cap": [3e12, 2.5e12, 2e12],
            "avg_volume": [5e7, 4e7, 3e7],
            "avg_dollar_volume": [1e10, 8e9, 6e9],
            "short_able": [True, True, True],
        })
        universe_provider.upsert_members("u100", initial, date(2026, 1, 1))

        # 移除 GOOG
        updated = initial[initial["ticker"] != "GOOG"]
        universe_provider.upsert_members("u100", updated, date(2026, 2, 1))

        result = universe_provider.get_universe("u100")
        assert len(result) == 2
        assert "GOOG" not in result["ticker"].values

    def test_time_slice(self, universe_provider):
        """测试 as_of 时间切片（避免前看偏差）。"""
        v1 = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "sector": ["Tech", "Tech"],
            "market_cap": [3e12, 2.5e12],
            "avg_volume": [5e7, 4e7],
            "avg_dollar_volume": [1e10, 8e9],
            "short_able": [True, True],
        })
        universe_provider.upsert_members("u100", v1, date(2026, 1, 1))

        v2 = pd.DataFrame({
            "ticker": ["AAPL", "GOOG"],
            "sector": ["Tech", "Tech"],
            "market_cap": [3e12, 2e12],
            "avg_volume": [5e7, 3e7],
            "avg_dollar_volume": [1e10, 6e9],
            "short_able": [True, True],
        })
        universe_provider.upsert_members("u100", v2, date(2026, 2, 1))

        # 查看 1 月的快照：应该有 AAPL, MSFT
        jan = universe_provider.get_universe("u100", as_of=date(2026, 1, 15))
        assert set(jan["ticker"].values) == {"AAPL", "MSFT"}

        # 查看 2 月的快照：应该有 AAPL, GOOG
        feb = universe_provider.get_universe("u100", as_of=date(2026, 2, 15))
        assert "GOOG" in feb["ticker"].values

    def test_get_all_level_counts(self, universe_provider, sample_universe):
        universe_provider.upsert_members("broad_all", sample_universe)
        counts = universe_provider.get_all_level_counts()
        assert counts["broad_all"] == 100
        assert counts["broad"] == 0
        assert counts["u100"] == 0

    def test_health_check(self, universe_provider):
        assert universe_provider.health_check() is True

    def test_shutdown_and_reinit(self, universe_provider, sample_universe):
        universe_provider.upsert_members("broad_all", sample_universe)
        universe_provider.shutdown()
        assert not universe_provider.is_initialized

        universe_provider.initialize()
        result = universe_provider.get_universe("broad_all")
        assert len(result) == 100


# =========================================================================
# UniverseFunnel Tests
# =========================================================================

class TestUniverseFunnel:

    def test_broad_filter(self, universe_provider, sample_universe):
        universe_provider.upsert_members("broad_all", sample_universe)

        funnel = UniverseFunnel(
            universe_provider,
            FunnelConfig(min_avg_dollar_volume=1.95e7),
        )
        result = funnel.run_broad_filter()
        # avg_dollar_volume 从 2e7 到 (2e7 - 99*1e5)=1.01e7
        # 过滤 >= 1.95e7 的：前几只
        assert len(result) < 100
        assert len(result) > 0

    def test_candidate_filter(self, universe_provider, sample_universe):
        universe_provider.upsert_members("broad_all", sample_universe)

        funnel = UniverseFunnel(
            universe_provider,
            FunnelConfig(
                min_avg_dollar_volume=0,   # 不过滤流动性
                candidate_size=20,
                min_market_cap=0,          # 不过滤市值
            ),
        )
        funnel.run_broad_filter()
        result = funnel.run_candidate_filter()
        assert len(result) == 20

    def test_u100_selection_by_market_cap(self, universe_provider, sample_universe):
        """无因子评分时按市值排名。"""
        universe_provider.upsert_members("broad_all", sample_universe)

        funnel = UniverseFunnel(
            universe_provider,
            FunnelConfig(
                min_avg_dollar_volume=0,
                candidate_size=50,
                min_market_cap=0,
                u100_size=10,
            ),
        )
        funnel.run_broad_filter()
        funnel.run_candidate_filter()
        result = funnel.run_u100_selection()
        assert len(result) == 10

    def test_u100_selection_with_factor_scores(self, universe_provider, sample_universe):
        """有因子评分时按评分排名。"""
        universe_provider.upsert_members("broad_all", sample_universe)

        funnel = UniverseFunnel(
            universe_provider,
            FunnelConfig(
                min_avg_dollar_volume=0,
                candidate_size=50,
                min_market_cap=0,
                u100_size=5,
            ),
        )
        funnel.run_broad_filter()
        funnel.run_candidate_filter()

        # 构造因子评分：最后几只股票评分最高
        candidate = universe_provider.get_universe("candidate")
        scores = pd.DataFrame({
            "ticker": candidate["ticker"],
            "composite_score": range(len(candidate)),  # 递增评分
        })

        result = funnel.run_u100_selection(factor_scores=scores)
        assert len(result) == 5
        # 评分最高的应该被选中
        assert result["ticker"].iloc[0] in candidate["ticker"].tail(5).values

    def test_full_pipeline(self, universe_provider, sample_universe):
        universe_provider.upsert_members("broad_all", sample_universe)

        funnel = UniverseFunnel(
            universe_provider,
            FunnelConfig(
                min_avg_dollar_volume=0,
                candidate_size=30,
                min_market_cap=0,
                u100_size=10,
            ),
        )
        counts = funnel.run_full_pipeline()
        assert counts["broad_all"] == 100
        assert counts["broad"] == 100    # min_avg_dollar_volume=0, 全部通过
        assert counts["candidate"] == 30
        assert counts["u100"] == 10

    def test_empty_broad_all(self, universe_provider):
        funnel = UniverseFunnel(universe_provider)
        result = funnel.run_broad_filter()
        assert result.empty
