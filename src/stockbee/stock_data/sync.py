"""StockDataSyncer — 数据同步管道。

负责从外部数据源拉取数据，写入 Parquet + SQLite：
1. 同步全部美股列表 → broad_all
2. 同步 OHLCV 日线 → Parquet
3. 补充流动性指标（avg_volume, avg_dollar_volume）
4. 运行四层漏斗筛选

典型用法（周度调度）：
    syncer = StockDataSyncer(parquet, universe, alpaca)
    syncer.sync_assets()           # 更新 broad_all
    syncer.sync_ohlcv(days=5)      # 增量同步最近5天
    syncer.update_liquidity()      # 更新流动性指标
    syncer.run_funnel()            # 运行漏斗筛选
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .alpaca_market import AlpacaMarketData
from .funnel import FunnelConfig, UniverseFunnel
from .parquet_store import ParquetMarketData
from .universe_store import SqliteUniverseProvider

logger = logging.getLogger(__name__)


class StockDataSyncer:
    """数据同步管道，编排各组件完成周度数据更新。"""

    def __init__(
        self,
        parquet: ParquetMarketData,
        universe: SqliteUniverseProvider,
        alpaca: AlpacaMarketData,
        funnel_config: FunnelConfig | None = None,
    ) -> None:
        self._parquet = parquet
        self._universe = universe
        self._alpaca = alpaca
        self._funnel = UniverseFunnel(universe, funnel_config)

    def sync_assets(self, snapshot_date: date | None = None) -> int:
        """从 Alpaca 同步全部可交易美股到 broad_all。

        Returns:
            broad_all 的总成员数
        """
        logger.info("Syncing all tradable assets from Alpaca...")
        assets = self._alpaca.get_all_tradable_assets()

        if assets.empty:
            logger.warning("No assets returned from Alpaca")
            return 0

        count = self._universe.upsert_members(
            "broad_all", assets, snapshot_date=snapshot_date
        )
        logger.info("broad_all synced: %d assets", count)
        return count

    def sync_ohlcv(
        self,
        tickers: list[str] | None = None,
        days: int = 5,
        end_date: date | None = None,
    ) -> int:
        """增量同步 OHLCV 数据到 Parquet。

        Args:
            tickers: 要同步的 ticker 列表，None 表示 broad 层全部
            days: 回溯天数（默认5，即一周增量）
            end_date: 结束日期（默认今天）

        Returns:
            成功同步的 ticker 数
        """
        end_date = end_date or date.today()
        start_date = end_date - timedelta(days=days)

        if tickers is None:
            df = self._universe.get_universe("broad")
            if df.empty:
                logger.warning("broad universe is empty, nothing to sync")
                return 0
            tickers = df["ticker"].tolist()

        logger.info(
            "Syncing OHLCV for %d tickers: %s to %s",
            len(tickers), start_date, end_date,
        )

        # 分批同步，Alpaca API 限制每次请求的 symbol 数量
        batch_size = 100
        synced = 0

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                bars = self._alpaca.get_daily_bars(batch, start_date, end_date)
                if bars.empty:
                    continue

                # 按 ticker 拆分写入 Parquet
                for ticker in batch:
                    try:
                        ticker_data = bars.xs(ticker, level="ticker")
                    except KeyError:
                        continue
                    if not ticker_data.empty:
                        self._parquet.write_ticker(ticker, ticker_data)
                        synced += 1
            except Exception:
                logger.exception("Failed to sync batch starting at index %d", i)

        logger.info("OHLCV sync complete: %d/%d tickers", synced, len(tickers))
        return synced

    def sync_ohlcv_full(
        self,
        tickers: list[str] | None = None,
        years: int = 5,
    ) -> int:
        """全量同步历史 OHLCV 数据（初始化时使用）。

        Args:
            tickers: ticker 列表，None 表示 broad 层全部
            years: 历史年数

        Returns:
            成功同步的 ticker 数
        """
        end_date = date.today()
        try:
            start_date = end_date.replace(year=end_date.year - years)
        except ValueError:  # Feb 29 in non-leap year
            start_date = end_date.replace(year=end_date.year - years, day=28)
        days = (end_date - start_date).days
        return self.sync_ohlcv(tickers=tickers, days=days, end_date=end_date)

    def update_liquidity(
        self,
        lookback_days: int = 60,
        level: str = "broad_all",
    ) -> int:
        """从 Parquet 计算流动性指标，更新到 SQLite。

        计算 avg_volume 和 avg_dollar_volume（近 lookback_days 均值）。

        Returns:
            更新的 ticker 数
        """
        df = self._universe.get_universe(level)
        if df.empty:
            return 0

        tickers = df["ticker"].tolist()
        end = date.today()
        start = end - timedelta(days=lookback_days + 30)  # 多取一些以确保有足够交易日

        updated = 0
        for ticker in tickers:
            result = self._parquet.get_daily_bars([ticker], start, end, ["close", "volume"])
            if result.empty:
                continue

            # 提取单 ticker，取最近 lookback_days 个交易日
            bars = result.xs(ticker.upper(), level="ticker")
            bars = bars.tail(lookback_days)
            if len(bars) < 10:  # 至少需要 10 天数据
                continue

            avg_vol = bars["volume"].mean()
            avg_dollar_vol = (bars["close"] * bars["volume"]).mean()

            # 更新到 universe_members
            idx = df["ticker"] == ticker
            df.loc[idx, "avg_volume"] = avg_vol
            df.loc[idx, "avg_dollar_volume"] = avg_dollar_vol
            updated += 1

        if updated > 0:
            self._universe.upsert_members(level, df)

        logger.info("Liquidity updated: %d/%d tickers", updated, len(tickers))
        return updated

    def run_funnel(
        self,
        factor_scores: pd.DataFrame | None = None,
        snapshot_date: date | None = None,
    ) -> dict[str, int]:
        """运行四层漏斗筛选。

        Returns:
            每层的成员数 dict
        """
        return self._funnel.run_full_pipeline(factor_scores, snapshot_date)

    def run_weekly_sync(
        self,
        factor_scores: pd.DataFrame | None = None,
    ) -> dict[str, int | dict]:
        """完整的周度同步流程。

        1. sync_assets → broad_all
        2. sync_ohlcv(days=7) → Parquet
        3. update_liquidity → SQLite
        4. run_funnel → 四层漏斗

        Returns:
            各步骤的结果摘要
        """
        today = date.today()
        results: dict[str, int | dict] = {}

        results["assets"] = self.sync_assets(snapshot_date=today)
        results["ohlcv_synced"] = self.sync_ohlcv(days=7, end_date=today)
        results["liquidity_updated"] = self.update_liquidity()
        results["funnel_counts"] = self.run_funnel(
            factor_scores=factor_scores, snapshot_date=today
        )

        logger.info("Weekly sync complete: %s", results)
        return results
