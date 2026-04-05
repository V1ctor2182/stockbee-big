"""MacroDataSyncer — 宏观数据同步管道。

编排 FRED API → Parquet + SQLite 的完整同步流程：
1. sync_all: 全量历史拉取（初始化）
2. sync_latest: 增量拉取最近数据
3. rebuild_z_scores: 重建 Z-score 索引

典型用法（周度调度）：
    syncer = MacroDataSyncer(parquet, fred)
    syncer.sync_latest()    # 增量同步
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from .fred_macro import FredMacroProvider
from .parquet_macro import ParquetMacroProvider

logger = logging.getLogger(__name__)


class MacroDataSyncer:
    """宏观数据同步管道。"""

    def __init__(
        self,
        parquet: ParquetMacroProvider,
        fred: FredMacroProvider,
    ) -> None:
        self._parquet = parquet
        self._fred = fred

    def sync_all(self, years: int = 5) -> dict[str, int | str]:
        """全量历史同步（初始化时使用）。

        Args:
            years: 历史年数

        Returns:
            同步结果摘要
        """
        end = date.today()
        try:
            start = end.replace(year=end.year - years)
        except ValueError:  # Feb 29 in non-leap year
            start = end.replace(year=end.year - years, day=28)

        logger.info("Full macro sync: %s to %s", start, end)

        df = self._fred.fetch_all(start=start, end=end)
        if df.empty:
            logger.warning("No data returned from FRED")
            return {"indicators": 0, "rows": 0, "status": "empty"}

        self._parquet.write_indicators(df)
        z_count = self._parquet.rebuild_z_score_index()

        result = {
            "indicators": len(df.columns),
            "rows": len(df),
            "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
            "z_scores_updated": z_count,
            "status": "ok",
        }
        logger.info("Full macro sync complete: %s", result)
        return result

    def sync_latest(self, days: int = 30) -> dict[str, int | str]:
        """增量同步最近数据。

        Args:
            days: 回溯天数（默认 30 天，覆盖月频指标的发布延迟）

        Returns:
            同步结果摘要
        """
        end = date.today()
        start = end - timedelta(days=days)

        logger.info("Incremental macro sync: %s to %s", start, end)

        df = self._fred.fetch_all(start=start, end=end)
        if df.empty:
            logger.warning("No new data from FRED")
            return {"indicators": 0, "rows": 0, "status": "empty"}

        self._parquet.write_indicators(df)
        z_count = self._parquet.rebuild_z_score_index()

        result = {
            "indicators": len(df.columns),
            "rows": len(df),
            "z_scores_updated": z_count,
            "status": "ok",
        }
        logger.info("Incremental macro sync complete: %s", result)
        return result

    def run_weekly_sync(self) -> dict[str, int | str]:
        """周度同步入口。拉取最近 30 天覆盖所有频率。"""
        return self.sync_latest(days=30)
