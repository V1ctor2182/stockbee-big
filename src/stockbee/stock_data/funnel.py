"""UniverseFunnel — 四层漏斗筛选逻辑。

8000→4000→500→100 的逐级筛选：
- broad_all → broad:     流动性 + 可融资过滤
- broad → candidate:     初始因子筛选（市值、成交量排名）
- candidate → u100:      综合因子评分 Top 100

每层筛选条件可通过配置调整，支持回测时的时间切片。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .universe_store import SqliteUniverseProvider

logger = logging.getLogger(__name__)


@dataclass
class FunnelConfig:
    """漏斗筛选参数。"""
    # broad_all → broad
    min_avg_dollar_volume: float = 500_000.0   # 日均成交额 > 50万美元
    require_shortable: bool = False             # broad 层不强制要求可做空

    # broad → candidate
    candidate_size: int = 500
    min_market_cap: float = 300_000_000.0       # 最小市值 3 亿美元

    # candidate → u100
    u100_size: int = 100


class UniverseFunnel:
    """四层漏斗筛选器。

    使用方式：
        funnel = UniverseFunnel(universe_provider, config=FunnelConfig())
        funnel.run_broad_filter(snapshot_date=date(2026, 3, 28))
        funnel.run_candidate_filter(snapshot_date=date(2026, 3, 28))
        funnel.run_u100_selection(factor_scores, snapshot_date=date(2026, 3, 28))
    """

    def __init__(
        self,
        universe: SqliteUniverseProvider,
        config: FunnelConfig | None = None,
    ) -> None:
        self._universe = universe
        self._config = config or FunnelConfig()

    def run_broad_filter(self, snapshot_date: date | None = None) -> pd.DataFrame:
        """broad_all → broad: 过滤流动性不足的股票。

        条件：
        - avg_dollar_volume >= min_avg_dollar_volume
        - 可选：require_shortable

        Returns:
            筛选后的 broad 层 DataFrame
        """
        df = self._universe.get_universe("broad_all")
        if df.empty:
            logger.warning("broad_all is empty, cannot filter to broad")
            return df

        initial = len(df)
        cfg = self._config

        # 流动性过滤
        if "avg_dollar_volume" in df.columns:
            df = df[df["avg_dollar_volume"] >= cfg.min_avg_dollar_volume]

        # 可融资过滤
        if cfg.require_shortable and "short_able" in df.columns:
            df = df[df["short_able"]]

        logger.info(
            "Broad filter: %d → %d (removed %d illiquid)",
            initial, len(df), initial - len(df),
        )

        self._universe.upsert_members("broad", df, snapshot_date=snapshot_date)
        return df

    def run_candidate_filter(
        self, snapshot_date: date | None = None
    ) -> pd.DataFrame:
        """broad → candidate: 市值 + 成交量排名筛选。

        条件：
        - market_cap >= min_market_cap
        - 按 avg_dollar_volume 排名取 top candidate_size

        Returns:
            筛选后的 candidate 层 DataFrame
        """
        df = self._universe.get_universe("broad")
        if df.empty:
            logger.warning("broad is empty, cannot filter to candidate")
            return df

        initial = len(df)
        cfg = self._config

        # 市值过滤
        if "market_cap" in df.columns:
            df = df[df["market_cap"] >= cfg.min_market_cap]

        # 按流动性排名截取
        if "avg_dollar_volume" in df.columns and len(df) > cfg.candidate_size:
            df = df.nlargest(cfg.candidate_size, "avg_dollar_volume")

        logger.info(
            "Candidate filter: %d → %d",
            initial, len(df),
        )

        self._universe.upsert_members("candidate", df, snapshot_date=snapshot_date)
        return df

    def run_u100_selection(
        self,
        factor_scores: pd.DataFrame | None = None,
        snapshot_date: date | None = None,
    ) -> pd.DataFrame:
        """candidate → u100: 基于因子综合评分选取 Top 100。

        Args:
            factor_scores: DataFrame with columns [ticker, composite_score]
                           如果为 None，按 market_cap 排名（Phase 1 简化）
            snapshot_date: 快照日期

        Returns:
            u100 DataFrame
        """
        df = self._universe.get_universe("candidate")
        if df.empty:
            logger.warning("candidate is empty, cannot select u100")
            return df

        cfg = self._config

        if factor_scores is not None and "composite_score" in factor_scores.columns:
            # 用因子评分排名
            merged = df.merge(
                factor_scores[["ticker", "composite_score"]],
                on="ticker",
                how="left",
            )
            merged["composite_score"] = merged["composite_score"].fillna(0.0)
            selected = merged.nlargest(cfg.u100_size, "composite_score")
            selected = selected.drop(columns=["composite_score"])
        else:
            # Phase 1 简化：按市值排名
            if "market_cap" in df.columns:
                selected = df.nlargest(cfg.u100_size, "market_cap")
            else:
                selected = df.head(cfg.u100_size)

        logger.info(
            "U100 selection: %d → %d",
            len(df), len(selected),
        )

        self._universe.upsert_members("u100", selected, snapshot_date=snapshot_date)
        return selected

    def run_full_pipeline(
        self,
        factor_scores: pd.DataFrame | None = None,
        snapshot_date: date | None = None,
    ) -> dict[str, int]:
        """运行完整的四层漏斗流程，返回每层的成员数。"""
        self.run_broad_filter(snapshot_date)
        self.run_candidate_filter(snapshot_date)
        self.run_u100_selection(factor_scores, snapshot_date)
        return self._universe.get_all_level_counts()
