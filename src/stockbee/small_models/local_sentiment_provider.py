"""LocalSentimentProvider — 实现 providers.SentimentProvider 契约。

核心功能:
- score_texts: 透传 m2a FinBERTScorer singleton (共享推理, 消除重复)
- get_ticker_sentiment: SQLite 查询 + confidence-weighted 均值 (见 spec.md "加权公式")
- backfill: 读 fine5_importance 那一批类似的方式 - 本 provider 补齐缺失的
  finbert_negative/neutral/confidence, 但**不触** sentiment_score (g2 已写)

决策:
- P3 共享 FinBERTScorer singleton: g2_classifier 和本 provider 跑同一个权重副本
- 加权公式: Σ(conf_i × prob_i) / Σ conf_i, 权重 = finbert_confidence
- Σ conf = 0 → 返回 zeros + count=0, 不 raise (PRD 可用性约束)
- backfill 只处理 finbert_confidence IS NULL 的行, 旧数据兼容
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import SentimentProvider

from .finbert_scorer import FinBERTScorer, get_default_scorer

logger = logging.getLogger(__name__)

_ZERO_SENTIMENT: dict[str, float] = {
    "positive": 0.0,
    "negative": 0.0,
    "neutral": 0.0,
    "confidence": 0.0,
    "count": 0.0,
}


class LocalSentimentProvider(SentimentProvider):
    """本地 FinBERT 情绪 Provider。

    依赖注入:
        scorer:     m2a FinBERTScorer, 默认 get_default_scorer() singleton
        news_store: 已 initialize 的 SqliteNewsProvider (get_ticker_sentiment /
                    backfill 需要)

    使用::

        provider = LocalSentimentProvider(news_store=store)
        provider.initialize()
        provider.score_texts(["company beats earnings"])
        provider.get_ticker_sentiment("AAPL", lookback_days=7)
        n = provider.backfill(limit=1000)
    """

    def __init__(
        self,
        config: ProviderConfig | None = None,
        scorer: FinBERTScorer | None = None,
        news_store: Any | None = None,
    ) -> None:
        super().__init__(config)
        self._scorer = scorer
        self._news_store = news_store

    # ------ 生命周期 ------

    def _do_initialize(self) -> None:
        if self._scorer is None:
            self._scorer = get_default_scorer()
        if self._news_store is not None and not getattr(
            self._news_store, "is_initialized", False
        ):
            self._news_store.initialize()

    def _do_shutdown(self) -> None:
        # scorer 是 singleton,不在此关闭
        # news_store 生命周期由注入方管理
        self._scorer = None

    # ------ SentimentProvider 接口 ------

    def score_texts(self, texts: list[str]) -> list[dict[str, float]]:
        """透传到 m2a FinBERTScorer。"""
        self._require_initialized()
        return self._scorer.score_texts(texts)

    def get_ticker_sentiment(
        self, ticker: str, lookback_days: int = 7
    ) -> dict[str, float]:
        """取 ticker 近期 (confidence-weighted) 情绪聚合。

        公式: agg_prob = Σ(conf_i × prob_i) / Σ conf_i,权重 = finbert_confidence

        返回键: positive / negative / neutral / confidence / count
        无匹配或 Σconf=0 → 全零 + count=0 (不 raise)。
        """
        self._require_news_store()
        if not ticker or not isinstance(ticker, str):
            raise ValueError(f"ticker must be non-empty str, got {ticker!r}")
        if lookback_days < 0:
            raise ValueError(f"lookback_days must be >= 0, got {lookback_days}")

        end_ts = datetime.now(timezone.utc)
        start_ts = end_ts - timedelta(days=lookback_days)
        rows = self._news_store._conn.execute(
            """
            SELECT DISTINCT e.id, e.sentiment_score, e.finbert_negative,
                   e.finbert_neutral, e.finbert_confidence
            FROM news_events e
            JOIN news_tickers nt ON nt.news_id = e.id
            WHERE nt.ticker = ?
              AND e.timestamp >= ?
              AND e.timestamp <= ?
              AND e.finbert_confidence IS NOT NULL
            """,
            (ticker.upper(), start_ts.isoformat(), end_ts.isoformat()),
        ).fetchall()
        if not rows:
            return dict(_ZERO_SENTIMENT)

        sum_conf = 0.0
        weighted = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        count = 0
        skipped_partial = 0
        for _id, pos, neg, neu, conf in rows:
            if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (pos, neg, neu, conf)):
                # review H5: 部分 NULL 行静默跳过,计数后调试可见
                skipped_partial += 1
                continue
            c = float(conf)
            if c <= 0.0:
                continue
            sum_conf += c
            weighted["positive"] += c * float(pos)
            weighted["negative"] += c * float(neg)
            weighted["neutral"] += c * float(neu)
            count += 1
        if skipped_partial:
            logger.debug(
                "get_ticker_sentiment(%s): skipped %d rows with partial NULL",
                ticker,
                skipped_partial,
            )
        if sum_conf == 0.0 or count == 0:
            return dict(_ZERO_SENTIMENT)
        out = {k: v / sum_conf for k, v in weighted.items()}
        # review H1 / M1: confidence = max 聚合概率 (与 FinBERTScorer 单行 confidence
        # = max 三项 softmax 的语义对齐),便于下游当"置信度"使用
        out["confidence"] = max(out["positive"], out["negative"], out["neutral"])
        out["count"] = float(count)
        return out

    # ------ Backfill ------

    def backfill(
        self,
        since: str | datetime | None = None,
        limit: int = 1000,
    ) -> int:
        """补齐任一情绪字段为 NULL 的行; 批量 scorer.score_texts 后 UPDATE。

        Args:
            since: 仅处理 timestamp >= since 的行 (None = 全量)
            limit: 单次 LIMIT 上限 (分页调用时传不同值)

        Returns:
            实际写入的行数。sentiment_score 仅在当前为 NULL 时由 scorer 的
            positive 概率填入 (COALESCE);若 g2 已写入,保持不动。
        """
        # review H6: 只按 finbert_confidence IS NULL 选行会永久冻结 sentiment_score
        # 仍 NULL 但 finbert_* 已填的"部分 G1 / 部分 backfill"行 —
        # get_ticker_sentiment 因 positive IS NULL 跳过,backfill 又不再命中。
        # 改为任一字段 NULL 即候选,并用 COALESCE 保留 g2 写入的 positive。
        self._require_initialized()
        self._require_news_store()
        if limit < 1:
            raise ValueError(f"limit must be >= 1, got {limit}")

        conn = self._news_store._conn
        where_since = ""
        params: list[Any] = []
        if since is not None:
            where_since = " AND timestamp >= ?"
            params.append(_normalize_since(since))
        sql = f"""
            SELECT id, headline, snippet FROM news_events
            WHERE (
                      finbert_confidence IS NULL
                   OR finbert_negative IS NULL
                   OR finbert_neutral IS NULL
                   OR sentiment_score IS NULL
                  )
                  {where_since}
            ORDER BY timestamp
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            return 0

        texts = [_join_text(h, s) for _id, h, s in rows]
        scored = self._scorer.score_texts(texts)
        assert len(scored) == len(rows)

        updated = 0
        with self._news_store._cursor() as cur:
            for (row_id, _h, _s), sc in zip(rows, scored):
                cur.execute(
                    """UPDATE news_events
                       SET sentiment_score = COALESCE(sentiment_score, ?),
                           finbert_negative = COALESCE(finbert_negative, ?),
                           finbert_neutral = COALESCE(finbert_neutral, ?),
                           finbert_confidence = COALESCE(finbert_confidence, ?)
                       WHERE id = ? AND (
                                 finbert_confidence IS NULL
                              OR finbert_negative IS NULL
                              OR finbert_neutral IS NULL
                              OR sentiment_score IS NULL
                             )""",
                    (
                        float(sc["positive"]),
                        float(sc["negative"]),
                        float(sc["neutral"]),
                        float(sc["confidence"]),
                        row_id,
                    ),
                )
                # review H4: cur.rowcount==-1 (某些驱动) 当 0 处理,不回退计数
                updated += max(cur.rowcount, 0)
        logger.info(
            "LocalSentimentProvider.backfill: %d candidates, %d written",
            len(rows),
            updated,
        )
        return updated

    # ------ Registry ------

    @classmethod
    def register_default(
        cls,
        registry: Any,
        news_store: Any | None = None,
    ) -> "LocalSentimentProvider":
        """注册本 Provider 类到 Registry 并可选创建默认实例。

        Args:
            registry: ProviderRegistry 实例
            news_store: 注入给 LocalSentimentProvider 的 news_store;
                        None 时调用方负责后续注入

        Returns:
            创建并注册的 LocalSentimentProvider 实例 (注入了 registry.cache)
        """
        registry.register("LocalSentimentProvider", cls)
        cfg = ProviderConfig(implementation="LocalSentimentProvider")
        instance = cls(cfg, news_store=news_store)
        # review H3: 手工创建的实例也必须注入 cache,
        # 与 registry.create() 行为一致 (registry.py:70)
        instance.cache = registry.cache
        if news_store is not None and not getattr(news_store, "is_initialized", False):
            news_store.initialize()
        instance.initialize()
        registry._instances["SentimentProvider"] = instance
        return instance

    # ------ 内部 ------

    def _require_initialized(self) -> None:
        if not self.is_initialized:
            raise RuntimeError(
                "LocalSentimentProvider not initialized; call .initialize()"
            )

    def _require_news_store(self) -> None:
        if self._news_store is None:
            raise RuntimeError(
                "news_store not injected; pass via ctor or setter before calling"
            )
        if getattr(self._news_store, "_conn", None) is None:
            raise RuntimeError(
                "news_store._conn is None; call news_store.initialize() first"
            )


# ---- 模块级辅助 ----


def _join_text(headline: str | None, snippet: str | None) -> str:
    h = (headline or "").strip()
    s = (snippet or "").strip()
    if not s:
        return h
    return f"{h} {s}" if h else s


def _normalize_since(since: Any) -> str:
    import pandas as pd

    ts = pd.Timestamp(since)
    if ts.tz is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.isoformat()


__all__ = ["LocalSentimentProvider"]
