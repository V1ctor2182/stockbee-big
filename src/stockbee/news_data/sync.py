"""NewsDataSyncer — fetch → G1 → store → G2 → G3 编排管道。

编排多数据源的新闻拉取和三级漏斗处理：
  fetch (多源) → 跨源去重 → G1 filter → store (g_level=1)
  → G2 classify (g_level=2) → G3 analyze (条件触发, g_level=3)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from stockbee.news_data.g1_filter import G1Filter
from stockbee.news_data.g2_classifier import G2Classifier, G2Result
from stockbee.news_data.g3_analyzer import G3Analyzer, G3Result
from stockbee.news_data.news_store import SqliteNewsProvider
from stockbee.news_data.sources import NewsSource

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """同步结果摘要。"""
    fetched: int = 0
    after_dedup: int = 0
    g1_passed: int = 0
    stored: int = 0
    g2_classified: int = 0
    g3_analyzed: int = 0
    errors: list[str] = field(default_factory=list)
    sources_used: list[str] = field(default_factory=list)


def _normalize_headline(headline: str) -> str:
    """Headline 归一化用于去重比较。lowercase + collapse whitespace。"""
    return " ".join(headline.lower().split())


class NewsDataSyncer:
    """新闻数据同步管道。

    编排 fetch → G1 → store → G2 → G3 的完整流程。
    支持多数据源，graceful degradation（单源失败不影响其他源）。
    """

    def __init__(
        self,
        store: SqliteNewsProvider,
        g1: G1Filter,
        g2: G2Classifier,
        g3: G3Analyzer,
        sources: list[NewsSource] | None = None,
    ) -> None:
        self._store = store
        self._g1 = g1
        self._g2 = g2
        self._g3 = g3
        self._sources: list[NewsSource] = sources or []

    def add_source(self, source: NewsSource) -> None:
        """动态添加数据源。"""
        self._sources.append(source)

    # ------ 主入口 ------

    def ingest_news(
        self,
        source: str = "all",
        keywords: list[str] | None = None,
        tickers: list[str] | None = None,
        days: int = 1,
    ) -> SyncResult:
        """拉取新闻并走完 G1/G2/G3 管道。

        Args:
            source: "all" 或具体 source_name ("newsapi"/"perigon"/"perplexity")
            keywords: 搜索关键词
            tickers: 按 ticker 搜索
            days: 回溯天数
        """
        now = datetime.now(timezone.utc)
        from_dt = now - timedelta(days=days)
        result = SyncResult()

        # 1. Fetch from sources
        raw_events = self._fetch_all(source, keywords, tickers, from_dt, now, result)
        result.fetched = len(raw_events)
        if not raw_events:
            return result

        # 2. Cross-source dedup (headline normalize)
        deduped = self._dedup_cross_source(raw_events)
        result.after_dedup = len(deduped)

        # 3. G1 filter
        g1_passed = self._run_g1(deduped)
        result.g1_passed = len(g1_passed)
        if not g1_passed:
            return result

        # 4. Store (g_level=1)
        stored = self._store_g1(g1_passed)
        result.stored = len(stored)
        if not stored:
            return result

        # 5. G2 classify → update g_level=2
        g2_items = self._run_g2(stored)
        result.g2_classified = len(g2_items)

        # 6. G3 analyze (conditional) → update g_level=3
        result.g3_analyzed = self._run_g3(g2_items)

        return result

    def sync_latest(
        self,
        keywords: list[str] | None = None,
        tickers: list[str] | None = None,
    ) -> SyncResult:
        """拉取最近 1 天的新闻。便捷方法。"""
        return self.ingest_news(source="all", keywords=keywords, tickers=tickers, days=1)

    # ------ 内部 pipeline 步骤 ------

    def _fetch_all(
        self,
        source_filter: str,
        keywords: list[str] | None,
        tickers: list[str] | None,
        from_dt: datetime,
        to_dt: datetime,
        result: SyncResult,
    ) -> list[dict]:
        """从多源拉取，单源失败不影响其他源。"""
        all_events: list[dict] = []
        for src in self._sources:
            if source_filter != "all" and src.source_name != source_filter:
                continue
            try:
                events = src.fetch(
                    keywords=keywords, tickers=tickers,
                    from_dt=from_dt, to_dt=to_dt,
                )
                all_events.extend(events)
                result.sources_used.append(src.source_name)
                logger.info("Fetched %d events from %s", len(events), src.source_name)
            except Exception as e:
                logger.warning("Source %s failed: %s", src.source_name, e)
                result.errors.append(f"{src.source_name}: {e}")
        return all_events

    def _dedup_cross_source(self, events: list[dict]) -> list[dict]:
        """跨源去重：headline normalize 后精确匹配，保留 snippet 最长的一条。"""
        seen: dict[str, dict] = {}
        for event in events:
            headline = event.get("headline", "")
            if not headline:
                continue
            key = _normalize_headline(headline)
            if key not in seen:
                seen[key] = event
            else:
                # 保留 snippet 更长的，或时间更早的
                existing = seen[key]
                existing_snippet = existing.get("snippet") or ""
                new_snippet = event.get("snippet") or ""
                if len(new_snippet) > len(existing_snippet):
                    seen[key] = event
        return list(seen.values())

    def _run_g1(self, events: list[dict]) -> list[dict]:
        """G1 过滤，给通过的 event 附加 tickers。"""
        passed: list[dict] = []
        for event in events:
            g1r = self._g1.filter(
                headline=event.get("headline", ""),
                source=event.get("source", ""),
                timestamp=event.get("timestamp", ""),
                snippet=event.get("snippet"),
                source_url=event.get("source_url"),
            )
            if g1r.passed:
                event["tickers"] = g1r.tickers
                event["g_level"] = 1
                passed.append(event)
        return passed

    def _store_g1(self, events: list[dict]) -> list[tuple[dict, int]]:
        """批量写入 g_level=1（单事务），返回 (event, news_id) 对。"""
        return self._store.insert_news_batch_with_ids(events)

    def _run_g2(self, items: list[tuple[dict, int]]) -> list[tuple[dict, int, G2Result]]:
        """G2 分类并更新 g_level=2。"""
        results: list[tuple[dict, int, G2Result]] = []
        # 批量分类
        batch_input = [
            {"headline": ev.get("headline", ""), "snippet": ev.get("snippet")}
            for ev, _ in items
        ]
        g2_results = self._g2.classify_batch(batch_input)

        for (event, news_id), g2r in zip(items, g2_results):
            self._store.update_g_level(
                news_id, g_level=2,
                sentiment_score=g2r.sentiment_score,
                importance_score=g2r.importance_score,
            )
            results.append((event, news_id, g2r))
        return results

    def _run_g3(self, items: list[tuple[dict, int, G2Result]]) -> int:
        """G3 条件分析，更新 g_level=3。返回分析数量。"""
        count = 0
        for event, news_id, g2r in items:
            if not self._g3.should_analyze(g2r):
                continue
            g3r = self._g3.analyze(
                headline=event.get("headline", ""),
                snippet=event.get("snippet"),
                tickers=event.get("tickers"),
                g2_result=g2r,
            )
            if g3r is None:
                continue
            self._store.update_g_level(
                news_id, g_level=3,
                reliability_score=g3r.reliability_score,
                analysis=json.dumps({
                    "weight_action": g3r.weight_action,
                    "weight_magnitude": g3r.weight_magnitude,
                    "reliability_score": g3r.reliability_score,
                    "reasoning": g3r.reasoning,
                    "confidence": g3r.confidence,
                }),
            )
            count += 1
        return count
