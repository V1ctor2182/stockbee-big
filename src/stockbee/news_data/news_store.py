"""SqliteNewsProvider — SQLite news_events 表 + 查询接口。

news_events 表存储所有经过 G1/G2/G3 处理的新闻事件。
支持按 ticker、时间范围、重要度、g_level 多条件查询。
WAL 模式保证并发读写安全。

Schema:
    news_events: id, timestamp, source, source_url, tickers, headline,
                 snippet, sentiment_score, importance_score,
                 reliability_score, g_level, analysis, created_at
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import NewsProvider

logger = logging.getLogger(__name__)

MAX_HEADLINE_LENGTH = 500
MAX_SNIPPET_LENGTH = 2000

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS news_events (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp         TEXT    NOT NULL,
    source            TEXT    NOT NULL,
    source_url        TEXT,
    tickers           TEXT    NOT NULL DEFAULT '[]',
    headline          TEXT    NOT NULL,
    snippet           TEXT,
    sentiment_score   REAL,
    importance_score  REAL,
    reliability_score REAL,
    g_level           INTEGER NOT NULL DEFAULT 0,
    analysis          TEXT,
    created_at        TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_news_timestamp
    ON news_events(timestamp);

CREATE INDEX IF NOT EXISTS idx_news_g_level
    ON news_events(g_level);

CREATE INDEX IF NOT EXISTS idx_news_importance
    ON news_events(importance_score);

CREATE TABLE IF NOT EXISTS g3_daily_counts (
    date  TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
);
"""


def _normalize_timestamp(ts: str | datetime) -> str:
    """归一化时间戳为 UTC ISO 8601 字符串。"""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.astimezone(timezone.utc).isoformat()
    # 字符串：尝试解析后归一化
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except (ValueError, TypeError):
        logger.warning("Invalid timestamp format: %r, using current UTC time", ts)
        return datetime.now(timezone.utc).isoformat()


def _truncate(text: str | None, max_len: int) -> str | None:
    """截断超长文本。"""
    if text is None:
        return None
    return text[:max_len] if len(text) > max_len else text


def _normalize_tickers(tickers: list[str] | str | None) -> str:
    """归一化 tickers 为 JSON 字符串。"""
    if tickers is None:
        return "[]"
    if isinstance(tickers, str):
        try:
            parsed = json.loads(tickers)
            if isinstance(parsed, list):
                return json.dumps(sorted(set(t.upper() for t in parsed if t)))
        except (json.JSONDecodeError, TypeError):
            # 可能是逗号分隔的字符串
            parts = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            return json.dumps(sorted(set(parts)))
    if isinstance(tickers, list):
        cleaned = [t.upper() for t in tickers if isinstance(t, str) and t.strip()]
        return json.dumps(sorted(set(cleaned)))
    return "[]"


class SqliteNewsProvider(NewsProvider):
    """SQLite 实现的新闻数据 Provider。

    功能:
    - news_events 表 CRUD
    - 按 ticker/时间/重要度/g_level 多条件查询
    - 插入时自动去重（相同 headline + source 组合）
    - G3 每日分析计数器
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        self._db_path = Path(params.get("db_path", "data/news.db"))
        self._conn: sqlite3.Connection | None = None

    def _do_initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        count = self._count_all()
        logger.info("SqliteNewsProvider ready: %s (%d events)", self._db_path, count)

    def _do_shutdown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        if not self._conn:
            raise RuntimeError("Provider not initialized")
        cur = self._conn.cursor()
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # ------ NewsProvider 接口实现 ------

    def get_news(
        self,
        tickers: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        min_importance: float = 0.0,
        g_level: int | None = None,
    ) -> pd.DataFrame:
        """查询新闻事件，支持多条件组合过滤。"""
        conditions: list[str] = []
        params: list[Any] = []

        if tickers:
            # 查找 tickers JSON 列中包含任意指定 ticker 的行
            ticker_conditions = []
            for t in tickers:
                ticker_conditions.append("tickers LIKE ?")
                params.append(f'%"{t.upper()}"%')
            conditions.append(f"({' OR '.join(ticker_conditions)})")

        if start:
            conditions.append("timestamp >= ?")
            params.append(_normalize_timestamp(start))

        if end:
            conditions.append("timestamp <= ?")
            params.append(_normalize_timestamp(end))

        if min_importance > 0.0:
            conditions.append("importance_score >= ?")
            params.append(min_importance)

        if g_level is not None:
            conditions.append("g_level >= ?")
            params.append(g_level)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT id, timestamp, source, tickers, headline, snippet,
                   sentiment_score, importance_score, reliability_score, g_level
            FROM news_events
            {where}
            ORDER BY timestamp DESC
        """

        with self._cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["id", "timestamp", "source", "tickers", "headline",
                         "snippet", "sentiment_score", "importance_score",
                         "reliability_score", "g_level"]
            )

        df = pd.DataFrame(rows, columns=[
            "id", "timestamp", "source", "tickers", "headline",
            "snippet", "sentiment_score", "importance_score",
            "reliability_score", "g_level",
        ])
        # 解析 tickers JSON 列为 list
        df["tickers"] = df["tickers"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else []
        )
        return df

    def ingest_news(self, source: str = "all") -> int:
        """占位：实际拉取逻辑由 NewsDataSyncer 编排。"""
        logger.info("ingest_news called (source=%s) — no-op, use syncer", source)
        return 0

    # ------ 写入接口（供 G1/syncer 使用）------

    def insert_news(
        self,
        headline: str,
        source: str,
        timestamp: str | datetime,
        tickers: list[str] | str | None = None,
        snippet: str | None = None,
        source_url: str | None = None,
        sentiment_score: float | None = None,
        importance_score: float | None = None,
        reliability_score: float | None = None,
        g_level: int = 0,
        analysis: str | None = None,
    ) -> int | None:
        """插入一条新闻事件，自动去重。

        去重规则：相同 headline + source 的组合视为重复。

        Returns:
            插入的行 id，如果是重复则返回 None
        """
        if not headline or not headline.strip():
            logger.debug("Skipping news with empty headline")
            return None

        headline = _truncate(headline.strip(), MAX_HEADLINE_LENGTH)
        snippet = _truncate(snippet, MAX_SNIPPET_LENGTH)
        ts_normalized = _normalize_timestamp(timestamp)
        tickers_json = _normalize_tickers(tickers)
        now = datetime.now(timezone.utc).isoformat()

        with self._cursor() as cur:
            # 去重检查
            cur.execute(
                "SELECT id FROM news_events WHERE headline = ? AND source = ?",
                (headline, source),
            )
            if cur.fetchone():
                logger.debug("Duplicate news skipped: %s [%s]", headline[:60], source)
                return None

            cur.execute(
                """INSERT INTO news_events
                   (timestamp, source, source_url, tickers, headline, snippet,
                    sentiment_score, importance_score, reliability_score,
                    g_level, analysis, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts_normalized, source, source_url, tickers_json,
                    headline, snippet, sentiment_score, importance_score,
                    reliability_score, g_level, analysis, now,
                ),
            )
            row_id = cur.lastrowid

        logger.debug("Inserted news #%d: %s", row_id, headline[:60])
        return row_id

    def insert_news_batch(self, events: list[dict[str, Any]]) -> int:
        """批量插入新闻事件，跳过重复。

        Args:
            events: 字典列表，每个字典的 key 对应 insert_news 的参数

        Returns:
            实际插入的条数
        """
        inserted = 0
        for event in events:
            result = self.insert_news(**event)
            if result is not None:
                inserted += 1
        return inserted

    def update_g_level(
        self,
        news_id: int,
        g_level: int,
        sentiment_score: float | None = None,
        importance_score: float | None = None,
        reliability_score: float | None = None,
        analysis: str | None = None,
    ) -> bool:
        """更新新闻的 G 级别和评分（G2/G3 处理后回写）。"""
        sets: list[str] = ["g_level = ?"]
        params: list[Any] = [g_level]

        if sentiment_score is not None:
            sets.append("sentiment_score = ?")
            params.append(sentiment_score)
        if importance_score is not None:
            sets.append("importance_score = ?")
            params.append(importance_score)
        if reliability_score is not None:
            sets.append("reliability_score = ?")
            params.append(reliability_score)
        if analysis is not None:
            sets.append("analysis = ?")
            params.append(analysis)

        params.append(news_id)

        with self._cursor() as cur:
            cur.execute(
                f"UPDATE news_events SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            updated = cur.rowcount > 0

        if not updated:
            logger.warning("News #%d not found for g_level update", news_id)
        return updated

    # ------ G3 每日计数器 ------

    def get_g3_daily_count(self, date_str: str | None = None) -> int:
        """获取某日的 G3 分析次数。"""
        date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._cursor() as cur:
            cur.execute(
                "SELECT count FROM g3_daily_counts WHERE date = ?", (date_str,)
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def increment_g3_daily_count(self, date_str: str | None = None) -> int:
        """递增某日的 G3 分析次数，返回新计数。"""
        date_str = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO g3_daily_counts (date, count) VALUES (?, 1)
                   ON CONFLICT(date) DO UPDATE SET count = count + 1""",
                (date_str,),
            )
            cur.execute(
                "SELECT count FROM g3_daily_counts WHERE date = ?", (date_str,)
            )
            return cur.fetchone()[0]

    # ------ 辅助查询 ------

    def _count_all(self) -> int:
        """返回 news_events 总条数。"""
        if not self._conn:
            return 0
        cur = self._conn.execute("SELECT COUNT(*) FROM news_events")
        return cur.fetchone()[0]

    def get_news_by_id(self, news_id: int) -> dict[str, Any] | None:
        """按 id 查询单条新闻。"""
        with self._cursor() as cur:
            cur.execute(
                """SELECT id, timestamp, source, source_url, tickers, headline,
                          snippet, sentiment_score, importance_score,
                          reliability_score, g_level, analysis, created_at
                   FROM news_events WHERE id = ?""",
                (news_id,),
            )
            row = cur.fetchone()

        if not row:
            return None

        columns = [
            "id", "timestamp", "source", "source_url", "tickers", "headline",
            "snippet", "sentiment_score", "importance_score",
            "reliability_score", "g_level", "analysis", "created_at",
        ]
        result = dict(zip(columns, row))
        result["tickers"] = json.loads(result["tickers"]) if result["tickers"] else []
        return result

    def count_by_g_level(self) -> dict[int, int]:
        """返回各 g_level 的新闻条数。"""
        with self._cursor() as cur:
            cur.execute(
                "SELECT g_level, COUNT(*) FROM news_events GROUP BY g_level"
            )
            return dict(cur.fetchall())
