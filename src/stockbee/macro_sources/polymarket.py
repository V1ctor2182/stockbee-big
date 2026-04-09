"""PolymarketFetcher — Polymarket 宏观事件概率抓取。

通过 Gamma API (https://gamma-api.polymarket.com) 抓取宏观事件概率。
检测"概率悬崖"（前后概率突变 > 阈值），作为 MacroTiltEngine 的外生输入。

来源：Tech Design §4.3
- 定期爬取 Polymarket 前 30 大宏观事件的隐含概率
- 与前期概率对比，检测"概率悬崖"
- 将概率作为 MacroTiltEngine 的补充因子
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
DEFAULT_CLIFF_THRESHOLD = 0.15  # 概率变化 > 15% 视为悬崖

# 宏观事件关键词过滤（question 中包含任一即视为宏观相关）
MACRO_KEYWORDS = [
    "fed ", "federal reserve", "fomc", "interest rate", "rate cut", "rate hike",
    "recession", "inflation", "cpi", "gdp", "unemployment", "tariff",
    "trade war", "treasury", "debt ceiling", "deficit", "stimulus",
    "central bank", "monetary policy", "fiscal",
]


@dataclass
class MarketEvent:
    """单个 Polymarket 事件。"""
    event_id: str
    question: str
    slug: str
    probability: float
    previous_probability: float | None
    volume: float
    liquidity: float
    is_cliff: bool
    fetched_at: str


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS polymarket_events (
    event_id    TEXT NOT NULL,
    question    TEXT NOT NULL,
    slug        TEXT,
    probability REAL NOT NULL,
    previous_probability REAL,
    volume      REAL,
    liquidity   REAL,
    is_cliff    INTEGER DEFAULT 0,
    fetched_at  TEXT NOT NULL,
    PRIMARY KEY (event_id, fetched_at)
);

CREATE INDEX IF NOT EXISTS idx_polymarket_fetched
    ON polymarket_events(fetched_at);
"""


class PolymarketFetcher:
    """Polymarket 宏观事件概率抓取器。

    使用方式：
        fetcher = PolymarketFetcher(db_path="data/macro_sources.db")
        fetcher.initialize()
        events = fetcher.fetch_macro_events(limit=30)
        cliffs = fetcher.detect_cliffs(events)
    """

    def __init__(
        self,
        db_path: str | Path = "data/macro_sources.db",
        cliff_threshold: float = DEFAULT_CLIFF_THRESHOLD,
    ) -> None:
        self._db_path = Path(db_path)
        self._cliff_threshold = cliff_threshold
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("PolymarketFetcher ready: %s", self._db_path)

    def shutdown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def fetch_macro_events(self, limit: int = 30, fetch_size: int = 200) -> list[MarketEvent]:
        """从 Gamma API 抓取宏观相关事件。

        拉取 fetch_size 个热门市场，本地用关键词过滤出宏观事件，返回前 limit 个。

        Args:
            limit: 最多返回的宏观事件数
            fetch_size: 从 API 拉取的总市场数（过滤前）

        Returns:
            MarketEvent 列表，按 volume 降序
        """
        raw_markets = self._call_api(
            "/markets",
            params={"limit": fetch_size, "active": "true", "order": "volume", "ascending": "false"},
        )
        if not raw_markets:
            return []

        # 本地关键词过滤
        raw_markets = [m for m in raw_markets if self._is_macro_related(m)]

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        events: list[MarketEvent] = []

        for market in raw_markets[:limit]:
            event_id = market.get("id", "")
            question = market.get("question", "")
            if not event_id or not question:
                continue

            # 提取概率（outcomePrices 是 JSON 字符串 "[yes_price, no_price]"）
            probability = self._extract_probability(market)
            if probability is None:
                continue

            previous = self._get_previous_probability(event_id)

            event = MarketEvent(
                event_id=event_id,
                question=question,
                slug=market.get("slug", ""),
                probability=probability,
                previous_probability=previous,
                volume=float(market.get("volume", 0) or 0),
                liquidity=float(market.get("liquidity", 0) or 0),
                is_cliff=self._is_cliff(probability, previous),
                fetched_at=now,
            )
            events.append(event)

        logger.info("Fetched %d events from Polymarket", len(events))
        return events

    def save_events(self, events: list[MarketEvent]) -> int:
        """保存事件到 SQLite。返回保存的数量。"""
        if not self._conn or not events:
            return 0

        cur = self._conn.cursor()
        for e in events:
            cur.execute(
                """INSERT OR REPLACE INTO polymarket_events
                   (event_id, question, slug, probability, previous_probability,
                    volume, liquidity, is_cliff, fetched_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (e.event_id, e.question, e.slug, e.probability,
                 e.previous_probability, e.volume, e.liquidity,
                 int(e.is_cliff), e.fetched_at),
            )
        self._conn.commit()
        return len(events)

    def detect_cliffs(self, events: list[MarketEvent] | None = None) -> list[MarketEvent]:
        """返回概率悬崖事件（概率变化 > threshold）。"""
        if events is None:
            events = self.fetch_macro_events()
        return [e for e in events if e.is_cliff]

    def get_latest_events(self, limit: int = 30) -> list[dict[str, Any]]:
        """从 SQLite 读取最近一次抓取的事件。"""
        if not self._conn:
            return []

        cur = self._conn.execute(
            """SELECT event_id, question, probability, previous_probability,
                      volume, is_cliff, fetched_at
               FROM polymarket_events
               WHERE fetched_at = (SELECT MAX(fetched_at) FROM polymarket_events)
               ORDER BY volume DESC
               LIMIT ?""",
            (limit,),
        )
        return [
            {
                "event_id": row[0], "question": row[1], "probability": row[2],
                "previous_probability": row[3], "volume": row[4],
                "is_cliff": bool(row[5]), "fetched_at": row[6],
            }
            for row in cur.fetchall()
        ]

    # ------ Internal ------

    def _call_api(self, path: str, params: dict | None = None) -> list[dict]:
        """调用 Gamma API。"""
        url = GAMMA_API_BASE + path
        if params:
            url += "?" + urlencode(params)

        req = Request(url, headers={
            "Accept": "application/json",
            "User-Agent": "StockBEE/1.0",
        })
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                if isinstance(data, list):
                    return data
                return data.get("data", data.get("markets", []))
        except (URLError, json.JSONDecodeError):
            logger.exception("Polymarket API error: %s", url)
            return []

    def _extract_probability(self, market: dict) -> float | None:
        """从 market 数据提取 Yes 概率。"""
        prices = market.get("outcomePrices")
        if prices:
            try:
                parsed = json.loads(prices) if isinstance(prices, str) else prices
                return float(parsed[0])  # Yes price = probability
            except (json.JSONDecodeError, IndexError, TypeError):
                pass
        # Fallback
        best_ask = market.get("bestAsk")
        if best_ask:
            return float(best_ask)
        return None

    def _get_previous_probability(self, event_id: str) -> float | None:
        """从 SQLite 获取该事件的上一次概率。"""
        if not self._conn:
            return None
        cur = self._conn.execute(
            """SELECT probability FROM polymarket_events
               WHERE event_id = ?
               ORDER BY fetched_at DESC LIMIT 1""",
            (event_id,),
        )
        row = cur.fetchone()
        return row[0] if row else None

    @staticmethod
    def _is_macro_related(market: dict) -> bool:
        """判断一个市场是否和宏观经济相关。"""
        text = (market.get("question", "") + " " + market.get("description", "")).lower()
        return any(kw in text for kw in MACRO_KEYWORDS)

    def _is_cliff(self, current: float, previous: float | None) -> bool:
        """判断是否为概率悬崖。"""
        if previous is None:
            return False
        return abs(current - previous) >= self._cliff_threshold
