"""EconomicCalendar — FRED release calendar 经济事件日历。

通过 FRED API (/fred/releases/dates) 获取经济数据发布日历。
标记高波动事件日（FOMC、非农、CPI 等），供 risk control 和 portfolio rebalancing 使用。

关键发布日（高波动）：
- FOMC 会议（每年 8 次）
- 非农就业 (NFP)（每月第一个周五）
- CPI（每月中旬）
- GDP（每季度末）
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)

FRED_API_BASE = "https://api.stlouisfed.org/fred"

# 高波动发布的 FRED release IDs
HIGH_VOLATILITY_RELEASES: dict[int, str] = {
    10:  "CPI (Consumer Price Index)",
    50:  "Employment Situation (NFP)",
    53:  "GDP (Gross Domestic Product)",
    21:  "PPI (Producer Price Index)",
    46:  "Federal Funds Rate (FOMC)",
    268: "Treasury Yield Curve Rates",
    11:  "Industrial Production and Capacity Utilization",
    22:  "M2 Money Stock",
}


@dataclass
class ReleaseEvent:
    """单个经济数据发布事件。"""
    release_id: int
    release_name: str
    release_date: date
    high_volatility: bool


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS economic_calendar (
    release_id    INTEGER NOT NULL,
    release_name  TEXT NOT NULL,
    release_date  TEXT NOT NULL,
    high_volatility INTEGER DEFAULT 0,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (release_id, release_date)
);

CREATE INDEX IF NOT EXISTS idx_calendar_date
    ON economic_calendar(release_date);
"""


class EconomicCalendar:
    """FRED 经济数据发布日历。

    使用方式：
        cal = EconomicCalendar(api_key="xxx", db_path="data/macro_sources.db")
        cal.initialize()
        cal.sync(days_ahead=90)
        events = cal.get_events(start=date.today(), end=date.today() + timedelta(days=30))
        is_volatile = cal.is_high_volatility_day(date.today())
    """

    def __init__(
        self,
        api_key: str = "",
        db_path: str | Path = "data/macro_sources.db",
    ) -> None:
        self._api_key = api_key
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("EconomicCalendar ready: %s", self._db_path)

    def shutdown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def sync(self, days_ahead: int = 90) -> int:
        """从 FRED API 同步经济数据发布日历。

        Args:
            days_ahead: 同步未来多少天的事件

        Returns:
            同步的事件数
        """
        if not self._api_key:
            logger.warning("No FRED API key, skipping calendar sync")
            return 0

        end = date.today() + timedelta(days=days_ahead)
        start = date.today() - timedelta(days=30)  # 也拉最近 30 天的

        dates_data = self._fetch_release_dates(start, end)
        if not dates_data:
            return 0

        events = self._parse_release_dates(dates_data)
        saved = self._save_events(events)
        logger.info("Calendar synced: %d events (%d high-volatility)",
                     saved, sum(1 for e in events if e.high_volatility))
        return saved

    def get_events(
        self,
        start: date | None = None,
        end: date | None = None,
        high_volatility_only: bool = False,
    ) -> list[ReleaseEvent]:
        """查询日历事件。"""
        if not self._conn:
            return []

        query = "SELECT release_id, release_name, release_date, high_volatility FROM economic_calendar WHERE 1=1"
        params: list[Any] = []

        if start:
            query += " AND release_date >= ?"
            params.append(start.isoformat())
        if end:
            query += " AND release_date <= ?"
            params.append(end.isoformat())
        if high_volatility_only:
            query += " AND high_volatility = 1"

        query += " ORDER BY release_date"

        cur = self._conn.execute(query, params)
        return [
            ReleaseEvent(
                release_id=row[0],
                release_name=row[1],
                release_date=date.fromisoformat(row[2]),
                high_volatility=bool(row[3]),
            )
            for row in cur.fetchall()
        ]

    def is_high_volatility_day(self, d: date) -> bool:
        """判断某天是否有高波动数据发布。"""
        events = self.get_events(start=d, end=d, high_volatility_only=True)
        return len(events) > 0

    def get_next_high_volatility(self, after: date | None = None) -> ReleaseEvent | None:
        """获取下一个高波动事件。"""
        after = after or date.today()
        events = self.get_events(
            start=after,
            end=after + timedelta(days=90),
            high_volatility_only=True,
        )
        return events[0] if events else None

    # ------ Internal ------

    def _fetch_release_dates(self, start: date, end: date) -> list[dict]:
        """从 FRED API 获取发布日期。"""
        url = (
            f"{FRED_API_BASE}/releases/dates"
            f"?api_key={self._api_key}"
            f"&realtime_start={start.isoformat()}"
            f"&realtime_end={end.isoformat()}"
            f"&include_release_dates_with_no_data=true"
            f"&file_type=json"
        )

        req = Request(url, headers={"Accept": "application/json"})
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
                return data.get("release_dates", [])
        except (URLError, json.JSONDecodeError):
            logger.exception("FRED release dates API error")
            return []

    def _parse_release_dates(self, dates_data: list[dict]) -> list[ReleaseEvent]:
        """解析 FRED release dates 响应。"""
        events: list[ReleaseEvent] = []
        for item in dates_data:
            release_id = item.get("release_id")
            release_name = item.get("release_name", "")
            release_date_str = item.get("date", "")

            if not release_id or not release_date_str:
                continue

            try:
                release_date = date.fromisoformat(release_date_str)
            except ValueError:
                continue

            events.append(ReleaseEvent(
                release_id=int(release_id),
                release_name=release_name,
                release_date=release_date,
                high_volatility=int(release_id) in HIGH_VOLATILITY_RELEASES,
            ))
        return events

    def _save_events(self, events: list[ReleaseEvent]) -> int:
        """保存事件到 SQLite。"""
        if not self._conn or not events:
            return 0

        now = date.today().isoformat()
        cur = self._conn.cursor()
        for e in events:
            cur.execute(
                """INSERT OR REPLACE INTO economic_calendar
                   (release_id, release_name, release_date, high_volatility, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (e.release_id, e.release_name, e.release_date.isoformat(),
                 int(e.high_volatility), now),
            )
        self._conn.commit()
        return len(events)
