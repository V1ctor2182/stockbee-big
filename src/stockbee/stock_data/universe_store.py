"""SqliteUniverseProvider — SQLite 四层漏斗股票池管理。

四层漏斗设计（8000→4000→500→100→8）：
- broad_all:  广域宇宙，全部美股
- broad:      广泛宇宙，过滤流动性不足/不可融资
- candidate:  候选池，初始因子筛选
- u100:       精选 100 只，月度因子评分

表结构:
- universe_members: 股票池成员（支持 as_of 时间切片）
- universe_snapshots: 快照历史（避免前看偏差）
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Generator

import pandas as pd

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import UniverseProvider

logger = logging.getLogger(__name__)

VALID_LEVELS = ("broad_all", "broad", "candidate", "u100")

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS universe_members (
    ticker       TEXT    NOT NULL,
    level        TEXT    NOT NULL CHECK(level IN ('broad_all','broad','candidate','u100')),
    sector       TEXT,
    market_cap   REAL,
    avg_volume   REAL,
    avg_dollar_volume REAL,
    short_able   INTEGER DEFAULT 1,
    added_at     TEXT    NOT NULL,
    removed_at   TEXT,
    PRIMARY KEY (ticker, level, added_at)
);

CREATE INDEX IF NOT EXISTS idx_universe_level_active
    ON universe_members(level, removed_at);

CREATE TABLE IF NOT EXISTS universe_snapshots (
    snapshot_date TEXT NOT NULL,
    level         TEXT NOT NULL,
    ticker_count  INTEGER NOT NULL,
    tickers_json  TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, level)
);
"""


class SqliteUniverseProvider(UniverseProvider):
    """SQLite 实现的股票池 Provider，支持四层漏斗和时间切片。"""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        self._db_path = Path(params.get("db_path", "data/universe.db"))
        self._conn: sqlite3.Connection | None = None

    def _do_initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("SqliteUniverseProvider ready: %s", self._db_path)

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

    def get_universe(
        self, level: str = "u100", as_of: date | None = None
    ) -> pd.DataFrame:
        if level not in VALID_LEVELS:
            raise ValueError(f"Invalid level: {level!r}. Must be one of {VALID_LEVELS}")

        if as_of:
            # 时间切片：查找该日期最近的快照
            return self._get_universe_snapshot(level, as_of)

        # 当前活跃成员
        query = """
            SELECT ticker, sector, market_cap, avg_volume, avg_dollar_volume, short_able
            FROM universe_members
            WHERE level = ? AND removed_at IS NULL
            ORDER BY ticker
        """
        with self._cursor() as cur:
            cur.execute(query, (level,))
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["ticker", "sector", "market_cap", "avg_volume",
                         "avg_dollar_volume", "short_able"]
            )

        df = pd.DataFrame(rows, columns=[
            "ticker", "sector", "market_cap", "avg_volume",
            "avg_dollar_volume", "short_able",
        ])
        df["short_able"] = df["short_able"].astype(bool)
        return df

    def _get_universe_snapshot(self, level: str, as_of: date) -> pd.DataFrame:
        """通过快照历史实现时间切片，避免前看偏差。"""
        query = """
            SELECT ticker, sector, market_cap, avg_volume, avg_dollar_volume, short_able
            FROM universe_members
            WHERE level = ?
              AND added_at <= ?
              AND (removed_at IS NULL OR removed_at > ?)
            ORDER BY ticker
        """
        as_of_str = as_of.isoformat()
        with self._cursor() as cur:
            cur.execute(query, (level, as_of_str, as_of_str))
            rows = cur.fetchall()

        if not rows:
            return pd.DataFrame(
                columns=["ticker", "sector", "market_cap", "avg_volume",
                         "avg_dollar_volume", "short_able"]
            )

        df = pd.DataFrame(rows, columns=[
            "ticker", "sector", "market_cap", "avg_volume",
            "avg_dollar_volume", "short_able",
        ])
        df["short_able"] = df["short_able"].astype(bool)
        return df

    def refresh_universe(self, level: str = "broad_all") -> int:
        """占位：实际刷新逻辑由 StockDataSyncer 调用 upsert_members 实现。"""
        logger.info("refresh_universe called for level=%s (no-op, use syncer)", level)
        return 0

    # ------ 写入接口（供 sync 管道和漏斗逻辑使用）------

    def upsert_members(
        self,
        level: str,
        members: pd.DataFrame,
        snapshot_date: date | None = None,
    ) -> int:
        """批量更新某层级的股票池成员。

        Args:
            level: 漏斗层级
            members: DataFrame with columns: ticker, sector, market_cap,
                     avg_volume, avg_dollar_volume, short_able
            snapshot_date: 快照日期（默认今天）

        Returns:
            更新的成员数
        """
        if level not in VALID_LEVELS:
            raise ValueError(f"Invalid level: {level!r}")

        snapshot_date = snapshot_date or date.today()
        date_str = snapshot_date.isoformat()

        new_tickers = set(members["ticker"].str.upper())

        with self._cursor() as cur:
            # 标记已移除的成员
            cur.execute(
                "SELECT ticker FROM universe_members WHERE level = ? AND removed_at IS NULL",
                (level,),
            )
            existing_tickers = {row[0] for row in cur.fetchall()}

            removed = existing_tickers - new_tickers
            if removed:
                cur.executemany(
                    "UPDATE universe_members SET removed_at = ? WHERE ticker = ? AND level = ? AND removed_at IS NULL",
                    [(date_str, t, level) for t in removed],
                )

            # Upsert 新成员
            added = new_tickers - existing_tickers
            for _, row in members.iterrows():
                ticker = row["ticker"].upper()
                if ticker in added:
                    cur.execute(
                        """INSERT INTO universe_members
                           (ticker, level, sector, market_cap, avg_volume, avg_dollar_volume, short_able, added_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ticker, level,
                            row.get("sector"),
                            row.get("market_cap"),
                            row.get("avg_volume"),
                            row.get("avg_dollar_volume"),
                            int(row.get("short_able", True)),
                            date_str,
                        ),
                    )
                else:
                    # 更新现有成员的指标
                    cur.execute(
                        """UPDATE universe_members
                           SET sector = ?, market_cap = ?, avg_volume = ?,
                               avg_dollar_volume = ?, short_able = ?
                           WHERE ticker = ? AND level = ? AND removed_at IS NULL""",
                        (
                            row.get("sector"),
                            row.get("market_cap"),
                            row.get("avg_volume"),
                            row.get("avg_dollar_volume"),
                            int(row.get("short_able", True)),
                            ticker, level,
                        ),
                    )

            # 保存快照
            import json
            cur.execute(
                """INSERT OR REPLACE INTO universe_snapshots
                   (snapshot_date, level, ticker_count, tickers_json)
                   VALUES (?, ?, ?, ?)""",
                (date_str, level, len(new_tickers), json.dumps(sorted(new_tickers))),
            )

        logger.info(
            "Upserted %s level=%s: %d members (+%d added, -%d removed)",
            date_str, level, len(new_tickers), len(added), len(removed),
        )
        return len(new_tickers)

    def get_member_count(self, level: str) -> int:
        """返回某层级的当前活跃成员数。"""
        with self._cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM universe_members WHERE level = ? AND removed_at IS NULL",
                (level,),
            )
            return cur.fetchone()[0]

    def get_all_level_counts(self) -> dict[str, int]:
        """返回所有层级的成员数。"""
        return {level: self.get_member_count(level) for level in VALID_LEVELS}
