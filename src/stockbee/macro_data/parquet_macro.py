"""ParquetMacroProvider — Parquet 时间序列 + SQLite Z-score 索引。

回测模式下的 MacroProvider 实现。
- 历史数据：data/macro/macro_all.parquet（单文件，19 列 × N 天）
- Z-score 索引：data/macro_index.db（最新值 + Z-score 缓存）
- Point-in-time 对齐：get_macro_indicators(end=T) 只返回 T-1 及之前的数据
- 不同频率指标 forward-fill 到日频
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import MacroProvider

from .indicators import ALL_CODES, FRED_INDICATORS

logger = logging.getLogger(__name__)

Z_SCORE_WINDOW = 252  # 默认 Z-score 滚动窗口（交易日）
Z_SCORE_CLIP = 3.0    # Z-score 截断范围

_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS macro_latest (
    indicator   TEXT PRIMARY KEY,
    value       REAL,
    z_score     REAL,
    date        TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


class ParquetMacroProvider(MacroProvider):
    """从 Parquet 读取宏观指标历史，SQLite 缓存最新 Z-score。"""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        self._data_path = Path(params.get("data_path", "data/macro"))
        self._index_db_path = Path(params.get("index_db", "data/macro_index.db"))
        self._parquet_path = self._data_path / "macro_all.parquet"
        self._index_conn: sqlite3.Connection | None = None

    def _do_initialize(self) -> None:
        self._data_path.mkdir(parents=True, exist_ok=True)
        self._index_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._index_conn = sqlite3.connect(str(self._index_db_path))
        self._index_conn.execute("PRAGMA journal_mode=WAL")
        self._index_conn.executescript(_INDEX_SCHEMA)
        self._index_conn.commit()
        exists = "exists" if self._parquet_path.exists() else "empty"
        logger.info("ParquetMacroProvider ready: %s (%s)", self._data_path, exists)

    def _do_shutdown(self) -> None:
        if self._index_conn:
            self._index_conn.close()
            self._index_conn = None

    def get_macro_indicators(
        self,
        indicators: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        indicators = indicators or ALL_CODES
        cache_key = f"macro:{','.join(sorted(indicators))}:{start}:{end}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        df = self._read_parquet(indicators)
        if df.empty:
            return df

        # Point-in-time 对齐：end 日期使用 end-1 的数据
        if end:
            pit_end = end - timedelta(days=1)
            df = df.loc[:str(pit_end)]
        if start:
            df = df.loc[str(start):]

        if self.cache and not df.empty:
            self.cache.set(cache_key, df)
        return df

    def get_latest_z_scores(
        self, indicators: list[str] | None = None, window: int = Z_SCORE_WINDOW
    ) -> dict[str, float]:
        indicators = indicators or ALL_CODES

        # 先尝试从 SQLite 索引读取
        cached = self._read_z_scores_from_index(indicators)
        if len(cached) == len(indicators):
            return cached

        # 回退到从 Parquet 计算
        df = self._read_parquet(indicators)
        if df.empty:
            return {}

        z_scores = self._compute_z_scores(df, window)
        latest = {col: float(z_scores[col].iloc[-1]) for col in z_scores.columns
                  if not np.isnan(z_scores[col].iloc[-1])}

        # 更新 SQLite 索引
        self._update_z_score_index(df, latest)
        return latest

    def refresh(self) -> int:
        """ParquetMacroProvider 不主动刷新，由 syncer 驱动。"""
        logger.info("ParquetMacroProvider.refresh() is no-op, use MacroDataSyncer")
        return 0

    # ------ 内部方法 ------

    def _read_parquet(self, indicators: list[str]) -> pd.DataFrame:
        if not self._parquet_path.exists():
            return pd.DataFrame()

        df = pd.read_parquet(self._parquet_path)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        available = [c for c in indicators if c in df.columns]
        if not available:
            return pd.DataFrame()

        df = df[available]
        # Forward-fill 不同频率指标的缺失值
        df = df.ffill()
        return df

    def _compute_z_scores(
        self, df: pd.DataFrame, window: int = Z_SCORE_WINDOW
    ) -> pd.DataFrame:
        """计算滚动 Z-score，clip 到 [-3, +3]。"""
        rolling_mean = df.rolling(window=window, min_periods=20).mean()
        rolling_std = df.rolling(window=window, min_periods=20).std()

        # 避免除零
        rolling_std = rolling_std.replace(0, np.nan)
        z_scores = (df - rolling_mean) / rolling_std
        z_scores = z_scores.clip(-Z_SCORE_CLIP, Z_SCORE_CLIP)
        return z_scores

    def _read_z_scores_from_index(self, indicators: list[str]) -> dict[str, float]:
        if not self._index_conn:
            return {}
        placeholders = ",".join("?" for _ in indicators)
        cur = self._index_conn.execute(
            f"SELECT indicator, z_score FROM macro_latest WHERE indicator IN ({placeholders})",
            indicators,
        )
        return {row[0]: row[1] for row in cur.fetchall() if row[1] is not None}

    def _update_z_score_index(
        self, df: pd.DataFrame, z_scores: dict[str, float]
    ) -> None:
        if not self._index_conn:
            return
        now = date.today().isoformat()
        for col in df.columns:
            if col not in z_scores:
                continue
            last_valid = df[col].last_valid_index()
            if last_valid is None:
                continue
            self._index_conn.execute(
                """INSERT OR REPLACE INTO macro_latest
                   (indicator, value, z_score, date, updated_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (col, float(df[col].loc[last_valid]),
                 z_scores[col], str(last_valid.date()), now),
            )
        self._index_conn.commit()

    # ------ 写入接口（供 syncer 使用）------

    def write_indicators(self, df: pd.DataFrame) -> None:
        """写入/合并宏观指标数据到 Parquet。

        Args:
            df: DatetimeIndex, columns = FRED 指标代码
        """
        self._data_path.mkdir(parents=True, exist_ok=True)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        tmp_path = self._parquet_path.with_suffix(".parquet.tmp")

        if self._parquet_path.exists():
            existing = pd.read_parquet(self._parquet_path)
            existing.index = pd.to_datetime(existing.index)
            # Column-level merge: union columns, keep existing values where df has
            # no column, and let df overwrite only the columns it actually provides.
            # A plain concat+drop_duplicates would overwrite whole rows and nullify
            # columns absent from the increment — corrupting historical data when
            # callers fetch a subset of indicators.
            combined = existing.combine_first(df)
            combined.update(df)
            combined.sort_index(inplace=True)
            combined.to_parquet(tmp_path)
            tmp_path.replace(self._parquet_path)  # cross-platform atomic replace
            logger.debug("Updated macro parquet: %d rows, %d columns",
                         len(combined), len(combined.columns))
        else:
            df.sort_index(inplace=True)
            df.to_parquet(tmp_path)
            tmp_path.replace(self._parquet_path)  # cross-platform atomic replace
            logger.debug("Created macro parquet: %d rows, %d columns",
                         len(df), len(df.columns))

    def rebuild_z_score_index(self, window: int = Z_SCORE_WINDOW) -> int:
        """重建全部 Z-score 索引。返回更新的指标数量。"""
        df = self._read_parquet(ALL_CODES)
        if df.empty:
            return 0
        z_scores = self._compute_z_scores(df, window)
        latest = {col: float(z_scores[col].iloc[-1]) for col in z_scores.columns
                  if not np.isnan(z_scores[col].iloc[-1])}
        self._update_z_score_index(df, latest)
        return len(latest)

    def indicator_date_range(self) -> tuple[date, date] | None:
        """返回数据的日期范围。"""
        if not self._parquet_path.exists():
            return None
        df = pd.read_parquet(self._parquet_path)
        if df.empty:
            return None
        idx = pd.to_datetime(df.index)
        return idx.min().date(), idx.max().date()
