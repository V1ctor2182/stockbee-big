"""FredMacroProvider — FRED API 实盘实现。

通过 fredapi 包连接 FRED (Federal Reserve Economic Data)。
支持增量和全量拉取 19 个宏观指标。
Fallback 到 ParquetMacroProvider。

依赖：pip install fredapi（可选依赖）
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any

import pandas as pd

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import MacroProvider

from .indicators import ALL_CODES, FRED_INDICATORS

logger = logging.getLogger(__name__)


class FredMacroProvider(MacroProvider):
    """通过 FRED API 获取宏观指标数据。"""

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        self._api_key = params.get("api_key", "")
        self._fred: Any = None

    def _do_initialize(self) -> None:
        try:
            from fredapi import Fred
            self._fred = Fred(api_key=self._api_key)
            logger.info("FredMacroProvider connected")
        except ImportError:
            raise ImportError(
                "fredapi is required for FredMacroProvider. "
                "Install with: pip install fredapi"
            )

    def _do_shutdown(self) -> None:
        self._fred = None

    def get_macro_indicators(
        self,
        indicators: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        indicators = indicators or ALL_CODES
        cache_key = f"fred:macro:{','.join(sorted(indicators))}:{start}:{end}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Point-in-time：end 使用 end-1
        pit_end = (end - timedelta(days=1)) if end else None

        frames: dict[str, pd.Series] = {}
        for code in indicators:
            series = self._fetch_series(code, start, pit_end)
            if series is not None:
                frames[code] = series

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.sort_index(inplace=True)
        # Forward-fill 不同频率的缺失值
        df = df.ffill()

        if self.cache and not df.empty:
            self.cache.set(cache_key, df)
        return df

    def get_latest_z_scores(
        self, indicators: list[str] | None = None, window: int = 252
    ) -> dict[str, float]:
        indicators = indicators or ALL_CODES

        # 拉取足够的历史数据计算 Z-score
        end = date.today()
        start = end - timedelta(days=int(window * 2.0))  # 252 交易日 ≈ 504 日历天
        df = self.get_macro_indicators(indicators, start=start, end=end)

        if df.empty:
            return {}

        import numpy as np
        rolling_mean = df.rolling(window=window, min_periods=20).mean()
        rolling_std = df.rolling(window=window, min_periods=20).std().replace(0, np.nan)
        z_scores = ((df - rolling_mean) / rolling_std).clip(-3.0, 3.0)

        return {col: float(z_scores[col].iloc[-1]) for col in z_scores.columns
                if not pd.isna(z_scores[col].iloc[-1])}

    def refresh(self) -> int:
        """从 FRED 拉取全部 19 个指标的最新数据。返回成功拉取的指标数量。"""
        count = 0
        for code in ALL_CODES:
            try:
                series = self._fetch_series(code)
                if series is not None and len(series) > 0:
                    count += 1
            except Exception:
                logger.exception("Failed to refresh %s", code)
        logger.info("FRED refresh: %d/%d indicators updated", count, len(ALL_CODES))
        return count

    def _fetch_series(
        self, code: str, start: date | None = None, end: date | None = None
    ) -> pd.Series | None:
        """从 FRED API 拉取单个指标的时间序列。"""
        try:
            kwargs: dict = {}
            if start:
                kwargs["observation_start"] = str(start)
            if end:
                kwargs["observation_end"] = str(end)

            series = self._fred.get_series(code, **kwargs)
            if series is None or series.empty:
                logger.debug("No data for %s", code)
                return None
            series.name = code
            return series
        except Exception:
            logger.exception("FRED API error fetching %s", code)
            return None

    def fetch_all(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """拉取全部 19 个指标，返回合并的 DataFrame。供 syncer 使用。"""
        frames: dict[str, pd.Series] = {}
        for code in ALL_CODES:
            series = self._fetch_series(code, start, end)
            if series is not None:
                frames[code] = series

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df.sort_index(inplace=True)
        logger.info("Fetched %d indicators from FRED, %d rows", len(frames), len(df))
        return df
