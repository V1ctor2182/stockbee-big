"""ParquetMarketData — Parquet 列存 OHLCV 实现。

回测模式下的 MarketDataProvider 实现。
按 ticker 分文件存储：data/ohlcv/{TICKER}.parquet
每个文件的 schema: date(index), open, high, low, close, volume, adj_close
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import MarketDataProvider

logger = logging.getLogger(__name__)

OHLCV_FIELDS = ["open", "high", "low", "close", "volume", "adj_close"]


class ParquetMarketData(MarketDataProvider):
    """从 Parquet 文件读取 OHLCV 日线数据。

    文件布局: {data_path}/{TICKER}.parquet
    每个文件: DatetimeIndex, columns=[open, high, low, close, volume, adj_close]
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        self._data_path = Path(params.get("data_path", "data/ohlcv"))

    def _do_initialize(self) -> None:
        self._data_path.mkdir(parents=True, exist_ok=True)
        count = len(list(self._data_path.glob("*.parquet")))
        logger.info("ParquetMarketData ready: %d ticker files in %s", count, self._data_path)

    def _ticker_path(self, ticker: str) -> Path:
        return self._data_path / f"{ticker.upper()}.parquet"

    def get_daily_bars(
        self,
        tickers: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        fields = fields or OHLCV_FIELDS
        cache_key = f"ohlcv:{','.join(sorted(tickers))}:{start}:{end}:{','.join(fields)}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        frames: list[pd.DataFrame] = []
        for ticker in tickers:
            df = self._read_ticker(ticker, start, end, fields)
            if df is not None:
                frames.append(df)

        if not frames:
            result = pd.DataFrame(
                columns=pd.MultiIndex.from_tuples([], names=["field"]),
                index=pd.MultiIndex.from_tuples([], names=["date", "ticker"]),
            )
        else:
            result = pd.concat(frames)
            result.sort_index(inplace=True)

        if self.cache:
            self.cache.set(cache_key, result)
        return result

    def get_latest_price(self, tickers: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for ticker in tickers:
            path = self._ticker_path(ticker)
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["close"])
            if not df.empty:
                prices[ticker.upper()] = float(df["close"].iloc[-1])
        return prices

    def _read_ticker(
        self, ticker: str, start: date, end: date, fields: list[str]
    ) -> pd.DataFrame | None:
        path = self._ticker_path(ticker)
        if not path.exists():
            logger.debug("No parquet file for %s", ticker)
            return None

        available_fields = [f for f in fields if f in OHLCV_FIELDS]
        df = pd.read_parquet(path, columns=available_fields)
        df.index = pd.to_datetime(df.index)
        df = df.loc[str(start) : str(end)]

        if df.empty:
            return None

        df["ticker"] = ticker.upper()
        df.index.name = "date"
        df = df.reset_index().set_index(["date", "ticker"])
        return df

    # ------ 写入接口（供 sync 管道使用）------

    def write_ticker(self, ticker: str, df: pd.DataFrame) -> None:
        """写入单只股票的 OHLCV 数据到 Parquet。

        Args:
            ticker: 股票代码
            df: DatetimeIndex, columns 包含 OHLCV 字段
        """
        self._data_path.mkdir(parents=True, exist_ok=True)
        path = self._ticker_path(ticker)

        tmp_path = path.with_suffix(".parquet.tmp")
        df.index = pd.to_datetime(df.index)

        if path.exists():
            existing = pd.read_parquet(path)
            existing.index = pd.to_datetime(existing.index)
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            combined.to_parquet(tmp_path)
            tmp_path.replace(path)  # cross-platform atomic replace
            logger.debug("Updated %s: %d rows", ticker, len(combined))
        else:
            df.sort_index(inplace=True)
            df.to_parquet(tmp_path)
            tmp_path.replace(path)  # cross-platform atomic replace
            logger.debug("Created %s: %d rows", ticker, len(df))

    def list_tickers(self) -> list[str]:
        """返回所有已存储的 ticker 列表。"""
        return [p.stem for p in self._data_path.glob("*.parquet")]

    def ticker_date_range(self, ticker: str) -> tuple[date, date] | None:
        """返回某只 ticker 的数据日期范围。"""
        path = self._ticker_path(ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path, columns=["close"])
        if df.empty:
            return None
        idx = pd.to_datetime(df.index)
        return idx.min().date(), idx.max().date()
