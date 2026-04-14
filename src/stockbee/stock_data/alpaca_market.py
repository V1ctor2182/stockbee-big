"""AlpacaMarketData — Alpaca API 实现的 MarketDataProvider。

实盘模式下的 OHLCV 数据源，fallback 到 ParquetMarketData 缓存。
使用 alpaca-py SDK 访问 Alpaca Market Data API。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import MarketDataProvider

logger = logging.getLogger(__name__)

OHLCV_FIELDS = ["open", "high", "low", "close", "volume", "adj_close"]


class AlpacaMarketData(MarketDataProvider):
    """通过 Alpaca API 获取 OHLCV 数据。

    依赖 alpaca-py 包（可选依赖，回测模式不需要）。
    实盘配置中可指定 fallback: ParquetMarketData。
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        self._api_key = params.get("api_key", "")
        self._api_secret = params.get("api_secret", "")
        self._base_url = params.get("base_url", "https://paper-api.alpaca.markets")
        self._client: Any = None
        self._trading_client: Any = None

    def _do_initialize(self) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
            self._client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
            self._trading_client = TradingClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
            logger.info("AlpacaMarketData connected")
        except ImportError:
            raise ImportError(
                "alpaca-py is required for AlpacaMarketData. "
                "Install with: pip install alpaca-py"
            )

    def _do_shutdown(self) -> None:
        self._client = None
        self._trading_client = None

    def get_daily_bars(
        self,
        tickers: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        fields = fields or OHLCV_FIELDS
        cache_key = f"alpaca:ohlcv:{','.join(sorted(tickers))}:{start}:{end}:{','.join(fields)}"

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        result = self._fetch_bars(tickers, start, end, fields)

        if self.cache and not result.empty:
            self.cache.set(cache_key, result)
        return result

    def _fetch_bars(
        self, tickers: list[str], start: date, end: date, fields: list[str]
    ) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        request = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=datetime.combine(start, datetime.min.time()),
            end=datetime.combine(end, datetime.min.time()),
        )

        try:
            bars = self._client.get_stock_bars(request)
        except Exception:
            logger.exception("Alpaca API error fetching bars for %s", tickers)
            return pd.DataFrame()

        df = bars.df
        if df.empty:
            return df

        # Alpaca 返回的列名映射
        rename_map = {
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
            "vwap": "adj_close",  # 近似用 VWAP
        }
        df = df.rename(columns=rename_map)

        # 如果没有 adj_close，用 close 代替
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        available = [f for f in fields if f in df.columns]
        df = df[available]

        # 规范化索引为 (date, ticker)
        df.index = df.index.set_names(["ticker", "timestamp"])
        df = df.reset_index()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date
        df = df.drop(columns=["timestamp"])
        df = df.set_index(["date", "ticker"])

        return df

    def get_latest_price(self, tickers: list[str]) -> dict[str, float]:
        from alpaca.data.requests import StockLatestQuoteRequest

        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=tickers)
            quotes = self._client.get_stock_latest_quote(request)
            return {
                symbol: float(quote.ask_price)
                for symbol, quote in quotes.items()
                if quote.ask_price
            }
        except Exception:
            logger.exception("Alpaca API error fetching latest prices")
            return {}

    # ------ 资产列表（供 sync 管道获取全部美股列表）------

    def get_all_tradable_assets(self) -> pd.DataFrame:
        """获取 Alpaca 上所有可交易美股资产。

        Returns:
            DataFrame: ticker, name, exchange, status, tradable,
                       shortable, market_cap (None from this API)
        """
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus

        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE,
        )
        assets = self._trading_client.get_all_assets(request)

        records = []
        for asset in assets:
            if not asset.tradable:
                continue
            records.append({
                "ticker": asset.symbol,
                "name": asset.name,
                "exchange": asset.exchange.value if asset.exchange else None,
                "tradable": asset.tradable,
                "short_able": asset.shortable,
                "sector": None,      # Alpaca 不提供 sector
                "market_cap": None,  # 需要从其他数据源补充
                "avg_volume": None,
                "avg_dollar_volume": None,
            })

        df = pd.DataFrame(records)
        logger.info("Fetched %d tradable US equities from Alpaca", len(df))
        return df
