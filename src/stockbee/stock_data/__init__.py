"""Stock Data 模块 — ParquetMarketData + SqliteUniverseProvider + AlpacaMarketData。

实现 01-stock-data Room 的全部功能：
- Parquet 列存 OHLCV（4000只 × 5年）
- SQLite 元数据（四层漏斗宇宙管理）
- Alpaca API 数据同步管道
"""

from .parquet_store import ParquetMarketData
from .universe_store import SqliteUniverseProvider
from .alpaca_market import AlpacaMarketData
from .funnel import FunnelConfig, UniverseFunnel
from .sync import StockDataSyncer

__all__ = [
    "ParquetMarketData",
    "SqliteUniverseProvider",
    "AlpacaMarketData",
    "FunnelConfig",
    "UniverseFunnel",
    "StockDataSyncer",
]
