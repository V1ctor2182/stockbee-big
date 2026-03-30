"""Stockbee Provider 架构。

10 个 Provider + Registry + 3 层缓存 + YAML 配置。
支持 backtest/live 环境通过配置文件无缝切换。

Quick Start:
    from stockbee.providers import (
        ProviderRegistry, CacheManager,
        load_env_config, ProviderEnv,
    )

    # 1. 创建 Registry
    cache = CacheManager(cache_dir="data/cache")
    registry = ProviderRegistry(cache_manager=cache)

    # 2. 注册实现类（各数据 Room 提供）
    registry.register("ParquetMarketData", ParquetMarketData)

    # 3. 从配置创建实例
    configs = load_env_config(env=ProviderEnv.BACKTEST)
    registry.create_from_config(configs)

    # 4. 使用
    market = registry.get("MarketDataProvider")
    market.initialize()
    df = market.get_daily_bars(["AAPL"], date(2025, 1, 1), date(2025, 12, 31))
"""

from .base import BaseProvider, ProviderConfig, ProviderEnv
from .cache import CacheManager, CacheStats, L1MemoryCache, L2ParquetCache
from .config import load_env_config, load_provider_configs
from .interfaces import (
    BrokerProvider,
    CacheProviderInterface,
    CalendarProvider,
    FactorProvider,
    FundamentalProvider,
    MacroProvider,
    MarketDataProvider,
    NewsProvider,
    SentimentProvider,
    UniverseProvider,
)
from .registry import ProviderRegistry

__all__ = [
    # Core
    "BaseProvider",
    "ProviderConfig",
    "ProviderEnv",
    "ProviderRegistry",
    # Cache
    "CacheManager",
    "CacheStats",
    "L1MemoryCache",
    "L2ParquetCache",
    # Config
    "load_env_config",
    "load_provider_configs",
    # 10 Provider Interfaces
    "CalendarProvider",
    "UniverseProvider",
    "MarketDataProvider",
    "FactorProvider",
    "FundamentalProvider",
    "NewsProvider",
    "MacroProvider",
    "SentimentProvider",
    "BrokerProvider",
    "CacheProviderInterface",
]
