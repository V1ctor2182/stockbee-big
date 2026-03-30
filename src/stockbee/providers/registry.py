"""ProviderRegistry — Provider 全局注册中心。

职责：
- 注册 Provider 实现类（class），按名称查找
- 按环境（backtest/live）配置批量实例化
- 为所有 Provider 注入共享的 CacheManager
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseProvider, ProviderConfig, ProviderEnv
from .cache import CacheManager

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Provider 注册中心 + 实例容器。

    用法：
        registry = ProviderRegistry()
        registry.register("ParquetMarketData", ParquetMarketData)
        registry.register("AlpacaMarketData", AlpacaMarketData)

        # 从配置批量创建
        registry.create_from_config(configs)

        # 按抽象名获取实例
        market_data = registry.get("MarketDataProvider")
    """

    def __init__(self, cache_manager: CacheManager | None = None) -> None:
        self._classes: dict[str, type[BaseProvider]] = {}
        self._instances: dict[str, BaseProvider] = {}
        self._cache = cache_manager or CacheManager()

    @property
    def cache(self) -> CacheManager:
        return self._cache

    def register(self, name: str, cls: type[BaseProvider]) -> None:
        """注册一个 Provider 实现类。"""
        if not (isinstance(cls, type) and issubclass(cls, BaseProvider)):
            raise TypeError(f"{cls} is not a subclass of BaseProvider")
        self._classes[name] = cls
        logger.debug("Registered provider class: %s", name)

    def create(
        self,
        provider_name: str,
        config: ProviderConfig,
    ) -> BaseProvider:
        """根据配置创建并注册一个 Provider 实例。

        Args:
            provider_name: 抽象 Provider 名称（如 "MarketDataProvider"）
            config: 包含 implementation 类名和参数
        """
        impl_name = config.implementation
        cls = self._classes.get(impl_name)
        if cls is None:
            raise KeyError(
                f"Unknown provider implementation: {impl_name!r}. "
                f"Registered: {list(self._classes.keys())}"
            )
        instance = cls(config)
        instance.cache = self._cache
        self._instances[provider_name] = instance
        logger.info("Created provider: %s -> %s", provider_name, impl_name)
        return instance

    def create_from_config(
        self,
        configs: dict[str, ProviderConfig],
    ) -> None:
        """从配置字典批量创建 Provider 实例。"""
        for provider_name, config in configs.items():
            self.create(provider_name, config)

    def get(self, provider_name: str) -> BaseProvider:
        """按抽象名获取 Provider 实例。"""
        instance = self._instances.get(provider_name)
        if instance is None:
            raise KeyError(
                f"Provider not found: {provider_name!r}. "
                f"Available: {list(self._instances.keys())}"
            )
        return instance

    def get_or_none(self, provider_name: str) -> BaseProvider | None:
        """按抽象名获取 Provider 实例，不存在返回 None。"""
        return self._instances.get(provider_name)

    def initialize_all(self) -> None:
        """初始化所有已注册的 Provider 实例。"""
        for name, instance in self._instances.items():
            try:
                instance.initialize()
            except Exception:
                logger.exception("Failed to initialize provider: %s", name)
                raise

    def shutdown_all(self) -> None:
        """关闭所有 Provider。"""
        for name, instance in self._instances.items():
            try:
                instance.shutdown()
            except Exception:
                logger.exception("Error shutting down provider: %s", name)

    def health_check_all(self) -> dict[str, bool]:
        """检查所有 Provider 健康状态。"""
        return {name: inst.health_check() for name, inst in self._instances.items()}

    @property
    def registered_classes(self) -> list[str]:
        return list(self._classes.keys())

    @property
    def active_providers(self) -> list[str]:
        return list(self._instances.keys())

    def __repr__(self) -> str:
        return (
            f"<ProviderRegistry classes={len(self._classes)} "
            f"instances={len(self._instances)}>"
        )
