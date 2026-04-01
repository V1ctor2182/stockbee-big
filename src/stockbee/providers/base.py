"""BaseProvider — 所有 Provider 的抽象基类。

设计思想借鉴 Qlib Provider 模式，解耦数据源与业务逻辑，
支持 backtest/live 环境通过 YAML 配置无缝切换。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ProviderEnv(str, Enum):
    """Provider 运行环境。"""
    BACKTEST = "backtest"
    LIVE = "live"


@dataclass
class ProviderConfig:
    """Provider 实例化配置，从 YAML 解析而来。"""
    implementation: str
    params: dict[str, Any] = field(default_factory=dict)
    fallback: str | None = None


class BaseProvider(ABC):
    """所有 Provider 的抽象基类。

    职责：
    - 定义公共生命周期（initialize / shutdown / health_check）
    - 集成 CacheManager（子类通过 self.cache 访问）
    - 提供 fallback 机制骨架
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        self._config = config or ProviderConfig(implementation=self.__class__.__name__)
        self._initialized = False
        self._cache: Any = None  # 由 Registry 注入 CacheManager

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def cache(self) -> Any:
        return self._cache

    @cache.setter
    def cache(self, cache_manager: Any) -> None:
        self._cache = cache_manager

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """初始化 Provider（连接、加载文件等）。子类重写 _do_initialize。"""
        if self._initialized:
            return
        logger.info("Initializing provider: %s", self.name)
        self._do_initialize()
        self._initialized = True

    def shutdown(self) -> None:
        """关闭 Provider，释放资源。子类重写 _do_shutdown。"""
        if not self._initialized:
            return
        logger.info("Shutting down provider: %s", self.name)
        self._do_shutdown()
        self._initialized = False

    def health_check(self) -> bool:
        """检查 Provider 是否健康。默认返回 initialized 状态。"""
        return self._initialized

    @abstractmethod
    def _do_initialize(self) -> None:
        """子类实现具体初始化逻辑。"""
        ...

    def _do_shutdown(self) -> None:
        """子类可选重写关闭逻辑。默认无操作。"""
        pass

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not initialized"
        return f"<{self.name} ({status})>"
