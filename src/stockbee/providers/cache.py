"""CacheManager — 3 层透明缓存。

L1: 内存 LRU（热数据，毫秒级）
L2: Parquet 磁盘（周期数据，10ms 级）
L3: 远程 API（更新数据，秒级，由调用方提供 fetcher）

多数查询止于 L1/L2，仅在无缓存时触发 L3 API 调用。
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """缓存命中统计。"""
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0

    @property
    def total_requests(self) -> int:
        return self.l1_hits + self.l2_hits + self.l3_hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_requests
        if total == 0:
            return 0.0
        return (self.l1_hits + self.l2_hits + self.l3_hits) / total


@dataclass
class CacheEntry:
    """L1 缓存条目。"""
    value: Any
    created_at: float
    ttl_seconds: float

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False  # 永不过期
        return (time.time() - self.created_at) > self.ttl_seconds


class L1MemoryCache:
    """L1 内存 LRU 缓存。"""

    def __init__(self, max_size: int = 1024, default_ttl: float = 3600) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._store[key]
            return None
        self._store.move_to_end(key)
        return entry.value

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        if key in self._store:
            del self._store[key]
        elif len(self._store) >= self._max_size:
            # 先尝试清理一个过期 entry，避免驱逐有效热数据
            evicted = False
            for k, v in list(self._store.items()):
                if v.is_expired:
                    del self._store[k]
                    evicted = True
                    break
            if not evicted:
                self._store.popitem(last=False)
        self._store[key] = CacheEntry(
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl if ttl is not None else self._default_ttl,
        )

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class L2ParquetCache:
    """L2 Parquet 磁盘缓存。

    将 DataFrame 类型的数据持久化到 Parquet 文件，
    非 DataFrame 数据不走 L2。
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        safe_name = hashlib.sha256(key.encode()).hexdigest()
        return self._cache_dir / f"{safe_name}.parquet"

    def get(self, key: str) -> pd.DataFrame | None:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            logger.warning("L2 cache read failed for key: %s", key)
            return None

    def set(self, key: str, value: pd.DataFrame) -> None:
        if not isinstance(value, pd.DataFrame):
            return  # L2 只缓存 DataFrame
        path = self._key_to_path(key)
        try:
            value.to_parquet(path, index=True)
        except Exception:
            logger.warning("L2 cache write failed for key: %s", key)

    def invalidate(self, key: str) -> None:
        path = self._key_to_path(key)
        path.unlink(missing_ok=True)

    def clear(self) -> None:
        for f in self._cache_dir.glob("*.parquet"):
            f.unlink()


class CacheManager:
    """3 层缓存管理器。

    使用方式：
        value = cache.get(key, fetcher=lambda: api.fetch(params))
        # 自动按 L1 → L2 → L3(fetcher) 顺序查找
        # 命中后自动回填上层缓存
    """

    def __init__(
        self,
        cache_dir: str | Path = "data/cache",
        l1_max_size: int = 1024,
        l1_default_ttl: float = 3600,
    ) -> None:
        self._l1 = L1MemoryCache(max_size=l1_max_size, default_ttl=l1_default_ttl)
        self._l2 = L2ParquetCache(cache_dir=cache_dir)
        self.stats = CacheStats()

    def get(
        self,
        key: str,
        fetcher: Callable[[], Any] | None = None,
        ttl: float | None = None,
    ) -> Any | None:
        """按 L1 → L2 → L3(fetcher) 顺序查找，命中后回填上层。"""
        # L1
        value = self._l1.get(key)
        if value is not None:
            self.stats.l1_hits += 1
            return value

        # L2 (仅 DataFrame)
        value = self._l2.get(key)
        if value is not None:
            self.stats.l2_hits += 1
            self._l1.set(key, value, ttl=ttl)  # 回填 L1
            return value

        # L3 (远程 API via fetcher)
        if fetcher is not None:
            try:
                value = fetcher()
            except Exception:
                logger.warning("L3 fetcher failed for key: %s", key)
                self.stats.misses += 1
                return None
            if value is not None:
                self.stats.l3_hits += 1
                self._l1.set(key, value, ttl=ttl)  # 回填 L1
                self._l2.set(key, value)  # 回填 L2（仅 DataFrame）
                return value

        self.stats.misses += 1
        return None

    def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """手动写入缓存（同时写 L1 和 L2）。"""
        self._l1.set(key, value, ttl=ttl)
        self._l2.set(key, value)

    def invalidate(self, key: str) -> None:
        """使指定 key 在所有层失效。"""
        self._l1.invalidate(key)
        self._l2.invalidate(key)

    def clear(self) -> None:
        """清空所有缓存。"""
        self._l1.clear()
        self._l2.clear()
        self.stats = CacheStats()

    def report(self) -> dict[str, Any]:
        """返回缓存统计摘要。"""
        return {
            "l1_size": self._l1.size,
            "l1_hits": self.stats.l1_hits,
            "l2_hits": self.stats.l2_hits,
            "l3_hits": self.stats.l3_hits,
            "misses": self.stats.misses,
            "hit_rate": f"{self.stats.hit_rate:.1%}",
        }
