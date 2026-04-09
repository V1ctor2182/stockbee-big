"""新闻数据源 Protocol + MockSource。

NewsSource 定义了所有新闻数据源的统一接口。
MockSource 用于测试和 backtest 环境。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@runtime_checkable
class NewsSource(Protocol):
    """新闻数据源统一接口。"""

    @property
    def source_name(self) -> str:
        """数据源名称，用于日志和去重追踪。"""
        ...

    def fetch(
        self,
        keywords: list[str] | None = None,
        tickers: list[str] | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[dict]:
        """拉取新闻。返回标准化字典列表。

        每个字典包含: headline, source, timestamp, snippet, source_url
        """
        ...


@dataclass
class MockNewsSource:
    """测试/backtest 用的 mock 数据源。"""

    _articles: list[dict] = field(default_factory=list)

    @property
    def source_name(self) -> str:
        return "mock"

    def fetch(
        self,
        keywords: list[str] | None = None,
        tickers: list[str] | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[dict]:
        return list(self._articles)
