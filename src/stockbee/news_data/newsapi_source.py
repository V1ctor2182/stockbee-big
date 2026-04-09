"""NewsAPI 数据源 — newsapi.org 免费层 ~500 请求/天。

依赖: newsapi-python（lazy import，未安装时 _available=False）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class NewsAPIConfig:
    """NewsAPI 配置。"""
    api_key: str | None = None
    page_size: int = 100
    max_pages: int = 5
    language: str = "en"


class NewsAPISource:
    """NewsAPI 数据源。

    使用 newsapi-python SDK 调用 newsapi.org/v2/everything。
    API key 缺失或 SDK 未安装时 graceful degradation。
    """

    def __init__(self, config: NewsAPIConfig | None = None) -> None:
        self._config = config or NewsAPIConfig()
        self._client = None
        self._available: bool | None = None

    @property
    def source_name(self) -> str:
        return "newsapi"

    def fetch(
        self,
        keywords: list[str] | None = None,
        tickers: list[str] | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[dict]:
        """从 NewsAPI 拉取新闻，返回标准化字典列表。"""
        if not self._ensure_client():
            return []

        query = self._build_query(keywords, tickers)
        if not query:
            return []

        all_articles: list[dict] = []
        for page in range(1, self._config.max_pages + 1):
            try:
                resp = self._client.get_everything(
                    q=query,
                    language=self._config.language,
                    page_size=self._config.page_size,
                    page=page,
                    from_param=from_dt.strftime("%Y-%m-%dT%H:%M:%S") if from_dt else None,
                    to=to_dt.strftime("%Y-%m-%dT%H:%M:%S") if to_dt else None,
                    sort_by="publishedAt",
                )
            except Exception as e:
                logger.warning("NewsAPI request failed (page %d): %s", page, e)
                break

            articles = resp.get("articles", [])
            if not articles:
                break

            for article in articles:
                normalized = self._normalize(article)
                if normalized:
                    all_articles.append(normalized)

            # 不足一页说明没有更多了
            if len(articles) < self._config.page_size:
                break

        logger.info("NewsAPI fetched %d articles", len(all_articles))
        return all_articles

    # ------ 内部方法 ------

    def _build_query(self, keywords: list[str] | None, tickers: list[str] | None) -> str:
        """构建 NewsAPI 查询字符串。"""
        parts: list[str] = []
        if keywords:
            parts.extend(keywords)
        if tickers:
            parts.extend(tickers)
        return " OR ".join(parts) if parts else ""

    def _normalize(self, article: dict) -> dict | None:
        """将 NewsAPI 文章格式转为标准字典。"""
        headline = (article.get("title") or "").strip()
        if not headline or headline == "[Removed]":
            return None

        source_name = ""
        if isinstance(article.get("source"), dict):
            source_name = article["source"].get("name", "")

        return {
            "headline": headline,
            "source": source_name.lower() if source_name else "newsapi",
            "timestamp": article.get("publishedAt", ""),
            "snippet": (article.get("description") or "").strip(),
            "source_url": article.get("url", ""),
        }

    def _ensure_client(self) -> bool:
        """Lazy init NewsAPI client。"""
        if self._available is False:
            return False
        if self._client is not None:
            return True
        try:
            from newsapi import NewsApiClient

            api_key = self._config.api_key
            if not api_key:
                import os
                api_key = os.environ.get("NEWSAPI_KEY")
            if not api_key:
                logger.info("NEWSAPI_KEY not set, NewsAPI source disabled")
                self._available = False
                return False

            self._client = NewsApiClient(api_key=api_key)
            self._available = True
            return True
        except ImportError:
            logger.info("newsapi-python not installed, NewsAPI source disabled")
            self._available = False
            return False
        except Exception as e:
            logger.warning("NewsAPI client init failed: %s", e)
            self._available = False
            return False
