"""Perigon 数据源 — goperigon.com 新闻聚合 API。

依赖: requests（项目已有）。
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class PerigonConfig:
    """Perigon API 配置。"""
    api_key: str | None = None
    base_url: str = "https://api.goperigon.com/v1"
    page_size: int = 100
    max_pages: int = 5
    timeout: float = 30.0


class PerigonSource:
    """Perigon 新闻数据源。

    REST API 调用 goperigon.com。
    API key 缺失时 graceful degradation。
    """

    def __init__(self, config: PerigonConfig | None = None) -> None:
        self._config = config or PerigonConfig()
        self._api_key = self._config.api_key or os.environ.get("PERIGON_API_KEY")
        self._available = bool(self._api_key)
        if not self._available:
            logger.info("PERIGON_API_KEY not set, Perigon source disabled")

    @property
    def source_name(self) -> str:
        return "perigon"

    def fetch(
        self,
        keywords: list[str] | None = None,
        tickers: list[str] | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[dict]:
        """从 Perigon 拉取新闻。"""
        if not self._available:
            return []

        query = " OR ".join((keywords or []) + (tickers or []))
        if not query:
            return []

        all_articles: list[dict] = []
        for page in range(1, self._config.max_pages + 1):
            params: dict = {
                "apiKey": self._api_key,
                "q": query,
                "size": self._config.page_size,
                "page": page,
                "language": "en",
                "sortBy": "date",
            }
            if from_dt:
                params["from"] = from_dt.strftime("%Y-%m-%dT%H:%M:%S")
            if to_dt:
                params["to"] = to_dt.strftime("%Y-%m-%dT%H:%M:%S")

            try:
                import requests
                resp = requests.get(
                    f"{self._config.base_url}/all",
                    params=params,
                    timeout=self._config.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
            except ImportError:
                logger.info("requests not installed, Perigon source disabled")
                self._available = False
                return all_articles
            except Exception as e:
                # 避免 API key 泄露到日志（key 在 URL query params 里）
                err_msg = str(e)
                if self._api_key and self._api_key in err_msg:
                    err_msg = err_msg.replace(self._api_key, "***")
                logger.warning("Perigon request failed (page %d): %s", page, err_msg)
                break

            articles = data.get("articles", [])
            if not articles:
                break

            for article in articles:
                normalized = self._normalize(article)
                if normalized:
                    all_articles.append(normalized)

            if len(articles) < self._config.page_size:
                break

        logger.info("Perigon fetched %d articles", len(all_articles))
        return all_articles

    def _normalize(self, article: dict) -> dict | None:
        """将 Perigon 文章格式转为标准字典。"""
        headline = (article.get("title") or "").strip()
        if not headline:
            return None

        source_name = ""
        if isinstance(article.get("source"), dict):
            source_name = article["source"].get("domain", "")
        elif isinstance(article.get("source"), str):
            source_name = article["source"]

        return {
            "headline": headline,
            "source": source_name.lower() if source_name else "perigon",
            "timestamp": article.get("pubDate") or article.get("publishedAt", ""),
            "snippet": (article.get("description") or "").strip(),
            "source_url": article.get("url", ""),
        }
