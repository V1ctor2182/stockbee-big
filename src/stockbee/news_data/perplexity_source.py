"""Perplexity 数据源 — AI 搜索 API，通过 citations 获取原始新闻来源。

Perplexity 返回 AI 生成的摘要 + citations（原始来源 URL 和标题）。
通过 prompt 要求返回结构化新闻列表，提取 citation 中的原始 publisher 作为 source。
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_SEARCH_PROMPT = """\
Find the latest financial news articles about: {query}

For each article, provide:
1. The exact original headline
2. The publisher name (e.g. Reuters, Bloomberg, CNBC)
3. The publication date in ISO 8601 format

Respond ONLY with a JSON array of objects:
[{{"headline": "...", "publisher": "...", "date": "..."}}]

Return at most {max_results} articles. Only include real published articles, not summaries."""


@dataclass
class PerplexityConfig:
    """Perplexity API 配置。"""
    api_key: str | None = None
    base_url: str = "https://api.perplexity.ai"
    model: str = "sonar"
    max_results: int = 20
    timeout: float = 60.0


class PerplexitySource:
    """Perplexity AI 搜索数据源。

    通过 prompt 要求 Perplexity 返回结构化新闻列表。
    利用 citations 获取原始新闻来源 URL。
    source 字段使用原始 publisher（不是 "perplexity"），
    使得 UNIQUE(headline, source) 可以和 NewsAPI/Perigon 正常去重。

    如果无法提取原始 publisher，fallback 到 "perplexity"。
    """

    def __init__(self, config: PerplexityConfig | None = None) -> None:
        self._config = config or PerplexityConfig()
        self._api_key = self._config.api_key or os.environ.get("PERPLEXITY_API_KEY")
        self._available = bool(self._api_key)
        if not self._available:
            logger.info("PERPLEXITY_API_KEY not set, Perplexity source disabled")

    @property
    def source_name(self) -> str:
        return "perplexity"

    def fetch(
        self,
        keywords: list[str] | None = None,
        tickers: list[str] | None = None,
        from_dt: datetime | None = None,
        to_dt: datetime | None = None,
    ) -> list[dict]:
        """从 Perplexity 搜索新闻。"""
        if not self._available:
            return []

        query = " ".join((keywords or []) + (tickers or []))
        if not query:
            return []

        prompt = _SEARCH_PROMPT.format(query=query, max_results=self._config.max_results)

        try:
            import requests
            resp = requests.post(
                f"{self._config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._config.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "return_citations": True,
                },
                timeout=self._config.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except ImportError:
            logger.info("requests not installed, Perplexity source disabled")
            self._available = False
            return []
        except Exception as e:
            logger.warning("Perplexity request failed: %s", e)
            return []

        # 提取 citations URLs（用于匹配 source）
        citations = self._extract_citations(data)

        # 解析 AI 响应中的结构化新闻
        content = self._extract_content(data)
        articles = self._parse_articles(content)

        # 用 citations 丰富 source_url
        normalized = self._enrich_with_citations(articles, citations)
        logger.info("Perplexity fetched %d articles", len(normalized))
        return normalized

    # ------ 内部方法 ------

    def _extract_content(self, data: dict) -> str:
        """从 Perplexity 响应提取 AI 生成内容。"""
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""

    def _extract_citations(self, data: dict) -> list[str]:
        """从 Perplexity 响应提取 citations URL 列表。"""
        try:
            return data.get("citations", [])
        except (KeyError, TypeError):
            return []

    def _parse_articles(self, content: str) -> list[dict]:
        """从 AI 内容解析 JSON 文章列表。"""
        # Tier 1: 直接 json.loads
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass

        # Tier 2: regex 提取 JSON array
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass

        logger.warning("Perplexity response not parseable: %.200s", content)
        return []

    def _enrich_with_citations(self, articles: list[dict], citations: list[str]) -> list[dict]:
        """用 citations 补充 source_url，提取原始 publisher。"""
        result: list[dict] = []
        for i, article in enumerate(articles):
            headline = (article.get("headline") or "").strip()
            if not headline:
                continue

            # publisher 来自 AI 输出或 citation URL 的 domain
            publisher = (article.get("publisher") or "").strip().lower()
            source_url = citations[i] if i < len(citations) else ""

            if not publisher and source_url:
                publisher = self._domain_to_publisher(source_url)
            if not publisher:
                publisher = "perplexity"

            # Perplexity AI 输出不保证有 date 字段，fallback 到当前时间
            timestamp = (article.get("date") or "").strip()
            if not timestamp:
                timestamp = datetime.now(timezone.utc).isoformat()

            result.append({
                "headline": headline,
                "source": publisher,
                "timestamp": timestamp,
                "snippet": "",
                "source_url": source_url,
            })
        return result

    @staticmethod
    def _domain_to_publisher(url: str) -> str:
        """从 URL 提取 publisher 名。"""
        try:
            domain = urlparse(url).netloc.lower()
            # 去掉 www. 前缀和 .com 等后缀
            domain = domain.removeprefix("www.")
            # 常见映射
            mapping = {
                "reuters.com": "reuters",
                "bloomberg.com": "bloomberg",
                "cnbc.com": "cnbc",
                "wsj.com": "wsj",
                "ft.com": "ft",
                "nytimes.com": "nytimes",
                "marketwatch.com": "marketwatch",
                "seekingalpha.com": "seekingalpha",
                "yahoo.com": "yahoo finance",
                "finance.yahoo.com": "yahoo finance",
            }
            return mapping.get(domain, domain.split(".")[0])
        except Exception:
            return ""
