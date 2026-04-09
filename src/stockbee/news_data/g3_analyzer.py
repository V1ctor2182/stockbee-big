"""G3 深度分析 — Claude Haiku 影响评估 + 权重建议 + 真伪评分。

G3 是新闻处理漏斗的第三级，对少量高价值新闻做 LLM 深度分析。
每日限额控制 API 成本（默认 10 篇/天 ≈ $0.30/月）。

依赖: anthropic SDK（lazy import，未安装时跳过 G3）。
模型: Claude Haiku 4.5。
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from stockbee.news_data.g2_classifier import G2Result
    from stockbee.news_data.news_store import SqliteNewsProvider

logger = logging.getLogger(__name__)

HAIKU_MODEL = "claude-haiku-4-5-20251001"
MIN_ASCII_RATIO = 0.5
MAX_INPUT_CHARS = 4096

# G3Result JSON schema（嵌入 prompt 供 Haiku 参考）
_G3_JSON_SCHEMA = """\
{
  "weight_action": "increase | decrease | hedge | hold",
  "weight_magnitude": 0.0 to 1.0,
  "reliability_score": 0.0 to 1.0,
  "reasoning": "2-3 sentence analysis in English",
  "confidence": 0.0 to 1.0
}"""

_SYSTEM_PROMPT = f"""\
You are a senior financial analyst. Analyze the news article below and assess \
its impact on the mentioned stock(s).

Respond ONLY with valid JSON matching this exact schema:
{_G3_JSON_SCHEMA}

Field definitions:
- weight_action: recommended portfolio action for the primary ticker
- weight_magnitude: strength of the recommendation (0.0 = no action, 1.0 = maximum)
- reliability_score: trustworthiness of the news source and claims (0.0 = unverified rumor, 1.0 = official filing)
- reasoning: concise 2-3 sentence explanation grounded in the article, no speculation
- confidence: your confidence in this assessment (0.0 = uncertain, 1.0 = very confident)

Do NOT include any text outside the JSON object. Do NOT wrap in markdown fences."""


# =========================================================================
# 数据类
# =========================================================================

@dataclass
class G3Result:
    """G3 深度分析结果。"""
    weight_action: str = "hold"          # increase / decrease / hedge / hold
    weight_magnitude: float = 0.0        # 0.0 to 1.0
    reliability_score: float = 0.5       # 0.0 to 1.0
    reasoning: str = ""                  # 2-3 句英文分析
    confidence: float = 0.0              # 0.0 to 1.0


@dataclass
class G3Config:
    """G3 分析器配置。"""
    model: str = HAIKU_MODEL
    api_key: str | None = None           # None = 从 ANTHROPIC_API_KEY 环境变量读取
    daily_limit: int = 10
    importance_threshold: float = 0.7
    urgency_trigger: str = "high"
    sentiment_high: float = 0.9
    sentiment_low: float = 0.1
    max_retries: int = 3
    retry_base_delay: float = 1.0
    max_input_chars: int = MAX_INPUT_CHARS
    max_tokens: int = 512
    timeout_seconds: float = 30.0


# =========================================================================
# G3Analyzer
# =========================================================================

class G3Analyzer:
    """G3 深度分析器。

    Claude Haiku 对高价值新闻做影响评估、权重建议、真伪评分。
    API 不可用时优雅降级（返回 None，新闻留在 g_level=2）。
    """

    def __init__(
        self,
        config: G3Config | None = None,
        provider: SqliteNewsProvider | None = None,
    ) -> None:
        self._config = config or G3Config()
        self._provider = provider
        self._client: Any = None
        self._haiku_available: bool | None = None  # None = 未检测

    # ------ 公开接口 ------

    def should_analyze(self, g2_result: G2Result) -> bool:
        """检查 G2 结果是否满足 G3 触发条件。

        触发条件（可通过 G3Config 调整阈值）：
        - importance >= threshold AND urgency == trigger
        - OR sentiment_score > high threshold (极端正面)
        - OR sentiment_score < low threshold (极端负面)
        """
        cfg = self._config
        if (
            g2_result.importance_score >= cfg.importance_threshold
            and g2_result.urgency == cfg.urgency_trigger
        ):
            return True
        if g2_result.sentiment_score > cfg.sentiment_high:
            return True
        if g2_result.sentiment_score < cfg.sentiment_low:
            return True
        return False

    def analyze(
        self,
        headline: str,
        snippet: str | None = None,
        tickers: list[str] | None = None,
        g2_result: G2Result | None = None,
        force: bool = False,
    ) -> G3Result | None:
        """对单条新闻进行 G3 深度分析。

        Returns:
            G3Result — 分析成功
            None — API 不可用 / 触发条件不满足 / 日限额已满 / 分析失败
        """
        # 0. 检查 Haiku 可用性
        if not self._ensure_haiku():
            return None

        # 1. 触发条件检查（force 绕过）
        if not force and g2_result is not None:
            if not self.should_analyze(g2_result):
                return None

        # 2. 可分类性检查
        text = self._prepare_text(headline, snippet)
        if not text or not self._is_classifiable(text):
            logger.debug("G3 skipped: text not classifiable")
            return None

        # 3. 日限额检查（force 绕过）
        if not force and self._provider is not None:
            count = self._provider.get_g3_daily_count()
            if count >= self._config.daily_limit:
                logger.info("G3 daily limit reached (%d/%d)", count, self._config.daily_limit)
                return None

        # 4. 构建 prompt 并调用 API
        messages = self._build_messages(text, tickers)
        try:
            response = self._call_with_retry(messages)
        except Exception as e:
            logger.warning("G3 API call failed after retries: %s", e)
            return None

        # 5. 解析响应
        if not response.content:
            logger.warning("G3 API returned empty content (content filtering?)")
            return None
        raw_text = response.content[0].text
        result = self._parse_response(raw_text)

        # 6. 只在有意义的结果时递增日计数（tier 3 garbage 不消耗额度）
        if result.reliability_score > 0.0 and self._provider is not None:
            self._provider.increment_g3_daily_count()

        return result

    # ------ 文本处理 ------

    def _prepare_text(self, headline: str | None, snippet: str | None = None) -> str:
        """拼接并截断输入文本。"""
        parts = []
        if headline and headline.strip():
            parts.append(headline.strip())
        if snippet and snippet.strip():
            parts.append(snippet.strip())
        text = " ".join(parts)
        if len(text) > self._config.max_input_chars:
            text = text[: self._config.max_input_chars]
        return text

    @staticmethod
    def _is_classifiable(text: str) -> bool:
        """检查文本是否适合 LLM 分析（英语、有实质内容）。"""
        if not text:
            return False
        ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
        non_space = sum(1 for c in text if not c.isspace())
        if non_space == 0:
            return False
        return ascii_letters / non_space >= MIN_ASCII_RATIO

    # ------ Prompt 构建 ------

    def _build_messages(self, text: str, tickers: list[str] | None = None) -> list[dict]:
        """构建 Haiku API 消息列表。不传 G2 结果，避免 anchoring bias。"""
        user_content = f"Headline: {text}"
        if tickers:
            user_content += f"\nTickers: {', '.join(tickers)}"
        return [{"role": "user", "content": user_content}]

    # ------ API 调用 ------

    def _call_with_retry(self, messages: list[dict]) -> Any:
        """带指数退避重试的 API 调用。

        只重试 429 (rate limit) / 529 (overloaded) / 5xx。
        400/401/403 等永久错误立即抛出。
        """
        last_exc: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                return self._client.messages.create(
                    model=self._config.model,
                    max_tokens=self._config.max_tokens,
                    system=_SYSTEM_PROMPT,
                    messages=messages,
                )
            except Exception as e:
                status = getattr(e, "status_code", None)
                # 永久错误不重试
                if status is not None and status < 500 and status != 429:
                    raise
                last_exc = e
                if attempt < self._config.max_retries - 1:
                    delay = self._config.retry_base_delay * (2 ** attempt)
                    delay += random.uniform(0, 0.5)
                    logger.info("G3 API retry %d/%d after %.1fs", attempt + 1, self._config.max_retries, delay)
                    time.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ------ 响应解析 ------

    def _parse_response(self, raw_text: str) -> G3Result:
        """3-tier 解析：json.loads → regex 提取 → 降级默认值。"""
        # Tier 1: 直接解析
        try:
            data = json.loads(raw_text)
            return self._dict_to_result(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        # Tier 2: strip markdown fences then re-parse
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            # Remove ```json ... ``` or ``` ... ```
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                data = json.loads(cleaned)
                logger.warning("G3 parse tier 2: extracted JSON from fenced response")
                return self._dict_to_result(data)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass

        # Tier 2b: find outermost { ... } using brace balancing
        start = raw_text.find("{")
        if start != -1:
            depth, end = 0, start
            for i in range(start, len(raw_text)):
                if raw_text[i] == "{":
                    depth += 1
                elif raw_text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if depth == 0:
                try:
                    data = json.loads(raw_text[start:end])
                    logger.warning("G3 parse tier 2b: extracted JSON via brace matching")
                    return self._dict_to_result(data)
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    pass

        # Tier 3: 降级 — 标记为不可靠
        logger.warning("G3 parse tier 3: returning default (raw: %.200s)", raw_text)
        return G3Result(
            reliability_score=0.0,
            reasoning=raw_text[:500],
            confidence=0.0,
        )

    @staticmethod
    def _safe_float(value: object, default: float) -> float:
        """安全的 float 转换，处理 LLM 返回的非数值字符串。"""
        try:
            f = float(value)
            if f != f:  # NaN check
                return default
            return max(0.0, min(1.0, f))  # clamp to [0, 1]
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _dict_to_result(data: dict) -> G3Result:
        """从字典构建 G3Result，容忍缺失字段和错误类型。"""
        action = str(data.get("weight_action", "hold")).lower()
        if action not in ("increase", "decrease", "hedge", "hold"):
            action = "hold"
        return G3Result(
            weight_action=action,
            weight_magnitude=G3Analyzer._safe_float(data.get("weight_magnitude"), 0.0),
            reliability_score=G3Analyzer._safe_float(data.get("reliability_score"), 0.5),
            reasoning=str(data.get("reasoning", ""))[:500],
            confidence=G3Analyzer._safe_float(data.get("confidence"), 0.0),
        )

    # ------ 懒加载 ------

    def _ensure_haiku(self) -> bool:
        """Lazy init Anthropic client。返回是否可用。"""
        if self._haiku_available is False:
            return False
        if self._client is not None:
            return True
        try:
            import anthropic

            api_key = self._config.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logger.info("ANTHROPIC_API_KEY not set, G3 analysis disabled")
                self._haiku_available = False
                return False

            self._client = anthropic.Anthropic(api_key=api_key)
            self._haiku_available = True
            return True
        except ImportError:
            logger.info("anthropic SDK not installed, G3 analysis disabled")
            self._haiku_available = False
            return False
        except Exception as e:
            logger.warning("Anthropic client init failed: %s", e)
            self._haiku_available = False
            return False
