"""G2 分类评分 — FinBERT 情绪 + 主题分类 + 重要度/紧急度评分。

G2 是新闻处理漏斗的第二级，对所有通过 G1 的新闻进行智能分类。
FinBERT 做情绪三分类，规则引擎做主题分类和重要度/紧急度评分。

依赖: transformers + torch（lazy import，未安装时降级为规则引擎）。
模型: ProsusAI/finbert（首次推理时自动下载 ~440MB）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_MAX_LENGTH = 512
MIN_ASCII_RATIO = 0.5  # ASCII 字母占比低���此值视为不可分类（非英语/纯符号）


def _build_keyword_patterns(keywords: list[str]) -> list[re.Pattern]:
    """为关键词列表构建 word-boundary 正则，避免子串误匹配。"""
    return [re.compile(r"\b" + re.escape(kw) + r"\b") for kw in keywords]


def _label_from_softmax(
    row: dict[str, float],
) -> tuple[str, float, float, float, float]:
    """从 FinBERTScorer 3-way softmax dict 取 (label, positive, negative, neutral, conf)。

    argmax 决定 sentiment 标签;conf = row["confidence"] = max 三项。
    """
    pos = float(row["positive"])
    neg = float(row["negative"])
    neu = float(row["neutral"])
    conf = float(row.get("confidence", max(pos, neg, neu)))
    pair = max(("positive", pos), ("negative", neg), ("neutral", neu), key=lambda x: x[1])
    return pair[0], pos, neg, neu, conf


def _count_keyword_hits(text_lower: str, patterns: list[re.Pattern]) -> int:
    """统计文本中匹配的关键词数量。"""
    return sum(1 for p in patterns if p.search(text_lower))


def _any_keyword_hit(text_lower: str, patterns: list[re.Pattern]) -> bool:
    """检查文本是否匹配任一关键词。"""
    return any(p.search(text_lower) for p in patterns)


# ------ 主题分类规则 ------

_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "earnings": [
        "earnings", "revenue", "profit", "profits", "loss", "losses",
        "eps", "quarterly", "fiscal", "guidance",
        "beat", "beats", "miss", "misses", "missed",
        "forecast", "outlook",
    ],
    "regulatory": [
        "sec", "fda", "regulation", "regulations", "compliance",
        "fine", "fined", "fines", "penalty", "penalties",
        "lawsuit", "lawsuits", "settlement", "antitrust",
        "probe", "probes", "investigation", "investigations",
    ],
    "merger": [
        "merger", "mergers", "acquisition", "acquisitions",
        "acquire", "acquired", "acquires",
        "takeover", "buyout", "deal", "deals",
        "bid", "bids", "offer", "offers", "combine", "merge", "merged",
    ],
    "litigation": [
        "lawsuit", "lawsuits", "sue", "sued", "suing",
        "court", "verdict", "trial", "trials", "legal",
        "class action", "patent", "patents", "litigation", "ruling", "rulings",
    ],
    "policy": [
        "fed", "federal reserve", "interest rate", "interest rates",
        "inflation", "tariff", "tariffs", "trade war",
        "sanctions", "stimulus", "policy", "policies",
    ],
    "product": [
        "launch", "launched", "launches",
        "release", "released", "releases",
        "announce", "announced", "announces",
        "unveil", "unveiled", "unveils",
        "product", "products", "feature", "features",
        "update", "upgrade", "new model", "ai",
    ],
}

# 预编译主题关键词为 word-boundary 正则
_TOPIC_PATTERNS: dict[str, list[re.Pattern]] = {
    topic: _build_keyword_patterns(keywords)
    for topic, keywords in _TOPIC_KEYWORDS.items()
}

# 紧急度关键词
_URGENCY_HIGH_KEYWORDS = [
    "breaking", "urgent", "alert", "alerts",
    "crash", "crashes", "crashed", "crashing",
    "plunge", "plunges", "plunged", "plunging",
    "surge", "surges", "surged", "surging",
    "halt", "halted", "halts",
    "bankrupt", "bankruptcy",
    "default", "defaults", "defaulted",
    "recall", "recalls", "recalled",
    "emergency", "fraud", "fraudulent", "scandal",
]

_URGENCY_MEDIUM_KEYWORDS = [
    "downgrade", "downgrades", "downgraded",
    "upgrade", "upgrades", "upgraded",
    "cut", "cuts", "cutting",
    "raise", "raises", "raised", "raising",
    "target", "targets", "targeted",
    "warning", "warnings", "warned",
    "restructure", "restructures", "restructured", "restructuring",
    "layoff", "layoffs",
    "resign", "resigns", "resigned", "resignation",
    "appoint", "appoints", "appointed", "appointment",
]

# 预编译紧急度关键词
_URGENCY_HIGH_PATTERNS = _build_keyword_patterns(_URGENCY_HIGH_KEYWORDS)
_URGENCY_MEDIUM_PATTERNS = _build_keyword_patterns(_URGENCY_MEDIUM_KEYWORDS)


@dataclass
class G2Result:
    """G2 分类结果。

    P3 重构 (m2b): sentiment_score 语义切换为 FinBERT positive softmax 概率 (0-1)。
    新增 finbert_negative / finbert_neutral 两列,方便下游直接读到 3-way softmax
    而无需再跑一次推理。规则引擎降级路径保留旧语义 (0-1 线性 score),
    finbert_negative / finbert_neutral 在规则降级路径为 None。
    """
    sentiment: str = "neutral"       # positive / neutral / negative
    sentiment_score: float = 0.5     # FinBERT positive softmax (used_finbert=True) 或 规则 score (False)
    confidence: float = 0.0          # max 三项 softmax (FinBERT) / heuristic conf (rules)
    topic: str = "other"             # earnings / regulatory / merger / litigation / policy / product / other
    importance_score: float = 0.5    # 0.0 to 1.0
    urgency: str = "low"             # high / medium / low
    used_finbert: bool = False       # 是否使用了 FinBERT（vs 降级为规则）
    finbert_negative: float | None = None  # P3: FinBERT negative softmax,规则降级为 None
    finbert_neutral: float | None = None   # P3: FinBERT neutral softmax


@dataclass
class G2Config:
    """G2 分类器配置。

    P3 重构后,FinBERT 推理经 m2a `FinBERTScorer` singleton,不再在 g2 本地加载;
    finbert_model / device 字段保留兼容老调用方,但实际推理由 singleton 管理。
    """
    use_finbert: bool = True         # 是否使用 FinBERT（False 时纯规则引擎）
    finbert_model: str = FINBERT_MODEL
    device: str = "auto"             # 保留兼容,实际 device 由 FinBERTScorer singleton 决定
    batch_size: int = 16


class G2Classifier:
    """G2 分类器。

    两层架构：
    1. FinBERT — 情绪三分类 (positive/neutral/negative) + confidence
    2. 规则引擎 — 主题分类、重要度评分、紧急度评分

    FinBERT 不可用时（未安装 torch/transformers、模型加载失败），
    自动降级为纯规则模式。
    """

    def __init__(self, config: G2Config | None = None) -> None:
        self._config = config or G2Config()
        # P3: 复用 m2a FinBERTScorer singleton,消除同一进程多次加载 ProsusAI/finbert
        self._scorer: Any = None
        self._scorer_available: bool | None = None

    def classify(self, headline: str, snippet: str | None = None) -> G2Result:
        """对单条新闻进行 G2 分类。"""
        text = self._prepare_text(headline, snippet)
        if not text:
            return G2Result()

        # 1. 情绪分类
        (
            sentiment,
            score,
            confidence,
            used_finbert,
            fb_neg,
            fb_neu,
        ) = self._classify_sentiment(text)

        # 2. 主题分类
        topic = self._classify_topic(text)

        # 3. 重要度评分
        importance = self._score_importance(text, topic, confidence)

        # 4. 紧急度评分
        urgency = self._score_urgency(text)

        return G2Result(
            sentiment=sentiment,
            sentiment_score=score,
            confidence=confidence,
            topic=topic,
            importance_score=round(importance, 3),
            urgency=urgency,
            used_finbert=used_finbert,
            finbert_negative=fb_neg,
            finbert_neutral=fb_neu,
        )

    def classify_batch(self, items: list[dict[str, str]]) -> list[G2Result]:
        """批量分类。items 为 [{"headline": ..., "snippet": ...}, ...]。

        如果 FinBERT 可用，使用 pipeline batch 推理优化。
        """
        if not items:
            return []

        texts = [self._prepare_text(it.get("headline", ""), it.get("snippet")) for it in items]

        # 批量情绪分类
        sentiments = self._classify_sentiment_batch(texts)

        results = []
        for text, sent_tuple in zip(texts, sentiments):
            sentiment, score, confidence, used_finbert, fb_neg, fb_neu = sent_tuple
            if not text:
                results.append(G2Result())
                continue
            topic = self._classify_topic(text)
            importance = self._score_importance(text, topic, confidence)
            urgency = self._score_urgency(text)
            results.append(G2Result(
                sentiment=sentiment,
                sentiment_score=round(score, 3),
                confidence=round(confidence, 3),
                topic=topic,
                importance_score=round(importance, 3),
                urgency=urgency,
                used_finbert=used_finbert,
                finbert_negative=(
                    round(fb_neg, 3) if fb_neg is not None else None
                ),
                finbert_neutral=(
                    round(fb_neu, 3) if fb_neu is not None else None
                ),
            ))
        return results

    # ------ 文本预处理 ------

    def _prepare_text(self, headline: str, snippet: str | None = None) -> str:
        """拼接标题和摘要，截断到 FinBERT 最大长度。"""
        text = (headline or "").strip()
        if snippet:
            text = f"{text} {snippet.strip()}"
        # 粗略截断到 ~FINBERT_MAX_LENGTH 个 token（按字符估算，1 token ≈ 4 chars）
        max_chars = FINBERT_MAX_LENGTH * 4
        if len(text) > max_chars:
            text = text[:max_chars]
        return text

    # ------ 可分类性检查 ------

    @staticmethod
    def _is_classifiable(text: str) -> bool:
        """检查文本是否适合 FinBERT 分类。

        FinBERT 是英语模型，非英语文本和纯符号/数字文本会产生无意义结果。
        用 ASCII 字母占比作为快速判定：低于 MIN_ASCII_RATIO 视为不可分类。
        """
        if not text:
            return False
        ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
        non_space = sum(1 for c in text if not c.isspace())
        if non_space == 0:
            return False
        return ascii_letters / non_space >= MIN_ASCII_RATIO

    # ------ 情绪分类 ------

    # P3 重构: 复用 m2a FinBERTScorer singleton。返回 tuple 扩展为 6 元素:
    # (sentiment, score, confidence, used_finbert, finbert_negative, finbert_neutral)
    #   - used_finbert=True 时: score = FinBERT positive softmax, fb_neg/fb_neu 有值
    #   - used_finbert=False 时 (规则降级或不可分类): fb_neg/fb_neu = None

    _NEUTRAL_TUPLE = ("neutral", 0.5, 0.0, False, None, None)
    _NEUTRAL_LOWCONF = ("neutral", 0.5, 0.1, False, None, None)

    def _classify_sentiment(
        self, text: str
    ) -> tuple[str, float, float, bool, float | None, float | None]:
        """单条情绪分类。"""
        if not text:
            return self._NEUTRAL_TUPLE
        if not self._is_classifiable(text):
            return self._NEUTRAL_LOWCONF

        if self._config.use_finbert:
            result = self._scorer_classify([text])
            if result:
                return result[0]

        rule = self._rule_sentiment(text)
        # 扩展旧 4 元组到新 6 元组
        return (rule[0], rule[1], rule[2], rule[3], None, None)

    def _classify_sentiment_batch(
        self, texts: list[str]
    ) -> list[tuple[str, float, float, bool, float | None, float | None]]:
        """批量情绪分类。"""
        if not texts:
            return []

        if self._config.use_finbert and self._ensure_scorer():
            valid_pairs = [
                (i, t) for i, t in enumerate(texts) if t and self._is_classifiable(t)
            ]
            results: list[
                tuple[str, float, float, bool, float | None, float | None]
            ] = [self._NEUTRAL_TUPLE] * len(texts)
            if not valid_pairs:
                # 全部不可分类,填 low-conf neutral
                return [
                    self._NEUTRAL_LOWCONF if t else self._NEUTRAL_TUPLE
                    for t in texts
                ]
            valid_texts = [t for _, t in valid_pairs]
            try:
                scored = self._scorer.score_texts(
                    valid_texts, batch_size=self._config.batch_size
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("FinBERT batch failed, falling back to rules: %s", e)
                return self._rule_batch(texts)

            # 把 valid 结果填回原位置
            for (orig_idx, _text), score in zip(valid_pairs, scored):
                label, pos, neg, neu, conf = _label_from_softmax(score)
                results[orig_idx] = (label, pos, conf, True, neg, neu)
            # 其余非 valid 位置按文本状态填 neutral 或 low-conf
            for i, t in enumerate(texts):
                if results[i] is self._NEUTRAL_TUPLE:
                    if not t:
                        results[i] = self._NEUTRAL_TUPLE
                    elif not self._is_classifiable(t):
                        results[i] = self._NEUTRAL_LOWCONF
            return results

        return self._rule_batch(texts)

    def _rule_batch(
        self, texts: list[str]
    ) -> list[tuple[str, float, float, bool, float | None, float | None]]:
        out: list[tuple[str, float, float, bool, float | None, float | None]] = []
        for t in texts:
            if not t:
                out.append(self._NEUTRAL_TUPLE)
            elif not self._is_classifiable(t):
                out.append(self._NEUTRAL_LOWCONF)
            else:
                r = self._rule_sentiment(t)
                out.append((r[0], r[1], r[2], r[3], None, None))
        return out

    def _scorer_classify(
        self, texts: list[str]
    ) -> list[tuple[str, float, float, bool, float | None, float | None]] | None:
        """使用 m2a singleton 推理;失败返回 None 以触发规则降级。"""
        if not self._ensure_scorer():
            return None
        try:
            scored = self._scorer.score_texts(
                texts, batch_size=self._config.batch_size
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("FinBERTScorer inference failed: %s", e)
            return None
        out: list[tuple[str, float, float, bool, float | None, float | None]] = []
        for row in scored:
            label, pos, neg, neu, conf = _label_from_softmax(row)
            out.append((label, pos, conf, True, neg, neu))
        return out

    def _ensure_scorer(self) -> bool:
        """Lazy 取 m2a FinBERTScorer singleton。"""
        if self._scorer_available is False:
            return False
        if self._scorer is not None:
            return True
        try:
            from stockbee.small_models.finbert_scorer import get_default_scorer

            self._scorer = get_default_scorer()
            self._scorer_available = True
            return True
        except ImportError:
            logger.info(
                "stockbee.small_models not available, using rule-based sentiment"
            )
            self._scorer_available = False
            return False
        except Exception as e:  # noqa: BLE001
            logger.warning("FinBERTScorer init failed: %s, using rule-based", e)
            self._scorer_available = False
            return False

    # 预编译情绪关键词（类级别，所有实例共享）
    # 包含常见变形（beats/surges/crashed 等），因为 word boundary 要求精确匹配
    _POSITIVE_PATTERNS = _build_keyword_patterns([
        "beat", "beats", "beating",
        "surge", "surges", "surging", "surged",
        "gain", "gains", "gaining", "gained",
        "rise", "rises", "rising", "rose",
        "profit", "profits", "profitable",
        "growth",
        "upgrade", "upgrades", "upgraded",
        "record",
        "strong", "stronger", "strongest",
        "rally", "rallies", "rallied", "rallying",
        "outperform", "outperforms", "outperformed",
        "boost", "boosts", "boosted",
        "bullish",
        "optimistic",
        "upside",
        "positive",
        "exceed", "exceeds", "exceeded", "exceeding",
        "soar", "soars", "soared", "soaring",
    ])
    _NEGATIVE_PATTERNS = _build_keyword_patterns([
        "miss", "misses", "missed", "missing",
        "drop", "drops", "dropped", "dropping",
        "fall", "falls", "falling", "fell",
        "loss", "losses",
        "decline", "declines", "declined", "declining",
        "crash", "crashes", "crashed", "crashing",
        "plunge", "plunges", "plunged", "plunging",
        "weak", "weaker", "weakest",
        "downgrade", "downgrades", "downgraded",
        "cut", "cuts", "cutting",
        "layoff", "layoffs",
        "bankrupt", "bankruptcy",
        "default", "defaults", "defaulted",
        "pessimistic",
        "bearish",
        "negative",
        "warning", "warnings", "warned",
        "fraud", "fraudulent",
    ])

    def _rule_sentiment(self, text: str) -> tuple[str, float, float, bool]:
        """规则引擎情绪分类（FinBERT 降级模式）。"""
        text_lower = text.lower()

        pos_count = _count_keyword_hits(text_lower, self._POSITIVE_PATTERNS)
        neg_count = _count_keyword_hits(text_lower, self._NEGATIVE_PATTERNS)
        total = pos_count + neg_count

        if total == 0:
            return ("neutral", 0.5, 0.3, False)

        ratio = pos_count / total
        if ratio > 0.6:
            score = 0.5 + (ratio - 0.5) * 0.8
            return ("positive", min(score, 0.95), 0.5, False)
        elif ratio < 0.4:
            score = 0.5 - (0.5 - ratio) * 0.8
            return ("negative", max(score, 0.05), 0.5, False)
        else:
            return ("neutral", 0.5, 0.4, False)

    # ------ 主题分类 ------

    def _classify_topic(self, text: str) -> str:
        """规则引擎主题分类。匹配关键词最多的类别（word-boundary 匹配）。"""
        text_lower = text.lower()
        best_topic = "other"
        best_count = 0

        for topic, patterns in _TOPIC_PATTERNS.items():
            count = _count_keyword_hits(text_lower, patterns)
            if count > best_count:
                best_count = count
                best_topic = topic

        return best_topic

    # ------ 重要度评分 ------

    def _score_importance(self, text: str, topic: str, confidence: float) -> float:
        """重要度评分 0-1。基于主题权重 + 情绪置信度 + 文本长度。"""
        # 基础分：主题权重
        topic_weights = {
            "earnings": 0.7, "regulatory": 0.8, "merger": 0.9,
            "litigation": 0.6, "policy": 0.7, "product": 0.5, "other": 0.3,
        }
        base = topic_weights.get(topic, 0.3)

        # 置信度加分
        conf_bonus = confidence * 0.2

        # 文本长度加分（更长的文章通常更重要）
        length_bonus = min(len(text) / 2000, 0.1)

        score = base + conf_bonus + length_bonus
        return min(max(score, 0.0), 1.0)

    # ------ 紧急度评分 ------

    def _score_urgency(self, text: str) -> str:
        """紧急度评分：high / medium / low（word-boundary 匹配）。"""
        text_lower = text.lower()

        if _any_keyword_hit(text_lower, _URGENCY_HIGH_PATTERNS):
            return "high"
        if _any_keyword_hit(text_lower, _URGENCY_MEDIUM_PATTERNS):
            return "medium"
        return "low"
