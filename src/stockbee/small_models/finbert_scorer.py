"""FinBERT 情绪评分器 — 纯推理层。

加载 ProsusAI/finbert (或兼容模型)，batch tokenize + forward + softmax，
返回 {positive, negative, neutral, confidence}。

PRD 硬约束: batch=32, max_length=128 推理 <100ms (GPU) / <300ms (CPU)。
性能基准见 tests/small_models/test_finbert_scorer.py::TestFinBERTPerfGate。

设计:
- lazy load: 首次 score_texts 才加载模型 (空列表短路,不触发加载)
- device=None: cuda → mps → cpu 自动选择
- device="mock": 走确定性关键词 heuristic,不加载 HF (m2b / g2 单测复用,避免下 440MB)
- 标签顺序从 model.config.id2label 动态读 (不硬编码 ProsusAI/finbert 特定顺序)
- singleton: get_default_scorer() 供 m2b LocalSentimentProvider + g2_classifier 共享

**架构契约 (P3)**: m2b 和 g2 必须走 get_default_scorer(),消除重复推理。
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

_LABELS = ("positive", "negative", "neutral")
_DEFAULT_MODEL = "ProsusAI/finbert"
_MAX_LENGTH = 128  # Tech Design §3.3 约束
_MOCK_POS_PATTERN = re.compile(
    r"\b(beat|beats|beating|surge|surged|gain|gains|gained|rise|rises|rose|"
    r"record|upgrade|upgraded|buy|strong|grow|grows|growing|growth|"
    r"boost|boosts|boosted|hike|hikes|raised|raising|high|highs|"
    r"profit|profits|profitable|dividend|positive|bullish)\b",
    re.IGNORECASE,
)
_MOCK_NEG_PATTERN = re.compile(
    r"\b(miss|misses|missed|plunge|plunged|drop|drops|dropped|fall|fell|"
    r"crash|crashed|cut|cuts|slash|slashed|slashes|layoff|layoffs|"
    r"fine|fined|fines|fraud|scandal|investigation|violation|violations|"
    r"resign|resigned|sue|sued|loss|losses|collapse|collapsed|collapses|"
    r"weak|weakness|decline|declines|declining|bearish|"
    r"warn|warns|warning|concern|concerns|downgrade|downgraded)\b",
    re.IGNORECASE,
)


class FinBERTScorer:
    """FinBERT 情绪评分器。

    示例::

        scorer = FinBERTScorer()          # 自动选 device
        out = scorer.score_texts(["stock beats earnings", "layoffs announced"])
        # [{"positive": 0.55, "negative": 0.23, "neutral": 0.21, "confidence": 0.55},
        #  {"positive": 0.02, "negative": 0.77, "neutral": 0.20, "confidence": 0.77}]

    测试/下游共享 FinBERT 实例::

        scorer = FinBERTScorer(device="mock")   # 不加载 HF, 关键词 heuristic
    """

    def __init__(
        self,
        device: str | None = None,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._model_name = model_name
        self._device_arg = device
        self._mock = (device == "mock")
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._resolved_device: str | None = "mock" if self._mock else None
        # id2label 顺序可能和 _LABELS 不同,加载后按 index 映射
        self._label_index: dict[str, int] | None = None
        self._load_lock = threading.Lock()

    def __repr__(self) -> str:
        loaded = self._model is not None or self._mock
        dev = self._resolved_device or "unloaded"
        return f"FinBERTScorer(model={self._model_name!r}, device={dev!r}, loaded={loaded})"

    @property
    def device(self) -> str | None:
        """已解析的 device (None = 尚未 lazy load)。"""
        return self._resolved_device

    def score_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """批量情绪评分。

        Args:
            texts: 待评分文本列表;空列表直接返回 []
            batch_size: 每次 forward 的样本数,默认 32

        Returns:
            长度等于 texts 的 list[dict],每个 dict 含 positive/negative/neutral/confidence
            (confidence = max(三项); 三项 softmax sum ≈ 1.0)
        """
        # 类型检查必须先于空检查: 否则 "" / None 会触发 `not texts` 短路,
        # 把非法输入静默当空列表返回
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list[str], got {type(texts).__name__}")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be list[str]; found non-str element")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not texts:
            return []

        if self._mock:
            return [_mock_score(t) for t in texts]

        self._ensure_loaded()
        out: list[dict[str, float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            out.extend(self._score_batch(batch))
        return out

    # ---- 内部 ----

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:  # 双检锁
                return
            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:  # pragma: no cover - 依赖检查
                raise RuntimeError(
                    "FinBERTScorer requires transformers + torch. "
                    "Install with `pip install transformers torch` "
                    "or pass device='mock' for test-only fake scoring."
                ) from exc

            self._torch = torch
            device = self._resolve_device(torch, self._device_arg)
            logger.info(
                "Loading FinBERT: model=%s device=%s", self._model_name, device
            )
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            )
            model.to(device)
            model.eval()

            id2label = getattr(model.config, "id2label", None) or {}
            label_index: dict[str, int] = {}
            for idx, lbl in id2label.items():
                key = str(lbl).lower()
                if key in _LABELS:
                    label_index[key] = int(idx)
            missing = [lbl for lbl in _LABELS if lbl not in label_index]
            if missing:
                raise RuntimeError(
                    f"model {self._model_name} id2label {id2label!r} "
                    f"missing labels: {missing}"
                )

            self._tokenizer = tokenizer
            self._model = model
            self._resolved_device = device
            self._label_index = label_index

    @staticmethod
    def _resolve_device(torch_mod: Any, device_arg: str | None) -> str:
        if device_arg in ("cpu", "cuda", "mps"):
            return device_arg
        if device_arg is None:
            if torch_mod.cuda.is_available():
                return "cuda"
            mps = getattr(torch_mod.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        raise ValueError(
            f"Invalid device {device_arg!r}; expected None/'cpu'/'cuda'/'mps'/'mock'"
        )

    def _score_batch(self, batch: list[str]) -> list[dict[str, float]]:
        torch = self._torch
        tok = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=_MAX_LENGTH,
            return_tensors="pt",
        )
        tok = {k: v.to(self._resolved_device) for k, v in tok.items()}
        with torch.no_grad():
            logits = self._model(**tok).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        idx = self._label_index or {}
        return [
            {
                "positive": float(row[idx["positive"]]),
                "negative": float(row[idx["negative"]]),
                "neutral": float(row[idx["neutral"]]),
                "confidence": float(max(row)),
            }
            for row in probs
        ]


# ---- singleton (供 m2b + g2_classifier 共享,消除重复推理) ----

_default_scorer: FinBERTScorer | None = None
_default_lock = threading.Lock()


def get_default_scorer() -> FinBERTScorer:
    """返回进程级 FinBERTScorer singleton。

    首次调用 lazy 创建 FinBERTScorer() (device=None 自动选择)。
    m2b LocalSentimentProvider 和 g2_classifier (P3 重构后) 共享同一实例,
    避免同一进程两次加载 FinBERT 权重 (~440MB) 及重复推理。
    """
    global _default_scorer
    if _default_scorer is not None:
        return _default_scorer
    with _default_lock:
        if _default_scorer is None:
            _default_scorer = FinBERTScorer()
    return _default_scorer


def reset_default_scorer() -> None:
    """清除 singleton (测试 hook)。下次 get_default_scorer 重建。"""
    global _default_scorer
    with _default_lock:
        _default_scorer = None


# ---- mock scoring (device="mock") ----


def _mock_score(text: str) -> dict[str, float]:
    """确定性关键词 heuristic,不加载 HF。

    用于 m2b / g2 单元测试,避免每次测试下载 440MB。
    - 含正向关键词 → positive ≈ 0.70, neutral 0.20, negative 0.10
    - 含负向关键词 → negative ≈ 0.70, neutral 0.20, positive 0.10
    - 两边都有或都没 → neutral ≈ 0.60, 其余各 0.20
    confidence = max 三项。
    """
    has_pos = bool(_MOCK_POS_PATTERN.search(text))
    has_neg = bool(_MOCK_NEG_PATTERN.search(text))
    if has_pos and not has_neg:
        probs = {"positive": 0.70, "negative": 0.10, "neutral": 0.20}
    elif has_neg and not has_pos:
        probs = {"positive": 0.10, "negative": 0.70, "neutral": 0.20}
    else:
        probs = {"positive": 0.20, "negative": 0.20, "neutral": 0.60}
    probs["confidence"] = max(probs.values())
    return probs


__all__ = [
    "FinBERTScorer",
    "get_default_scorer",
    "reset_default_scorer",
]
