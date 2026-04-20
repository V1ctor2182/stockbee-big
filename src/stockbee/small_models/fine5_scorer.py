"""Fin-E5 Embedding Scorer — 加载 + batch encode + cosine 去重。

决策 (见 spec.md):
- Fin-E5 本 room scope 降级为 embedding infra + 去重 + rule baseline importance;
  精确 importance fine-tune 推迟到 04-alpha-mining
- HF 无 canonical `fin-e5`,用 `intfloat/e5-large-v2` 作 fallback (hidden_size=1024)
- embedding_dim 从 model.config.hidden_size 动态读,不硬编码
- cosine_dedup 语义 = union-find 传递闭包 (A~B, B~C → A/B/C 同 cluster)

E5 模型输入惯例可加 "passage: " 前缀,本 scorer 不自动加;调用方自选一致即可
(同批文本同样处理,cosine 距离相对可比)。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "intfloat/e5-large-v2"
_DEFAULT_MAX_LENGTH = 512
_DEFAULT_BATCH = 16


class FinE5Scorer:
    """Fin-E5 / E5-large-v2 encoder。

    用法::

        scorer = FinE5Scorer()              # 自动 device
        emb = scorer.encode(["news1", "news2"])   # shape (2, hidden_size)
        sim = scorer.cosine_sim(emb)              # (2, 2) 对称矩阵
        keep = scorer.cosine_dedup(emb, 0.95)     # list[int] 去重后索引
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
        self._hidden_size: int | None = None
        self._load_lock = threading.Lock()

    def __repr__(self) -> str:
        loaded = self._model is not None or self._mock
        return (
            f"FinE5Scorer(model={self._model_name!r}, "
            f"device={self._resolved_device or 'unloaded'!r}, loaded={loaded})"
        )

    @property
    def device(self) -> str | None:
        return self._resolved_device

    @property
    def embedding_dim(self) -> int:
        """hidden_size。lazy-load 前对 mock 返回 1024,对真实模型触发加载。"""
        if self._mock:
            return self._hidden_size or 1024
        self._ensure_loaded()
        assert self._hidden_size is not None
        return self._hidden_size

    def encode(
        self,
        texts: list[str],
        batch_size: int = _DEFAULT_BATCH,
        max_length: int = _DEFAULT_MAX_LENGTH,
    ) -> np.ndarray:
        """将文本编码为 L2-normalized embedding 矩阵。

        Args:
            texts: list[str]
            batch_size: 默认 16
            max_length: 默认 512

        Returns:
            np.ndarray shape (N, hidden_size), dtype float32, 每行 L2 范数 ≈ 1
        """
        if not isinstance(texts, list):
            raise TypeError(f"texts must be list[str], got {type(texts).__name__}")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("texts must be list[str]; found non-str element")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}")

        if not texts:
            dim = self.embedding_dim if self._mock or self._model else 1
            return np.zeros((0, dim), dtype=np.float32)

        if self._mock:
            return _mock_encode(texts, self._hidden_size or 1024)

        self._ensure_loaded()
        out_batches: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            out_batches.append(self._encode_batch(batch, max_length))
        emb = np.vstack(out_batches).astype(np.float32)
        return _l2_normalize(emb)

    def cosine_sim(
        self,
        a: np.ndarray,
        b: np.ndarray | None = None,
    ) -> np.ndarray:
        """余弦相似度矩阵。假设行已 L2 归一化 → 直接 a @ b.T。

        Args:
            a: (M, D)
            b: (N, D) 或 None; None → 返回 (M, M) 对称矩阵

        Returns:
            np.ndarray shape (M, N)
        """
        if a.ndim != 2:
            raise ValueError(f"a must be 2D, got shape {a.shape}")
        a_n = _l2_normalize(a)
        if b is None:
            b_n = a_n
        else:
            if b.ndim != 2 or b.shape[1] != a.shape[1]:
                raise ValueError(
                    f"b shape {b.shape} incompatible with a shape {a.shape}"
                )
            b_n = _l2_normalize(b)
        return a_n @ b_n.T

    def cosine_dedup(
        self,
        embeds: np.ndarray,
        threshold: float = 0.95,
    ) -> list[int]:
        """Union-find 去重。

        对 i<j,若 cosine_sim(i,j) >= threshold 则 i,j 合并到同 cluster
        (**传递闭包**: A~B, B~C → A/B/C 同组,即使 A-C 直接距离 < threshold)。

        复杂度: O(N²) 时间 + 内存 (构造 N×N sim 矩阵),适合 N ≲ 数千 的批次。
        更大规模去重请用 FAISS + blocking 策略 (本 room scope 不覆盖)。

        Args:
            embeds: (N, D) 行 L2 归一化 (非归一化也允许,内部重新算 cosine)
            threshold: 合并阈值 ∈ [0, 1],默认 0.95 (相等也合并,使用 >=)

        Returns:
            list[int]: 去重后保留的行索引 (每 cluster 取最小下标作代表),升序返回
        """
        if embeds.ndim != 2:
            raise ValueError(f"embeds must be 2D, got shape {embeds.shape}")
        n = embeds.shape[0]
        if n == 0:
            return []
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")

        sim = self.cosine_sim(embeds)
        parent = list(range(n))

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:  # 路径压缩
                nxt = parent[x]
                parent[x] = root
                x = nxt
            return root

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            # 始终让较小索引做根,保证 "每 cluster 代表 = 最小下标"
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= threshold:
                    union(i, j)

        representatives = sorted({find(i) for i in range(n)})
        return representatives

    # ---- 内部 ----

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "FinE5Scorer requires transformers + torch. "
                    "Install with `pip install transformers torch` "
                    "or pass device='mock' for test-only fake encoding."
                ) from exc

            self._torch = torch
            device = _resolve_device(torch, self._device_arg)
            logger.info(
                "Loading Fin-E5: model=%s device=%s", self._model_name, device
            )
            tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            model = AutoModel.from_pretrained(self._model_name)
            model.to(device)
            model.eval()

            self._tokenizer = tokenizer
            self._model = model
            self._resolved_device = device
            self._hidden_size = int(model.config.hidden_size)

    def _encode_batch(self, batch: list[str], max_length: int) -> np.ndarray:
        torch = self._torch
        tok = self._tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(self._resolved_device) for k, v in tok.items()}
        with torch.no_grad():
            out = self._model(**tok)
        # mean-pool with attention mask
        last_hidden = out.last_hidden_state  # (B, L, H)
        mask = tok["attention_mask"].unsqueeze(-1).float()  # (B, L, 1)
        summed = (last_hidden * mask).sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        pooled = summed / counts  # (B, H)
        return pooled.cpu().numpy()


# ---- 模块级辅助 ----


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


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # < 1e-12 兜底 denormal,避免 float32 近零除法产出 inf
    norms = np.where(norms < 1e-12, 1.0, norms)
    return (x / norms).astype(np.float32)


def _mock_encode(texts: list[str], dim: int) -> np.ndarray:
    """确定性哈希 → 随机向量 → L2 归一化。

    同文本返回同向量;不同文本几乎正交 (dim=1024 下 cosine ~ 0)。
    测试 cosine_dedup 传递闭包需要手工构造相似向量,不走 _mock_encode。
    """
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        rng = np.random.default_rng(hash(t) & 0xFFFF_FFFF)
        v = rng.standard_normal(dim).astype(np.float32)
        out[i] = v
    return _l2_normalize(out)


__all__ = ["FinE5Scorer"]
