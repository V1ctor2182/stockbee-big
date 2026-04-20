"""m5 Fin-E5 Embedding Scorer + Importance Baseline 测试。

覆盖:
- FinE5Scorer API: encode / cosine_sim / cosine_dedup
- mock encode 确定性
- cosine_dedup 传递闭包 + 空输入 + threshold 边界
- baseline_importance 公式: 值域 [0, 1], 手算 golden, NaN 传播
- backfill_importance: 只写 NULL 行, since 过滤, 批次分页, 空 DB
- 真实 e5-large-v2 (@pytest.mark.perf): encode shape, batch 一致性
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from stockbee.small_models.fine5_scorer import (
    FinE5Scorer,
    _l2_normalize,
    _mock_encode,
)
from stockbee.small_models.importance_baseline import (
    backfill_importance,
    baseline_importance,
)

_RUN_REAL = bool(os.getenv("RUN_PERF_TESTS"))
_real_only = pytest.mark.skipif(
    not _RUN_REAL,
    reason="set RUN_PERF_TESTS=1 to load real e5-large-v2 weights",
)


# ============================================================================
# FinE5Scorer — mock 模式
# ============================================================================


class TestFinE5Mock:
    def test_construct_mock_no_hf_load(self):
        s = FinE5Scorer(device="mock")
        assert s.device == "mock"
        assert s._model is None
        assert s._tokenizer is None

    def test_embedding_dim_default_mock(self):
        s = FinE5Scorer(device="mock")
        assert s.embedding_dim == 1024  # mock 默认

    def test_encode_shape(self):
        s = FinE5Scorer(device="mock")
        emb = s.encode(["text one", "text two", "text three"])
        assert emb.shape == (3, 1024)
        assert emb.dtype == np.float32

    def test_encode_l2_normalized(self):
        s = FinE5Scorer(device="mock")
        emb = s.encode(["foo", "bar"])
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_encode_deterministic(self):
        s = FinE5Scorer(device="mock")
        e1 = s.encode(["same text"])
        e2 = s.encode(["same text"])
        assert np.allclose(e1, e2)

    def test_encode_different_texts_different_vectors(self):
        s = FinE5Scorer(device="mock")
        emb = s.encode(["text A", "text B wholly different"])
        # 高维随机向量 cosine ~ 0
        assert not np.allclose(emb[0], emb[1])

    def test_encode_empty_returns_empty_no_load(self):
        s = FinE5Scorer()  # 非 mock 也应短路
        out = s.encode([])
        assert out.shape[0] == 0
        assert s._model is None

    def test_encode_non_list_rejected(self):
        s = FinE5Scorer(device="mock")
        with pytest.raises(TypeError):
            s.encode("single")  # type: ignore[arg-type]

    def test_encode_invalid_batch_size(self):
        s = FinE5Scorer(device="mock")
        with pytest.raises(ValueError):
            s.encode(["x"], batch_size=0)

    def test_encode_invalid_max_length(self):
        s = FinE5Scorer(device="mock")
        with pytest.raises(ValueError):
            s.encode(["x"], max_length=0)


# ============================================================================
# cosine_sim
# ============================================================================


class TestCosineSim:
    def test_self_similarity_is_identity(self):
        s = FinE5Scorer(device="mock")
        emb = s.encode(["a", "b", "c"])
        sim = s.cosine_sim(emb)
        assert sim.shape == (3, 3)
        # 对角 ≈ 1
        assert np.allclose(np.diag(sim), 1.0, atol=1e-5)
        # 对称
        assert np.allclose(sim, sim.T, atol=1e-5)

    def test_cross_matrix_shape(self):
        s = FinE5Scorer(device="mock")
        a = s.encode(["a1", "a2", "a3"])
        b = s.encode(["b1", "b2"])
        sim = s.cosine_sim(a, b)
        assert sim.shape == (3, 2)

    def test_shape_mismatch_raises(self):
        s = FinE5Scorer(device="mock")
        with pytest.raises(ValueError):
            s.cosine_sim(np.zeros((3, 10)), np.zeros((2, 8)))

    def test_1d_input_raises(self):
        s = FinE5Scorer(device="mock")
        with pytest.raises(ValueError):
            s.cosine_sim(np.zeros(10))


# ============================================================================
# cosine_dedup — 重点: 传递闭包
# ============================================================================


def _vec_near(base: np.ndarray, angle_frac: float) -> np.ndarray:
    """给定 base 单位向量,返回与之 cosine ≈ cos(angle_frac*π/2) 的向量。"""
    # 构造正交扰动
    rng = np.random.default_rng(0)
    perp = rng.standard_normal(len(base)).astype(np.float32)
    perp -= (perp @ base) * base
    perp /= np.linalg.norm(perp)
    # cosine theta = cos(angle)
    theta = angle_frac * (np.pi / 2.0)
    v = np.cos(theta) * base + np.sin(theta) * perp
    return (v / np.linalg.norm(v)).astype(np.float32)


class TestCosineDedup:
    def test_two_near_texts_one_cluster(self):
        """两条高相似向量 → dedup 保留 1 个 representative。"""
        s = FinE5Scorer(device="mock")
        base = np.zeros(32, dtype=np.float32)
        base[0] = 1.0
        v1 = base.copy()
        v2 = _vec_near(base, 0.05)  # cosine ≈ cos(4.5°) ≈ 0.9969
        emb = np.stack([v1, v2])
        keep = s.cosine_dedup(emb, threshold=0.99)
        assert keep == [0]

    def test_unrelated_texts_stay_separate(self):
        s = FinE5Scorer(device="mock")
        dim = 32
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((3, dim)).astype(np.float32)
        emb = _l2_normalize(emb)
        keep = s.cosine_dedup(emb, threshold=0.99)
        assert keep == [0, 1, 2]

    def test_transitive_closure(self):
        """A~B, B~C, 但 A-C 直接 < threshold → 传递闭包仍同组 (union-find)。"""
        s = FinE5Scorer(device="mock")
        dim = 64
        base = np.zeros(dim, dtype=np.float32)
        base[0] = 1.0
        A = base.copy()
        B = _vec_near(base, 0.18)
        C = _vec_near(B, 0.18)
        emb = np.stack([A, B, C])

        sim = s.cosine_sim(emb)
        # 选一个 threshold 使 A-B 和 B-C 都 >= threshold,且 A-C < threshold
        # 保证本测试专门走"传递闭包"分支,而非普通三元合并
        threshold = min(sim[0, 1], sim[1, 2]) - 1e-4
        assert sim[0, 1] >= threshold
        assert sim[1, 2] >= threshold
        assert sim[0, 2] < threshold, (
            f"测试设计失效: A-C {sim[0,2]:.4f} 应低于 threshold {threshold:.4f}"
        )
        keep_transitive = s.cosine_dedup(emb, threshold=threshold)
        assert keep_transitive == [0], (
            f"传递闭包失败: sim={sim} keep={keep_transitive}"
        )

    def test_threshold_exact_equality_merges(self):
        """sim == threshold 时使用 >= 合并 (不漏)。"""
        s = FinE5Scorer(device="mock")
        dim = 8
        base = np.zeros(dim, dtype=np.float32)
        base[0] = 1.0
        # 构造 cosine 恰好 = 0.8 的两向量: v = 0.8 * base + 0.6 * perp
        perp = np.zeros(dim, dtype=np.float32)
        perp[1] = 1.0
        v = 0.8 * base + 0.6 * perp
        v = (v / np.linalg.norm(v)).astype(np.float32)
        emb = np.stack([base, v])
        sim = s.cosine_sim(emb)
        assert sim[0, 1] == pytest.approx(0.8, abs=1e-5)
        keep = s.cosine_dedup(emb, threshold=0.8)
        assert keep == [0], "相等即合并 (>= threshold),不应漏合"
        # 略高于 sim → 不合并
        keep_strict = s.cosine_dedup(emb, threshold=0.81)
        assert keep_strict == [0, 1]

    def test_empty_input(self):
        s = FinE5Scorer(device="mock")
        assert s.cosine_dedup(np.zeros((0, 10))) == []

    def test_1d_embeds_raises(self):
        s = FinE5Scorer(device="mock")
        with pytest.raises(ValueError):
            s.cosine_dedup(np.zeros(10))

    def test_threshold_out_of_range(self):
        s = FinE5Scorer(device="mock")
        with pytest.raises(ValueError):
            s.cosine_dedup(np.ones((2, 4), dtype=np.float32), threshold=1.5)

    def test_threshold_boundary_at_zero_merges_nonnegative_pairs(self):
        """threshold=0 → 所有正 cosine 对合并 (mock 向量高维几乎正交,可能部分合并)。"""
        s = FinE5Scorer(device="mock")
        emb = s.encode(["a", "b", "c"])
        keep = s.cosine_dedup(emb, threshold=0.0)
        # 至少合并到 1 组 (不大于 3)
        assert 1 <= len(keep) <= 3

    def test_threshold_one_keeps_all_unless_identical(self):
        """threshold=1.0 → 几乎不合并 (除非完全相同)。"""
        s = FinE5Scorer(device="mock")
        emb = s.encode(["a", "b"])
        keep = s.cosine_dedup(emb, threshold=1.0 - 1e-9)
        # "a" 和 "b" mock 出的向量不会一模一样
        assert len(keep) == 2


# ============================================================================
# Device resolution (mirrors FinBERT)
# ============================================================================


class TestDeviceResolution:
    def test_mock_repr(self):
        s = FinE5Scorer(device="mock")
        assert "FinE5Scorer" in repr(s)
        assert "mock" in repr(s)

    def test_unloaded_repr(self):
        s = FinE5Scorer()
        assert "unloaded" in repr(s)

    def test_invalid_device(self):
        from stockbee.small_models.fine5_scorer import _resolve_device
        from unittest.mock import MagicMock

        stub = MagicMock()
        with pytest.raises(ValueError):
            _resolve_device(stub, "tpu")


# ============================================================================
# baseline_importance — 公式 + 值域 + NaN
# ============================================================================


class TestBaselineImportance:
    @staticmethod
    def _df(**cols):
        return pd.DataFrame(cols)

    def test_value_range_in_0_1(self):
        df = self._df(
            count_30d=[0, 5, 10, 20, 100],
            sentiment_score=[0.0, 0.25, 0.5, 0.75, 1.0],
            reliability_score=[0.0, 0.5, 1.0, 1.2, -0.3],
        )
        s = baseline_importance(df)
        assert (s >= 0.0).all()
        assert (s <= 1.0).all()

    def test_high_extreme_positive(self):
        """高 count + 极端情绪 (sent=1.0) + 高 reliability → 接近 1。"""
        df = self._df(
            count_30d=[20],
            sentiment_score=[1.0],
            reliability_score=[1.0],
        )
        assert baseline_importance(df).iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_high_extreme_negative(self):
        """sent=0.0 同样极端。"""
        df = self._df(
            count_30d=[20],
            sentiment_score=[0.0],
            reliability_score=[1.0],
        )
        assert baseline_importance(df).iloc[0] == pytest.approx(1.0, abs=1e-6)

    def test_neutral_sentiment_zero(self):
        """sent=0.5 → importance=0 (无论 count/reliability)。"""
        df = self._df(
            count_30d=[100],
            sentiment_score=[0.5],
            reliability_score=[1.0],
        )
        assert baseline_importance(df).iloc[0] == 0.0

    def test_low_count_low_score(self):
        df = self._df(
            count_30d=[0],
            sentiment_score=[1.0],
            reliability_score=[1.0],
        )
        assert baseline_importance(df).iloc[0] == 0.0

    def test_manual_golden(self):
        """count=5 (factor=0.5), sent=0.8 (factor=0.6), rel=0.5 → 0.5*0.6*0.5 = 0.15。"""
        df = self._df(
            count_30d=[5],
            sentiment_score=[0.8],
            reliability_score=[0.5],
        )
        assert baseline_importance(df).iloc[0] == pytest.approx(0.15, abs=1e-9)

    def test_reliability_negative_clipped_to_zero(self):
        df = self._df(
            count_30d=[10],
            sentiment_score=[1.0],
            reliability_score=[-0.5],
        )
        assert baseline_importance(df).iloc[0] == 0.0

    def test_count_saturation(self):
        """count >= 10 和 count = 1000 产出相同 (满饱和)。"""
        a = self._df(count_30d=[10], sentiment_score=[1.0], reliability_score=[1.0])
        b = self._df(count_30d=[1000], sentiment_score=[1.0], reliability_score=[1.0])
        assert baseline_importance(a).iloc[0] == baseline_importance(b).iloc[0]

    def test_nan_in_count_30d_propagates(self):
        df = self._df(
            count_30d=[np.nan, 10],
            sentiment_score=[0.8, 0.8],
            reliability_score=[0.9, 0.9],
        )
        out = baseline_importance(df)
        assert np.isnan(out.iloc[0])
        assert not np.isnan(out.iloc[1])

    def test_nan_in_sentiment_propagates(self):
        df = self._df(
            count_30d=[10, 10],
            sentiment_score=[np.nan, 0.8],
            reliability_score=[0.9, 0.9],
        )
        out = baseline_importance(df)
        assert np.isnan(out.iloc[0])
        assert not np.isnan(out.iloc[1])

    def test_nan_in_reliability_propagates(self):
        df = self._df(
            count_30d=[10, 10],
            sentiment_score=[0.8, 0.8],
            reliability_score=[np.nan, 0.9],
        )
        out = baseline_importance(df)
        assert np.isnan(out.iloc[0])
        assert not np.isnan(out.iloc[1])

    def test_missing_column_raises(self):
        df = self._df(
            count_30d=[10],
            sentiment_score=[0.8],
            # reliability_score 缺失
        )
        with pytest.raises(ValueError, match="reliability_score"):
            baseline_importance(df)


# ============================================================================
# backfill_importance — SQLite 集成
# ============================================================================


@pytest.fixture
def populated_news_store(tmp_path, monkeypatch):
    """新建临时 SqliteNewsProvider, 插入 5 条新闻 (不同 ticker/时间) 并初始化。"""
    from stockbee.news_data.news_store import SqliteNewsProvider
    from stockbee.providers.config import ProviderConfig

    db_path = tmp_path / "news.db"
    cfg = ProviderConfig(implementation="test", params={"db_path": str(db_path)})
    store = SqliteNewsProvider(cfg)
    store.initialize()

    base = datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
    # 5 条: 2 AAPL 近期 + 2 MSFT 近期 + 1 旧 AAPL
    events = [
        {
            "headline": "AAPL beats earnings",
            "source": "reuters",
            "timestamp": base.isoformat(),
            "tickers": ["AAPL"],
            "sentiment_score": 0.9,
            "reliability_score": 0.8,
        },
        {
            "headline": "AAPL new product",
            "source": "bloomberg",
            "timestamp": (base + timedelta(hours=6)).isoformat(),
            "tickers": ["AAPL"],
            "sentiment_score": 0.7,
            "reliability_score": 0.9,
        },
        {
            "headline": "MSFT cloud revenue",
            "source": "reuters",
            "timestamp": (base + timedelta(hours=12)).isoformat(),
            "tickers": ["MSFT"],
            "sentiment_score": 0.85,
            "reliability_score": 0.7,
        },
        {
            "headline": "MSFT cost cuts",
            "source": "bloomberg",
            "timestamp": (base + timedelta(hours=18)).isoformat(),
            "tickers": ["MSFT"],
            "sentiment_score": 0.3,
            "reliability_score": 0.6,
        },
        {
            "headline": "AAPL ancient news",
            "source": "reuters",
            "timestamp": (base - timedelta(days=60)).isoformat(),
            "tickers": ["AAPL"],
            "sentiment_score": 0.6,
            "reliability_score": 0.5,
        },
    ]
    store.insert_news_batch(events)
    yield store
    store.shutdown()


class TestBackfillImportance:
    def test_writes_null_rows(self, populated_news_store):
        updated = backfill_importance(populated_news_store, batch=100)
        assert updated == 5
        # 再次跑应 0 (都已写)
        updated2 = backfill_importance(populated_news_store, batch=100)
        assert updated2 == 0

    def test_does_not_overwrite_non_null(self, populated_news_store):
        # 手动在一条上预置非空值
        conn = populated_news_store._conn
        conn.execute(
            "UPDATE news_events SET fine5_importance = 0.777 WHERE id = 1"
        )
        conn.commit()
        backfill_importance(populated_news_store, batch=100)
        row = conn.execute(
            "SELECT fine5_importance FROM news_events WHERE id = 1"
        ).fetchone()
        assert row[0] == pytest.approx(0.777, abs=1e-9)

    def test_since_filter(self, populated_news_store):
        # since = base - 30d → ancient news (60 days old) 被排除
        since = "2026-02-15T00:00:00+00:00"
        updated = backfill_importance(
            populated_news_store, since=since, batch=100
        )
        # 5 条中 1 条在 since 之前 → 只处理 4 条
        assert updated == 4

    def test_since_filter_naive_tz_treated_as_utc(self, populated_news_store):
        """review H1: 无 tz 的 since 字符串自动按 UTC 处理。"""
        updated = backfill_importance(
            populated_news_store, since="2026-02-15T00:00:00", batch=100
        )
        assert updated == 4

    def test_since_filter_non_utc_tz_converted(self, populated_news_store):
        """review H1: 非 UTC tz 的 since 转成 UTC 再与 DB 字符串比较。"""
        # 2026-02-15 00:00:00 -05:00  等价  2026-02-15 05:00:00 UTC
        # 5 条新闻里 "ancient" 是 base - 60d (~2026-01-01 UTC), 在此 since 之前
        # 近期 4 条 (base ~ 2026-03-01 UTC) 在 since 之后
        updated = backfill_importance(
            populated_news_store,
            since="2026-02-15T00:00:00-05:00",
            batch=100,
        )
        assert updated == 4

    def test_batch_pagination(self, populated_news_store):
        u1 = backfill_importance(populated_news_store, batch=2)
        u2 = backfill_importance(populated_news_store, batch=2)
        u3 = backfill_importance(populated_news_store, batch=2)
        assert u1 + u2 + u3 == 5
        assert u1 == 2 and u2 == 2 and u3 == 1

    def test_empty_db(self, tmp_path):
        from stockbee.news_data.news_store import SqliteNewsProvider
        from stockbee.providers.config import ProviderConfig

        cfg = ProviderConfig(implementation="t", params={"db_path": str(tmp_path / "x.db")})
        store = SqliteNewsProvider(cfg)
        store.initialize()
        assert backfill_importance(store) == 0
        store.shutdown()

    def test_skip_rows_without_sentiment(self, populated_news_store):
        """sentiment_score 为 NULL 的行不处理。"""
        conn = populated_news_store._conn
        conn.execute("UPDATE news_events SET sentiment_score = NULL WHERE id = 1")
        conn.commit()
        updated = backfill_importance(populated_news_store, batch=100)
        assert updated == 4  # 5 条中 1 条缺 sentiment → 跳过

    def test_invalid_batch(self, populated_news_store):
        with pytest.raises(ValueError):
            backfill_importance(populated_news_store, batch=0)

    def test_store_not_initialized(self, tmp_path):
        from stockbee.news_data.news_store import SqliteNewsProvider
        from stockbee.providers.config import ProviderConfig

        cfg = ProviderConfig(implementation="t", params={"db_path": str(tmp_path / "x.db")})
        store = SqliteNewsProvider(cfg)
        # 不 initialize
        with pytest.raises(RuntimeError, match="initialize"):
            backfill_importance(store)


# ============================================================================
# 内部工具直测
# ============================================================================


class TestInternalHelpers:
    def test_l2_normalize_zero_vector_safe(self):
        x = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        n = _l2_normalize(x)
        # 零行原样保留 (除数 fallback=1)
        assert np.allclose(n[0], 0.0)
        assert np.allclose(n[1], [1.0, 0.0])

    def test_mock_encode_same_text_same_vector(self):
        e1 = _mock_encode(["x"], 32)
        e2 = _mock_encode(["x"], 32)
        assert np.allclose(e1, e2)

    def test_mock_encode_l2_normalized(self):
        e = _mock_encode(["a", "b", "c"], 64)
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


# ============================================================================
# 真实 e5-large-v2 (@pytest.mark.perf)
# ============================================================================


@pytest.mark.perf
@_real_only
class TestFinE5RealModel:
    def test_encode_hidden_size_dynamic(self):
        """embedding_dim 从 model.config 动态读,而非硬编码。"""
        scorer = FinE5Scorer()
        dim = scorer.embedding_dim
        assert dim == 1024  # e5-large-v2 实际值

    def test_batch_consistency(self):
        scorer = FinE5Scorer()
        texts = [
            "company beats earnings estimates",
            "stock crashes on bad news",
            "board approves buyback",
        ]
        e_b1 = scorer.encode(texts, batch_size=1)
        e_b16 = scorer.encode(texts, batch_size=16)
        for a, b in zip(e_b1, e_b16):
            assert np.allclose(a, b, atol=1e-4)

    def test_similar_texts_high_cosine(self):
        scorer = FinE5Scorer()
        emb = scorer.encode([
            "Apple beats earnings by 12%",
            "Apple reports strong earnings beat",
            "Company announces massive layoffs",
        ])
        sim = scorer.cosine_sim(emb)
        # 前两条相关,应比与第三条更高
        assert sim[0, 1] > sim[0, 2]
        assert sim[0, 1] > sim[1, 2]
