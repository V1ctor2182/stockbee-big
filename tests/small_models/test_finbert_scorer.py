"""m2a FinBERTScorer 测试。

覆盖:
- mock 模式 (device="mock"): 不加载 HF, 确定性关键词路由
- singleton: get_default_scorer / reset_default_scorer
- API 形状: score_texts 返回 list[dict{positive,negative,neutral,confidence}]
- softmax sum / confidence = max 不变式
- 输入边界: 空列表 / 非 list / batch_size 非法 / 长文本截断
- lazy load: 空列表不触发 transformers import
- device 解析: None/cpu/cuda/mps/mock/invalid
- 真实模型 (@pytest.mark.perf, 默认 skip): finbert_golden 准确率 / 性能 gate / batch 一致性
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from stockbee.small_models import finbert_scorer as fbs
from stockbee.small_models.finbert_scorer import (
    FinBERTScorer,
    _mock_score,
    get_default_scorer,
    reset_default_scorer,
)

_RUN_REAL = bool(os.getenv("RUN_PERF_TESTS"))
_real_only = pytest.mark.skipif(
    not _RUN_REAL,
    reason="set RUN_PERF_TESTS=1 to run tests that load real FinBERT weights",
)


@pytest.fixture(autouse=True)
def _reset_singleton_between_tests():
    """每个测试独立 singleton,避免测试间状态泄漏。"""
    reset_default_scorer()
    yield
    reset_default_scorer()


# ----------------------------------------------------------------------------
# Mock 模式基础 API
# ----------------------------------------------------------------------------


class TestMockMode:
    def test_construct_mock_no_hf_load(self):
        """device='mock' 构造不触发 transformers import。"""
        s = FinBERTScorer(device="mock")
        assert s.device == "mock"
        # 内部状态: 既没加载 model 也没加载 tokenizer
        assert s._model is None
        assert s._tokenizer is None

    def test_mock_score_shape(self):
        s = FinBERTScorer(device="mock")
        out = s.score_texts(["earnings beat estimates", "layoffs announced"])
        assert len(out) == 2
        for row in out:
            assert set(row.keys()) == {"positive", "negative", "neutral", "confidence"}

    def test_mock_softmax_invariant(self):
        s = FinBERTScorer(device="mock")
        out = s.score_texts([
            "beats earnings",
            "stock crashed",
            "company reports tomorrow",
        ])
        for row in out:
            total = row["positive"] + row["negative"] + row["neutral"]
            assert total == pytest.approx(1.0, abs=1e-6)
            expected_conf = max(row["positive"], row["negative"], row["neutral"])
            assert row["confidence"] == pytest.approx(expected_conf, abs=1e-9)

    def test_mock_positive_routing(self):
        s = FinBERTScorer(device="mock")
        [row] = s.score_texts(["company beats earnings and raises guidance"])
        assert row["positive"] > row["negative"]
        assert row["positive"] > row["neutral"]

    def test_mock_negative_routing(self):
        s = FinBERTScorer(device="mock")
        [row] = s.score_texts(["company announces massive layoffs and fraud investigation"])
        assert row["negative"] > row["positive"]
        assert row["negative"] > row["neutral"]

    def test_mock_neutral_routing(self):
        s = FinBERTScorer(device="mock")
        [row] = s.score_texts(["company will report earnings next Tuesday at 4pm"])
        # 无明确正/负关键词 → neutral 占优
        assert row["neutral"] >= row["positive"]
        assert row["neutral"] >= row["negative"]

    def test_mock_accuracy_on_golden(self, finbert_golden):
        """mock 在 golden 集上精确准确率 (regression guard against 关键词 list 误改)。

        当前关键词表下 mock 应该打中 10/10;若任一关键词被改动导致 miss,
        本测试立即失败 (优于 loose 70% 阈值隐藏退化)。
        """
        s = FinBERTScorer(device="mock")
        texts = [row["text"] for row in finbert_golden]
        out = s.score_texts(texts)
        predictions = [
            max(("positive", "negative", "neutral"), key=lambda k: row[k])
            for row in out
        ]
        truths = [row["label"] for row in finbert_golden]
        mismatches = [
            (i, finbert_golden[i]["text"], p, t)
            for i, (p, t) in enumerate(zip(predictions, truths))
            if p != t
        ]
        assert not mismatches, (
            f"mock heuristic regressed on golden: {mismatches}"
        )

    def test_mock_mixed_keywords_falls_back_to_neutral(self):
        """同时含正/负关键词 → 落 neutral 分支。"""
        s = FinBERTScorer(device="mock")
        [row] = s.score_texts(["company beats estimates but warns of layoffs"])
        assert row["neutral"] >= row["positive"]
        assert row["neutral"] >= row["negative"]


# ----------------------------------------------------------------------------
# 输入/参数边界
# ----------------------------------------------------------------------------


class TestInputBoundaries:
    def test_empty_list_returns_empty_no_load(self):
        """空列表短路返回,不尝试加载模型。"""
        s = FinBERTScorer()  # 非 mock,也不应加载
        assert s.score_texts([]) == []
        assert s._model is None

    def test_non_list_raises(self):
        s = FinBERTScorer(device="mock")
        with pytest.raises(TypeError):
            s.score_texts("single string")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [None, "", 0, (), {"x": 1}])
    def test_falsy_non_list_rejected_not_silently_empty(self, bad):
        """回归测试: 空字符串/None 等 falsy 非 list 值必须 raise TypeError,
        不能走 `not texts` 短路当空列表返回。"""
        s = FinBERTScorer(device="mock")
        with pytest.raises(TypeError):
            s.score_texts(bad)  # type: ignore[arg-type]

    def test_non_str_element_raises(self):
        s = FinBERTScorer(device="mock")
        with pytest.raises(TypeError):
            s.score_texts(["ok", 123])  # type: ignore[list-item]

    def test_invalid_batch_size(self):
        s = FinBERTScorer(device="mock")
        with pytest.raises(ValueError):
            s.score_texts(["x"], batch_size=0)

    def test_whitespace_only(self):
        s = FinBERTScorer(device="mock")
        [row] = s.score_texts(["   \n\t  "])
        total = row["positive"] + row["negative"] + row["neutral"]
        assert total == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------------------
# device 解析
# ----------------------------------------------------------------------------


class TestDeviceResolution:
    def test_explicit_cpu_valid(self):
        torch_stub = _make_torch_stub(cuda_available=True, mps_available=True)
        assert FinBERTScorer._resolve_device(torch_stub, "cpu") == "cpu"

    def test_explicit_cuda_valid(self):
        torch_stub = _make_torch_stub(cuda_available=False, mps_available=False)
        assert FinBERTScorer._resolve_device(torch_stub, "cuda") == "cuda"

    def test_auto_prefers_cuda(self):
        torch_stub = _make_torch_stub(cuda_available=True, mps_available=True)
        assert FinBERTScorer._resolve_device(torch_stub, None) == "cuda"

    def test_auto_falls_back_mps(self):
        torch_stub = _make_torch_stub(cuda_available=False, mps_available=True)
        assert FinBERTScorer._resolve_device(torch_stub, None) == "mps"

    def test_auto_cpu_when_no_accel(self):
        torch_stub = _make_torch_stub(cuda_available=False, mps_available=False)
        assert FinBERTScorer._resolve_device(torch_stub, None) == "cpu"

    def test_invalid_device_string(self):
        torch_stub = _make_torch_stub()
        with pytest.raises(ValueError):
            FinBERTScorer._resolve_device(torch_stub, "tpu")


def _make_torch_stub(cuda_available: bool = False, mps_available: bool = False):
    stub = MagicMock()
    stub.cuda.is_available.return_value = cuda_available
    stub.backends.mps.is_available.return_value = mps_available
    return stub


# ----------------------------------------------------------------------------
# Singleton
# ----------------------------------------------------------------------------


class TestSingleton:
    def test_get_default_scorer_identity(self):
        a = get_default_scorer()
        b = get_default_scorer()
        assert a is b

    def test_reset_creates_new_instance(self):
        a = get_default_scorer()
        reset_default_scorer()
        b = get_default_scorer()
        assert a is not b

    def test_reset_is_idempotent(self):
        reset_default_scorer()  # 空状态 reset 不报错
        reset_default_scorer()


# ----------------------------------------------------------------------------
# HF 加载失败 → 明确错误
# ----------------------------------------------------------------------------


class TestImportFailure:
    """_ensure_loaded 的 import 失败路径都应 raise RuntimeError 明确提示,
    不裸 ImportError。用户可参考消息装依赖或切 device='mock'。

    注意: 两个测试都把 torch 和 transformers 同时 patch 为 None,避免真实加载
    torch (macOS 上 torch 的 libomp 和 lightgbm 的 libomp 在同进程冲突会 abort)。
    """

    def test_ensure_loaded_raises_when_both_missing(self):
        """torch + transformers 都缺 → RuntimeError 提示安装。"""
        s = FinBERTScorer()
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises(RuntimeError, match="transformers"):
                s._ensure_loaded()

    def test_ensure_loaded_raises_when_transformers_missing(self):
        """transformers 缺失 → 同一错误分支 (torch 同样 patch 掉以防实际加载)。"""
        s = FinBERTScorer()
        with patch.dict("sys.modules", {"transformers": None, "torch": None}):
            with pytest.raises(RuntimeError, match="transformers"):
                s._ensure_loaded()


# ----------------------------------------------------------------------------
# __repr__ / 属性
# ----------------------------------------------------------------------------


class TestRepr:
    def test_mock_repr(self):
        s = FinBERTScorer(device="mock")
        r = repr(s)
        assert "FinBERTScorer" in r
        assert "mock" in r
        assert "loaded=True" in r

    def test_unloaded_repr(self):
        s = FinBERTScorer()
        r = repr(s)
        assert "loaded=False" in r
        assert "unloaded" in r

    def test_custom_model_name_in_repr(self):
        s = FinBERTScorer(device="mock", model_name="custom/finbert-ish")
        assert "custom/finbert-ish" in repr(s)


# ----------------------------------------------------------------------------
# _mock_score 工具函数直测
# ----------------------------------------------------------------------------


class TestMockScoreFunction:
    def test_positive_shape(self):
        out = _mock_score("beats earnings estimates")
        assert out["positive"] == 0.70
        assert out["confidence"] == 0.70

    def test_negative_shape(self):
        out = _mock_score("company reports massive loss and layoffs")
        assert out["negative"] == 0.70
        assert out["confidence"] == 0.70

    def test_neutral_shape(self):
        out = _mock_score("board approves routine share buyback")
        assert out["neutral"] == 0.60
        assert out["confidence"] == 0.60


# ----------------------------------------------------------------------------
# 真实 FinBERT 加载 (@pytest.mark.perf, 默认 skip)
# ----------------------------------------------------------------------------


@pytest.mark.perf
@_real_only
class TestFinBERTRealModel:
    """需要 RUN_PERF_TESTS=1 + 网络下载 ~440MB ProsusAI/finbert。"""

    def test_real_accuracy_on_golden(self, finbert_golden):
        """真实模型在 finbert_golden 10 条上 argmax 准确率 >= 8/10。"""
        reset_default_scorer()
        scorer = FinBERTScorer()
        texts = [row["text"] for row in finbert_golden]
        out = scorer.score_texts(texts, batch_size=8)
        correct = 0
        for scored, truth in zip(out, finbert_golden):
            pred = max(("positive", "negative", "neutral"), key=lambda k: scored[k])
            if pred == truth["label"]:
                correct += 1
        assert correct >= 8, f"accuracy {correct}/10 < 8/10"

    def test_batch_consistency(self, finbert_golden):
        """batch=1 / batch=4 / batch=32 对同输入结果一致 (tol 1e-4)。"""
        reset_default_scorer()
        scorer = FinBERTScorer()
        texts = [row["text"] for row in finbert_golden]
        out_b1 = scorer.score_texts(texts, batch_size=1)
        out_b4 = scorer.score_texts(texts, batch_size=4)
        out_b32 = scorer.score_texts(texts, batch_size=32)
        for a, b, c in zip(out_b1, out_b4, out_b32):
            for k in ("positive", "negative", "neutral"):
                assert a[k] == pytest.approx(b[k], abs=1e-4)
                assert b[k] == pytest.approx(c[k], abs=1e-4)

    def test_real_softmax_invariant(self):
        reset_default_scorer()
        scorer = FinBERTScorer()
        out = scorer.score_texts(["company beats earnings"])
        row = out[0]
        total = row["positive"] + row["negative"] + row["neutral"]
        assert total == pytest.approx(1.0, abs=1e-4)
        assert row["confidence"] == pytest.approx(
            max(row["positive"], row["negative"], row["neutral"]), abs=1e-6
        )

    def test_long_text_truncation_no_crash(self):
        """远超 128 token 的文本,tokenizer truncation=True 不报错。"""
        reset_default_scorer()
        scorer = FinBERTScorer()
        long_text = "earnings beat " * 200  # 远超 128 token
        out = scorer.score_texts([long_text])
        assert len(out) == 1
        row = out[0]
        assert 0.0 <= row["positive"] <= 1.0


@pytest.mark.perf
@_real_only
class TestFinBERTPerfGate:
    """PRD §3.3 硬约束: batch=32, max_length=128 推理 <100ms GPU / <300ms CPU。

    采样 3 warm-up + 10 中位数,降低冷启动/GC 噪声。
    """

    @staticmethod
    def _percentile(data: list[float], p: float) -> float:
        data_sorted = sorted(data)
        k = max(0, min(len(data_sorted) - 1, int(len(data_sorted) * p)))
        return data_sorted[k]

    def test_batch32_latency_gate(self):
        import time

        reset_default_scorer()
        scorer = FinBERTScorer()
        texts = [
            "earnings beat estimates by a wide margin; guidance raised." for _ in range(32)
        ]
        # warm-up
        for _ in range(3):
            scorer.score_texts(texts, batch_size=32)
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            scorer.score_texts(texts, batch_size=32)
            samples.append((time.perf_counter() - t0) * 1000.0)
        median_ms = self._percentile(samples, 0.5)

        dev = scorer.device
        tolerance_ms = 100.0 if dev == "cuda" else 300.0
        assert median_ms < tolerance_ms, (
            f"FinBERT batch=32 median {median_ms:.1f}ms exceeds "
            f"{tolerance_ms:.0f}ms on device={dev}"
        )
