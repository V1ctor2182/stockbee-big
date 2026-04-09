# tests/test_llm_routing.py
"""LLM Routing 模块测试。
# PYTHONPATH=src .venv/bin/python -m pytest tests/test_llm_routing.py -v

测试对象：LLMRouter（路由 + 降级链 + JSON 解析）、CostTracker（成本 + 缓存）。
全部使用 mock completion function，不调真实 LLM API。
"""

import json

import pytest

from stockbee.llm_routing.router import LLMRouter, LLMResponse
from stockbee.llm_routing.task_config import (
    TaskType, TaskConfig, DEFAULT_TASK_CONFIGS, TOTAL_MONTHLY_BUDGET,
)
from stockbee.llm_routing.cost_tracker import CostTracker


# =========================================================================
# Helpers
# =========================================================================

def make_mock_fn(content: str = '{"relevant": true}', cost: float = 0.001):
    """创建一个固定返回值的 mock completion function。"""
    def fn(model, messages, max_tokens, temperature):
        return {
            "content": content,
            "input_tokens": 50,
            "output_tokens": 10,
            "cost": cost,
        }
    return fn


def make_failing_fn(fail_models: set[str], fallback_content: str = '{"ok": true}'):
    """创建一个对指定模型失败的 mock。"""
    def fn(model, messages, max_tokens, temperature):
        if any(fm in model for fm in fail_models):
            raise ConnectionError(f"{model} unavailable")
        return {
            "content": fallback_content,
            "input_tokens": 30,
            "output_tokens": 5,
            "cost": 0.0005,
        }
    return fn


# =========================================================================
# TaskConfig Tests
# =========================================================================

class TestTaskConfig:

    def test_5_task_types(self):
        assert len(TaskType) == 5
        assert len(DEFAULT_TASK_CONFIGS) == 5

    def test_total_budget_15(self):
        assert TOTAL_MONTHLY_BUDGET == 15.0

    def test_all_configs_have_model(self):
        for tt, cfg in DEFAULT_TASK_CONFIGS.items():
            assert cfg.model, f"{tt} missing model"
            assert cfg.max_tokens > 0

    def test_g1_uses_nano(self):
        cfg = DEFAULT_TASK_CONFIGS[TaskType.G1_FILTER]
        assert "nano" in cfg.model
        assert cfg.output_format == "json"

    def test_macro_uses_sonnet(self):
        cfg = DEFAULT_TASK_CONFIGS[TaskType.MACRO_ANALYSIS]
        assert "sonnet" in cfg.model
        assert cfg.output_format == "text"


# =========================================================================
# LLMRouter Tests
# =========================================================================

class TestLLMRouter:

    def test_basic_route(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn('{"relevant": true}'))

        resp = router.route(TaskType.G1_FILTER, "test prompt")
        assert resp.parsed == {"relevant": True}
        assert resp.model_used == "openai/gpt-4.1-nano"
        assert resp.from_fallback is False
        assert resp.cost == 0.001

    def test_fallback_on_primary_failure(self):
        router = LLMRouter(max_retries=1)
        router.set_completion_fn(make_failing_fn({"nano"}, '{"relevant": false}'))

        resp = router.route(TaskType.G1_FILTER, "test")
        assert resp.from_fallback is True
        assert resp.model_used == "google/gemini-2.0-flash"
        assert resp.parsed == {"relevant": False}

    def test_all_models_fail_raises(self):
        router = LLMRouter(max_retries=1)
        router.set_completion_fn(make_failing_fn({"nano", "gemini", "flash"}))

        with pytest.raises(RuntimeError, match="LLM routing failed"):
            router.route(TaskType.G1_FILTER, "test")

    def test_text_output_format(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn("The economy is expanding."))

        resp = router.route(TaskType.MACRO_ANALYSIS, "Analyze macro")
        assert resp.content == "The economy is expanding."
        assert resp.parsed is None  # text format, no JSON parsing

    def test_json_with_code_block(self):
        router = LLMRouter()
        content = '```json\n{"sentiment": "positive", "urgency": 0.8}\n```'
        router.set_completion_fn(make_mock_fn(content))

        resp = router.route(TaskType.G2_CLASSIFY, "classify this")
        assert resp.parsed == {"sentiment": "positive", "urgency": 0.8}

    def test_invalid_json_returns_none_parsed(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn("not valid json at all"))

        resp = router.route(TaskType.G1_FILTER, "test")
        assert resp.parsed is None
        assert resp.content == "not valid json at all"

    def test_schema_validation_pass(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn('{"relevant": true}'))

        schema = {"required": ["relevant"]}
        resp = router.route(TaskType.G1_FILTER, "test", output_schema=schema)
        assert resp.parsed == {"relevant": True}

    def test_schema_validation_fail_still_returns(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn('{"foo": "bar"}'))

        schema = {"required": ["relevant"]}
        resp = router.route(TaskType.G1_FILTER, "test", output_schema=schema)
        # Still returns, just logs warning
        assert resp.parsed == {"foo": "bar"}

    def test_unknown_task_type_raises(self):
        router = LLMRouter(task_configs={TaskType.G2_CLASSIFY: DEFAULT_TASK_CONFIGS[TaskType.G2_CLASSIFY]})
        router.set_completion_fn(make_mock_fn())
        with pytest.raises(ValueError, match="Unknown task type"):
            router.route(TaskType.G1_FILTER, "test")  # G1 not in configs

    def test_system_prompt_passed(self):
        received = {}

        def capture_fn(model, messages, max_tokens, temperature):
            received["messages"] = messages
            return {"content": '{"ok": true}', "input_tokens": 10, "output_tokens": 5, "cost": 0}

        router = LLMRouter()
        router.set_completion_fn(capture_fn)
        router.route(TaskType.G1_FILTER, "user msg", system_prompt="be brief")

        assert len(received["messages"]) == 2
        assert received["messages"][0] == {"role": "system", "content": "be brief"}
        assert received["messages"][1] == {"role": "user", "content": "user msg"}

    def test_retry_count(self):
        call_count = 0

        def counting_fn(model, messages, max_tokens, temperature):
            nonlocal call_count
            call_count += 1
            raise Exception("fail")

        router = LLMRouter(max_retries=3, task_configs={
            TaskType.G1_FILTER: TaskConfig(
                task_type=TaskType.G1_FILTER,
                model="test-model",
                fallback_model=None,
                max_tokens=100,
            )
        })
        router.set_completion_fn(counting_fn)

        with pytest.raises(RuntimeError):
            router.route(TaskType.G1_FILTER, "test")

        assert call_count == 3  # 3 retries on primary, no fallback

    def test_list_task_types(self):
        router = LLMRouter()
        types = router.list_task_types()
        assert len(types) == 5
        assert all("task_type" in t for t in types)

    def test_get_config(self):
        router = LLMRouter()
        cfg = router.get_config(TaskType.G3_ANALYSIS)
        assert "gpt-4.1" in cfg.model
        assert cfg.max_tokens == 2000


# =========================================================================
# CostTracker Tests
# =========================================================================

class TestCostTracker:

    @pytest.fixture
    def tracker(self, tmp_path) -> CostTracker:
        t = CostTracker(db_path=str(tmp_path / "costs.db"), monthly_budget=15.0)
        t.initialize()
        return t

    def test_record_and_report(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "prompt1", '{"r":true}', 50, 10, 0.001)
        tracker.record_call(TaskType.G2_CLASSIFY, "sonnet", "prompt2", '{"s":"pos"}', 200, 50, 0.01)

        report = tracker.monthly_report()
        assert report["total_calls"] == 2
        assert report["total_cost"] == 0.011
        assert report["over_budget"] is False
        assert "g1_filter" in report["breakdown"]
        assert "g2_classify" in report["breakdown"]

    def test_cache_hit(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "same prompt", "cached response", 50, 10, 0.001)

        cached = tracker.get_cached(TaskType.G1_FILTER, "same prompt")
        assert cached == "cached response"

    def test_cache_miss(self, tracker):
        cached = tracker.get_cached(TaskType.G1_FILTER, "never seen")
        assert cached is None

    def test_cache_different_task_type(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "prompt", "response", 50, 10, 0.001)

        # Same prompt, different task type → cache miss
        cached = tracker.get_cached(TaskType.G2_CLASSIFY, "prompt")
        assert cached is None

    def test_over_budget(self, tracker):
        assert not tracker.is_over_budget()

        # Record expensive calls
        for i in range(20):
            tracker.record_call(TaskType.G3_ANALYSIS, "gpt4", f"prompt{i}", "resp", 1000, 500, 1.0)

        assert tracker.is_over_budget()
        assert tracker.monthly_spent() == 20.0

    def test_monthly_spent_filters_by_month(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 5.0)

        # Current month should show the cost
        assert tracker.monthly_spent() == 5.0

        # A different month should be 0
        assert tracker.monthly_spent("2025-01") == 0.0

    def test_cache_size(self, tracker):
        assert tracker.cache_size() == 0

        tracker.record_call(TaskType.G1_FILTER, "nano", "p1", "r1", 50, 10, 0.001)
        tracker.record_call(TaskType.G2_CLASSIFY, "sonnet", "p2", "r2", 100, 20, 0.005)

        assert tracker.cache_size() == 2

    def test_cache_update_on_same_prompt(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "same", "old response", 50, 10, 0.001)
        tracker.record_call(TaskType.G1_FILTER, "nano", "same", "new response", 50, 10, 0.001)

        cached = tracker.get_cached(TaskType.G1_FILTER, "same")
        assert cached == "new response"
        assert tracker.cache_size() == 1  # same hash, replaced

    def test_shutdown_and_reinit(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 0.5)
        tracker.shutdown()

        tracker.initialize()
        assert tracker.monthly_spent() == 0.5
        assert tracker.get_cached(TaskType.G1_FILTER, "p") == "r"

    def test_empty_report(self, tracker):
        report = tracker.monthly_report()
        assert report["total_calls"] == 0
        assert report["total_cost"] == 0
        assert report["remaining"] == 15.0
