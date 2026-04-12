# tests/test_llm_routing.py
"""LLM Routing 模块测试。
# PYTHONPATH=src .venv/bin/python -m pytest tests/test_llm_routing.py -v

测试对象：
  - LLMRouter（路由 + 降级链 + JSON 解析 + facade with tracker）
  - CostTracker（成本 + 缓存 + 线程安全 + 时区 + per-task budget + TTL）
  - _default_completion（litellm 适配路径，mock litellm）

全部使用 mock completion function 或 mock litellm，不调真实 LLM API。
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from stockbee.llm_routing.cost_tracker import CostTracker
from stockbee.llm_routing.router import (
    LLMParseError,
    LLMResponse,
    LLMRouter,
)
from stockbee.llm_routing.task_config import (
    DEFAULT_TASK_CONFIGS,
    TOTAL_MONTHLY_BUDGET,
    TaskConfig,
    TaskType,
)


# =========================================================================
# Helpers
# =========================================================================

def make_mock_fn(content: str = '{"relevant": true}', cost: float = 0.001):
    """Mock completion function returning a fixed payload. Accepts **kwargs
    so callers can add params (timeout, etc.) without breaking the mock."""
    def fn(model, messages, max_tokens, temperature, **kwargs):
        return {
            "content": content,
            "input_tokens": 50,
            "output_tokens": 10,
            "cost": cost,
        }
    return fn


def make_failing_fn(fail_models: set[str], fallback_content: str = '{"ok": true}'):
    """Mock that raises ConnectionError for models matching any substring in
    ``fail_models``, otherwise returns ``fallback_content``."""
    def fn(model, messages, max_tokens, temperature, **kwargs):
        if any(fm in model for fm in fail_models):
            raise ConnectionError(f"{model} unavailable")
        return {
            "content": fallback_content,
            "input_tokens": 30,
            "output_tokens": 5,
            "cost": 0.0005,
        }
    return fn


def make_conditional_fn(responses: dict[str, str], cost: float = 0.001):
    """Mock that returns a different content per model substring match.
    Useful for "primary returns dirty JSON, fallback returns clean JSON"."""
    def fn(model, messages, max_tokens, temperature, **kwargs):
        for key, content in responses.items():
            if key in model:
                return {
                    "content": content,
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "cost": cost,
                }
        raise ConnectionError(f"no mock response for {model}")
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
            assert cfg.timeout_seconds > 0

    def test_g1_uses_nano(self):
        cfg = DEFAULT_TASK_CONFIGS[TaskType.G1_FILTER]
        assert "nano" in cfg.model
        assert cfg.output_format == "json"

    def test_macro_uses_flagship(self):
        """MACRO_ANALYSIS runs weekly, so it gets the flagship Anthropic model."""
        cfg = DEFAULT_TASK_CONFIGS[TaskType.MACRO_ANALYSIS]
        assert "claude" in cfg.model
        assert cfg.output_format == "text"

    def test_model_ids_are_not_stale_snapshots(self):
        """Regression: PR #17 v1 shipped `claude-*-20250514` snapshot IDs
        that are already obsolete. Lock in that we use at least the 4.5/4.6
        family going forward."""
        for cfg in DEFAULT_TASK_CONFIGS.values():
            if "anthropic" in cfg.model:
                # Must use the 4.5 / 4.6 family, not 2025-05 snapshots.
                assert "20250514" not in cfg.model, \
                    f"{cfg.task_type} still uses obsolete 2025-05 snapshot: {cfg.model}"
                assert any(ver in cfg.model for ver in ("4-6", "4-5")), \
                    f"{cfg.task_type} not on Claude 4.5/4.6: {cfg.model}"


# =========================================================================
# LLMRouter — happy path + fallback on failure
# =========================================================================

class TestLLMRouterBasic:

    def test_basic_route(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn('{"relevant": true}'))

        resp = router.route(TaskType.G1_FILTER, "test prompt")
        assert resp.parsed == {"relevant": True}
        assert resp.model_used == "openai/gpt-4.1-nano"
        assert resp.from_fallback is False
        assert resp.cost == 0.001

    def test_fallback_on_primary_connection_error(self):
        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(make_failing_fn({"nano"}, '{"relevant": false}'))

        resp = router.route(TaskType.G1_FILTER, "test")
        assert resp.from_fallback is True
        assert resp.parsed == {"relevant": False}
        # Fallback path: cost reflects the *fallback* call, not primary.
        assert resp.cost == 0.0005

    def test_all_models_fail_raises(self):
        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(make_failing_fn({"nano", "haiku", "gpt"}))

        with pytest.raises(RuntimeError, match="LLM routing failed"):
            router.route(TaskType.G1_FILTER, "test")

    def test_text_output_format(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn("The economy is expanding."))

        resp = router.route(TaskType.MACRO_ANALYSIS, "Analyze macro")
        assert resp.content == "The economy is expanding."
        assert resp.parsed is None  # text format, no JSON parsing

    def test_system_prompt_passed(self):
        received = {}

        def capture_fn(model, messages, max_tokens, temperature, **kwargs):
            received["messages"] = messages
            received["timeout"] = kwargs.get("timeout")
            return {"content": '{"ok": true}', "input_tokens": 10, "output_tokens": 5, "cost": 0}

        router = LLMRouter()
        router.set_completion_fn(capture_fn)
        router.route(TaskType.G1_FILTER, "user msg", system_prompt="be brief")

        assert len(received["messages"]) == 2
        assert received["messages"][0] == {"role": "system", "content": "be brief"}
        assert received["messages"][1] == {"role": "user", "content": "user msg"}

    def test_timeout_is_passed_to_completion_fn(self):
        """TaskConfig.timeout_seconds must reach the downstream call so that
        a hung model doesn't block the fallback chain."""
        received = {}

        def capture_fn(model, messages, max_tokens, temperature, **kwargs):
            received["timeout"] = kwargs.get("timeout")
            return {"content": '{"ok": true}', "input_tokens": 10, "output_tokens": 5, "cost": 0}

        router = LLMRouter()
        router.set_completion_fn(capture_fn)
        router.route(TaskType.G1_FILTER, "test")
        assert received["timeout"] == DEFAULT_TASK_CONFIGS[TaskType.G1_FILTER].timeout_seconds

    def test_list_task_types(self):
        router = LLMRouter()
        types = router.list_task_types()
        assert len(types) == 5
        assert all("task_type" in t for t in types)

    def test_get_config(self):
        router = LLMRouter()
        cfg = router.get_config(TaskType.G3_ANALYSIS)
        assert cfg.max_tokens == 2000
        assert cfg.timeout_seconds > 0

    def test_unknown_task_type_raises(self):
        router = LLMRouter(
            task_configs={TaskType.G2_CLASSIFY: DEFAULT_TASK_CONFIGS[TaskType.G2_CLASSIFY]}
        )
        router.set_completion_fn(make_mock_fn())
        with pytest.raises(ValueError, match="Unknown task type"):
            router.route(TaskType.G1_FILTER, "test")  # G1 not in configs


# =========================================================================
# LLMRouter — JSON / schema failure MUST trigger fallback
# =========================================================================

class TestLLMRouterJsonFailureTriggersFallback:
    """These tests lock in the behaviour of P0 #1: a successful call that
    returns dirty JSON (or misses required schema keys) on a json-format task
    must move on to the fallback model instead of returning parsed=None."""

    def test_dirty_json_on_primary_triggers_fallback(self):
        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(make_conditional_fn({
            "nano": "this is not json at all",
            "haiku": '{"relevant": true}',
        }))

        resp = router.route(TaskType.G1_FILTER, "test")
        assert resp.from_fallback is True
        assert resp.parsed == {"relevant": True}

    def test_dirty_json_on_both_raises(self):
        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(make_conditional_fn({
            "nano": "garbage",
            "haiku": "also garbage",
        }))
        with pytest.raises(RuntimeError, match="LLM routing failed"):
            router.route(TaskType.G1_FILTER, "test")

    def test_missing_required_schema_field_triggers_fallback(self):
        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(make_conditional_fn({
            "nano": '{"foo": "bar"}',                 # missing `relevant`
            "haiku": '{"relevant": true, "foo": "bar"}',
        }))
        schema = {"required": ["relevant"]}
        resp = router.route(TaskType.G1_FILTER, "test", output_schema=schema)
        assert resp.from_fallback is True
        assert resp.parsed == {"relevant": True, "foo": "bar"}

    def test_missing_schema_on_both_raises(self):
        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(make_conditional_fn({
            "nano": '{"foo": "bar"}',
            "haiku": '{"baz": "qux"}',
        }))
        schema = {"required": ["relevant"]}
        with pytest.raises(RuntimeError, match="LLM routing failed"):
            router.route(TaskType.G1_FILTER, "test", output_schema=schema)

    def test_valid_schema_passes(self):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn('{"relevant": true}'))
        schema = {"required": ["relevant"]}
        resp = router.route(TaskType.G1_FILTER, "test", output_schema=schema)
        assert resp.from_fallback is False
        assert resp.parsed == {"relevant": True}


# =========================================================================
# LLMRouter — robust markdown JSON extraction
# =========================================================================

class TestMarkdownJsonExtraction:

    @pytest.mark.parametrize("content,expected", [
        ('{"ok": true}', {"ok": True}),
        ('```json\n{"ok": true}\n```', {"ok": True}),
        ('```\n{"ok": true}\n```', {"ok": True}),
        ("Here's the result:\n```json\n{\"ok\": true}\n```\nHope that helps!", {"ok": True}),
        ("Prefix text {\"ok\": true} suffix text", {"ok": True}),
        ('```JSON\n{"ok": true}\n```', {"ok": True}),
        ('  \n```json\n{"ok": true}\n```\n  ', {"ok": True}),
    ])
    def test_extraction(self, content, expected):
        router = LLMRouter()
        router.set_completion_fn(make_mock_fn(content))
        resp = router.route(TaskType.G1_FILTER, "test")
        assert resp.parsed == expected

    def test_pure_garbage_still_fails_over_to_fallback(self):
        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(make_conditional_fn({
            "nano": "not any form of json",
            "haiku": '{"ok": true}',
        }))
        resp = router.route(TaskType.G1_FILTER, "test")
        assert resp.from_fallback is True


# =========================================================================
# LLMRouter — retry / exception handling
# =========================================================================

class TestRouterRetry:

    def test_max_attempts_semantics(self):
        call_count = 0

        def counting_fn(model, messages, max_tokens, temperature, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        router = LLMRouter(max_attempts=3, task_configs={
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
        assert call_count == 3  # max_attempts is literally "attempts"

    def test_max_retries_alias_still_works(self):
        """Backwards compatibility for old `max_retries=` kwarg."""
        call_count = 0

        def counting_fn(model, messages, max_tokens, temperature, **kwargs):
            nonlocal call_count
            call_count += 1
            raise ConnectionError("fail")

        router = LLMRouter(max_retries=2, task_configs={
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
        assert call_count == 2

    def test_bug_exceptions_are_not_swallowed(self):
        """AttributeError / KeyError / TypeError indicate bugs in our code
        (not a dead model). Router must re-raise, not silently fall back."""
        def buggy_fn(model, messages, max_tokens, temperature, **kwargs):
            # simulate our own code path raising a real bug
            raise AttributeError("'NoneType' object has no attribute 'choices'")

        router = LLMRouter(max_attempts=1)
        router.set_completion_fn(buggy_fn)
        with pytest.raises(AttributeError):
            router.route(TaskType.G1_FILTER, "test")

    def test_timeout_error_triggers_retry(self):
        call_count = 0

        def flakey_fn(model, messages, max_tokens, temperature, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError("slow")
            return {"content": '{"ok": true}', "input_tokens": 1, "output_tokens": 1, "cost": 0}

        router = LLMRouter(max_attempts=5)
        router.set_completion_fn(flakey_fn)
        resp = router.route(TaskType.G1_FILTER, "test")
        assert resp.parsed == {"ok": True}
        assert call_count == 3


# =========================================================================
# LLMRouter — _default_completion (litellm path)
# =========================================================================

class TestDefaultCompletionLiteLLMPath:

    def test_default_completion_parses_openai_style_usage(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"ok": true}'
        mock_response.usage.prompt_tokens = 12
        mock_response.usage.completion_tokens = 3

        with patch.dict("sys.modules", {"litellm": MagicMock()}) as modules:
            litellm_mock = modules["litellm"]
            litellm_mock.completion.return_value = mock_response
            litellm_mock.completion_cost.return_value = 0.0042

            result = LLMRouter._default_completion(
                model="openai/gpt-4.1-nano",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0.0,
                timeout=30.0,
            )

        assert result["content"] == '{"ok": true}'
        assert result["input_tokens"] == 12
        assert result["output_tokens"] == 3
        assert result["cost"] == 0.0042

    def test_default_completion_handles_anthropic_style_usage(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        # Some Anthropic responses expose input_tokens/output_tokens instead
        mock_response.usage = MagicMock(spec=["input_tokens", "output_tokens"])
        mock_response.usage.input_tokens = 7
        mock_response.usage.output_tokens = 2

        with patch.dict("sys.modules", {"litellm": MagicMock()}) as modules:
            litellm_mock = modules["litellm"]
            litellm_mock.completion.return_value = mock_response
            litellm_mock.completion_cost.return_value = 0.001

            result = LLMRouter._default_completion(
                model="anthropic/claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0.0,
                timeout=30.0,
            )

        assert result["input_tokens"] == 7
        assert result["output_tokens"] == 2

    def test_default_completion_tolerates_completion_cost_failure(self):
        """Unknown model → litellm.completion_cost raises. Must not nuke the
        whole call — log and record cost=0."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1

        with patch.dict("sys.modules", {"litellm": MagicMock()}) as modules:
            litellm_mock = modules["litellm"]
            litellm_mock.completion.return_value = mock_response
            litellm_mock.completion_cost.side_effect = Exception("unknown model")

            result = LLMRouter._default_completion(
                model="made/up-model",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=100,
                temperature=0.0,
                timeout=30.0,
            )

        assert result["content"] == "hello"
        assert result["cost"] == 0.0  # fallback, not crash

    def test_default_completion_handles_missing_usage(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "x"
        mock_response.usage = None

        with patch.dict("sys.modules", {"litellm": MagicMock()}) as modules:
            litellm_mock = modules["litellm"]
            litellm_mock.completion.return_value = mock_response
            litellm_mock.completion_cost.return_value = 0.0

            result = LLMRouter._default_completion(
                model="any", messages=[], max_tokens=1, temperature=0.0, timeout=30.0,
            )

        assert result["input_tokens"] == 0
        assert result["output_tokens"] == 0


# =========================================================================
# CostTracker — fixtures
# =========================================================================

@pytest.fixture
def tracker(tmp_path) -> CostTracker:
    t = CostTracker(db_path=str(tmp_path / "costs.db"), monthly_budget=15.0)
    t.initialize()
    return t


# =========================================================================
# CostTracker — basic record / report
# =========================================================================

class TestCostTrackerBasic:

    def test_record_and_report(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "prompt1", '{"r":true}', 50, 10, 0.001)
        tracker.record_call(TaskType.G2_CLASSIFY, "haiku", "prompt2", '{"s":"pos"}', 200, 50, 0.01)

        report = tracker.monthly_report()
        assert report["total_calls"] == 2
        assert abs(report["total_cost"] - 0.011) < 1e-9
        assert report["over_budget"] is False
        assert "g1_filter" in report["breakdown"]
        assert "g2_classify" in report["breakdown"]

    def test_monthly_spent_filters_by_month(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 5.0)
        assert tracker.monthly_spent() == 5.0
        assert tracker.monthly_spent("2025-01") == 0.0

    def test_over_budget_total(self, tracker):
        assert not tracker.is_over_budget()
        for i in range(20):
            tracker.record_call(TaskType.G3_ANALYSIS, "sonnet", f"p{i}", "resp", 1000, 500, 1.0)
        assert tracker.is_over_budget()
        assert tracker.monthly_spent() == 20.0

    def test_empty_report(self, tracker):
        report = tracker.monthly_report()
        assert report["total_calls"] == 0
        assert report["total_cost"] == 0
        assert report["remaining"] == 15.0

    def test_cache_size(self, tracker):
        assert tracker.cache_size() == 0
        tracker.record_call(TaskType.G1_FILTER, "nano", "p1", "r1", 50, 10, 0.001)
        tracker.record_call(TaskType.G2_CLASSIFY, "haiku", "p2", "r2", 100, 20, 0.005)
        assert tracker.cache_size() == 2

    def test_shutdown_and_reinit(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 0.5)
        tracker.shutdown()

        tracker.initialize()
        assert tracker.monthly_spent() == 0.5
        # Must use the same model as the one in record_call — the cache key
        # now includes model so passing the wrong one is a miss.
        assert tracker.get_cached(TaskType.G1_FILTER, "p", model="nano") == "r"


# =========================================================================
# CostTracker — cache key includes all context (P1 #4)
# =========================================================================

class TestCostTrackerCacheKey:

    def test_cache_hit_with_same_context(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "same prompt", "cached response", 50, 10, 0.001)
        cached = tracker.get_cached(TaskType.G1_FILTER, "same prompt", model="nano")
        assert cached == "cached response"

    def test_cache_miss_when_system_prompt_changes(self, tracker):
        tracker.record_call(
            TaskType.G1_FILTER, "nano", "prompt", "r", 50, 10, 0.001,
            system_prompt="you are concise",
        )
        assert tracker.get_cached(
            TaskType.G1_FILTER, "prompt", system_prompt="you are concise", model="nano",
        ) == "r"
        # Different system prompt → cache miss
        assert tracker.get_cached(
            TaskType.G1_FILTER, "prompt", system_prompt="you are verbose", model="nano",
        ) is None
        # Missing system prompt → also cache miss
        assert tracker.get_cached(TaskType.G1_FILTER, "prompt", model="nano") is None

    def test_cache_miss_when_output_schema_changes(self, tracker):
        tracker.record_call(
            TaskType.G1_FILTER, "nano", "prompt", "r", 50, 10, 0.001,
            output_schema={"required": ["a"]},
        )
        # Same schema → hit
        assert tracker.get_cached(
            TaskType.G1_FILTER, "prompt", output_schema={"required": ["a"]}, model="nano",
        ) == "r"
        # Different schema → miss
        assert tracker.get_cached(
            TaskType.G1_FILTER, "prompt", output_schema={"required": ["b"]}, model="nano",
        ) is None

    def test_cache_hits_regardless_of_schema_key_order(self, tracker):
        """Schema canonicalization: {'a':1,'b':2} and {'b':2,'a':1} must
        produce the same cache key."""
        schema_a = {"required": ["x", "y"], "type": "object"}
        schema_b = {"type": "object", "required": ["x", "y"]}
        tracker.record_call(
            TaskType.G1_FILTER, "nano", "prompt", "r", 50, 10, 0.001,
            output_schema=schema_a,
        )
        assert tracker.get_cached(
            TaskType.G1_FILTER, "prompt", output_schema=schema_b, model="nano",
        ) == "r"

    def test_cache_miss_when_model_changes(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "prompt", "r", 50, 10, 0.001)
        assert tracker.get_cached(TaskType.G1_FILTER, "prompt", model="nano") == "r"
        # Same prompt, different model → miss
        assert tracker.get_cached(TaskType.G1_FILTER, "prompt", model="haiku") is None

    def test_cache_different_task_type(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "prompt", "response", 50, 10, 0.001)
        # Same prompt+model, different task type → miss
        assert tracker.get_cached(TaskType.G2_CLASSIFY, "prompt", model="nano") is None

    def test_cache_miss(self, tracker):
        assert tracker.get_cached(TaskType.G1_FILTER, "never seen", model="nano") is None

    def test_cache_update_on_same_prompt(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "same", "old", 50, 10, 0.001)
        tracker.record_call(TaskType.G1_FILTER, "nano", "same", "new", 50, 10, 0.001)

        cached = tracker.get_cached(TaskType.G1_FILTER, "same", model="nano")
        assert cached == "new"
        assert tracker.cache_size() == 1  # same hash, replaced


# =========================================================================
# CostTracker — TTL (P2 #12)
# =========================================================================

class TestCostTrackerCacheTtl:

    def test_expired_entries_are_not_returned(self, tracker, monkeypatch):
        from stockbee.llm_routing import cost_tracker as ct_mod

        # Record at T=0.
        t0 = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0)
        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 0.001)

        # 23h later → still fresh
        from datetime import timedelta
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0 + timedelta(hours=23))
        assert tracker.get_cached(TaskType.G1_FILTER, "p", model="nano") == "r"

        # 25h later → expired (default TTL = 24h)
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0 + timedelta(hours=25))
        assert tracker.get_cached(TaskType.G1_FILTER, "p", model="nano") is None

    def test_custom_max_age_overrides_default(self, tracker, monkeypatch):
        from stockbee.llm_routing import cost_tracker as ct_mod

        t0 = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0)
        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 0.001)

        from datetime import timedelta
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0 + timedelta(minutes=30))
        # Tight TTL → even a 30-minute-old entry is stale
        assert tracker.get_cached(TaskType.G1_FILTER, "p", model="nano", max_age_hours=0.25) is None
        # Generous TTL → fresh
        assert tracker.get_cached(TaskType.G1_FILTER, "p", model="nano", max_age_hours=24) == "r"

    def test_clear_expired_cache(self, tracker, monkeypatch):
        from stockbee.llm_routing import cost_tracker as ct_mod

        t0 = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0)
        tracker.record_call(TaskType.G1_FILTER, "nano", "p1", "r1", 50, 10, 0.001)

        from datetime import timedelta
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0 + timedelta(hours=48))
        tracker.record_call(TaskType.G1_FILTER, "nano", "p2", "r2", 50, 10, 0.001)

        # Clear with 24h TTL removes the first one.
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: t0 + timedelta(hours=49))
        deleted = tracker.clear_expired_cache(max_age_hours=24)
        assert deleted == 1
        assert tracker.cache_size() == 1


# =========================================================================
# CostTracker — thread safety (P1 #3)
# =========================================================================

class TestCostTrackerThreadSafety:

    def test_concurrent_record_call_does_not_raise(self, tracker):
        """Regression: PR #17 v1 used the default ``check_same_thread=True``
        which raises sqlite3.ProgrammingError on any cross-thread use. The
        Tech Design §3.2 news pipeline runs G1/G2 concurrently, so this
        must work."""
        errors: list[BaseException] = []

        def worker(idx: int):
            try:
                for j in range(20):
                    tracker.record_call(
                        TaskType.G1_FILTER, "nano",
                        f"prompt-{idx}-{j}", "resp",
                        10, 5, 0.0001,
                    )
            except BaseException as exc:  # noqa: BLE001 — test wants to surface anything
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"threaded record_call raised: {errors}"
        # 4 threads × 20 calls
        report = tracker.monthly_report()
        assert report["total_calls"] == 80

    def test_concurrent_reads_and_writes(self, tracker):
        tracker.record_call(TaskType.G1_FILTER, "nano", "initial", "ok", 10, 5, 0.0001)
        errors: list[BaseException] = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    tracker.monthly_spent()
                    tracker.get_cached(TaskType.G1_FILTER, "initial", model="nano")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        def writer():
            try:
                for i in range(50):
                    tracker.record_call(
                        TaskType.G1_FILTER, "nano", f"p{i}", "r", 10, 5, 0.0001,
                    )
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        readers = [threading.Thread(target=reader) for _ in range(2)]
        writers = [threading.Thread(target=writer) for _ in range(2)]
        for t in readers + writers:
            t.start()
        for t in writers:
            t.join()
        stop.set()
        for t in readers:
            t.join()

        assert errors == [], f"concurrent rw raised: {errors}"


# =========================================================================
# CostTracker — UTC-consistent month boundary (P1 #5)
# =========================================================================

class TestCostTrackerUtcMonthBoundary:

    def test_end_of_month_utc_call_lands_in_utc_month(self, tracker, monkeypatch):
        """PR #17 v1 recorded in UTC but queried via local ``date.today()``.
        On US/Eastern 2026-04-30 20:30 that converts to UTC 2026-05-01 00:30.
        The $ spent must appear in the UTC month report (2026-05), and the
        default report with no month argument must use UTC too."""
        from stockbee.llm_routing import cost_tracker as ct_mod

        # Simulate: real wall clock is UTC 2026-05-01 00:30:00
        fake_now = datetime(2026, 5, 1, 0, 30, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: fake_now)

        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 2.50)

        # Default month — must be UTC "2026-05", not local "2026-04".
        report = tracker.monthly_report()
        assert report["month"] == "2026-05"
        assert report["total_cost"] == 2.50
        # Explicitly querying "2026-04" must find nothing.
        assert tracker.monthly_spent("2026-04") == 0.0
        # Explicitly querying "2026-05" must find the $.
        assert tracker.monthly_spent("2026-05") == 2.50

    def test_monthly_spent_default_uses_utc(self, tracker, monkeypatch):
        from stockbee.llm_routing import cost_tracker as ct_mod

        fake_now = datetime(2026, 4, 15, 10, 0, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(ct_mod, "_utcnow", lambda: fake_now)
        tracker.record_call(TaskType.G1_FILTER, "nano", "p", "r", 50, 10, 1.5)
        assert tracker.monthly_spent() == 1.5


# =========================================================================
# CostTracker — per-task budget (P1 #9)
# =========================================================================

class TestCostTrackerPerTaskBudget:

    def test_per_task_budget_enforced(self, tracker):
        # G1_FILTER has monthly_budget=$1 in DEFAULT_TASK_CONFIGS.
        # Spend $0.90 under G1 — not over yet.
        tracker.record_call(TaskType.G1_FILTER, "nano", "p1", "r", 50, 10, 0.90)
        assert not tracker.is_over_budget(task_type=TaskType.G1_FILTER)
        assert not tracker.is_over_budget()  # total is $0.90 < $15

        # Push G1 over its $1 cap.
        tracker.record_call(TaskType.G1_FILTER, "nano", "p2", "r", 50, 10, 0.15)
        assert tracker.is_over_budget(task_type=TaskType.G1_FILTER)
        # But other tasks (and total) are still fine.
        assert not tracker.is_over_budget(task_type=TaskType.G2_CLASSIFY)
        assert not tracker.is_over_budget()

    def test_task_budget_does_not_block_other_tasks(self, tracker):
        """A G3 overrun must not affect G1 availability."""
        # G3 cap is $5.
        for i in range(6):
            tracker.record_call(TaskType.G3_ANALYSIS, "sonnet", f"p{i}", "r", 500, 200, 1.0)
        assert tracker.is_over_budget(task_type=TaskType.G3_ANALYSIS)
        assert not tracker.is_over_budget(task_type=TaskType.G1_FILTER)

    def test_report_shows_per_task_over_budget_flag(self, tracker):
        for i in range(2):
            tracker.record_call(TaskType.G1_FILTER, "nano", f"p{i}", "r", 50, 10, 0.6)
        report = tracker.monthly_report()
        assert report["breakdown"]["g1_filter"]["task_over_budget"] is True


# =========================================================================
# LLMRouter + CostTracker facade (P1 #7)
# =========================================================================

class TestRouterTrackerFacade:

    def test_router_auto_records_calls(self, tmp_path):
        tracker = CostTracker(db_path=str(tmp_path / "c.db"), monthly_budget=15.0)
        tracker.initialize()
        router = LLMRouter(tracker=tracker)
        router.set_completion_fn(make_mock_fn('{"ok": true}', cost=0.002))

        resp = router.route(TaskType.G1_FILTER, "hi")
        assert resp.parsed == {"ok": True}

        # CostTracker must have seen this call without caller doing anything.
        assert tracker.monthly_spent() == 0.002
        report = tracker.monthly_report()
        assert report["breakdown"]["g1_filter"]["calls"] == 1

    def test_router_auto_serves_cache_hit(self, tmp_path):
        tracker = CostTracker(db_path=str(tmp_path / "c.db"), monthly_budget=15.0)
        tracker.initialize()
        router = LLMRouter(tracker=tracker)

        calls = 0

        def counting_fn(model, messages, max_tokens, temperature, **kwargs):
            nonlocal calls
            calls += 1
            return {"content": '{"ok": true}', "input_tokens": 1, "output_tokens": 1, "cost": 0.001}

        router.set_completion_fn(counting_fn)

        resp1 = router.route(TaskType.G1_FILTER, "same prompt")
        resp2 = router.route(TaskType.G1_FILTER, "same prompt")

        # Only one real call, second was served from cache.
        assert calls == 1
        assert resp1.from_cache is False
        assert resp2.from_cache is True
        assert resp2.parsed == {"ok": True}

    def test_router_refuses_when_task_over_budget(self, tmp_path):
        tracker = CostTracker(db_path=str(tmp_path / "c.db"), monthly_budget=15.0)
        tracker.initialize()
        # Push G1 over its $1 budget ahead of time.
        tracker.record_call(TaskType.G1_FILTER, "nano", "seed", "r", 50, 10, 1.5)

        router = LLMRouter(tracker=tracker)
        router.set_completion_fn(make_mock_fn('{"ok": true}'))
        with pytest.raises(RuntimeError, match="Over budget for task g1_filter"):
            router.route(TaskType.G1_FILTER, "new prompt")

    def test_router_refuses_when_total_over_budget(self, tmp_path):
        tracker = CostTracker(db_path=str(tmp_path / "c.db"), monthly_budget=1.0)
        tracker.initialize()
        tracker.record_call(TaskType.G3_ANALYSIS, "sonnet", "seed", "r", 100, 50, 2.0)

        router = LLMRouter(tracker=tracker)
        router.set_completion_fn(make_mock_fn("plain text"))
        with pytest.raises(RuntimeError, match="Over total monthly budget"):
            router.route(TaskType.MACRO_ANALYSIS, "analyze")

    def test_cache_respects_system_prompt_through_facade(self, tmp_path):
        """End-to-end: router auto-records with system_prompt → second call
        with different system_prompt must miss cache."""
        tracker = CostTracker(db_path=str(tmp_path / "c.db"), monthly_budget=15.0)
        tracker.initialize()
        router = LLMRouter(tracker=tracker)
        calls = 0

        def counting_fn(model, messages, max_tokens, temperature, **kwargs):
            nonlocal calls
            calls += 1
            return {"content": '{"ok": true}', "input_tokens": 1, "output_tokens": 1, "cost": 0.001}

        router.set_completion_fn(counting_fn)

        router.route(TaskType.G1_FILTER, "p", system_prompt="be brief")
        router.route(TaskType.G1_FILTER, "p", system_prompt="be verbose")
        assert calls == 2  # different system prompts → no cache reuse

        router.route(TaskType.G1_FILTER, "p", system_prompt="be brief")  # repeat first
        assert calls == 2  # cache hit for the original combo
