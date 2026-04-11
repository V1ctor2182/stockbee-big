"""LLMRouter — 多模型路由器 + 降级链。

统一调用接口（via litellm），按 TaskType 选模型。
Router 是 task-agnostic 的 — 不管 G1/G2/G3 业务逻辑，
只提供 route(task_type, prompt) → response 接口。

可选集成 CostTracker：传入 tracker 后，route() 会自动
- 预检缓存
- 预检预算
- 成功后记录 call

依赖：pip install litellm（可选依赖）

来源：Tech Design §3.2
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .task_config import DEFAULT_TASK_CONFIGS, TaskConfig, TaskType

if TYPE_CHECKING:
    from .cost_tracker import CostTracker

logger = logging.getLogger(__name__)


# Exception classes that indicate transient provider-side failure and
# should trigger retry / fallback. Bugs in our own code (AttributeError,
# KeyError, TypeError…) must not be silently swallowed as "model down".
_RETRY_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
)


class LLMParseError(RuntimeError):
    """Raised when an LLM returned content but it fails JSON parse or schema
    validation for a json-format task. Caught internally by ``route()`` so
    the fallback chain continues instead of returning a broken response."""


@dataclass
class LLMResponse:
    """LLM 调用的统一响应。"""
    content: str                        # 原始文本响应
    parsed: dict[str, Any] | None       # JSON 解析后的结果（output_format=json 时）
    model_used: str                     # 实际使用的模型
    task_type: TaskType
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0                   # 本次调用成本（美元）
    from_fallback: bool = False         # 是否使用了降级模型
    from_cache: bool = False            # 是否来自缓存


# Regex extracts the first fenced block (```json ... ``` or plain ```...```)
# anywhere in the content, even when wrapped in prose like
# "Here's the result:\n```json\n{...}\n```\nHope that helps!".
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Fallback: greedy match for an outermost { ... } object.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMRouter:
    """多模型路由器。

    使用方式（不带 tracker）：
        router = LLMRouter()
        response = router.route(
            TaskType.G1_FILTER,
            "Is this news relevant to AAPL? Headline: ...",
        )
        print(response.parsed)  # {"relevant": true}

    使用方式（带 tracker，自动缓存 + 预算 + 记录）：
        tracker = CostTracker()
        tracker.initialize()
        router = LLMRouter(tracker=tracker)
        response = router.route(TaskType.G1_FILTER, "...")
    """

    def __init__(
        self,
        task_configs: dict[TaskType, TaskConfig] | None = None,
        max_attempts: int = 2,
        tracker: "CostTracker | None" = None,
        *,
        max_retries: int | None = None,  # backward-compat alias
    ) -> None:
        self._configs = task_configs or DEFAULT_TASK_CONFIGS
        # `max_retries` kept as a deprecated alias; if both are provided
        # the explicit `max_attempts` wins.
        if max_retries is not None and max_attempts == 2:
            max_attempts = max_retries
        self._max_attempts = max_attempts
        self._tracker = tracker
        self._completion_fn = self._default_completion

    # Backwards-compat: some old call sites may still read `max_retries`.
    @property
    def max_retries(self) -> int:
        return self._max_attempts

    def set_completion_fn(self, fn: Any) -> None:
        """替换底层调用函数（用于测试 mock）。"""
        self._completion_fn = fn

    def route(
        self,
        task_type: TaskType,
        prompt: str,
        system_prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """路由 LLM 调用。

        当 ``tracker`` 被注入时，流程是：
          1. 查缓存（带 system_prompt/schema/model 维度）→ 命中直接返回
          2. 查任务级预算 → 超预算抛 RuntimeError
          3. 调模型（含 fallback chain）
          4. 成功后 record_call + 更新缓存

        否则只做第 3 步。
        """
        config = self._configs.get(task_type)
        if config is None:
            raise ValueError(f"Unknown task type: {task_type}")

        # --- Step 1: cache lookup (only if tracker wired) ---
        if self._tracker is not None:
            cached_content = self._tracker.get_cached(
                task_type,
                prompt,
                system_prompt=system_prompt,
                output_schema=output_schema,
                model=config.model,
            )
            if cached_content is not None:
                parsed = self._try_parse_json(cached_content) if config.output_format == "json" else None
                return LLMResponse(
                    content=cached_content,
                    parsed=parsed,
                    model_used=config.model,
                    task_type=task_type,
                    from_cache=True,
                )

            # --- Step 2: budget gate (per-task first, then total) ---
            if self._tracker.is_over_budget(task_type=task_type):
                raise RuntimeError(
                    f"Over budget for task {task_type.value}: "
                    f"${self._tracker.monthly_spent(task_type=task_type):.4f} "
                    f"spent against ${config.monthly_budget:.2f} cap"
                )
            if self._tracker.is_over_budget():
                raise RuntimeError(
                    f"Over total monthly budget: "
                    f"${self._tracker.monthly_spent():.4f} spent"
                )

        # --- Step 3: try primary then fallback ---
        response = self._try_model_and_finalize(
            config.model, config, prompt, system_prompt, output_schema, from_fallback=False,
        )
        if response is None and config.fallback_model:
            logger.warning(
                "Primary model %s failed or returned invalid output for %s, falling back to %s",
                config.model, task_type.value, config.fallback_model,
            )
            response = self._try_model_and_finalize(
                config.fallback_model, config, prompt, system_prompt, output_schema, from_fallback=True,
            )

        if response is None:
            logger.error("All models failed for task %s", task_type.value)
            raise RuntimeError(
                f"LLM routing failed for {task_type.value}: "
                f"primary={config.model}, fallback={config.fallback_model}"
            )

        # --- Step 4: record call on the tracker ---
        if self._tracker is not None:
            self._tracker.record_call(
                task_type=task_type,
                model=response.model_used,
                prompt=prompt,
                response_content=response.content,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cost=response.cost,
                from_fallback=response.from_fallback,
                system_prompt=system_prompt,
                output_schema=output_schema,
            )

        return response

    def get_config(self, task_type: TaskType) -> TaskConfig:
        """获取某任务类型的配置。"""
        config = self._configs.get(task_type)
        if config is None:
            raise ValueError(f"Unknown task type: {task_type}")
        return config

    def list_task_types(self) -> list[dict[str, str]]:
        """列出所有注册的任务类型及其模型。"""
        return [
            {
                "task_type": config.task_type.value,
                "model": config.model,
                "fallback": config.fallback_model or "none",
                "description": config.description,
            }
            for config in self._configs.values()
        ]

    # ------ Internal ------

    def _try_model_and_finalize(
        self,
        model: str,
        config: TaskConfig,
        prompt: str,
        system_prompt: str | None,
        output_schema: dict[str, Any] | None,
        from_fallback: bool,
    ) -> LLMResponse | None:
        """Try a single model end-to-end. Returns None on *any* failure —
        network error, exhausted retries, JSON parse failure, or schema
        mismatch — so the caller can move to the fallback model."""
        raw = self._try_model(model, config, prompt, system_prompt)
        if raw is None:
            return None
        try:
            return self._finalize(raw, config, output_schema, from_fallback=from_fallback)
        except LLMParseError as e:
            logger.warning(
                "Model %s returned invalid output for %s: %s",
                model, config.task_type.value, e,
            )
            return None

    def _try_model(
        self,
        model: str,
        config: TaskConfig,
        prompt: str,
        system_prompt: str | None,
    ) -> dict[str, Any] | None:
        """尝试调用指定模型，失败返回 None。只吞真正的 provider 异常，
        不吞 AttributeError / KeyError 这类自家 bug。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exc: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                result = self._completion_fn(
                    model=model,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                    timeout=config.timeout_seconds,
                )
                return {
                    "content": result["content"],
                    "model": model,
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "cost": result.get("cost", 0.0),
                }
            except _RETRY_EXCEPTIONS as exc:
                last_exc = exc
                logger.warning(
                    "Model %s attempt %d/%d failed for %s: %s",
                    model, attempt, self._max_attempts, config.task_type.value, exc,
                )
            except Exception as exc:
                # Try to recognise litellm's own provider-error hierarchy
                # without hard-importing it (litellm is an optional dep).
                exc_module = type(exc).__module__
                exc_name = type(exc).__name__
                if exc_module.startswith("litellm") or "APIError" in exc_name or "RateLimit" in exc_name:
                    last_exc = exc
                    logger.warning(
                        "Model %s attempt %d/%d failed for %s: %s(%s)",
                        model, attempt, self._max_attempts, config.task_type.value,
                        exc_name, exc,
                    )
                else:
                    # Genuine bug — re-raise so it surfaces in tests/CI.
                    raise
        if last_exc is not None:
            logger.warning(
                "Model %s exhausted %d attempts for %s",
                model, self._max_attempts, config.task_type.value,
            )
        return None

    def _finalize(
        self,
        raw: dict[str, Any],
        config: TaskConfig,
        output_schema: dict[str, Any] | None,
        from_fallback: bool,
    ) -> LLMResponse:
        """构建 LLMResponse，验证 JSON / schema。

        对 ``output_format == "json"`` 的 task：
        - JSON 解析失败 → 抛 LLMParseError（让外层切 fallback）
        - Schema 缺必需字段 → 抛 LLMParseError

        对 text task，直接包装返回。
        """
        content = raw["content"]
        parsed = None

        if config.output_format == "json":
            parsed = self._try_parse_json(content)
            if parsed is None:
                raise LLMParseError(
                    f"failed to parse JSON from content: {content[:120]!r}"
                )
            if output_schema and not self._validate_schema(parsed, output_schema):
                required = output_schema.get("required", [])
                missing = [k for k in required if k not in parsed]
                raise LLMParseError(
                    f"schema validation failed, missing required keys: {missing}"
                )

        return LLMResponse(
            content=content,
            parsed=parsed,
            model_used=raw["model"],
            task_type=config.task_type,
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cost=raw.get("cost", 0.0),
            from_fallback=from_fallback,
        )

    @staticmethod
    def _try_parse_json(content: str) -> dict[str, Any] | None:
        """尝试从 LLM 输出中解析 JSON。

        按顺序尝试：
        1. 整段直接 json.loads
        2. 第一个 ```json ... ``` 代码块
        3. 第一个裸 ``` ... ``` 代码块
        4. 第一个 {...} 区段（最大匹配）
        """
        if not content:
            return None
        text = content.strip()

        # 1. 直接解析
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # 2/3. fenced code block（含 language specifier 或不含）
        match = _CODE_FENCE_RE.search(text)
        if match:
            inner = match.group(1).strip()
            try:
                result = json.loads(inner)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        # 4. greedy outermost object
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def _validate_schema(data: dict, schema: dict) -> bool:
        """简单的 schema 验证（检查 required 字段是否存在）。"""
        required = schema.get("required", [])
        return all(k in data for k in required)

    @staticmethod
    def _default_completion(
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """默认调用函数 — 使用 litellm。"""
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "litellm is required for LLMRouter. "
                "Install with: pip install litellm"
            )

        response = litellm.completion(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )

        choice = response.choices[0]
        usage = getattr(response, "usage", None)

        # litellm normalises usage but older versions / some providers may
        # name the attrs differently. Be defensive.
        input_tokens = 0
        output_tokens = 0
        if usage is not None:
            input_tokens = (
                getattr(usage, "prompt_tokens", None)
                or getattr(usage, "input_tokens", None)
                or 0
            )
            output_tokens = (
                getattr(usage, "completion_tokens", None)
                or getattr(usage, "output_tokens", None)
                or 0
            )

        # completion_cost may raise for unknown models — don't let that nuke
        # the whole call, just log and return 0.
        cost = 0.0
        try:
            cost = litellm.completion_cost(completion_response=response) or 0.0
        except Exception as exc:  # noqa: BLE001 — litellm exception hierarchy varies
            logger.warning(
                "litellm.completion_cost failed for %s: %s (recording cost=0)",
                model, exc,
            )

        return {
            "content": choice.message.content or "",
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost": float(cost),
        }
