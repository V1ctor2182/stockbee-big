"""LLMRouter — 多模型路由器 + 降级链。

统一调用接口（via litellm），按 TaskType 选模型。
Router 是 task-agnostic 的 — 不管 G1/G2/G3 业务逻辑，
只提供 route(task_type, prompt) → response 接口。

依赖：pip install litellm（可选依赖）

来源：Tech Design §3.2
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .task_config import DEFAULT_TASK_CONFIGS, TaskConfig, TaskType

logger = logging.getLogger(__name__)


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


class LLMRouter:
    """多模型路由器。

    使用方式：
        router = LLMRouter()
        response = router.route(
            TaskType.G1_FILTER,
            "Is this news relevant to AAPL? Headline: ...",
        )
        print(response.parsed)  # {"relevant": true}
    """

    def __init__(
        self,
        task_configs: dict[TaskType, TaskConfig] | None = None,
        max_retries: int = 2,
    ) -> None:
        self._configs = task_configs or DEFAULT_TASK_CONFIGS
        self._max_retries = max_retries
        self._completion_fn = self._default_completion

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

        Args:
            task_type: 任务类型，决定使用哪个模型
            prompt: 用户 prompt
            system_prompt: 可选的 system prompt
            output_schema: 可选的 JSON schema（用于验证输出）

        Returns:
            LLMResponse
        """
        config = self._configs.get(task_type)
        if config is None:
            raise ValueError(f"Unknown task type: {task_type}")

        # 尝试主模型
        response = self._try_model(config.model, config, prompt, system_prompt)
        if response is not None:
            return self._finalize(response, config, output_schema, from_fallback=False)

        # 降级到备选模型
        if config.fallback_model:
            logger.warning(
                "Primary model %s failed for %s, falling back to %s",
                config.model, task_type.value, config.fallback_model,
            )
            response = self._try_model(config.fallback_model, config, prompt, system_prompt)
            if response is not None:
                return self._finalize(response, config, output_schema, from_fallback=True)

        # 全部失败
        logger.error("All models failed for task %s", task_type.value)
        raise RuntimeError(
            f"LLM routing failed for {task_type.value}: "
            f"primary={config.model}, fallback={config.fallback_model}"
        )

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

    def _try_model(
        self,
        model: str,
        config: TaskConfig,
        prompt: str,
        system_prompt: str | None,
    ) -> dict[str, Any] | None:
        """尝试调用指定模型，失败返回 None。"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(1, self._max_retries + 1):
            try:
                result = self._completion_fn(
                    model=model,
                    messages=messages,
                    max_tokens=config.max_tokens,
                    temperature=config.temperature,
                )
                return {
                    "content": result["content"],
                    "model": model,
                    "input_tokens": result.get("input_tokens", 0),
                    "output_tokens": result.get("output_tokens", 0),
                    "cost": result.get("cost", 0.0),
                }
            except Exception:
                logger.warning(
                    "Model %s attempt %d/%d failed for %s",
                    model, attempt, self._max_retries, config.task_type.value,
                    exc_info=attempt == self._max_retries,
                )
        return None

    def _finalize(
        self,
        raw: dict[str, Any],
        config: TaskConfig,
        output_schema: dict[str, Any] | None,
        from_fallback: bool,
    ) -> LLMResponse:
        """构建 LLMResponse，尝试 JSON 解析。"""
        content = raw["content"]
        parsed = None

        if config.output_format == "json":
            parsed = self._try_parse_json(content)
            if parsed is None:
                logger.warning("Failed to parse JSON output for %s", config.task_type.value)

            if parsed and output_schema:
                if not self._validate_schema(parsed, output_schema):
                    logger.warning(
                        "Output schema validation failed for %s",
                        config.task_type.value,
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

    def _try_parse_json(self, content: str) -> dict[str, Any] | None:
        """尝试从 LLM 输出中解析 JSON。"""
        content = content.strip()
        # 处理 markdown code block
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def _validate_schema(self, data: dict, schema: dict) -> bool:
        """简单的 schema 验证（检查必需字段是否存在）。"""
        required = schema.get("required", [])
        return all(k in data for k in required)

    @staticmethod
    def _default_completion(
        model: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
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
        )

        choice = response.choices[0]
        usage = response.usage

        return {
            "content": choice.message.content or "",
            "input_tokens": usage.prompt_tokens if usage else 0,
            "output_tokens": usage.completion_tokens if usage else 0,
            "cost": litellm.completion_cost(completion_response=response),
        }
