"""LLM Routing 模块 — 多模型路由 + 降级链 + 成本追踪。

按任务类型选模型，统一 OpenAI/Anthropic 调用接口（via litellm）。
Router 不管业务逻辑（G1/G2/G3），只提供 route(task_type, prompt) 接口。
"""

from .task_config import TaskType, TaskConfig, DEFAULT_TASK_CONFIGS
from .router import LLMRouter, LLMResponse
from .cost_tracker import CostTracker

__all__ = [
    "TaskType",
    "TaskConfig",
    "DEFAULT_TASK_CONFIGS",
    "LLMRouter",
    "LLMResponse",
    "CostTracker",
]
