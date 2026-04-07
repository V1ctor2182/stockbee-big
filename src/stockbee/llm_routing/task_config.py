"""TaskConfig — 任务类型定义 + 模型映射配置。

5 种任务类型，每种指定主模型、备选模型、token 限制、输出格式。
成本估算基于 Tech Design §3.2。

来源：Tech Design §3.2, research-llm-selection.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TaskType(str, Enum):
    """LLM 任务类型。"""
    G1_FILTER = "g1_filter"              # 新闻快速过滤（相关/无关）
    G2_CLASSIFY = "g2_classify"          # 新闻分类（情绪/主题/紧急度）
    G3_ANALYSIS = "g3_analysis"          # 深度基本面分析
    SEC_PARSE = "sec_parse"              # SEC 10-K/10-Q 解析
    MACRO_ANALYSIS = "macro_analysis"    # 宏观环境分析（周度）


@dataclass(frozen=True)
class TaskConfig:
    """单个任务类型的模型配置。"""
    task_type: TaskType
    model: str                     # litellm 模型 ID（如 "openai/gpt-4.1-nano"）
    fallback_model: str | None     # 备选模型
    max_tokens: int                # 最大输出 token
    temperature: float = 0.0       # 默认确定性输出
    output_format: str = "json"    # "json" | "text"
    monthly_budget: float = 0.0    # 月度预算上限（美元）
    description: str = ""


# 默认配置 — Tech Design §3.2
DEFAULT_TASK_CONFIGS: dict[TaskType, TaskConfig] = {
    TaskType.G1_FILTER: TaskConfig(
        task_type=TaskType.G1_FILTER,
        model="openai/gpt-4.1-nano",
        fallback_model="google/gemini-2.0-flash",
        max_tokens=100,
        temperature=0.0,
        output_format="json",
        monthly_budget=1.0,
        description="新闻快速过滤：标题+摘要 → {relevant: bool}",
    ),
    TaskType.G2_CLASSIFY: TaskConfig(
        task_type=TaskType.G2_CLASSIFY,
        model="anthropic/claude-sonnet-4-20250514",
        fallback_model="openai/gpt-4o-mini",
        max_tokens=500,
        temperature=0.0,
        output_format="json",
        monthly_budget=2.0,
        description="新闻分类：摘要+部分正文 → {sentiment, category, urgency}",
    ),
    TaskType.G3_ANALYSIS: TaskConfig(
        task_type=TaskType.G3_ANALYSIS,
        model="openai/gpt-4.1",
        fallback_model="anthropic/claude-opus-4-20250514",
        max_tokens=2000,
        temperature=0.3,
        output_format="text",
        monthly_budget=5.0,
        description="深度基本面分析：完整新闻+持仓情景 → 影响评估",
    ),
    TaskType.SEC_PARSE: TaskConfig(
        task_type=TaskType.SEC_PARSE,
        model="openai/gpt-4.1",
        fallback_model="anthropic/claude-sonnet-4-20250514",
        max_tokens=3000,
        temperature=0.0,
        output_format="json",
        monthly_budget=3.0,
        description="SEC 10-K/10-Q 解析 → 关键财务指标 JSON",
    ),
    TaskType.MACRO_ANALYSIS: TaskConfig(
        task_type=TaskType.MACRO_ANALYSIS,
        model="anthropic/claude-sonnet-4-20250514",
        fallback_model="openai/gpt-4.1",
        max_tokens=1500,
        temperature=0.2,
        output_format="text",
        monthly_budget=4.0,
        description="周度宏观环境分析：FRED 指标 → 经济周期判断",
    ),
}

TOTAL_MONTHLY_BUDGET = sum(c.monthly_budget for c in DEFAULT_TASK_CONFIGS.values())
# Should be $15
