# Stockbee 项目总控 (00-project-room)

## Intent

**AI 驱动的美股量化交易系统，Long 7+Short 1 周度调仓**

融合宏观分析、多因子评分、强化学习组合优化、LLM 新闻分析和人工审批机制，实现智能化的周度投资组合再平衡。核心策略：从 U100 精选宇宙中选股。

来源：PRD §1 产品概述

## Constraints

- **月度总运营成本 < $200** — 来源: PRD §4 非功能需求
- **LLM API 成本 < $15/月** — 来源: PRD §4
- **交易时间系统可用性 > 99.5%** — 来源: PRD §4
- **周度再平衡端到端 < 30 分钟** — 来源: PRD §4
- **所有 LLM 输出必须经过验证层** — 来源: PRD §3.4 CEO Review

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
