# LLM Agent 决策与安全 (06-llm-agent-decision)

## Intent

**LLM Agent 作为决策编排者，整合多信号源，经验证层后输出建议**

CEO Review 发现的 CRITICAL GAP：LLM 直接驱动交易决策缺乏安全护栏。所有 LLM 输出必须经过验证层。Decision Loop 是护城河的种子。

来源：PRD §3.4, Feature Hierarchy §6

## Constraints

- **LLM 输出经 JSON Schema 验证** — 来源: PRD §3.4
- **周度再平衡必须人工审批** — 来源: PRD §3.4
- **所有 LLM 输入/输出记录日志** — 来源: PRD §3.4

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
