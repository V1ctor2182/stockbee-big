# 验证层 (02-validation-layer)

## Intent

**Schema 验证 + 风控约束检查 + 异常检测 + Injection 防护**

JSON Schema 验证结构正确。风控自动检查仓位/行业/Beta 约束。偏离历史分配范围触发告警。新闻文本清洗防 Prompt Injection。

来源：PRD §3.4, Feature Hierarchy §6.2

## Constraints

- **新闻文本输入前清洗** — 来源: PRD §3.4
- **偏离历史分配范围的决策触发告警** — 来源: PRD §3.4

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
