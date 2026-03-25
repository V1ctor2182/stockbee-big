# 决策日志 (03-decision-log)

## Intent

**所有 LLM 输入/输出全记录，Decision Loop 护城河种子**

Decision Record Schema: timestamp, market_regime, signals_snapshot, llm_input/output, decision, rationale, risk_check, outcome_1d/5d/20d。存入 SQLite decision_log 表。

来源：PRD §5 护城河策略, Feature Hierarchy §6.3

## Constraints

- **Phase 1 积累 1000+ Decision Records** — 来源: PRD §6

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
