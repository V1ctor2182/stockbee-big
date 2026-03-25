# 投资组合管理 (01-portfolio-management)

## Intent

**从 U100 中自动选出 Long 7 + Short 1，生成调仓方案，经人工审批后执行交易**

U100 每月自动重构，覆盖 11 大 GICS 行业。包含宇宙管理、组合构建、交易执行、做空模式四个子模块。

来源：PRD §3.1, Feature Hierarchy §1

## Constraints

- **单股仓位 ≤ 20%** — 来源: PRD §3.1
- **同行业 ≤ 3 只** — 来源: PRD §3.1
- **净 Beta 目标 ~72%** — 来源: PRD §3.1

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
