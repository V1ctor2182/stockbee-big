# 投资组合构建 (02-portfolio-construction)

## Intent

**生成 Long 7 + Short 1 投资组合，约束优化**

整合多因子评分、宏观 Tilt、RL 权重优化，在行业/仓位/Beta 约束下构建最优组合。
score_final = factor_score + 0.1 × propagation_score + 0.2 × tilt_sector + 0.1 × tilt_style

来源：PRD §3.1, Tech Design §5.1

## Constraints

- **单股仓位 ≤ 20%（硬约束）** — 来源: PRD §3.1
- **同行业 ≤ 3 只（硬约束）** — 来源: PRD §3.1
- **净 Beta ~72%** — 来源: PRD §3.1

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
