# 风险控制 (04-risk-controls)

## Intent

**尾随止损 ATR×1.5 + 冷却期 + 头寸限制**

价格下跌 > 3σ 自动平仓（不需审批）。Tier-1 加速上限 2 次/天，冷却期 30 分钟。

来源：Tech Design §6.2, Feature Hierarchy §7.4

## Constraints

- **ATR×1.5 尾随止损** — 来源: Tech Design §6.2
- **加速冷却期 30 分钟** — 来源: Tech Design §6.2

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
