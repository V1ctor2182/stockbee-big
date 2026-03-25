# 交易执行 (03-trade-execution)

## Intent

**通过 Alpaca API 执行交易，幂等订单，完整日志**

周度再平衡：周一开盘市价单。实时信号：Tier-1 立即执行，Tier-2 待命。每笔交易记录完整理由。

来源：PRD §3.1, Tech Design §6.1

## Constraints

- **交易执行成功率 > 99.5%** — 来源: PRD §4
- **幂等执行（重试不重复下单）** — 来源: PRD §3.1
- **日度交易金额上限 $50k** — 来源: Tech Design §6.2

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
