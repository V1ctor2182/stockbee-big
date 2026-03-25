# 供应商架构 (07-provider-arch)

## Intent

**10 个 Provider + Registry，支持 backtest/live 切换**

Calendar/Universe/MarketData/Factor/Fundamental/News/Macro/Sentiment/Broker/Cache 10 个 Provider。YAML 配置切换实现，3 层缓存（L1 内存 → L2 Parquet → L3 API）。

来源：Tech Design §2.8
研究状态：研究完成 (research-provider-design.md)

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
