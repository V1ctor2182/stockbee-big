# LLM 多模型路由 (01-llm-routing)

## Intent

**按任务选模型，G1/G2/G3 分级路由，降级链**

G1 GPT-4.1 nano (<$1)，G2 Claude Sonnet 4 (<$2)，G3 GPT-4.1 (<$5)，SEC GPT-4.1 long (<$3)，宏观 Claude Sonnet 4 (<$4)。降级链：主模型→备选→缓存→禁用信号。

来源：Tech Design §3.2
研究状态：研究完成 (research-llm-selection.md)

## Constraints

- **月度合计 < $15** — 来源: Tech Design §3.2

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
