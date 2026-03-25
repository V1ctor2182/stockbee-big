# 知识图谱与传播引擎 (06-knowledge-graph)

## Intent

**两层传播图（宏观指标图 + 公司关系图），事件沿图谱传播生成信号**

宏观图：~30 节点、~70 有向边，Z-score > 1.0 触发传播，3 层衰减。
公司图：U100 的 100 节点、~500 边（供应链/竞争/同行业/母子/合作）。
Phase 1：静态图可查询。Phase 2：传播引擎上线。

来源：Tech Design §2.7
研究状态：研究完成 (research-knowledge-graph-20260324.md)

## Constraints

- **宏观传播 Tilt = 0.6 × 规则 + 0.4 × 传播** — 来源: Tech Design §2.7
- **传播日志可追溯** — 来源: PRD §3.5

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
