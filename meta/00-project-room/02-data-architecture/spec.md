# 数据架构 (02-data-architecture)

## Intent

**完整数据管道，覆盖行情、新闻、宏观、因子、图谱、RL 经验库**

10 个 Provider 抽象接口，支持 backtest/live 无缝切换。Parquet + SQLite 混合存储，3 层缓存策略。全部数据 < 2GB。

来源：PRD §3.2, Tech Design §2

## Constraints

- **SQLite 使用 WAL 模式** — 来源: PRD §3.2
- **全部数据 < 2GB** — 来源: Tech Design §2.9
- **10 个 Provider 抽象接口** — 来源: Tech Design §2.8

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
