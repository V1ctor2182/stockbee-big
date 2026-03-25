# 新闻数据存储 (02-news-data)

## Intent

**news_events 表，结构化字段，分级存储 G1/G2/G3**

所有新闻进入 G1 快速过滤，通过后持久化到 SQLite，触发 G2/G3 处理。

来源：Tech Design §2.3
研究状态：研究完成 (research-news-storage.md)

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
