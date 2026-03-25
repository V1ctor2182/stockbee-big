# 股票数据存储 (01-stock-data)

## Intent

**三层金字塔架构，Parquet 列存 + SQLite 元数据**

4000 只可投资美股，日线 OHLCV。四层漏斗避免幸存者偏差。Parquet 压缩率 8:1。

来源：Tech Design §2.1-§2.2
研究状态：研究完成 (research-stock-data-storage.md)

## Constraints

- **覆盖 ~4000 只可投资美股** — 来源: PRD §3.2

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
