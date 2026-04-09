# 补充宏观数据源 (01-macro-sources)

## Intent

**Polymarket 事件概率 + 经济日历**（原 FRED/BLS/RSS 已被 03-macro-data 覆盖）

两个补充数据源：
- **Polymarket 事件概率** — 爬取前 30 大宏观事件，检测"概率悬崖"，作为 MacroTiltEngine 外生因子
- **经济日历** — FOMC/非农/CPI 等事件日，标记高波动日，避免在数据发布前后再平衡

依赖：03-macro-data ✅（FRED 17 指标已连通）

## Constraints

_待开发时确定_

## Decisions

- **FRED/BLS/RSS 不在本 Room 实现** — 已被 03-macro-data 完全覆盖（2026-04-06 重新定义）

## Contracts

_待开发时确定_

---
_spec 状态: active（intent 已重新定义）_
_spec.md 最后更新: 2026-04-06_
