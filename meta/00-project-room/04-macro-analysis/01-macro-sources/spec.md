# 补充宏观数据源 (01-macro-sources)

## Intent

**Polymarket 事件概率 + FRED 经济日历**（原 FRED/BLS/RSS 已被 03-macro-data 覆盖）

2 个核心组件：
- **PolymarketFetcher** — Gamma API 抓取前 30 大宏观事件概率，SQLite 存储历史，概率悬崖检测（变化 > 15%）
- **EconomicCalendar** — FRED release dates API 同步经济数据发布日历，8 类高波动事件标记（FOMC/NFP/CPI/GDP 等）

实现状态：已完成 ✅（2026-04-06）
测试覆盖：22 个单元测试

## Constraints

- **Polymarket API 无需认证** — 公开 REST API (gamma-api.polymarket.com)
- **概率悬崖阈值 15%** — abs(current - previous) >= 0.15
- **高波动事件 8 类** — FOMC, NFP, CPI, PPI, GDP, Treasury Yield, Industrial Production, M2

## Decisions

- **FRED/BLS/RSS 不在本 Room 实现** — 已被 03-macro-data 完全覆盖
- **Polymarket REST API > 爬虫** — 公开 API 更稳定
- **FRED release calendar API > 静态维护** — 自动同步，无需手动更新

## Contracts

- **PolymarketFetcher** — fetch_macro_events(limit) → list[MarketEvent]; save_events; detect_cliffs; get_latest_events
- **EconomicCalendar** — sync(days_ahead) → int; get_events(start, end, hv_only); is_high_volatility_day(d); get_next_high_volatility(after)

---
_所有 spec 状态: active_
_spec.md 最后更新: 2026-04-06_
