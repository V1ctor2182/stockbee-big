# 补充宏观数据源 (01-macro-sources)

## Intent

**Polymarket 事件概率 + FRED 经济日历**（原 FRED/BLS/RSS 已被 03-macro-data 覆盖）

2 个核心组件：
- **PolymarketFetcher** — Gamma API 抓取宏观事件概率，**两步过滤**（API 拉前 200 热门 + 本地关键词过滤），SQLite 存储历史，概率悬崖检测（变化 ≥ 15%）
- **EconomicCalendar** — FRED Release Dates API 同步经济日历，标记 8 类高波动事件（FOMC/NFP/CPI/PPI/GDP/Treasury/Industrial/M2）

实现状态：已完成 ✅ + Polymarket API 实测验证（2026-04-09）
测试覆盖：22 个单元测试

## Constraints

- **概率悬崖阈值 15%** — abs(current - previous) >= 0.15
- **8 类高波动事件** — FOMC, NFP, CPI, PPI, GDP, Treasury Yield, Industrial Production, M2
- **Polymarket 两步搜索** — API limit=200 + 本地 MACRO_KEYWORDS 过滤
- **MACRO_KEYWORDS 20 个** — 货币政策(6) + 经济周期(5) + 贸易(2) + 财政(4) + 央行政策(3)
- **Polymarket 必须 User-Agent header** — 否则 403 Forbidden
- **Polymarket 完全免费，无需 API key** — Gamma API 是公开 REST

## Decisions

- **Room 重新定义**: FRED/BLS/RSS → Polymarket + 经济日历（原范围被 03-macro-data 覆盖）
- **两步过滤策略**: Gamma API 不支持文本搜索/tag 过滤，必须客户端关键词匹配
- **关键词列表硬编码 in code**: 简单可控，未来可扩展为配置文件

## Contracts

- **PolymarketFetcher**:
  - `fetch_macro_events(limit=30, fetch_size=200)` → `list[MarketEvent]`
  - `save_events(events)`, `detect_cliffs(events?)`, `get_latest_events(limit=30)`
- **EconomicCalendar**:
  - `sync(days_ahead=90)` → int (无 API key 时返回 0)
  - `get_events(start?, end?, hv_only=False)`, `is_high_volatility_day(d)`, `get_next_high_volatility(after?)`

## 实测结果（2026-04-09）

```
200 个热门市场 → 关键词过滤 → 3 个宏观事件：
  1%   联邦基金利率 5月会议预测
  78%  墨西哥央行 5月利率决策
  18%  阿根廷比索汇率
```

## 关键词列表

```python
MACRO_KEYWORDS = [
    # 美联储/利率 (6)
    "fed ", "federal reserve", "fomc", "interest rate", "rate cut", "rate hike",
    # 经济周期 (5)
    "recession", "inflation", "cpi", "gdp", "unemployment",
    # 贸易 (2)
    "tariff", "trade war",
    # 财政 (4)
    "treasury", "debt ceiling", "deficit", "stimulus",
    # 央行/政策 (3)
    "central bank", "monetary policy", "fiscal",
]
```

未覆盖（待扩展）：地缘冲突、大宗商品、大选、美元指数

---
_所有 spec 状态: active_
_specs 目录: 1 intent + 4 constraints + 1 decision + 1 contract + 1 change = 8 个 spec 文件_
_spec.md 最后更新: 2026-04-09_
