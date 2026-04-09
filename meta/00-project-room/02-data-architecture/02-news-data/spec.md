# 新闻数据存储 (02-news-data)

## Intent

**SQLite news_events 表 + G1/G2/G3 三级漏斗处理管道**

7 个组件组成完整的新闻数据基础设施：
- **SqliteNewsProvider** — WAL 模式 news_events + news_tickers junction table，支持多条件查询（ticker JOIN 索引/时间/重要度/g_level）
- **G1 Filter** — 来源校验、时间校验、去重、实体识别（ticker 提取）
- **G2 Classifier** — FinBERT 本地情绪分类 + 主题分类 + 重要度/紧急度评分
- **G3 Analyzer** — Claude Haiku 深度分析，日限 10 篇，影响评估 + 权重建议
- **NewsAPI Source** — 免费层 ~500 条/天实时拉取
- **Perplexity Source** — API 新闻拉取，与 NewsAPI 同级数据源
- **NewsDataSyncer** — 编排 fetch → G1 → store → G2 → G3（条件触发）

来源：Tech Design §2.3, §4.1-4.2
依赖：transformers + torch (FinBERT), anthropic SDK (Haiku), newsapi-python

## Milestones

1. SqliteNewsProvider — news_events 表 + 查询接口 (~200 行)
2. G1 快速过滤管道 (~150 行)
3. G2 FinBERT 分类评分 (~180 行)
4. G3 深度分析 Claude Haiku (~120 行)
5. NewsAPI + Perplexity + NewsDataSyncer (~220 行)
6. 测试 (~280 行)

## Decisions

- **G2 用本地 FinBERT** > 规则引擎 / LLM API — 准确率高、零 API 成本、离线可用
- **G3 用 Claude Haiku** > Sonnet — 成本更低（<$2/月），深度分析任务 Haiku 足够
- **数据源: NewsAPI + Perplexity** — 两个同级新闻拉取源，跨源去重
- **G3 Phase 1 包含** — 日限 10 篇，scope 可控
- **DB-level UNIQUE 去重** > 应用层 SELECT-then-INSERT — 消除 TOCTOU 并发竞态，由 /review 驱动
- **get_news() 默认 LIMIT 1000** — 防止无过滤查询 OOM，由 /review 驱动
- **batch 插入单事务** > 逐条 commit — 500 条从 ~1.5s 降到 ~5ms（200x），由 /review 驱动
- **Junction table (news_tickers)** > JSON 字符串 + LIKE — 可索引、不依赖序列化格式、SQL 标准写法，由 code review 驱动

## Contracts

- **SqliteNewsProvider** — get_news(tickers, start, end, min_importance, g_level, limit) -> DataFrame; insert_news(headline, source, timestamp, ...) -> id|None; insert_news_batch(events) -> int; update_g_level(news_id, g_level, scores) -> bool; get_g3_daily_count/increment; get_news_by_id; count_by_g_level

---
_spec 状态: intent (active), decision (active), change x5 (active)_
_spec.md 最后更新: 2026-04-07 (m6 edge case tests — Room 完成)_
_specs 目录: 1 intent + 1 decision + 5 change = 7 个 spec 文件_
