# Stockbee 技术设计 — 信息源与新闻处理

**版本**: 2.3 | **日期**: 2026-04-19 | ← [返回总览](00-architecture-overview.md)

---

## 1. 新闻数据源架构

Stockbee 的新闻处理采用多源融合策略，确保信息的广度、深度和权威性。→ 参考：信息.md §1-§3

| 来源 | 覆盖范围 | 更新频率 | 成本 | 用途 |
|------|---------|---------|------|------|
| NewsAPI | 全球新闻聚合（70k+ 来源） | 实时 | 免费（~500 条/天 limit） | 广泛扫描，G1 过滤基础 |
| Perplexity | AI 搜索聚合新闻 | 实时 | 按用量 | 与 NewsAPI 同级数据源，跨源去重 |
| SEC EDGAR | 8-K 速报、10-K/Q | 日度 | 免费 API | 权威基本面信息，无延迟 |
| Nasdaq RSS | 交易暂停、熔断、除牌 | 实时 | 免费 | 流动性事件 |
| FactSet/Bloomberg（未来扩展） | 分析师目标价、评级变化 | 日度 | 专业版 | 机构预期 |
| Prediction Market（Polymarket） | 宏观事件概率 | 实时 | 免费 | 市场共识的制度变化预测 |

→ 参考：信息.md §1, polymarket的数据如何运用.md

---

## 2. 新闻摄入与三级处理漏斗（G1/G2/G3）

**G1（快速过滤，95% 淘汰）**：
- 来源合法性检查（黑名单 domain）
- 发布时间检查（避免重复旧新闻）
- 去重（多源相同新闻只保留一条）
- 实体识别（检测新闻中的股票 ticker，支持模糊匹配）
- 输出：news_events 表写入，g_level = 'G1'

**G2（智能分类，50-70% 淘汰）**：
触发条件：所有通过 G1 的新闻。

本地 FinBERT 推理，零 API 成本，延迟 <5 秒。

- FinBERT (本地) 情绪分类：positive/neutral/negative
- 主题自动分类：收益、合规、并购、诉讼、政策、产品等
- 相关性打分：该新闻与当前持仓的相关度（0-1）
- 紧急度评分：high/medium/low（基于主题和持仓覆盖度）
- 输出：news_events.sentiment_score, importance_score, g_level = 'G2'

**G3（深度分析，10-30 篇/天）**：
触发条件：
- importance_score > 0.7 且 urgency = high，或
- 新闻涉及当前持仓且 sentiment_score 为极端（>0.9 or <0.1）

频率上限：每天最多 10 篇。

由 Claude Haiku 执行（成本更低，<$2/月，深度分析任务足够）：
- 完整阅读全文
- 估算对持仓价值的影响（±X%）
- 建议权重调整或风险对冲
- 评估新闻真伪（是否为讹传）
- 输出：news_events.analysis 字段，g_level = 'G3'，推送给 Victor 审批

月成本 <$5。→ 参考：research-news-storage.md §3-§4

---

## 3. Prediction Market 应用

Polymarket 等去中心化预测市场提供"群体智慧"式的事件概率。例如"美联储 2026 Q2 降息"的隐含概率。

Stockbee 使用方式：
- 定期爬取 Polymarket 前 30 大宏观事件的隐含概率
- 与前期概率对比，检测"概率悬崖"（突然大幅变化）
- 将概率作为 MacroTiltEngine 的补充因子，调整行业倾斜
- 例如降息概率从 20% 跳升至 70%，则增加高 duration 金融股的权重

优势：市场驱动，免费，实时性强。→ 参考：polymarket的数据如何运用.md
