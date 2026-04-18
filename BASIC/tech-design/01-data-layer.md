# Stockbee 技术设计 — 数据层

**版本**: 2.2 | **日期**: 2026-04-18 | ← [返回总览](00-architecture-overview.md)

---

## 1. 股票数据存储与分级架构

**四层漏斗设计**（8000→4000→500→100→8）：

第一层（广域宇宙，8000只）：每周从 Alpaca API 同步全部美股清单，包含流动性指标和融资可用性标志。用于监控市场变化。

第二层（广泛宇宙，约4000只）：筛选条件为流动性>100万美元/日、平均价格>2美元、非 OTC 市场。存储在 SQLite 的 broad_universe 表中，字段包括 ticker、sector、market_cap、avg_volume、short_able。此层作为因子计算的基础集合。→ 参考：research-stock-data-storage.md §2-§3

第三层（候选池，约500只）：从广泛宇宙按行业均衡和流动性顶部筛选。仅这一层及以上进行日线数据同步和因子计算。

第四层（U100，100只）：月度重新评估，基于因子综合评分和绩效指标。→ 参考：research-stock-data-storage.md §3

**为什么分层而不是直接从 U100 开始？** 如果直接观察历史 U100，会产生幸存者偏差。通过实时维护所有层级，系统可在回测时模拟真实的选股流程，而不是在历史数据中拣选。→ 参考：research-stock-data-storage.md §1

---

## 2. 存储技术选型与缓存策略

| 数据类型 | 存储方式 | 原因 | 引用 |
|---------|---------|------|------|
| 股票日线 OHLCV (4000只 × 5年) | Parquet (列存, 压缩) | 时间序列查询高效，压缩率 8:1，通用性强（vs Qlib .bin 私有格式） | research-stock-data-storage.md §2 |
| 宇宙元数据 (交易日历、黑名单、流动性) | SQLite broad_universe 表 | 支持快速过滤和关系查询，ACID 保证一致性 | research-stock-data-storage.md §3 |
| 新闻事件及评分 (G1/G2/G3) | SQLite news_events 表，字段：id, timestamp, source, tickers, headline, snippet, sentiment_score, importance_score, reliability_score, g_level | 全文搜索、结构化查询、关联查询支持 | research-news-storage.md §2 |
| 宏观指标 (19个FRED + 派生特征) | Parquet (时间序列) + SQLite (最新值索引) | Parquet 支持时间切片查询，SQLite 索引加速最新值查询 | research-macro-features.md §2 |
| 技术因子 (Alpha158集) | 表达式引擎 (动态计算) | 灵活性强，修改公式无需重新计算历史，支持 A/B 测试 | research-factor-storage.md §1-§2 |
| 基本面和ML融合因子 | Parquet (预计算) | 避免重复计算，推理时直接查表，降低 CPU 占用 | research-factor-storage.md §3 |
| 经验回放缓冲 (RL训练) | 内存 Ring Buffer (numpy 数组) | 高速随机访问，可转移至 GPU，50K 容量 ≈ 35MB | research-experience-buffer-20260317.md §2-§3 |
| 公司关系传播图 | NetworkX Graph + CSV 备份 (100 节点, ~500 边) | 内存常驻毫秒级查询，支持传播计算，CSV 方便手动维护 | research-knowledge-graph-20260324.md §3, research-entity-relationships-20260317.md §1 |
| 宏观指标传播图 | NetworkX DiGraph + CSV 备份 (~30 节点, ~70 边) | 有向图表达因果方向，传播机制计算行业 Tilt | research-knowledge-graph-20260324.md §2, research-entity-relationships-20260317.md §2 |

---

## 3. 新闻数据处理管道

所有新闻进入 G1 快速过滤：检查时间戳、数据源合法性、去重（同一新闻多个来源）。通过 G1 的新闻进入 SQLite 持久化，同时触发 G2 LLM 分类（GPT-4.1 nano，<$1/月）。G2 产生：情绪（正/中/负）、主题分类、紧急度评分。可选的 G3 由 Claude Sonnet 4 执行深度分析（仅限分数>70 的新闻，日限 10 篇，<$5/月）。→ 参考：research-news-storage.md §3-§4, research-llm-selection.md §3

---

## 4. 宏观数据与 MacroTiltEngine

19 个 FRED 指标分布在 8 个经济维度：利率（3个：10Y, 2Y, FFR）、就业（3个：NFP, 失业率, 初请）、物价（2个：CPI, PPI）、增长（2个：GDP, 工业产值）、信用（2个：高收益差, 杠杆率）、流动性（2个：M2, VIX）、汇率（1个：DXY）、商品（2个：原油, 黄金）。

每个指标实时更新至 Parquet 时间序列文件，同时在 SQLite 中维护最新值及其 Z-score（相对 252 天历史的标准化）。系统采用点对点对齐技术避免前看偏差：因子在日期 T 的计算仅使用 T 前一个交易日的宏观数据。→ 参考：research-macro-features.md §2, research-macro-features.md §3

---

## 5. 因子存储与 IC/ICIR 评估

因子分两类存储：

**表达式因子（技术因子）**：来自 Qlib 的 Alpha158 标准因子集。使用自研表达式引擎动态计算，支持 OHLCV、成交量加权、移动平均、相对强度等操作。公式存储在配置文件中，每日闭盘后重新计算。优势：灵活调整因子定义，无需重新处理历史数据。→ 参考：research-factor-storage.md §1

**预计算因子（融合因子）**：基本面因子（PE、PB、ROE）、情绪因子（FinBERT 得分）、ML 预测因子（LightGBM 收益预测）、宏观倾斜得分（MacroTiltEngine 输出）。这些无法用纯表达式表示，因此预计算后存储在 Parquet 中。每周更新一次。→ 参考：research-factor-storage.md §2-§3

**IC/ICIR 评估管道**：系统维护滚动窗口（252 日）的 IC（因子与收益的相关性）和 ICIR（IC 的信息比率）。在日终计算所有因子的 IC，当某个因子的 ICIR 从正转负时，自动禁用或降低其权重。这确保了仅有效的因子在组合中发挥作用。→ 参考：research-factor-storage.md §4

---

## 6. 经验回放缓冲区

Ring Buffer 存储 RL 训练的 transition（状态→动作→奖励→下一状态）。容量 50K，每条 transition 占用：state(74D float32) + action(8D or 100D) + reward(1D) + next_state(74D) + done(1D bool) ≈ 0.7KB，总计 ≈35MB，完全驻留内存。

Buffer 采用 FIFO 覆盖策略：当满时，新的 transition 覆盖最旧的数据。支持可选的 Priority Experience Replay（PER）以强调高 TD 误差的样本，留待 Phase 2 实现。→ 参考：research-experience-buffer-20260317.md §2-§3

---

## 7. 知识图谱与传播引擎

Stockbee 采用**两层知识图谱**架构，实现"一个事件进入，影响沿图谱传播"的信号放大机制。→ 参考：research-knowledge-graph-20260324.md

**宏观指标传播图**：~30 个节点（Fed Rate、10Y Treasury、CPI、失业率、油价、DXY 等）、~70 条有向边（因果关系，含极性、强度 0~1、时滞天数）。以 NetworkX DiGraph 常驻内存，CSV 备份。当某指标发布异常数据（Z-score > 1.0）时，触发传播引擎：沿因果链计算各下游节点的冲击信号（`impact = z × polarity × strength`），衰减因子 0.5，最大深度 3 层。传播终端映射到 GICS 行业 Tilt，与现有 MacroTiltEngine 的规则 Tilt 融合（0.6 规则 + 0.4 传播）。→ 参考：research-knowledge-graph-20260324.md §2, research-macro-features.md

**公司关系传播图**：U100 的 100 个公司节点、~500 条边（供应链、竞争、同行业、母子、合作）。以 NetworkX Graph 常驻内存，CSV 备份。当 G2 新闻分析输出某公司的情绪和重要性评分后，传播引擎沿关系边计算影响：供应链同向传播、竞争反向传播、同行业弱正传播。每家公司获得 propagation_score，作为新因子参与评分：`score_final = factor_score + 0.1 × propagation_score + 0.2 × tilt_sector + 0.1 × tilt_style`。→ 参考：research-knowledge-graph-20260324.md §3, research-entity-relationships-20260317.md §1

**传播日志**：每次传播事件记录到 SQLite propagation_log 表（timestamp、trigger_node、trigger_event、affected_node、impact_score、propagation_path JSON），用于审批时展示传播路径、回测传播信号准确率。→ 参考：research-knowledge-graph-20260324.md §5

**流动性指标**：每只股票维护历史流动性数据（日均成交量、成交额、买卖价差），存储在 Parquet（历史）和 SQLite（最新）。用于权重分配时确保 8 只多头+1 只空头的流动性满足最低要求（日均成交额 >50 万美元）。→ 参考：research-entity-relationships-20260317.md §3

---

## 8. Provider 架构与多实现切换

Stockbee 继承了 Qlib 的 Provider 设计思想，扩展到 10 个核心 Provider：

```
CalendarProvider          交易日历（回测用 Qlib 日历，实盘用 Alpaca 日历）
UniverseProvider          股票池（从 SQLite broad_universe 表）
MarketDataProvider        OHLCV 数据（Parquet 读取 vs Alpaca API）
FactorProvider            因子数据（表达式引擎 vs 预计算 Parquet）
FundamentalProvider       基本面数据（Polygon.io 或 Yahoo Finance fallback）
NewsProvider              新闻数据（SQLite news_events vs 实时 NewsAPI）
MacroProvider             宏观数据（Parquet 时间序列 vs FRED API）
SentimentProvider         情绪评分（本地 FinBERT vs 缓存）
BrokerProvider            交易执行（Alpaca 纸币 vs 实盘）
CacheProvider             缓存管理（3 层缓存统筹）
```

每个 Provider 通过 YAML 配置文件切换实现（backtest vs live）：

```yaml
# config/providers-backtest.yaml
MarketDataProvider:
  implementation: ParquetMarketData
  path: /data/ohlcv/

# config/providers-live.yaml
MarketDataProvider:
  implementation: AlpacaMarketData
  api_key: ${ALPACA_API_KEY}
  fallback: ParquetMarketData  # 缓存回源
```

**3 层缓存策略**：L1（内存，热数据），L2（Parquet 磁盘，周期数据），L3（远程 API，更新数据）。多数查询止于 L1/L2，仅在无缓存时触发 API。→ 参考：research-provider-design.md §2-§5

---

## 9. 存储容量总估算

| 类型 | 计算方式 | 容量 |
|------|---------|------|
| 股票日线 OHLCV | 4000只 × 5年 × 252交易日 × 6列（OHLCVA）× 4字节 | 250 MB |
| Alpha158 技术因子 | 158因子 × 4000只 × 5年 × 252天 × 4字节 | 1.3 GB |
| 基本面+情绪融合因子 | 50个因子 × 4000只 × 5年 × 252天 × 4字节 | 200 MB |
| 新闻数据 | 50篇/天 × 365天 × 5年，含标题(200B) + 摘要(500B) + 情绪(50B) | 100 MB |
| 宏观指标 | 19指标 × 5年 × 252天 × 8字节 | 200 KB |
| 知识图谱（宏观+公司） | NetworkX 内存 + CSV 备份 (~130 节点, ~570 边) | < 1 MB |
| 经验回放缓冲 | 50K transition × 0.7KB | 35 MB |
| 模型权重缓存 | DQN/PPO/SAC 三个模型，每个 5-50MB | 100 MB |
| **总计** | | **< 2 GB** |

考虑到实际压缩率和增长，建议分配 5GB SSD 空间。→ 参考：research-stock-data-storage.md §4
