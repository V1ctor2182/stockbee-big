# Stockbee 技术设计文档

**版本**: 2.1
**日期**: 2026-03-24
**作者**: Stockbee 研发团队

---

## 1. 系统架构总览

Stockbee 是一个 AI 驱动的量化交易系统，核心特点是 **LLM Agent 作为最终决策编排者**，整合宏观分析、多因子评分、强化学习组合优化、LLM 新闻分析，经过验证层后输出交易建议，人工审批后执行。

**设计哲学**：系统采用 Provider 抽象模式（借鉴 Qlib 的五大核心 Provider），允许数据源在不同环境下（回测/实盘）灵活切换，同时通过多层缓存最小化 API 调用成本。分层设计保证了各模块独立性，支持增量迭代。→ 参考：research-provider-design.md §1-§2

```
┌─────────────────────────────────────────────────────────────────┐
│                      外部数据源                                   │
│  (Alpaca API / Yahoo Finance / NewsAPI / FRED / Polygon.io)     │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      数据层 (Data Layer)         │
        │  SQLite (WAL) | Parquet | Buffer │
        │  知识图谱 (NetworkX) | 10 Provider│
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    特征层 (Feature Layer)        │
        │  因子计算引擎 | 表达式评估器     │
        │  MacroTiltEngine | 传播引擎      │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    模型层 (Model Layer)          │
        │  LLM 路由 | 小模型 | RL Agent   │
        │  DQN+PPO | SAC+Ensemble        │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   LLM Agent 决策层               │
        │  ┌─────────────────────────┐     │
        │  │ 验证层 (CRITICAL)        │     │
        │  │ Schema + 风控 + 异常检测 │     │
        │  └─────────────────────────┘     │
        │  周度再平衡 | 实时监控 | 审批    │
        │  决策日志（输入/输出全记录）      │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │   执行层 (Execution Layer)       │
        │  Alpaca API | 风控 | 日志记录   │
        │  幂等执行 | 防守机制             │
        │  做空模式: 融券/反向ETF/仅做多   │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │  仪表板与监控 (Dashboard)        │
        │  训练可视化 | 交易历史 | 告警   │
        │  实时 WebSocket | 审批界面      │
        └─────────────────────────────────┘
```

**相比 v1.0 新增** (CEO Review 发现):
- 决策层新增验证层（Schema + 风控约束 + 异常检测 + Prompt Injection 防护）
- 决策日志（所有 LLM 输入/输出全记录，支持事后审计）
- 数据层 SQLite 使用 WAL 模式（支持多进程并发）
- 执行层做空模式可配置（融券 / 反向 ETF / 仅做多）

---

## 2. 数据层设计

Stockbee 数据层采用分阶段存储策略，通过精心设计的漏斗型筛选流程，从 8000+ 个全球上市公司逐步筛选至 100 只候选股，最后锁定 8 只多头+1 只空头。这一设计避免了直接从 U100 开始带来的幸存者偏差。→ 参考：research-stock-data-storage.md §1

### 2.1 股票数据存储与分级架构

**四层漏斗设计**（8000→4000→500→100→8）：

第一层（广域宇宙，8000只）：每周从 Alpaca API 同步全部美股清单，包含流动性指标和融资可用性标志。用于监控市场变化。

第二层（广泛宇宙，约4000只）：筛选条件为流动性>100万美元/日、平均价格>2美元、非 OTC 市场。存储在 SQLite 的 broad_universe 表中，字段包括 ticker、sector、market_cap、avg_volume、short_able。此层作为因子计算的基础集合。→ 参考：research-stock-data-storage.md §2-§3

第三层（候选池，约500只）：从广泛宇宙按行业均衡和流动性顶部筛选。仅这一层及以上进行日线数据同步和因子计算。

第四层（U100，100只）：月度重新评估，基于因子综合评分和绩效指标。→ 参考：research-stock-data-storage.md §3

**为什么分层而不是直接从 U100 开始？** 如果直接观察历史 U100，会产生幸存者偏差。通过实时维护所有层级，系统可在回测时模拟真实的选股流程，而不是在历史数据中拣选。→ 参考：research-stock-data-storage.md §1

### 2.2 存储技术选型与缓存策略

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

### 2.3 新闻数据处理管道

news_events 表的完整设计（已详见 2.2，这里重申处理流程）：

所有新闻进入 G1 快速过滤：检查时间戳、数据源合法性、去重（同一新闻多个来源）。通过 G1 的新闻进入 SQLite 持久化，同时触发 G2 LLM 分类（GPT-4.1 nano，<$1/月）。G2 产生：情绪（正/中/负）、主题分类、紧急度评分。可选的 G3 由 Claude Sonnet 4 执行深度分析（仅限分数>70 的新闻，日限 10 篇，<$5/月）。→ 参考：research-news-storage.md §3-§4, research-llm-selection.md §3

### 2.4 宏观数据与 MacroTiltEngine

19 个 FRED 指标分布在 8 个经济维度：利率（3个：10Y, 2Y, FFR）、就业（3个：NFP, 失业率, 初请）、物价（2个：CPI, PPI）、增长（2个：GDP, 工业产值）、信用（2个：高收益差, 杠杆率）、流动性（2个：M2, VIX）、汇率（1个：DXY）、商品（2个：原油, 黄金）。

每个指标实时更新至 Parquet 时间序列文件，同时在 SQLite 中维护最新值及其 Z-score（相对 252 天历史的标准化）。系统采用点对点对齐技术避免前看偏差：因子在日期 T 的计算仅使用 T 前一个交易日的宏观数据。→ 参考：research-macro-features.md §2, research-macro-features.md §3

### 2.5 因子存储与 IC/ICIR 评估

因子分两类存储：

**表达式因子（技术因子）**：来自 Qlib 的 Alpha158 标准因子集。使用自研表达式引擎动态计算，支持 OHLCV、成交量加权、移动平均、相对强度等操作。公式存储在配置文件中，每日闭盘后重新计算。优势：灵活调整因子定义，无需重新处理历史数据。→ 参考：research-factor-storage.md §1

**预计算因子（融合因子）**：基本面因子（PE、PB、ROE）、情绪因子（FinBERT 得分）、ML 预测因子（LightGBM 收益预测）、宏观倾斜得分（MacroTiltEngine 输出）。这些无法用纯表达式表示，因此预计算后存储在 Parquet 中。每周更新一次。→ 参考：research-factor-storage.md §2-§3

**IC/ICIR 评估管道**：系统维护滚动窗口（252 日）的 IC（因子与收益的相关性）和 ICIR（IC 的信息比率）。在日终计算所有因子的 IC，当某个因子的 ICIR 从正转负时，自动禁用或降低其权重。这确保了仅有效的因子在组合中发挥作用。→ 参考：research-factor-storage.md §4

### 2.6 经验回放缓冲区

Ring Buffer 存储 RL 训练的 transition（状态→动作→奖励→下一状态）。容量 50K，每条 transition 占用：state(74D float32) + action(8D or 100D) + reward(1D) + next_state(74D) + done(1D bool) ≈ 0.7KB，总计 ≈35MB，完全驻留内存。

Buffer 采用 FIFO 覆盖策略：当满时，新的 transition 覆盖最旧的数据。支持可选的 Priority Experience Replay（PER）以强调高 TD 误差的样本，留待 Phase 2 实现。→ 参考：research-experience-buffer-20260317.md §2-§3

### 2.7 知识图谱与传播引擎

Stockbee 采用**两层知识图谱**架构，实现"一个事件进入，影响沿图谱传播"的信号放大机制。→ 参考：research-knowledge-graph-20260324.md

**宏观指标传播图**：~30 个节点（Fed Rate、10Y Treasury、CPI、失业率、油价、DXY 等）、~70 条有向边（因果关系，含极性、强度 0~1、时滞天数）。以 NetworkX DiGraph 常驻内存，CSV 备份。当某指标发布异常数据（Z-score > 1.0）时，触发传播引擎：沿因果链计算各下游节点的冲击信号（`impact = z × polarity × strength`），衰减因子 0.5，最大深度 3 层。传播终端映射到 GICS 行业 Tilt，与现有 MacroTiltEngine 的规则 Tilt 融合（0.6 规则 + 0.4 传播）。→ 参考：research-knowledge-graph-20260324.md §2, research-macro-features.md

**公司关系传播图**：U100 的 100 个公司节点、~500 条边（供应链、竞争、同行业、母子、合作）。以 NetworkX Graph 常驻内存，CSV 备份。当 G2 新闻分析输出某公司的情绪和重要性评分后，传播引擎沿关系边计算影响：供应链同向传播、竞争反向传播、同行业弱正传播。每家公司获得 propagation_score，作为新因子参与评分：`score_final = factor_score + 0.1 × propagation_score + 0.2 × tilt_sector + 0.1 × tilt_style`。→ 参考：research-knowledge-graph-20260324.md §3, research-entity-relationships-20260317.md §1

**传播日志**：每次传播事件记录到 SQLite propagation_log 表（timestamp、trigger_node、trigger_event、affected_node、impact_score、propagation_path JSON），用于审批时展示传播路径、回测传播信号准确率。→ 参考：research-knowledge-graph-20260324.md §5

**流动性指标**：每只股票维护历史流动性数据（日均成交量、成交额、买卖价差），存储在 Parquet（历史）和 SQLite（最新）。用于权重分配时确保 8 只多头+1 只空头的流动性满足最低要求（日均成交额 >50 万美元）。→ 参考：research-entity-relationships-20260317.md §3

### 2.8 Provider 架构与多实现切换

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

### 2.9 存储容量总估算

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


---

## 3. 模型层设计

### 3.1 多模型架构概览

系统采用分离式多模型设计，将离散选股（DQN）与连续权重分配（PPO/SAC）分离。这样做的原因是：选择 8 只股票本质上是组合优化中的离散问题，而权重分配是连续优化。分离后，DQN 可专注于选股，PPO/SAC 可专注于权重分配，各自达到更优的收敛性。→ 参考：research-rl-algorithms.md §2.1

```
新闻源 → [LLM 路由器]
            ├─→ G1 快速过滤 (GPT-4.1 nano, <$1/月)
            ├─→ G2 分类 (Claude Sonnet 4, <$2/月)
            └─→ G3 深度分析 (GPT-4.1, <$5/月，可选)

因子输入层：
  ├─ OHLCV 数据 → [Alpha158 技术因子]
  ├─ 新闻文本 → [FinBERT 情绪分类]
  ├─ 重要性信号 → [Fin-E5 嵌入]
  ├─ 宏观指标 → [MacroTiltEngine → 倾斜因子]
  └─ 历史收益 → [LightGBM 收益预测]
                           ↓
                    [因子融合器]
                    XGBoost 或等权
                           ↓
                    综合评分 (100维)
                           ↓
                   [RL 智能体]
                    ├─→ DQN (离散选股: 100→8)
                    ├─→ PPO/SAC (连续权重: Σ=1)
                    └─→ Ensemble (Phase 2: 应对制度切换)
                           ↓
                    [宏观倾斜覆盖层]
                    Soft overlay (±3%)
                           ↓
                    最终投资组合 (8L+1S)
```

→ 参考：research-rl-algorithms.md §1-§4

### 3.2 LLM 多模型路由策略

**核心设计理念**：不同的 NLP 任务对模型能力要求差异大。G1 只需布尔判断（相关/无关），用廉价模型足够；G2 需要细致的情绪/主题判断，用中档模型；G3 需要深度推理，才用最强模型。这样成本从 $50+/月（单一 GPT-4）降到 $15/月，同时准确率无显著差异。→ 参考：research-llm-selection.md §3

| 任务 | 选择模型 | 备选模型 | 月成本 | 输入规模 | 输出格式 |
|------|---------|---------|--------|---------|---------|
| G1 新闻快速过滤 | GPT-4.1 nano | Gemini Flash | < $1 | 标题+摘要 (100-300 tokens) | JSON: {relevant: bool} |
| G2 新闻分类 (情绪/主题) | Claude Sonnet 4 | GPT-4o mini | < $2 | 摘要+部分正文 (500-1k tokens) | JSON: {sentiment, category, urgency} |
| G3 深度基本面分析 | GPT-4.1 | Claude Opus 4 | < $5 | 完整新闻+持仓情景 (2-5k tokens) | Markdown：影响评估 |
| SEC 10-K/10-Q 解析 | GPT-4.1 (long context) | Claude Sonnet 4 | < $3 | 完整 10-K (60k tokens) | JSON：关键财务指标 |
| 宏观环境分析 (周度) | Claude Sonnet 4 | GPT-4.1 | < $4 | 19 个 FRED 指标 + 描述 (2k tokens) | Markdown：经济周期判断 |
| **月度合计** | | | **< $15** | ~2.5M 输入 + 420K 输出 token | 远低于 $200 预算 |

**降级链设计**：
```
主模型失败 → 备选模型重试（最多 2 次）
           → 备选失败 → 缓存上次成功响应
           → 缓存也失败 → 禁用该信号（保守处理）
```

**成本优化技巧**：
- 使用 OpenAI Batch API：G2/G3 文本打包为夜间批处理，节省 50%
- Prompt Caching：相同背景信息（如持仓列表）缓存，多个请求共用
- 函数调用而非 Markdown：LLM 以 JSON 返回结构化数据，减少 token 消耗

→ 参考：research-llm-selection.md §4-§5

### 3.3 小模型部署体系

所有小模型本地部署，无 API 依赖，支持 GPU 加速推理。

**FinBERT（情绪分类）**：微调的 BERT 模型，专为金融文本优化。输入：新闻标题+摘要（max 128 tokens），输出：[positive, negative, neutral] 三分类 + 置信度。推理延迟 <100ms。训练数据来自 Lexicon 与弱监督标签（新闻发布日期与股价涨跌的关联）。→ 参考：新闻小模型.md §1, finbert.md

**Fin-E5（重要性评估）**：嵌入模型（类似 E5），任务是区分"哪些新闻真正影响股价"。输入：完整新闻正文，输出：0-1 重要性分数。训练目标：最大化新闻嵌入与后 5 日涨幅的相似度。→ 参考：新闻小模型.md §2, fine5.md

**LightGBM（收益预测）**：梯度提升决策树，输入 Alpha158 因子（158维），输出：未来 5 日平均收益率（连续值）。在历史回测数据上训练（5 年 × 252 天），采用滚动窗口避免前看偏差。推理延迟 <50ms。为什么选 LightGBM 而非 XGBoost？Qlib 基准测试表明 LightGBM 在金融数据上收敛更快，IC 持久性更好。→ 参考：量化小模型.md §1, qlib 因子挖掘和量化模型.md

**XGBoost（因子融合）**：可选的非线性融合器。输入：6-8 个预计算因子（技术因子、情绪、基本面、宏观倾斜），输出：综合因子分数。相比简单等权，XGBoost 融合可根据市场体制动态调整因子权重，提升 Sharpe 率约 5-10%。→ 参考：量化小模型.md §2-§3

所有模型均支持定期 refit（每周重新训练），以适应市场制度变化。

### 3.4 强化学习双阶段训练

**阶段 1（MVP）：离散选股 + 连续权重**

*环境定义*：Gymnasium StockbeeEnv，两个独立子环境：
- **选股环境**：State 为 U100（100只候选）的各项因子评分，Action 为选择 8 只的离散决策（100 choose 8 = 1.76e14 种），Reward 为这 8 只的后续周平均收益率。
- **权重环境**：State 为已选 8 只的收益率、波动率、相关性矩阵，Action 为权重分配 (w1,...,w8, Σ=1），Reward 为投资组合收益减去风险成本。

*算法选择*：
- **选股用 DQN（离散优化）**：Q 值网络学习 U100 中每只股票的"选中价值"。Action space = 100（选/不选任意一只），远小于组合优化的指数空间。DQN 通过目标网络和经验回放稳定学习。→ 参考：research-rl-algorithms.md §2.1
- **权重用 PPO（连续优化）**：Policy 网络直接输出 8 个权重（softmax 确保和为 1），PPO 通过 Clipped Surrogate Loss 避免过大的策略更新。对比 A3C，PPO 收敛更稳定。→ 参考：research-rl-algorithms.md §2.2

*数据与评估*：
- 训练数据：5-10 年历史回测数据，周度再平衡频率
- 环境反馈：每周重新训练（Sunday 10:00），基于过去 60 周的实现收益
- 评价指标：Sharpe 率、Sortino 率、最大回撤、年化收益 vs 等权 U100 基准
- 反过拟合：Walk-Forward 评估（详见 research-rl-practical-guide-20260317.md §2），滚动窗口训练与测试

*硬件需求*：RTX 3070（8GB VRAM）约 16 小时完整训练（从零开始）。增量训练（周度微调）约 2 小时。→ 参考：research-rl-algorithms.md §3

**阶段 2（增强）：自适应与集合**

- **PPO → SAC**：SAC（Soft Actor-Critic）是 Off-Policy 算法，数据效率更高。当新闻事件触发快速权重微调时（Tier-1 信号），SAC 可在少量样本下快速适应。→ 参考：research-rl-algorithms.md §2.3
- **集合策略**：维护 3 个策略的异构集合（PPO + A2C + SAC），当市场体制切换时（如 VIX 从低到高），自动选择历史在该体制下表现最优的策略。集合权重基于滚动 Sharpe 率动态调整。
- **实时微调**：G3 新闻若预测大幅影响，触发权重微调（±5%），使用 SAC 的数据效率快速响应，然后在周日完整训练时重置。

→ 参考：research-rl-practical-guide-20260317.md §3-§5

### 3.5 宏观分析引擎（MacroTiltEngine）

**19 个 FRED 指标的经济分类**：

| 经济维度 | FRED 代码 | 含义 | 用途 |
|---------|---------|------|------|
| 利率 | DFF / T10Y2Y / DGS10 | Federal Funds Rate / 2-10Y 期限差 / 10Y 美债收益率 | 投资者风险偏好指标 |
| 就业 | PAYEMS / UNRATE / ICSA | 非农就业人数 / 失业率 / 初请周数 | 劳动力市场强度 |
| 物价 | CPIAUCSL / PPIACO | CPI / PPI | 通胀压力 |
| 增长 | A191RA1Q225SBEA / INDPRO | 实际 GDP | 经济增速 |
| 信用 | BAMLH0A0HYM2 / TCMILRSL | 高收益债差 / 杠杆率（衍生） | 信用风险 |
| 流动性 | M2 / VIXCLS | 货币供应 / VIX | 市场风险偏好 |
| 汇率 | DEXUSEU | USD/EUR | 贸易敏感度 |
| 商品 | DCOILWTICO / GOLDAMND | 原油 / 黄金现货 | 商品通胀预期 |

**Z-score 标准化**：每天计算所有指标相对 252 天历史的 Z-score，范围 [-3, +3]。正值表示当前指标高于历史平均，负值表示低于。

**经济周期分类**：根据 5 个关键指标的 Z-score 组合，分类为 5 种经济制度：

```
扩张期：GDP↑ + NFP↑ + 失业率↓ → 科技+周期股表现优，防守股滞后
过热期：CPI↑↑ + 利率↑ → 价值股防御，成长股承压
滞胀期：CPI↑ + GDP↓ → 分化严重，黄金+防守股优
衰退期：失业率↑ + GDP↓ → 极度风险规避，债券优
复苏期：GDP 底部 + 预期上升 → 周期股领导，防守滞后
```

**倾斜输出**：

- **sector_tilt**：11 个 GICS 行业对宏观指标的敏感度矩阵（预训练，存于 SQLite）。例如能源对油价敏感度 0.8，对利率敏感度 0.3。MacroTiltEngine 对每个行业计算加权倾斜分数，±3% 的权重调整。
- **style_tilt**：价值 vs 成长倾斜。利率高时倾斜价值（相对权重 +3%），反之倾斜成长。

这些倾斜作为"软覆盖层"施加在 RL Agent 的原始选股上，而非硬约束，保留 Agent 的灵活性。→ 参考：research-macro-features.md §3-§5

---

## 4. 信息源与新闻处理设计

Stockbee 的新闻处理采用多源融合策略，确保信息的广度、深度和权威性。→ 参考：信息.md §1-§3

### 4.1 新闻数据源架构

| 来源 | 覆盖范围 | 更新频率 | 成本 | 用途 |
|------|---------|---------|------|------|
| NewsAPI | 全球新闻聚合（70k+ 来源） | 实时 | 免费（~500 条/天 limit） | 广泛扫描，G1 过滤基础 |
| Perigon | 金融专业媒体聚合 | 实时 | 按用量（~$100-500/月） | 金融新闻深化，高信噪比 |
| SEC EDGAR | 8-K 速报、10-K/Q | 日度 | 免费 API | 权威基本面信息，无延迟 |
| Nasdaq RSS | 交易暂停、熔断、除牌 | 实时 | 免费 | 流动性事件 |
| FactSet/Bloomberg（未来扩展） | 分析师目标价、评级变化 | 日度 | 专业版 | 机构预期 |
| Prediction Market（Polymarket） | 宏观事件概率 | 实时 | 免费 | 市场共识的制度变化预测 |

→ 参考：信息.md §1, polymarket的数据如何运用.md

### 4.2 新闻摄入与三级处理漏斗（G1/G2/G3）

**G1（快速过滤，95% 淘汰）**：
- 来源合法性检查（黑名单 domain）
- 发布时间检查（避免重复旧新闻）
- 去重（多源相同新闻只保留一条）
- 实体识别（检测新闻中的股票 ticker，支持模糊匹配）
- 输出：news_events 表写入，g_level = 'G1'

**G2（智能分类，50-70% 淘汰）**：
触发条件：所有通过 G1 的新闻。

成本极低（GPT-4.1 nano，<$1/月），延迟 <5 秒。

- FinBERT 情绪分类：positive/neutral/negative
- 主题自动分类：收益、合规、并购、诉讼、政策、产品等
- 相关性打分：该新闻与当前持仓的相关度（0-1）
- 紧急度评分：high/medium/low（基于主题和持仓覆盖度）
- 输出：news_events.sentiment_score, importance_score, g_level = 'G2'

**G3（深度分析，10-30 篇/天）**：
触发条件：
- importance_score > 0.7 且 urgency = high，或
- 新闻涉及当前持仓且 sentiment_score 为极端（>0.9 or <0.1）

频率上限：每天最多 10 篇。

由 Claude Sonnet 4 执行：
- 完整阅读全文
- 估算对持仓价值的影响（±X%）
- 建议权重调整或风险对冲
- 评估新闻真伪（是否为讹传）
- 输出：news_events.analysis 字段，g_level = 'G3'，推送给 Victor 审批

月成本 <$5。→ 参考：research-news-storage.md §3-§4

### 4.3 Prediction Market 应用

Polymarket 等去中心化预测市场提供"群体智慧"式的事件概率。例如"美联储 2026 Q2 降息"的隐含概率。

Stockbee 使用方式：
- 定期爬取 Polymarket 前 30 大宏观事件的隐含概率
- 与前期概率对比，检测"概率悬崖"（突然大幅变化）
- 将概率作为 MacroTiltEngine 的补充因子，调整行业倾斜
- 例如降息概率从 20% 跳升至 70%，则增加高 duration 金融股的权重

优势：市场驱动，免费，实时性强。→ 参考：polymarket的数据如何运用.md

---

## 5. 决策层设计

### 5.1 周度再平衡流程 (Friday Close → Monday Open)

```
周五收盘
  │
  ├─→ 数据更新模块
  │   ├─ 日线 OHLCV (Alpaca Historical)
  │   ├─ 新闻摄入 (NewsAPI)
  │   └─ 宏观指标 (FRED latest)
  │
  ├─→ 因子计算模块
  │   ├─ 技术因子 (表达式引擎)
  │   ├─ 情绪因子 (FinBERT)
  │   └─ 宏观因子 (FRED 组合)
  │
  ├─→ RL 推理模块
  │   ├─ DQN 推理 (100→8 选股)
  │   └─ PPO/SAC 推理 (权重分配)
  │
  ├─→ 宏观倾斜覆盖层 (Soft Overlay)
  │   └─ 根据宏观景气调整权重 (±3%)
  │
  ├─→ 约束检查模块
  │   ├─ 行业集中度上限 (per sector cap)
  │   ├─ 单个持仓上限 (20%)
  │   └─ Beta 中性性 (target β < 0.1)
  │
  ├─→ 生成建议投资组合
  │   └─ 8 只多头 + 1 只空头
  │
  ├─→ 人工审批 (Dashboard)
  │   ├─ 展示逻辑：当前 vs 建议 (持仓变化、收益预期)
  │   ├─ Victor 可视化审查
  │   └─ 批准/拒绝/手动覆盖 (无超时)
  │
  └─→ 周一开盘交易执行 (Alpaca API)
      ├─ 生成平仓单 (旧持仓)
      └─ 生成建仓单 (新持仓)
```

### 5.2 实时监控流程 (Phase 2 - Fast Chain)

```
新闻/价格 Feed
  │
  ├─→ G1 快速过滤
  │   └─ 主题相关性 (相关/无关)
  │
  ├─→ G2 分类 (LLM)
  │   ├─ 情绪 (正面/中性/负面)
  │   ├─ 类别 (收益预警/拆分/诉讼等)
  │   └─ 紧急度 (低/中/高)
  │
  ├─→ G3 深度阅读 (可选，高紧急度)
  │   └─ Claude 深度分析 (仓位影响评估)
  │
  ├─→ 信号评分
  │   ├─ 情绪 × 重要度 × 紧急度 → 综合分数
  │   └─ 分级 (Tier-1/2/3)
  │
  ├─→ Tier-1 (自动执行，无审批)
  │   ├─ 条件：分数 > 85 且事件明确
  │   └─ 动作：±5% 权重微调或平仓应对
  │
  ├─→ Tier-2 (待审批，15分钟超时)
  │   ├─ 条件：分数 60-85
  │   ├─ 推送告警给 Victor
  │   └─ 无响应后自动拒绝
  │
  └─→ Tier-3 (观察列表)
      └─ 记录但不触发交易
```

防守机制：价格下跌 > 3σ (超过 ATR × 1.5) 时，自动平仓所有多头持仓（不需审批）。

### 5.3 审批工作流

- **周度再平衡**：推送通知 → Victor 审查逻辑 (展示收益预期、持仓变化对比) → 批准/拒绝/覆盖 → 无超时限制
- **Tier-2 加速信号**：告警 → 15分钟审批窗口 → 无响应自动拒绝 → 告警通知
- **权限设计**：Victor 为唯一审批者，防止越权交易

→ 参考：research-rl-algorithms.md §4（Phase 1 推荐）

---

## 6. 执行层设计

### 6.1 交易执行策略

- **Broker**：Alpaca API (免佣，API 友好)
- **执行方式**：
  - 周度再平衡：周一开盘市价单
  - 实时信号：立即执行（Tier-1）或待命（Tier-2）
  - 应急防守：立即市价平仓
- **幂等性**：每个交易任务生成唯一 order_id，支持重试而不重复下单
- **融资管制**：
  - 多头：直接持仓
  - 空头：使用反向 ETF 代理 (例 TQQQ/SQQQ) 或融券（如可用）

### 6.2 风险控制机制

| 风控措施 | 阈值 | 触发条件 | 动作 |
|---------|------|--------|------|
| 尾部止损 | ATR × 1.5 | 单个持仓下跌超过阈值 | 自动平仓 |
| 应急加速上限 | 2次/天 | Tier-1 信号触发加速 | 超过上限则停止 Tier-1 |
| 加速冷却期 | 30分钟 | 每次 Tier-1 执行后 | 进入冷却期禁止新的 Tier-1 |
| 单个持仓上限 | 20% | 任何时刻 | 强制上限，不可超越 |
| 行业集中度 | 3只/行业 | 任何时刻 | 防止行业过度集中 |
| 日度交易金额 | $50k | 应急措施 | 防止一天内过度交易 |

### 6.3 日志与审计

每笔交易记录：
- 执行时间、order_id、持仓变化
- 触发来源 (周度/Tier-1/Tier-2/防守)
- 信号得分、最终权重
- 成交价格、手续费、滑点估计

---

## 7. 基础设施设计

### 7.1 硬件配置

| 组件 | 规格 | 用途 |
|------|------|------|
| GPU | NVIDIA RTX 3070 (8GB VRAM) | RL 训练 + 小模型推理 |
| CPU | 16核 AMD Ryzen / Intel | 因子计算 + 数据处理 |
| RAM | 32GB | 数据缓存 + 训练缓冲 |
| 存储 | 500GB NVMe SSD | 数据库 + Parquet + 模型权重 |
| 网络 | 稳定有线网络 | API 连接 (Alpaca/LLM/新闻) |

### 7.2 技术栈选型

| 层级 | 技术 | 为什么选择 |
|------|------|----------|
| 编程语言 | Python 3.11+ | 最完整的量化交易生态 | research-rl-algorithms.md |
| RL 框架 | SB3 (开发) → ElegantRL (生产) | 易用过渡到稳定性能，Phase 1→2 | research-rl-algorithms.md §3 |
| 数据存储 | Parquet + SQLite | 轻量、无需额外服务、压缩高效 | research-stock-data-storage.md §2 |
| 因子引擎 | 自研表达式解释器 + Qlib 风格 | 灵活性强，支持动态修改公式 | research-factor-storage.md §1 |
| 小模型框架 | PyTorch (HuggingFace) | 推理优化成熟，GPU 加速 | 新闻小模型.md, 量化小模型.md |
| LLM 调用 | OpenAI/Anthropic SDK | 多模型路由、自动重试、降级链 | research-llm-selection.md §5 |
| Broker API | Alpaca (alpaca-trade-api) | 稳定、低延迟、支持融资 | research-stock-data-storage.md |
| 任务调度 | APScheduler + cron | 简单轻量，适合单机 | research-rl-practical-guide-20260317.md |
| 可视化/监控 | TensorBoard (训练) + Plotly (Dashboard) | 实时动态展示、WebSocket 推送 | - |
| 版本控制 | Git + 模型日期标签 | 简单可追溯、支持模型回滚 | - |

### 7.3 部署架构 (单机)

```
Host Machine
  ├─ 进程 1：Scheduler (cron 触发)
  │   ├─ 每日 18:00 → 数据更新任务
  │   ├─ 周五 16:00 → 周度再平衡流程
  │   └─ 周日 10:00 → RL 模型训练 (离线)
  │
  ├─ 进程 2：实时监控服务 (Phase 2)
  │   ├─ 持续监听新闻源
  │   ├─ 价格告警 (3σ 检测)
  │   └─ WebSocket 推送（仪表板）
  │
  ├─ 进程 3：Web Dashboard (FastAPI)
  │   ├─ 当前持仓展示
  │   ├─ 周度再平衡审批
  │   ├─ 实时交易历史
  │   └─ 模型性能看板
  │
  └─ 共享资源
      ├─ SQLite 数据库 (file lock)
      ├─ Parquet 数据目录
      └─ 模型权重缓存
```

无需容器化（初期），虚拟环境 + 系统 cron 充足。

---

## 8. 预算设计

### 8.1 月度运营成本

| 项目 | 月度预算 | 说明 |
|------|---------|------|
| LLM API 调用 | $15 | G1(1) + G2(2) + G3(5) + SEC(3) + Macro(4)，使用 Batch API 优化 | research-llm-selection.md §4 |
| 新闻数据源 | $0-30 | NewsAPI 免费层 + Perigon 按用量 | 信息.md §1 |
| 市场数据 | $0-20 | Alpaca 免费 + Polygon.io 基础版 | research-stock-data-storage.md |
| GPU 电力 | ~$10 | RTX 3070 周日训练约 20 小时 | research-rl-algorithms.md §3 |
| **合计** | **< $75** | 远低于 $200 月预算上限 | research-llm-selection.md §4 |

### 8.2 成本降级策略

| 等级 | 触发点 | 措施 | 影响 |
|------|--------|------|------|
| 绿色 (< $75) | - | 正常运营，全功能 | 无 |
| 黄色 (70-85%) | 接近 $150 | FMP → Yahoo fallback, 减少 G3 频率 | 基本面数据延迟 |
| 橙色 (85-95%) | 接近 $190 | 禁用 G3 深度分析，仅用缓存宏观数据 | 新闻深度分析停止 |
| 红色 (> 95%) | 接近 $200 | 禁用 Tier-2 自动加速，最小化运营 | 仅保留周度再平衡 |

---

## 9. 安全与风控设计

### 9.1 密钥与认证

- API 密钥存储在 `.env` 文件，从不写入代码
- 生产环境使用环境变量或密钥管理服务
- 账户密码从不涉及自动交易流程

### 9.2 交易权限

- **审批工作流**防止越权：所有 Tier-2 及以上需人工批准
- **周度再平衡**无自动执行权，必须人工批准后才能下单
- **Tier-1 自动执行**受严格条件限制（分数 > 85，且仅在有人监控时启用）

### 9.3 持仓限制

- 单个持仓上限：20% (硬约束)
- 行业集中度：最多 3 只/行业 (硬约束)
- 日度交易量：$50k (应急情况下的上限)

---

## 10. 扩展性与模型迭代

### 10.1 核心扩展点

**Provider 模式的灵活性**：易于更换数据源（例 Yahoo → CRSP、Alpaca → Interactive Brokers），只需修改 Provider 实现类而无需改动上层逻辑。→ 参考：research-provider-design.md §5

**模型版本控制**：每次 RL 训练产生日期标签版本（e.g., model-20260324-dqn.pt），支持快速回滚与 A/B 对比测试。维护一个版本清单，记录每个版本的 Sharpe 率、最大回撤、样本外表现。

**模块化因子体系**：Alpha158 技术因子集可增减，新增因子自动计入 IC 评估。融合器（XGBoost）权重动态调整，无需重新训练。

**多策略并行框架**：当前 Long 7 + Short 1 框架可扩展至：
- Long-only 版本（无空头，降低风险）
- 配对交易（同行业多空对冲）
- 海外市场版本（同代码库，不同 Universe Provider）

→ 参考：research-rl-algorithms.md §4

### 10.2 持续改进流程

1. **每周回测评估**：日终计算持仓的实现收益率，对比 RL Agent 预测，计算预测误差与新的 IC。
2. **月度因子评审**：禁用 ICIR < 0 的因子，新增待测因子，更新 MacroTiltEngine 的行业敏感度矩阵。
3. **季度模型重训**：新数据进入，历史回测窗口扩大，重新训练 DQN+PPO，对比历史最优模型。
4. **年度架构评估**：是否切换至 SAC+Ensemble？是否扩展到新市场或策略？基于过去 12 个月的表现和市场条件变化。

---

## 11. 关键技术决策记录 (ADR)

本节详细记录 Stockbee 架构中的所有重大技术决策，每个决策都基于研究团队对权衡因素的评估。

| 决策 | 最终选择 | 备选方案 | 核心权衡 | 参考文档 |
|------|---------|---------|--------|---------|
| **股票数据存储格式** | Parquet + SQLite | Qlib .bin + HDF5 | Parquet：通用性强、社区生态广、列式压缩（8:1 率）、支持部分列读取。vs Qlib .bin：快速但格式私有、难以跨语言使用、占用空间大。决策：Parquet 更符合长期可维护性。 | research-stock-data-storage.md §2 |
| **股票池分层策略** | 四层漏斗（8000→4000→500→100→8） | 直接从 U100 开始 | 四层漏斗允许在回测中模拟真实选股流程，避免幸存者偏差。直接 U100 会导致历史优化偏差（今天的 U100 在 5 年前可能不存在）。 | research-stock-data-storage.md §1, §3 |
| **Provider 架构** | 10 个 Provider + Registry | 直接 API 调用 + 单例模式 | Provider + Registry 模式解耦数据源，支持 backtest/live 快速切换（仅改配置）。单例模式会将 API 细节混入业务逻辑。 | research-provider-design.md §2-§5 |
| **因子存储混合策略** | 表达式引擎（技术因子）+ Parquet（预计算因子） | 纯表达式引擎 | 技术因子（Alpha158）可用表达式动态计算，灵活度高。但基本面、情绪、ML 预测因子无法用公式表示，必须预计算存储。混合方案既保留灵活性又保证性能。 | research-factor-storage.md §1-§3 |
| **LLM 多模型路由** | 按任务选模型（$15/月） | 单一 GPT-4（$50+/月） | 不同任务对模型能力要求不同：G1 仅需布尔判断用廉价模型，G3 需推理用强模型。路由策略成本降低 70% 同时准确率无显著差异（各用最佳模型）。 | research-llm-selection.md §3-§4 |
| **宏观指标数量** | 19 个 FRED 指标 | 更多指标（30+） | 19 个覆盖 8 大经济维度（利率、就业、物价等），已充分捕捉宏观信息。更多指标反而增加噪音与过拟合风险，IC 没有提升但延迟增加。 | research-macro-features.md §2 |
| **经验回放缓冲区** | 内存 Ring Buffer（50K） | 磁盘存储 / PER | 50K transition ≈ 35MB 完全驻留内存，高速随机访问。PER（优先级回放）留待 Phase 2 优化，初期简单 FIFO 足够收敛。 | research-experience-buffer-20260317.md §3 |
| **RL Phase 1 算法分解** | DQN（选股）+ PPO（权重） | 端到端 PPO / 多任务学习 | 选股（100→8 离散决策）与权重分配（连续优化）是不同性质的问题。分离后各自收敛更快，Action 空间更小，可解释性更好。 | research-rl-algorithms.md §2.1-§2.2 |
| **RL Phase 2 升级路线** | SAC + Ensemble | 继续 PPO / 多智能体 | SAC（Off-Policy）数据效率更高，当新闻触发快速微调时更适合。Ensemble 应对市场制度切换（如 VIX 从低到高）。PPO 单策略适应力不足。 | research-rl-algorithms.md §2.3 |
| **知识图谱存储** | NetworkX + CSV 备份 | Neo4j / SQLite 邻接表 | 两层传播图（宏观 30 节点+70 边，公司 100 节点+500 边）需要高效的图遍历和传播计算，NetworkX 内存图毫秒级完成。CSV 备份方便手动检查和版本控制。Neo4j 过重，SQLite 邻接表查询图遍历不便。 | research-knowledge-graph-20260324.md §5, §7 |
| **情绪分类模型** | FinBERT | FinGPT / 通用 BERT | FinBERT 专为金融文本微调，开箱即用无需额外训练。FinGPT 更强但计算贵、延迟高不适合日间处理。 | 新闻小模型.md §1, finbert.md |
| **收益预测模型** | LightGBM | XGBoost / LSTM | Qlib 基准测试中 LightGBM 在金融数据上表现最优（IC 持久性高），训练快。XGBoost 略慢，LSTM 易过拟合且需大量历史。 | 量化小模型.md §1, qlib 因子挖掘和量化模型.md |
| **新闻信息源多元化** | NewsAPI + Perigon + SEC EDGAR | 单一来源 | 多源互补：NewsAPI 提供广泛覆盖（70k+ 媒体），Perigon 聚焦金融深度，SEC EDGAR 提供权威基本面。单一来源会漏掉关键信息。 | 信息.md §1, §2, §3 |
| **Prediction Market 使用** | Polymarket 事件概率作补充因子 | 不使用 / 只作参考 | Polymarket 提供市场共识的制度变化预测（如降息概率），免费 API，可作为 MacroTiltEngine 的外生输入。与自有预测交叉验证。 | polymarket的数据如何运用.md |
| **成本控制上限** | $200/月 预算 | 无上限 | 即使全用最强/最昂贵模型（Opus 4 处理所有新闻、手动信息源等），月成本也不超 $75。$200 预算为充分安全边际，覆盖峰值 2.5 倍。 | research-llm-selection.md §4 |

### 11.1 决策过程透明度

每项决策的论证：
1. **问题陈述**：技术选项及权衡因素
2. **研究支撑**：引用的实证评估或行业标准
3. **预期效果**：采纳该选项的定量目标（如成本、性能、维护复杂度）
4. **回溯计划**：若该决策出现问题（如 Parquet 查询变慢），如何快速回滚或调整

---

## 文档总结

本设计文档定义了 Stockbee AI 量化交易系统的完整架构，包括：

**数据层**（第 2 节）：四层漏斗股票筛选，Parquet+SQLite 混合存储，10 个 Provider 抽象层支持 backtest/live 切换，3 层缓存最小化 API 成本。总数据量 <2GB。

**模型层**（第 3 节）：多模型 LLM 路由（$15/月），本地小模型体系（FinBERT+Fin-E5+LightGBM+XGBoost），双阶段 RL 训练（Phase 1：DQN 选股 + PPO 权重；Phase 2：SAC+Ensemble），MacroTiltEngine 进行宏观倾斜覆盖。

**信息源**（第 4 节）：多源融合（NewsAPI+Perigon+SEC EDGAR），G1/G2/G3 三级新闻漏斗，Polymarket 事件概率作外生输入。

**决策与执行**（第 5-6 节）：周度再平衡流程，实时 Tier-1/2/3 信号处理，人工审批工作流，完整的风控约束与日志审计。

**基础设施**（第 7 节）：单机部署（RTX 3070），Python 3.11+，SB3→ElegantRL 框架升级路径。

**成本与可持续性**（第 8 节）：月度总成本 <$75，预留成本降级策略应对预算压力，充分的安全边际。

**所有设计决策**（第 11 节）均有研究文献支撑，确保了科学性、可验证性和长期可维护性。系统设计遵循：
- **解耦性**：Provider 模式使数据源可快速切换
- **可追溯性**：模型版本化、决策记录详细
- **渐进性**：MVP→Phase 2 清晰升级路线，避免过度设计
- **成本意识**：多源优化（LLM 路由、缓存策略）将成本控制到极限

Stockbee 已具备从研究到实盘的完整技术栈，为持续迭代和长期运营奠定了坚实基础。
