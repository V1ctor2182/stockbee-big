# Stockbee 技术设计 — 系统架构总览

**版本**: 2.2
**日期**: 2026-04-18
**作者**: Stockbee 研发团队

---

## 1. 设计哲学

Stockbee 是一个 AI 驱动的量化交易系统，核心特点是 **LLM Agent 作为最终决策编排者**，整合宏观分析、多因子评分、强化学习组合优化、LLM 新闻分析，经过验证层后输出交易建议，人工审批后执行。

系统采用 Provider 抽象模式（借鉴 Qlib 的五大核心 Provider），允许数据源在不同环境下（回测/实盘）灵活切换，同时通过多层缓存最小化 API 调用成本。分层设计保证了各模块独立性，支持增量迭代。→ 参考：research-provider-design.md §1-§2

---

## 2. 分层架构

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

---

## 3. 三条 Pipeline

Stockbee 按响应速度将所有处理流程划分为三条 Pipeline：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          STOCKBEE PIPELINE 总览                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗   │
│  ║  SLOW CHAIN — 训练链 (周日离线)                                       ║   │
│  ║                                                                       ║   │
│  ║  Historical     Alpha158      RL Training    Model        Model       ║   │
│  ║  OHLCV+Factors → Feature   →  DQN+PPO/SAC → Evaluation → Checkpoint  ║   │
│  ║  (Parquet)      Engineering   (16h full /    (Walk-Fwd)   (.pt 文件)  ║   │
│  ║                               2h incremental)                         ║   │
│  ║  频率: 每周日 10:00 | 延迟: 小时级 | 成本: GPU 电力 ~$10/月           ║   │
│  ╚═══════════════════════════════════════════════════════════════════════╝   │
│                                        │ 模型权重                           │
│                                        ▼                                    │
│  ╔═══════════════════════════════════════════════════════════════════════╗   │
│  ║  MID CHAIN — 周度决策链 (周五收盘 → 周一开盘)                          ║   │
│  ║                                                                       ║   │
│  ║  Data Sync  → Factor    → RL Inference → Macro Tilt → Constraint     ║   │
│  ║  (OHLCV,      Calc        DQN: 100→8     ±3% Overlay   Check        ║   │
│  ║   FRED,        (Alpha158,  PPO: weights                 (sector,     ║   │
│  ║   News)        FinBERT,    SAC: Phase2                   position    ║   │
│  ║                LightGBM)                                 caps)       ║   │
│  ║       → Portfolio → Human Approval → Execution (Alpaca)              ║   │
│  ║         (8L+1S)     (Victor审批)      (周一开盘市价单)                ║   │
│  ║                                                                       ║   │
│  ║  频率: 每周五 16:00 | 延迟: ~1h 计算 + 人工审批 | 成本: LLM ~$4/周   ║   │
│  ╚═══════════════════════════════════════════════════════════════════════╝   │
│                                                                             │
│  ╔═══════════════════════════════════════════════════════════════════════╗   │
│  ║  FAST CHAIN — 实时响应链 (Phase 2, 交易时段持续运行)                    ║   │
│  ║                                                                       ║   │
│  ║  News/Price → G1 Filter → G2 Classify  → G3 Deep    → Signal Score  ║   │
│  ║  Feed         (去重,       (FinBERT,      (Claude,     (情绪×重要度   ║   │
│  ║  (WebSocket)   实体识别,    GPT-4.1 nano,  可选,        ×紧急度)      ║   │
│  ║               95%淘汰)     情绪/主题)     ≤10篇/天)                   ║   │
│  ║                                                                       ║   │
│  ║       ┌─────────────────────────────────────────┐                     ║   │
│  ║       │ Tier-1 (>85分): 自动执行 ±5% 权重微调   │                     ║   │
│  ║       │ Tier-2 (60-85): 推送Victor, 15min超时   │                     ║   │
│  ║       │ Tier-3 (<60):   仅记录观察列表           │                     ║   │
│  ║       └─────────────────────────────────────────┘                     ║   │
│  ║       防守: 价格 >3σ 自动平仓 (无需审批)                               ║   │
│  ║                                                                       ║   │
│  ║  频率: 实时 | 延迟: G1 <1s, G2 <5s, G3 <30s | 成本: LLM ~$8/月      ║   │
│  ╚═══════════════════════════════════════════════════════════════════════╝   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline 对比

| 维度 | Slow Chain (训练链) | Mid Chain (决策链) | Fast Chain (响应链) |
|------|--------------------|--------------------|---------------------|
| **触发** | 周日 10:00 cron | 周五 16:00 收盘后 | 实时新闻/价格事件 |
| **延迟** | 2-16 小时 | ~1 小时 + 人工审批 | 秒级 (G1) ~ 分钟级 (G3) |
| **目标** | 更新模型权重 | 生成周度投资组合 | 响应突发事件 |
| **输出** | .pt 模型文件 | 8L+1S 投资组合建议 | 权重微调 / 平仓指令 |
| **审批** | 无（离线训练） | Victor 手动审批 | Tier-1 自动 / Tier-2 审批 |
| **Phase** | Phase 1 | Phase 1 | Phase 2 |

---

## 4. 数据源全景

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES 全景图                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─── 市场数据 ────────────────────────────────────────────────────┐     │
│  │                                                                  │     │
│  │  Alpaca API ──────► OHLCV 日线 (4000只×5年)                     │     │
│  │    输入: ticker, date_range                                      │     │
│  │    输出: open, high, low, close, volume, adj_close               │     │
│  │    存储: Parquet (列存压缩)                                      │     │
│  │    → Mid Chain (因子计算), Slow Chain (训练数据)                  │     │
│  │                                                                  │     │
│  │  Alpaca API ──────► 股票池元数据 + 交易执行                      │     │
│  │    输入: 全美股清单查询 / 订单指令                                │     │
│  │    输出: ticker, sector, market_cap, avg_volume, short_able       │     │
│  │         / order confirmation                                     │     │
│  │    存储: SQLite broad_universe 表                                │     │
│  │    → Mid Chain (选股漏斗), Execution (下单)                      │     │
│  │                                                                  │     │
│  │  Polygon.io ──────► 基本面数据 (PE, PB, ROE, 财报)              │     │
│  │    输入: ticker                                                  │     │
│  │    输出: 财务比率, 季度营收, EPS                                  │     │
│  │    存储: Parquet (预计算因子)                                     │     │
│  │    → Mid Chain (因子融合)                                        │     │
│  │    备选: Yahoo Finance (免费 fallback)                            │     │
│  │                                                                  │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌─── 新闻与情报 ──────────────────────────────────────────────────┐     │
│  │                                                                  │     │
│  │  NewsAPI ─────────► 全球新闻聚合 (70k+ 来源, 500条/天)          │     │
│  │    输入: keyword query, category                                 │     │
│  │    输出: headline, snippet, source, published_at                 │     │
│  │    存储: SQLite news_events 表                                   │     │
│  │    → Fast Chain (G1 过滤入口), Mid Chain (情绪因子)              │     │
│  │                                                                  │     │
│  │  Perigon ─────────► 金融专业新闻 (高信噪比)                      │     │
│  │    输入: financial topic filters                                 │     │
│  │    输出: 结构化金融新闻 + 实体标注                                │     │
│  │    → Fast Chain (G2 分类补充)                                    │     │
│  │                                                                  │     │
│  │  SEC EDGAR ───────► 8-K 速报, 10-K/10-Q 年报季报                │     │
│  │    输入: CIK / ticker                                            │     │
│  │    输出: 完整 filing 文本 (60k tokens)                            │     │
│  │    → Mid Chain (LLM 解析关键财务指标)                            │     │
│  │                                                                  │     │
│  │  Nasdaq RSS ──────► 交易暂停, 熔断, 除牌通知                     │     │
│  │    输入: RSS feed subscription                                   │     │
│  │    输出: halt/resume events                                      │     │
│  │    → Fast Chain (流动性事件触发)                                  │     │
│  │                                                                  │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌─── 宏观经济 ────────────────────────────────────────────────────┐     │
│  │                                                                  │     │
│  │  FRED API ────────► 19个经济指标 (8大维度)                       │     │
│  │    输入: series_id (DFF, T10Y2Y, CPIAUCSL, ...)                 │     │
│  │    输出: date, value (时间序列)                                   │     │
│  │    存储: Parquet (历史) + SQLite (最新值+Z-score)                 │     │
│  │    → Mid Chain (MacroTiltEngine → sector/style tilt)             │     │
│  │    → Slow Chain (经济周期分类 → RL reward shaping)               │     │
│  │                                                                  │     │
│  │  Polymarket ──────► 宏观事件隐含概率                              │     │
│  │    输入: top 30 macro events                                     │     │
│  │    输出: event_name, implied_probability, delta                  │     │
│  │    → Mid Chain (MacroTiltEngine 补充因子)                        │     │
│  │                                                                  │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 数据源汇总表

| 数据源 | 类型 | 频率 | 成本 | 输入 | 输出 | 流向 Pipeline |
|--------|------|------|------|------|------|---------------|
| **Alpaca API** | 市场数据 | 日度 | 免费 | ticker, date_range | OHLCV, 股票池, order status | Mid + Slow + Execution |
| **Polygon.io** | 基本面 | 日度 | $0-20/月 | ticker | PE, PB, ROE, EPS | Mid Chain |
| **Yahoo Finance** | 基本面 | 日度 | 免费 | ticker | 财务比率 (fallback) | Mid Chain |
| **FRED API** | 宏观指标 | 日-月度 | 免费 | series_id (19个) | 经济时间序列 | Mid + Slow |
| **NewsAPI** | 新闻 | 实时 | 免费 | keyword query | headline, snippet, source | Fast + Mid |
| **Perigon** | 金融新闻 | 实时 | $100-500/月 | financial filters | 结构化新闻 + 实体 | Fast Chain |
| **SEC EDGAR** | 监管文件 | 日度 | 免费 | CIK / ticker | 10-K, 10-Q, 8-K 全文 | Mid Chain |
| **Nasdaq RSS** | 交易事件 | 实时 | 免费 | RSS subscription | halt/resume 事件 | Fast Chain |
| **Polymarket** | 预测市场 | 实时 | 免费 | top 30 events | 事件概率 + delta | Mid Chain |

---

## 5. 数据流转全链路

```
                    ┌─────────────────┐
                    │   External APIs  │
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │     10 Providers (抽象层)     │
              │  backtest ←YAML切换→ live     │
              │  L1(内存) → L2(磁盘) → L3(API)│
              └──────────────┬──────────────┘
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
   │  Parquet     │   │  SQLite     │   │  NetworkX   │
   │  (OHLCV,    │   │  (universe, │   │  (宏观图    │
   │   factors,  │   │   news,     │   │   30+70,    │
   │   macro)    │   │   z-scores) │   │   公司图    │
   │  ~1.7 GB    │   │  ~200 MB    │   │   100+500)  │
   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Feature Layer  │
                    │                  │
                    │  Alpha158 表达式 → 技术因子 (158D)
                    │  FinBERT        → 情绪因子
                    │  LightGBM      → 收益预测
                    │  MacroTilt     → sector/style tilt
                    │  传播引擎       → propagation_score
                    │  XGBoost       → 融合分数 (100D)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Model Layer    │
                    │                  │
                    │  DQN  → 选股 (100→8)
                    │  PPO  → 权重分配 (Σ=1)
                    │  SAC  → Phase 2 替代 PPO
                    │  Tilt → ±3% 宏观覆盖
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Decision Layer  │
                    │                  │
                    │  验证: Schema + 风控 + 异常
                    │  约束: sector cap + position cap
                    │  审批: Victor 手动确认
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Execution       │
                    │  Alpaca API      │
                    │  8L + 1S 组合    │
                    └─────────────────┘
```

---

## 6. 文档索引

本技术设计文档已拆分为以下子文档：

| 文档 | 内容 | 对应原始章节 |
|------|------|-------------|
| [00-architecture-overview.md](00-architecture-overview.md) | 系统架构总览、Pipeline、数据源 | §1 |
| [01-data-layer.md](01-data-layer.md) | 数据层：漏斗、存储、缓存、Provider | §2 |
| [02-model-layer.md](02-model-layer.md) | 模型层：LLM 路由、小模型、RL、MacroTilt | §3 |
| [03-news-processing.md](03-news-processing.md) | 信息源与新闻处理：G1/G2/G3、Polymarket | §4 |
| [04-decision-execution.md](04-decision-execution.md) | 决策与执行：再平衡、实时监控、风控 | §5-§6 |
| [05-infrastructure.md](05-infrastructure.md) | 基础设施、预算、安全、扩展性 | §7-§10 |
| [06-adr.md](06-adr.md) | 关键技术决策记录 (ADR) | §11 |
