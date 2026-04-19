# Stockbee 技术设计 — 模型层

**版本**: 2.2 | **日期**: 2026-04-18 | ← [返回总览](00-architecture-overview.md)

---

## 1. 多模型架构概览

系统采用分离式多模型设计，将离散选股（DQN）与连续权重分配（PPO/SAC）分离。这样做的原因是：选择 8 只股票本质上是组合优化中的离散问题，而权重分配是连续优化。分离后，DQN 可专注于选股，PPO/SAC 可专注于权重分配，各自达到更优的收敛性。→ 参考：research-rl-algorithms.md §2.1

```
新闻源 → [LLM 路由器]
            ├─→ G1 快速过滤 (规则引擎, 零成本)
            ├─→ G2 分类 (FinBERT 本地, 零 API 成本)
            └─→ G3 深度分析 (Claude Haiku, <$2/月，可选)

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

---

## 2. LLM 多模型路由策略

**核心设计理念**：不同的 NLP 任务对模型能力要求差异大。G1 只需布尔判断（相关/无关），用廉价模型足够；G2 需要细致的情绪/主题判断，用中档模型；G3 需要深度推理，才用最强模型。这样成本从 $50+/月（单一 GPT-4）降到 $15/月，同时准确率无显著差异。→ 参考：research-llm-selection.md §3

| 任务 | 选择模型 | 备选模型 | 月成本 | 输入规模 | 输出格式 |
|------|---------|---------|--------|---------|---------|
| G1 新闻快速过滤 | 规则引擎 (本地) | — | $0 | 标题+摘要 | 通过/淘汰 + ticker 提取 |
| G2 新闻分类 (情绪/主题) | FinBERT (本地) | — | $0 | 标题+摘要 (max 128 tokens) | JSON: {sentiment, category, urgency, importance} |
| G3 深度基本面分析 | Claude Haiku | Claude Sonnet 4 | < $2 | 完整新闻+持仓情景 (2-5k tokens) | JSON：影响评估 + 权重建议 |
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

---

## 3. 小模型部署体系

所有小模型本地部署，无 API 依赖，支持 GPU 加速推理。

**FinBERT（情绪分类）**：微调的 BERT 模型，专为金融文本优化。输入：新闻标题+摘要（max 128 tokens），输出：[positive, negative, neutral] 三分类 + 置信度。推理延迟 <100ms。训练数据来自 Lexicon 与弱监督标签（新闻发布日期与股价涨跌的关联）。→ 参考：新闻小模型.md §1, finbert.md

**Fin-E5（重要性评估）**：嵌入模型（类似 E5），任务是区分"哪些新闻真正影响股价"。输入：完整新闻正文，输出：0-1 重要性分数。训练目标：最大化新闻嵌入与后 5 日涨幅的相似度。→ 参考：新闻小模型.md §2, fine5.md

**LightGBM（收益预测）**：梯度提升决策树，输入 Alpha158 因子（158维），输出：未来 5 日平均收益率（连续值）。在历史回测数据上训练（5 年 × 252 天），采用滚动窗口避免前看偏差。推理延迟 <50ms。为什么选 LightGBM 而非 XGBoost？Qlib 基准测试表明 LightGBM 在金融数据上收敛更快，IC 持久性更好。→ 参考：量化小模型.md §1, qlib 因子挖掘和量化模型.md

**XGBoost（因子融合）**：可选的非线性融合器。输入：6-8 个预计算因子（技术因子、情绪、基本面、宏观倾斜），输出：综合因子分数。相比简单等权，XGBoost 融合可根据市场体制动态调整因子权重，提升 Sharpe 率约 5-10%。→ 参考：量化小模型.md §2-§3

所有模型均支持定期 refit（每周重新训练），以适应市场制度变化。

---

## 4. 强化学习双阶段训练

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

---

## 5. 宏观分析引擎（MacroTiltEngine）

**19 个 FRED 指标的经济分类**：

| 经济维度 | FRED 代码 | 含义 | 用途 |
|---------|---------|------|------|
| 利率 | DFF / T10Y2Y / DGS10 | Federal Funds Rate / 2-10Y 期限差 / 10Y 美债收益率 | 投资者风险偏好指标 |
| 就业 | PAYEMS / UNRATE / ICSA | 非农就业人数 / 失业率 / 初请周数 | 劳动力市场强度 |
| 物价 | CPIAUCSL / PPIACO | CPI / PPI | 通胀压力 |
| 增长 | A191RA1Q225SBEA / INDPRO | 实际 GDP / 工业产值 | 经济增速 |
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
