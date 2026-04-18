# Stockbee 技术设计 — 关键技术决策记录 (ADR)

**版本**: 2.2 | **日期**: 2026-04-18 | ← [返回总览](00-architecture-overview.md)

---

## 决策总览

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

---

## 决策过程透明度

每项决策的论证：
1. **问题陈述**：技术选项及权衡因素
2. **研究支撑**：引用的实证评估或行业标准
3. **预期效果**：采纳该选项的定量目标（如成本、性能、维护复杂度）
4. **回溯计划**：若该决策出现问题（如 Parquet 查询变慢），如何快速回滚或调整
