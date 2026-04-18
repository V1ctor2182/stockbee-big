# Stockbee 技术设计 — 基础设施、预算与安全

**版本**: 2.2 | **日期**: 2026-04-18 | ← [返回总览](00-architecture-overview.md)

---

## 1. 硬件配置

| 组件 | 规格 | 用途 |
|------|------|------|
| GPU | NVIDIA RTX 3070 (8GB VRAM) | RL 训练 + 小模型推理 |
| CPU | 16核 AMD Ryzen / Intel | 因子计算 + 数据处理 |
| RAM | 32GB | 数据缓存 + 训练缓冲 |
| 存储 | 500GB NVMe SSD | 数据库 + Parquet + 模型权重 |
| 网络 | 稳定有线网络 | API 连接 (Alpaca/LLM/新闻) |

---

## 2. 技术栈选型

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

---

## 3. 部署架构 (单机)

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

## 4. 月度运营成本

| 项目 | 月度预算 | 说明 |
|------|---------|------|
| LLM API 调用 | $15 | G1(1) + G2(2) + G3(5) + SEC(3) + Macro(4)，使用 Batch API 优化 | research-llm-selection.md §4 |
| 新闻数据源 | $0-30 | NewsAPI 免费层 + Perigon 按用量 | 信息.md §1 |
| 市场数据 | $0-20 | Alpaca 免费 + Polygon.io 基础版 | research-stock-data-storage.md |
| GPU 电力 | ~$10 | RTX 3070 周日训练约 20 小时 | research-rl-algorithms.md §3 |
| **合计** | **< $75** | 远低于 $200 月预算上限 | research-llm-selection.md §4 |

---

## 5. 成本降级策略

| 等级 | 触发点 | 措施 | 影响 |
|------|--------|------|------|
| 绿色 (< $75) | - | 正常运营，全功能 | 无 |
| 黄色 (70-85%) | 接近 $150 | FMP → Yahoo fallback, 减少 G3 频率 | 基本面数据延迟 |
| 橙色 (85-95%) | 接近 $190 | 禁用 G3 深度分析，仅用缓存宏观数据 | 新闻深度分析停止 |
| 红色 (> 95%) | 接近 $200 | 禁用 Tier-2 自动加速，最小化运营 | 仅保留周度再平衡 |

---

## 6. 安全与风控设计

### 密钥与认证
- API 密钥存储在 `.env` 文件，从不写入代码
- 生产环境使用环境变量或密钥管理服务
- 账户密码从不涉及自动交易流程

### 交易权限
- **审批工作流**防止越权：所有 Tier-2 及以上需人工批准
- **周度再平衡**无自动执行权，必须人工批准后才能下单
- **Tier-1 自动执行**受严格条件限制（分数 > 85，且仅在有人监控时启用）

### 持仓限制
- 单个持仓上限：20% (硬约束)
- 行业集中度：最多 3 只/行业 (硬约束)
- 日度交易量：$50k (应急情况下的上限)

---

## 7. 扩展性与模型迭代

### 核心扩展点

**Provider 模式的灵活性**：易于更换数据源（例 Yahoo → CRSP、Alpaca → Interactive Brokers），只需修改 Provider 实现类而无需改动上层逻辑。→ 参考：research-provider-design.md §5

**模型版本控制**：每次 RL 训练产生日期标签版本（e.g., model-20260324-dqn.pt），支持快速回滚与 A/B 对比测试。维护一个版本清单，记录每个版本的 Sharpe 率、最大回撤、样本外表现。

**模块化因子体系**：Alpha158 技术因子集可增减，新增因子自动计入 IC 评估。融合器（XGBoost）权重动态调整，无需重新训练。

**多策略并行框架**：当前 Long 7 + Short 1 框架可扩展至：
- Long-only 版本（无空头，降低风险）
- 配对交易（同行业多空对冲）
- 海外市场版本（同代码库，不同 Universe Provider）

→ 参考：research-rl-algorithms.md §4

### 持续改进流程

1. **每周回测评估**：日终计算持仓的实现收益率，对比 RL Agent 预测，计算预测误差与新的 IC。
2. **月度因子评审**：禁用 ICIR < 0 的因子，新增待测因子，更新 MacroTiltEngine 的行业敏感度矩阵。
3. **季度模型重训**：新数据进入，历史回测窗口扩大，重新训练 DQN+PPO，对比历史最优模型。
4. **年度架构评估**：是否切换至 SAC+Ensemble？是否扩展到新市场或策略？基于过去 12 个月的表现和市场条件变化。
