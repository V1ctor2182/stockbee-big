# 模型体系 (03-model-system)

## Intent

**LLM 多模型路由 + 小模型本地部署 + RL 选股/权重优化**

分离式多模型设计。LLM 路由降低成本到 <$15/月。小模型本地 GPU 推理。RL 双阶段：DQN 选股 + PPO/SAC 权重。

来源：PRD §3.3, Tech Design §3

## Constraints

- **LLM 月度 API 成本 < $15** — 来源: PRD §3.3
- **FinBERT 情绪准确率 > 80%** — 来源: PRD §3.3
- **RL Walk-forward Sharpe > 1.8** — 来源: PRD §3.3

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
