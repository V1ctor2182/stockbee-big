# 小模型体系 (02-small-models)

## Intent

**FinBERT 情绪 + Fin-E5 重要性 + LightGBM 收益预测**

所有模型本地部署，GPU 加速推理。FinBERT <100ms，LightGBM <50ms。每周 refit 适应市场变化。

来源：Tech Design §3.3
研究状态：研究完成 (新闻小模型.md, 量化小模型.md)

## Constraints

- **FinBERT 情绪准确率 > 80%** — 来源: PRD §3.3
- **LightGBM 预测 IC > 0.03** — 来源: PRD §3.3

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
