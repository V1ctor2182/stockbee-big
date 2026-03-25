# 宏观分析 (04-macro-analysis)

## Intent

**19 FRED 指标驱动的经济周期判断 + 行业/风格 Tilt**

MacroTiltEngine 将宏观信号转化为 soft bias 叠加在因子分数上。经济周期分类（扩张/过热/滞胀/衰退/复苏）准确率 > 70%。

来源：PRD §3.5, Tech Design §3.5

## Constraints

- **经济周期分类准确率 > 70%** — 来源: PRD §3.5
- **宏观 Tilt = 0.6 × 规则 + 0.4 × 传播图** — 来源: PRD §3.5

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
