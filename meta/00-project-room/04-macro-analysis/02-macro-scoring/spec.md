# 宏观评分 (02-macro-scoring)

## Intent

**日度看涨/看跌评分，经济周期分类**

5 种经济制度分类（扩张/过热/滞胀/衰退/复苏）。Z-score 标准化所有指标。与知识图谱传播 Tilt 融合。

来源：Feature Hierarchy §4.2, Tech Design §3.5

## Decisions

- **Regime 分类方法**: 规则引擎 (Plan A)。Phase 1 用阈值规则，简单可解释可调。Phase 2 可升级为 HMM 统计模型。
- **LLM 周度分析**: Phase 1 做。LLMRouter 已有 MACRO_ANALYSIS task type ($4/月预算)，早期迭代积累 prompt 经验。
- **融合权重**: Phase 1 propagation_weight=0.0 (纯规则)。知识图谱 Room 完成后改为 0.6 规则 + 0.4 传播图。1 行改动，完全可逆。

## Contracts

- **MacroScoringProvider**:
  - `get_macro_score(date) → float` — 日度评分 [-1.0, +1.0]
  - `get_regime(date) → str` — 经济周期 (expansion/overheating/stagflation/recession/recovery)
  - `get_history(start, end) → DataFrame` — 历史评分序列
- **下游消费者**: 03-sector-tilts, 04-style-tilts (用 regime + score 计算 tilt 幅度)

## 当前进度

4 milestones, 0% 完成:
- m1: MacroScorer 核心 — 经济周期分类 + 日度评分 (~250 行)
- m2: LLM 周度宏观分析 — Claude Opus 深度解读 (~200 行)
- m3: MacroScoringProvider — Provider 接口 + 持久化 (~180 行)
- m4: 测试 (~400 行)

总计: ~1030 行, 4-5 files

## Deferred (Phase 2)

- HMM/统计模型替换规则引擎 (如 regime 分类准确率 < 70%)
- 知识图谱传播信号融合 (propagation_weight → 0.4)

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 plan-milestones 更新, 2026-04-15_
_specs/*.yaml 为源数据_
