# 宏观评分 (02-macro-scoring)

## Intent

**日度看涨/看跌评分，经济周期分类**

5 种经济制度分类（扩张/过热/滞胀/衰退/复苏）。Z-score 标准化所有指标。与知识图谱传播 Tilt 融合。

来源：Feature Hierarchy §4.2, Tech Design §3.5
实现状态：已完成 ✅

## Decisions

- **Regime 分类方法**: 规则引擎 (Plan A)。Phase 1 用阈值规则，简单可解释可调。Phase 2 可升级为 HMM 统计模型。
- **LLM 周度分析**: Phase 1 做。LLMRouter 已有 MACRO_ANALYSIS task type ($4/月预算)，早期迭代积累 prompt 经验。
- **融合权重**: Phase 1 propagation_weight=0.0 (纯规则)。知识图谱 Room 完成后改为 0.6 规则 + 0.4 传播图。
- **Live 模式融合**: 0.7 rule-based + 0.3 LLM weekly outlook
- **Polymarket cliff**: 负向调整 (概率上升 → 不确定性 → bearish)，cap ±0.15
- **HV 日 dampening**: 0.8x 乘数 (FOMC/CPI/NFP 日降低 conviction)

## Contracts

- **MacroScoringProvider**:
  - `get_macro_score(date) → float | None` — 日度评分 [-1.0, +1.0]
  - `get_regime(date) → str | None` — 经济周期 (expansion/overheating/stagflation/recession/recovery)
  - `get_history(start, end) → DataFrame` — 历史评分序列
  - `score_and_store(date) → MacroScore | None` — 评分并持久化
- **下游消费者**: 03-sector-tilts, 04-style-tilts (用 regime + score 计算 tilt 幅度)

## 当前进度

4/4 milestones 完成，74 tests:
- ✅ m1: MacroScorer 核心 — 规则引擎 regime 分类 + 日度评分 (scorer.py)
- ✅ m2: LLM 周度宏观分析 — Claude Opus 深度解读 (llm_analyst.py)
- ✅ m3: MacroScoringProvider — Provider 接口 + SQLite 持久化 (provider.py)
- ✅ m4: 测试 — 74 tests (test_macro_scoring.py)

## Deferred (Phase 2)

- HMM/统计模型替换规则引擎 (如 regime 分类准确率 < 70%)
- 知识图谱传播信号融合 (propagation_weight → 0.4)

---
_所有 spec 状态: active_
_specs 目录: 1 intent + 1 change = 2 个 spec 文件_
_spec.md 最后更新: 2026-04-16_
