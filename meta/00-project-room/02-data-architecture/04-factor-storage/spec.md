# 因子存储 (04-factor-storage)

## Intent

**表达式引擎（技术因子）+ Parquet 预计算（融合因子）**

Alpha158 全部 158 个技术因子用表达式引擎动态计算（27 组 × 5 窗口 + 9 KBAR），灵活度高。基本面/情绪/ML 因子预计算存 Parquet。

来源：Tech Design §2.5
研究状态：研究完成 (research-factor-storage.md)

## Decisions

| 决策 | 结论 | 日期 |
|------|------|------|
| CLOSE 映射 | adj_close（复权价）| 2026-04-09 |
| MarketDataProvider 注入 | setter 注入（和 cache 模式一致）| 2026-04-09 |
| Alpha158 范围 | 全部 158 因子 | 2026-04-09 |
| Milestone 方案 | Plan B: 8 milestones（m1 拆 tokenizer/evaluator，m5/m6 并行，m7 集成测试）| 2026-04-11 |
| MAX/MIN 重载 | 第二参数整数≥2 → rolling, 否则 → element-wise | 2026-04-09 |
| RANK/QUANTILE 语义 | 横截面（cross-section at each date），非时序 | 2026-04-11 |
| IC shift 归属 | ic_evaluator.compute(shift=1) 层做，禁止 caller 自行 shift | 2026-04-11 |
| max_lookback 算法 | AST.walk() 沿嵌套链相加（MA(REF(c,5),20)→25），非取 max | 2026-04-11 |
| Parquet group 粒度 | 按数据来源（fundamental/sentiment/ml_score），每组一文件 | 2026-04-11 |
| 未知 factor 处理 | get_factors 收到未知 factor name → raise，不静默跳过 | 2026-04-11 |

## Contracts

**FactorProvider 接口** (`providers/interfaces.py:129-161`):
- `get_factors(tickers, factor_names, start, end) → MultiIndex DataFrame (date, ticker) x factors`
  - 列顺序与 factor_names 一致
  - unknown name → raise
  - 无 MarketDataProvider 时请求 expression 因子 → RuntimeError
- `list_factors() → list[dict]` (含 type: expression / precomputed)
- `get_ic_report(factor_name, window=252) → dict` (ic_mean, ic_std, icir)

**表达式引擎函数** (26 个):
- 基础 (m1b): REF/DELAY, MA/MEAN, STD, SUM, MAX, MIN, DELTA, ABS, LOG, SIGN
- 高级 (m2): EMA, SLOPE, RSQUARE, RESI, CORR, RANK, IF, QUANTILE, IDXMAX, IDXMIN

**AST 接口** (m1a 锁定):
- `AST.walk() → Iterator[Node]` — 遍历全树，供 max_lookback 和集成测试
- `AST.lookback() → int` — 沿嵌套链相加，返回该子树所需的最大 lookback 窗口

**IC Evaluator 接口**:
- `compute(factor_df, prices_df, shift=1) → dict(ic_mean, ic_std, icir)`
- shift 在内部做（factor at t vs return at t+1），禁止 caller 侧 shift

## 当前进度

8 milestones, ~2080 行, 9 files — 全部 pending

1. **m1a-tokenizer-parser** — Tokenizer + Parser + AST（含 walk/lookback 接口）(~200行)
2. **m1b-evaluator-basic-funcs** — Evaluator + 16 基础函数 + MAX/MIN 重载 (~250行)
3. **m2-advanced-functions** — 10 高级函数（rolling OLS / cross-section RANK 等）(~300行)
4. **m3-alpha158-full** — Alpha158 全量 158 因子定义 + max_lookback (~350行)
5. **m4-parquet-factor-store** — 预计算因子 Parquet 存储（可与 m1-m3 并行）(~250行)
6. **m5-ic-evaluator** — IC/ICIR 纯数值计算（可与 m1-m4 并行）(~200行)
7. **m6-local-provider** — LocalFactorProvider 路由 + 模块导出 (~280行)
8. **m7-integration-tests** — 集成层 edge case 测试（12 个场景）(~250行)

**并行路径**：m1a / m4 / m5 三条链第一天可同时开工。

---
_spec 状态: intent (draft), change (active)_
_spec.md 最后更新: 2026-04-11 (8-milestone 重规划 + contract 扩充)_
_specs 目录: 1 intent + 1 change = 2 个 spec 文件_
