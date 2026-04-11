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
| Milestone 方案 | Plan A: 6 milestones 精细拆分 | 2026-04-09 |
| MAX/MIN 重载 | 第二参数整数≥2 → rolling, 否则 → element-wise | 2026-04-09 |

## Contracts

**FactorProvider 接口** (`providers/interfaces.py:129-161`):
- `get_factors(tickers, factor_names, start, end) → MultiIndex DataFrame (date, ticker) x factors`
- `list_factors() → list[dict]` (含 type: expression / precomputed)
- `get_ic_report(factor_name, window=252) → dict` (ic_mean, ic_std, icir)

**表达式引擎函数** (26 个):
- 基础 (m1): REF/DELAY, MA/MEAN, STD, SUM, MAX, MIN, DELTA, ABS, LOG, SIGN
- 高级 (m2): EMA, SLOPE, RSQUARE, RESI, CORR, RANK, IF, QUANTILE, IDXMAX, IDXMIN

## 当前进度

6 milestones, ~1470 行, 7 files — 全部 pending

1. **m1-expression-engine-core** — 表达式引擎核心 (~220行)
2. **m2-advanced-functions** — 高级表达式函数 (~200行)
3. **m3-alpha158-full** — Alpha158 全量 158 因子定义 (~350行)
4. **m4-parquet-factor-store** — 预计算因子 Parquet 存储 (~150行)
5. **m5-local-provider** — LocalFactorProvider + IC 评估 (~200行)
6. **m6-tests** — 测试 + 模块导出 (~350行)

---
_spec 状态: intent (draft), change (active)_
_spec.md 最后更新: 2026-04-10 (milestone 规划 + pytest 配置)_
_specs 目录: 1 intent + 1 change = 2 个 spec 文件_
