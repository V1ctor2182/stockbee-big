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
| EMA 实现 | pandas ewm(adjust=False, α=2/(n+1))，IIR 递推；与 qlib FIR 截断加权在前 ~2n 行有差异 | 2026-04-11 |
| Alpha158 因子分布 | 9 KBAR + 4 price + 5 VMA + 28算子×5窗口 = 158（修正 PRD "27组×5"） | 2026-04-11 |
| TS_RANK/TS_QUANTILE | 新增时序滚动函数（与横截面 RANK/QUANTILE 共存），TS_RANK ties=average，TS_QUANTILE interpolation=linear | 2026-04-11 |
| NaN 比较语义 | 保持 pandas 默认（NaN 比较→False），与 qlib 行为一致，不改为 NaN 传播 | 2026-04-11 |
| max_lookback 实现 | 直接用 AST.lookback()（嵌套相加），不需要 walk() | 2026-04-11 |
| 因子定义源 | 全量 YAML（config/factors-alpha158.yaml），长表达式也写全，不混合 Python factory | 2026-04-11 |
| 混合价格 schema | CLOSE→adj_close + raw OHLV 是已知约束（数据源仅提供 adj_close）；m6 LocalFactorProvider 可用 adj_close/close ratio 推导 adj OHLV 解决 | 2026-04-11 |
| VWAP0 可选依赖 | VWAP0 属于 Alpha158 标准但 vwap 列在当前数据源中不保证存在；无 vwap 时该因子 raise，其余 157 因子正常 | 2026-04-11 |

## Contracts

**FactorProvider 接口** (`providers/interfaces.py:129-161`):
- `get_factors(tickers, factor_names, start, end) → MultiIndex DataFrame (date, ticker) x factors`
  - 列顺序与 factor_names 一致
  - unknown name → raise
  - 无 MarketDataProvider 时请求 expression 因子 → RuntimeError
- `list_factors() → list[dict]` (含 type: expression / precomputed)
- `get_ic_report(factor_name, window=252) → dict` (ic_mean, ic_std, icir)

**表达式引擎函数** (28 个):
- 基础 (m1b): REF/DELAY, MA/MEAN, STD, SUM, MAX, MIN, DELTA, ABS, LOG, SIGN
- 高级 (m2): EMA, SLOPE, RSQUARE, RESI, CORR, RANK, IF, QUANTILE, IDXMAX, IDXMIN
- 时序滚动 (m3): TS_RANK, TS_QUANTILE

**Alpha158 因子名→表达式函数 Alias**:
| Factor 前缀 | 表达式函数 | 参数 |
|-------------|-----------|------|
| ROC | REF | `REF($close,w)/$close` |
| MA | MEAN | `MEAN($close,w)/$close` |
| STD | STD | `STD($close,w)/$close` |
| BETA | SLOPE | `SLOPE($close,w)/$close` |
| RSQR | RSQUARE | `RSQUARE($close,w)` |
| RESI | RESI | `RESI($close,w)/$close` |
| MAX | MAX(rolling) | `MAX($high,w)/$close` |
| MIN | MIN(rolling) | `MIN($low,w)/$close` |
| QTLU | TS_QUANTILE | q=0.8 `TS_QUANTILE($close,w,0.8)/$close` |
| QTLD | TS_QUANTILE | q=0.2 `TS_QUANTILE($close,w,0.2)/$close` |
| RANK | TS_RANK | `TS_RANK($close,w)` |
| RSV | MAX/MIN | `($close-MIN($low,w))/(MAX($high,w)-MIN($low,w)+1e-12)` |
| IMAX | IDXMAX | `IDXMAX($high,w)/w` |
| IMIN | IDXMIN | `IDXMIN($low,w)/w` |
| IMXD | IDXMAX-IDXMIN | `(IDXMAX($high,w)-IDXMIN($low,w))/w` |
| CORR | CORR | `CORR($close,LOG($volume+1),w)` |
| CORD | CORR(returns) | `CORR($close/REF($close,1),LOG($volume/REF($volume,1)+1),w)` |
| CNTP | MEAN(bool) | `MEAN($close>REF($close,1),w)` |
| CNTN | MEAN(bool) | `MEAN($close<REF($close,1),w)` |
| CNTD | CNTP-CNTN | 差值 |
| SUMP/SUMN/SUMD | SUM+MAX | 比率 |
| VMA | MEAN(volume) | `MEAN($volume,w)/($volume+1e-12)` |
| VSTD | STD(volume) | `STD($volume,w)/($volume+1e-12)` |
| WVMA | STD/MEAN | 嵌套加权波动率 |
| VSUMP/VSUMN/VSUMD | SUM+MAX(volume) | 同 SUMP 但用 $volume |

**AST 接口** (m1a 锁定):
- `AST.walk() → Iterator[Node]` — 遍历全树，供集成测试
- `AST.lookback() → int` — 沿嵌套链相加，返回该子树所需的最大 lookback 窗口（m3 max_lookback 直接调用此接口）

**IC Evaluator 接口**:
- `compute(factor_df, prices_df, shift=1) → dict(ic_mean, ic_std, icir)`
- shift 在内部做（factor at t vs return at t+1），禁止 caller 侧 shift

## 当前进度

8 milestones — 6/8 完成 (m1a, m1b, m2, m3, m4, m5)

1. **m1a-tokenizer-parser** — ✅ Tokenizer + Parser + AST（740行）
2. **m1b-evaluator-basic-funcs** — ✅ Evaluator + 12 基础函数（866行）
3. **m2-advanced-functions** — ✅ 10 高级函数（687行）
4. **m3-alpha158-full** — ✅ Alpha158 全量 158 因子 + TS_RANK/TS_QUANTILE（516行, 167 tests）
5. **m4-parquet-factor-store** — ✅ 预计算因子 Parquet 存储（315行, 17 tests, 184 全量 pass）
6. **m5-ic-evaluator** — ✅ IC/ICIR 纯数值离线评估（248行, 19 tests, 203 全量 pass）
7. **m6-local-provider** — LocalFactorProvider 路由 + 模块导出 (~280行)
8. **m7-integration-tests** — 集成层 edge case 测试（12 个场景）(~250行)

**并行路径**：m1a / m4 / m5 三条链第一天可同时开工。

---
_spec 状态: intent (active), change (active)_
_spec.md 最后更新: 2026-04-12 (m5 完成)_
_specs 目录: 1 intent + 4 change = 5 个 spec 文件_
