# 宏观数据 (03-macro-data)

## Intent

**17 个 FRED 指标，Parquet 时间序列 + SQLite Z-score 索引，point-in-time 对齐**

3 个核心组件：
- **ParquetMacroProvider** — 单文件 Parquet 存储 + SQLite Z-score 缓存，252 天滚动 Z-score clip 到 [-3,+3]，forward-fill 混频指标
- **FredMacroProvider** — fredapi SDK 实盘数据拉取，支持单指标和批量 fetch
- **MacroDataSyncer** — 编排 sync_all（全量 5 年）和 sync_latest（增量 30 天）

17 个 FRED 指标覆盖 8 个经济维度：利率(3)、就业(3)、物价(2)、增长(2)、信用(2)、流动性(2)、汇率(1)、商品(2)。

实现状态：已完成 ✅（2026-04-04）
测试覆盖：23 个单元测试

## Constraints

- **point-in-time 对齐无未来偏差** — end=T 时只返回 T-1 及之前的数据
- **Z-score 252 天滚动窗口，clip [-3, +3]** — std=0 返回 NaN，min_periods=20
- **混频 forward-fill** — 日频/周频/月频/季频指标统一填充到日频
- **Parquet 单文件存储** — macro_all.parquet，数据量极小 (~200KB)
- **fredapi 为可选依赖** — 回测模式不需要安装

## Decisions

- **单文件 Parquet > 按指标分文件** — 17 个指标数据量极小，单文件简化 I/O
- **SQLite Z-score 索引** — 避免每次查询都从 Parquet 重新计算
- **17 个可映射 FRED 指标 vs Tech Design 中的 19 个** — 2 个为派生特征，非独立 FRED series

## Contracts

- **ParquetMacroProvider** — get_macro_indicators(indicators, start, end) → DataFrame(date x indicators); get_latest_z_scores(indicators, window) → dict; write_indicators(df); rebuild_z_score_index() → int
- **FredMacroProvider** — 同上 + fetch_all(start, end) → DataFrame
- **MacroDataSyncer** — sync_all(years) → dict; sync_latest(days) → dict; run_weekly_sync() → dict

---
_所有 spec 状态: active_
_spec.md 最后更新: 2026-04-04_
