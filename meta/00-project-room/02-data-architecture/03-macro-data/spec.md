# 宏观数据 (03-macro-data)

## Intent

**17 个 FRED 指标，Parquet 时间序列 + SQLite Z-score 索引，point-in-time 对齐**

4 个核心组件：
- **indicators.py** — 17 个 FRED 指标定义（8 维度），FredIndicator dataclass
- **ParquetMacroProvider** — 单文件 Parquet 存储 + SQLite Z-score 缓存，252 天滚动 Z-score clip [-3,+3]，forward-fill 混频，point-in-time 对齐
- **FredMacroProvider** — fredapi SDK 实盘数据拉取，支持单指标和批量 fetch
- **MacroDataSyncer** — 编排 sync_all（全量 5 年）和 sync_latest（增量 30 天）

实现状态：已完成 ✅ + FRED API 17/17 验证通过（2026-04-05）
测试覆盖：23 个单元测试，141 全项目测试通过

## Constraints

- **point-in-time 对齐无未来偏差** — end=T 只返回 T-1 及之前的数据
- **Z-score 252 天滚动窗口，clip [-3, +3]** — std=0 返回 NaN，min_periods=20
- **17 个 FRED 指标全部 API 验证通过** — 含 3 个修正 ID（见 decision-002）
- **混频 forward-fill + Parquet 单文件存储** — 日/周/月/季 → 日频，~200KB

## Decisions

- **单文件 Parquet > 按指标分文件** — 17 指标数据量极小，单文件简化 I/O
- **3 个 FRED ID 替换** — BAMLH0A0HY2→HYM2，EXHYLD→DRTSCILM，GOLDAMGBD228NLBM→PCOPPUSDM

## Contracts

- **ParquetMacroProvider** — get_macro_indicators → DataFrame(date x indicators); get_latest_z_scores → dict; write_indicators; rebuild_z_score_index
- **FredMacroProvider** — 同上 + fetch_all(start, end) → DataFrame
- **MacroDataSyncer** — sync_all(years); sync_latest(days); run_weekly_sync

---
_所有 spec 状态: active_
_spec.md 最后更新: 2026-04-05_
_specs 目录: 1 intent + 4 constraints + 2 decisions + 1 contract + 1 change = 9 个 spec 文件_
