# 行业倾斜 (03-sector-tilts)

## Intent

**11 个 GICS 行业动态加权**

行业对宏观指标的敏感度矩阵（SQLite 存储）。如能源对油价敏感度 0.8。±3% 权重调整。

来源：Feature Hierarchy §4.3, Tech Design §3.5
实现状态：已完成 ✅

## Decisions

- **敏感度矩阵存储**: SQLite。支持运行时调参，Phase 2 月度因子评审可动态更新。
- **模块位置**: `macro_scoring/sector_tilts.py`，和 style_tilts 对称。
- **Tilt cap**: ±3% (±0.03)，作为 soft overlay 不硬约束 RL Agent。
- **Regime 调整**: 5 种 regime 各有行业乘数 (e.g. recession: Staples 1.4x, Discretionary 0.5x)。

## Contracts

- **SectorTilter**:
  - `compute_tilts(as_of) → dict[str, float]` — 11 行业 tilt 权重 (±0.03)
  - `get_sensitivity_matrix() → DataFrame` — 当前敏感度矩阵
  - `update_sensitivity(sector, indicator, value)` — 更新单个敏感度

## 当前进度

2/2 milestones 完成，24 tests:
- ✅ m1: 敏感度矩阵 (SQLite) + SectorTilter 核心 (sector_tilts.py)
- ✅ m2: 测试 — 24 tests inline with m1 (test_sector_tilts.py)

---
_所有 spec 状态: active_
_specs 目录: 1 intent + 1 change = 2 个 spec 文件_
_spec.md 最后更新: 2026-04-17_
