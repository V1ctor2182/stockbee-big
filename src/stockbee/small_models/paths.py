"""small_models 路径常量。

统一模型 artifact 与 precomputed 因子落盘位置,供 m2a-m6 共享。
"""

from __future__ import annotations

from pathlib import Path

MODEL_ROOT: Path = Path("data/models")
"""模型 artifact 根目录。子目录按模型名组织: data/models/{name}/{version}.pkl"""

ML_SCORE_PARQUET: Path = Path("data/factors/ml_score.parquet")
"""LightGBM 推理结果 precomputed 因子落盘路径 (m4 写, FactorProvider 读)。"""
