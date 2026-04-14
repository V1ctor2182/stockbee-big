"""small_models — FinBERT / Fin-E5 / LightGBM 本地模型 (02-small-models Room)。

每个 milestone 向本文件增量 re-export 自己的对外符号。

当前:
- m1: paths, model_io (artifact 版本管理 + save/load pickle)
"""

from .model_io import (
    InvalidArtifactError,
    NotFoundError,
    artifact_path,
    list_versions,
    load_pickle,
    save_pickle,
    update_symlink,
)
from .paths import ML_SCORE_PARQUET, MODEL_ROOT

__all__ = [
    "InvalidArtifactError",
    "ML_SCORE_PARQUET",
    "MODEL_ROOT",
    "NotFoundError",
    "artifact_path",
    "list_versions",
    "load_pickle",
    "save_pickle",
    "update_symlink",
]
