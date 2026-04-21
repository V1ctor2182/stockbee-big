"""small_models — FinBERT / Fin-E5 / LightGBM 本地模型 (02-small-models Room)。

每个 milestone 向本文件增量 re-export 自己的对外符号。

当前:
- m1: paths, model_io (artifact 版本管理 + save/load pickle)
- m2a: FinBERTScorer, get_default_scorer, reset_default_scorer (纯推理 + singleton)
- m3: forward_return_5d (B3 label) + train_lightgbm + train_and_save + walk_forward_splits
- m5: FinE5Scorer (encode + cosine_sim + cosine_dedup) + baseline_importance + backfill_importance
- m2b: LocalSentimentProvider (SentimentProvider 实现 + g2 P3 重构)
- m4: LightGBMScorer (推理 + ml_score.parquet 落盘) + evaluate_ml_score(shift=5)

为避免 `from stockbee.small_models.finbert_scorer import ...` 这类直接导入
把 lightgbm_scorer → factor_data → pyarrow 的重依赖全量拉起,此处使用 PEP 562
lazy __getattr__: 仅当访问 `stockbee.small_models.<Name>` 时才触发对应子模块加载。
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .fine5_scorer import FinE5Scorer
    from .finbert_scorer import (
        FinBERTScorer,
        get_default_scorer,
        reset_default_scorer,
    )
    from .importance_baseline import backfill_importance, baseline_importance
    from .label_utils import LABEL_NAME, forward_return_5d
    from .lightgbm_scorer import (
        ML_SCORE_COLUMN,
        ML_SCORE_GROUP,
        LightGBMScorer,
        evaluate_ml_score,
    )
    from .lightgbm_trainer import (
        DEFAULT_PARAMS,
        train_and_save,
        train_lightgbm,
        walk_forward_splits,
    )
    from .local_sentiment_provider import LocalSentimentProvider
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


_LAZY_MAP: dict[str, str] = {
    "FinE5Scorer": "fine5_scorer",
    "FinBERTScorer": "finbert_scorer",
    "get_default_scorer": "finbert_scorer",
    "reset_default_scorer": "finbert_scorer",
    "backfill_importance": "importance_baseline",
    "baseline_importance": "importance_baseline",
    "LABEL_NAME": "label_utils",
    "forward_return_5d": "label_utils",
    "LightGBMScorer": "lightgbm_scorer",
    "ML_SCORE_COLUMN": "lightgbm_scorer",
    "ML_SCORE_GROUP": "lightgbm_scorer",
    "evaluate_ml_score": "lightgbm_scorer",
    "DEFAULT_PARAMS": "lightgbm_trainer",
    "train_and_save": "lightgbm_trainer",
    "train_lightgbm": "lightgbm_trainer",
    "walk_forward_splits": "lightgbm_trainer",
    "LocalSentimentProvider": "local_sentiment_provider",
    "InvalidArtifactError": "model_io",
    "NotFoundError": "model_io",
    "artifact_path": "model_io",
    "list_versions": "model_io",
    "load_pickle": "model_io",
    "save_pickle": "model_io",
    "update_symlink": "model_io",
    "ML_SCORE_PARQUET": "paths",
    "MODEL_ROOT": "paths",
}


def __getattr__(name: str) -> Any:
    submodule = _LAZY_MAP.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    mod = importlib.import_module(f".{submodule}", __name__)
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_MAP))


__all__ = [
    "DEFAULT_PARAMS",
    "FinBERTScorer",
    "FinE5Scorer",
    "InvalidArtifactError",
    "LABEL_NAME",
    "LightGBMScorer",
    "LocalSentimentProvider",
    "ML_SCORE_COLUMN",
    "ML_SCORE_GROUP",
    "ML_SCORE_PARQUET",
    "MODEL_ROOT",
    "NotFoundError",
    "artifact_path",
    "backfill_importance",
    "baseline_importance",
    "evaluate_ml_score",
    "forward_return_5d",
    "get_default_scorer",
    "list_versions",
    "load_pickle",
    "reset_default_scorer",
    "save_pickle",
    "train_and_save",
    "train_lightgbm",
    "update_symlink",
    "walk_forward_splits",
]
