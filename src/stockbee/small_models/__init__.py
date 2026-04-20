"""small_models — FinBERT / Fin-E5 / LightGBM 本地模型 (02-small-models Room)。

每个 milestone 向本文件增量 re-export 自己的对外符号。

当前:
- m1: paths, model_io (artifact 版本管理 + save/load pickle)
- m2a: FinBERTScorer, get_default_scorer, reset_default_scorer (纯推理 + singleton)
- m3: forward_return_5d (B3 label) + train_lightgbm + train_and_save + walk_forward_splits
- m5: FinE5Scorer (encode + cosine_sim + cosine_dedup) + baseline_importance + backfill_importance
- m2b: LocalSentimentProvider (SentimentProvider 实现 + g2 P3 重构)
"""

from .fine5_scorer import FinE5Scorer
from .finbert_scorer import (
    FinBERTScorer,
    get_default_scorer,
    reset_default_scorer,
)
from .importance_baseline import backfill_importance, baseline_importance
from .local_sentiment_provider import LocalSentimentProvider
from .label_utils import LABEL_NAME, forward_return_5d
from .lightgbm_trainer import (
    DEFAULT_PARAMS,
    train_and_save,
    train_lightgbm,
    walk_forward_splits,
)
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
    "DEFAULT_PARAMS",
    "FinBERTScorer",
    "FinE5Scorer",
    "InvalidArtifactError",
    "LABEL_NAME",
    "LocalSentimentProvider",
    "ML_SCORE_PARQUET",
    "MODEL_ROOT",
    "NotFoundError",
    "artifact_path",
    "backfill_importance",
    "baseline_importance",
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
