"""LightGBM Trainer — Alpha158 × B3 label 训练 + artifact 落盘。

Walk-forward 严格无前看切分 → train_lightgbm (drop NaN + valid split + early stop) →
save_pickle 落 data/models/lightgbm/{version}.pkl → update_symlink 指向 current。

决策:
- B3: label = forward_return_5d(adj_close, horizon=5)
- m1: save_pickle 默认 overwrite=False, 同版本再训需显式 overwrite=True
- OQ4: 不自动 update_symlink,训练流程显式调用 (保留"先保存再评审再上线")
- 本模块只依赖 FactorProvider + MarketDataProvider + lightgbm + m1 model_io;
  与 m2a / m5 完全解耦

训练产出的 booster 通过 save_pickle 持久化 lgb.Booster 对象本身。
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from .label_utils import LABEL_NAME, forward_return_5d
from .model_io import save_pickle, update_symlink

logger = logging.getLogger(__name__)

MODEL_NAME = "lightgbm"

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "seed": 42,
    "deterministic": True,
    "force_col_wise": True,
}

_DEFAULT_NUM_ROUNDS = 500
_DEFAULT_EARLY_STOP = 50
_VALID_FRACTION = 0.2


def walk_forward_splits(
    dates: pd.DatetimeIndex | pd.Index,
    train_size: int = 252,
    test_size: int = 21,
    step: int = 21,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """生成 walk-forward 训练/测试日期切分 (严格无前看)。

    Args:
        dates: 按升序排列的唯一日期序列 (DatetimeIndex 或类似)
        train_size: 训练窗口天数
        test_size: 测试窗口天数
        step: 每次滑动天数

    Yields:
        (train_dates, test_dates) 各为 pd.DatetimeIndex,
        train_dates.max() < test_dates.min() 严格成立。

    Raises:
        ValueError: 参数非正,或总长不够一个 train+test 窗
    """
    if train_size < 1 or test_size < 1 or step < 1:
        raise ValueError(
            f"train_size/test_size/step must be positive, "
            f"got {train_size}/{test_size}/{step}"
        )
    unique = pd.DatetimeIndex(dates).unique().sort_values()
    n = len(unique)
    if n < train_size + test_size:
        raise ValueError(
            f"dates length {n} < train_size + test_size = {train_size + test_size}"
        )

    start = 0
    while start + train_size + test_size <= n:
        train_end = start + train_size
        test_end = train_end + test_size
        train_dates = unique[start:train_end]
        test_dates = unique[train_end:test_end]
        # 不变式: 严格无前看
        assert train_dates.max() < test_dates.min()
        yield train_dates, test_dates
        start += step


def train_lightgbm(
    factor_df: pd.DataFrame,
    label_sr: pd.Series,
    params: dict[str, Any] | None = None,
    num_rounds: int = _DEFAULT_NUM_ROUNDS,
    early_stop: int = _DEFAULT_EARLY_STOP,
) -> "Any":
    """训练 LightGBM booster。

    Args:
        factor_df: MultiIndex(date, ticker) × feature 列
        label_sr: MultiIndex(date, ticker), 与 factor_df 索引对齐
        params: 覆盖 DEFAULT_PARAMS (浅合并)
        num_rounds: 最大迭代轮数
        early_stop: 在 valid set 上连续 n 轮无改进则停止

    Returns:
        lgb.Booster

    流程:
      1. inner-join factor_df 和 label_sr 索引
      2. drop NaN 行 (任意特征或 label NaN)
      3. 按时间顺序切 valid (最后 _VALID_FRACTION)
      4. lgb.train + early_stopping
    """
    import lightgbm as lgb

    if factor_df.empty:
        raise ValueError("factor_df 为空")
    if label_sr.empty:
        raise ValueError("label_sr 为空")

    merged = factor_df.join(label_sr.rename("_label"), how="inner")
    if merged.empty:
        # review H3: inner-join 为空 = 索引无交集,明确报错
        raise ValueError(
            "factor_df 和 label_sr 索引无交集; "
            "检查 MultiIndex(date, ticker) 是否一致"
        )
    feature_cols = [c for c in merged.columns if c != "_label"]
    if not feature_cols:
        raise ValueError("factor_df 无特征列")

    # drop 任一特征/label NaN
    before_drop = len(merged)
    clean = merged.dropna(subset=feature_cols + ["_label"])
    after_drop = len(clean)
    if clean.empty:
        raise ValueError("drop NaN 后无样本可训")

    # 按 (date, ticker) 全索引排序,保证同日 ticker 顺序在 pandas 版本间确定
    clean = clean.sort_index()

    # review H1: 按 **唯一日期** 切 train/valid,避免在 uneven panel (listings/delistings
    # /suspensions) 上把同一天的部分 ticker 划到 train、部分划到 valid 产生日内 look-ahead
    unique_dates = clean.index.get_level_values("date").unique().sort_values()
    if len(unique_dates) < 2:
        raise ValueError(
            f"unique dates {len(unique_dates)} < 2, 无法切 train+valid"
        )
    split_idx = max(1, int(len(unique_dates) * (1 - _VALID_FRACTION)))
    split_idx = min(split_idx, len(unique_dates) - 1)
    split_date = unique_dates[split_idx]
    dates_all = clean.index.get_level_values("date")
    train_part = clean[dates_all < split_date]
    valid_part = clean[dates_all >= split_date]
    if train_part.empty or valid_part.empty:
        raise ValueError(
            f"样本 {after_drop} (unique dates {len(unique_dates)}) "
            f"不足以切 train+valid"
        )
    logger.info(
        "lgb split: %d rows before drop, %d after; "
        "train=%d valid=%d (split_date=%s)",
        before_drop,
        after_drop,
        len(train_part),
        len(valid_part),
        split_date,
    )

    X_train = train_part[feature_cols].to_numpy()
    y_train = train_part["_label"].to_numpy()
    X_valid = valid_part[feature_cols].to_numpy()
    y_valid = valid_part["_label"].to_numpy()

    merged_params = {**DEFAULT_PARAMS, **(params or {})}
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    valid_ds = lgb.Dataset(
        X_valid, label=y_valid, reference=train_ds, feature_name=feature_cols
    )
    callbacks = [lgb.early_stopping(early_stop, verbose=False)]
    booster = lgb.train(
        merged_params,
        train_ds,
        num_boost_round=num_rounds,
        valid_sets=[valid_ds],
        valid_names=["valid"],
        callbacks=callbacks,
    )
    logger.info(
        "lgb trained: features=%d train=%d valid=%d best_iter=%d",
        len(feature_cols),
        len(train_part),
        len(valid_part),
        getattr(booster, "best_iteration", -1),
    )
    return booster


def train_and_save(
    factor_provider: Any,
    market_data_provider: Any,
    tickers: list[str],
    start: date,
    end: date,
    factor_names: list[str] | None = None,
    version: str | None = None,
    params: dict[str, Any] | None = None,
    num_rounds: int = _DEFAULT_NUM_ROUNDS,
    early_stop: int = _DEFAULT_EARLY_STOP,
    promote_current: bool = True,
    overwrite: bool = False,
) -> Path:
    """端到端: 拉因子 + 行情 → 计算 label → 训练 → save_pickle → update_symlink。

    Args:
        factor_provider: FactorProvider 实例 (get_factors + list_factors)
        market_data_provider: MarketDataProvider 实例 (get_daily_bars)
        tickers: 股票代码列表
        start / end: 日期范围 (需留足 horizon 尾部用于 label)
        factor_names: 特征列表;None → factor_provider.list_factors() 过滤 type=expression
        version: YYYYMMDD,None = today
        params / num_rounds / early_stop: 传给 train_lightgbm
        promote_current: 默认 True,训练完自动 update_symlink 把 current 指向本版本;
                         False 保留"先保存再上线"流程 (OQ4)
        overwrite: 默认 False,同版本已存在则 FileExistsError;调参重训需显式 True
                   (review H2: 避免同日重训静默失败)

    Returns:
        版本化 artifact 路径 (如 data/models/lightgbm/20260420.pkl)
    """
    if not tickers:
        raise ValueError("tickers 不能为空")

    if factor_names is None:
        listed = factor_provider.list_factors()
        factor_names = [
            e["name"] for e in listed if e.get("type") == "expression"
        ]
        if not factor_names:
            raise ValueError(
                "factor_names 未指定且 list_factors() 无 expression 因子"
            )

    factor_df = factor_provider.get_factors(tickers, factor_names, start, end)
    bars = market_data_provider.get_daily_bars(tickers, start, end)
    if "adj_close" not in bars.columns:
        raise ValueError(
            f"market_data_provider.get_daily_bars 缺 'adj_close' 列, "
            f"实得 {list(bars.columns)}"
        )
    label = forward_return_5d(bars["adj_close"])
    label.name = LABEL_NAME

    booster = train_lightgbm(
        factor_df,
        label,
        params=params,
        num_rounds=num_rounds,
        early_stop=early_stop,
    )

    artifact = save_pickle(booster, MODEL_NAME, version=version, overwrite=overwrite)
    if promote_current:
        version_resolved = artifact.stem
        update_symlink(MODEL_NAME, version_resolved)
    logger.info("lightgbm train_and_save: %s", artifact)
    return artifact


__all__ = [
    "DEFAULT_PARAMS",
    "MODEL_NAME",
    "train_and_save",
    "train_lightgbm",
    "walk_forward_splits",
]
