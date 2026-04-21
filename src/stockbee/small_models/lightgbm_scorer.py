"""LightGBM 推理 + ml_score Parquet 落盘 + evaluate_ml_score。

决策:
- m4 推理路径: load_pickle("lightgbm", version) → Booster → 批预测 →
  写 data/factors/ml_score.parquet (group="ml_score", ParquetFactorStore)
- LocalFactorProvider._build_precomputed_index 已泛化,ml_score 自动发现,
  m4 对 factor-storage 代码零改动
- D2: evaluate_ml_score(shift=5) 独立函数,不改 get_ic_report(shift=1 硬编码)
- 性能约束: 1000 rows × 158 cols 推理 <50ms GPU / <150ms CPU (@pytest.mark.perf)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stockbee.factor_data.ic_evaluator import compute as compute_ic
from stockbee.factor_data.parquet_factor import ParquetFactorStore

from .lightgbm_trainer import MODEL_NAME
from .model_io import load_pickle
from .paths import ML_SCORE_PARQUET

logger = logging.getLogger(__name__)

ML_SCORE_GROUP = "ml_score"
ML_SCORE_COLUMN = "ml_score"
# 交易日 → 日历日保守换算, 与 factor_data.local_provider 一致
_CALENDAR_MULTIPLIER = 2


class LightGBMScorer:
    """LightGBM artifact 加载 + 批推理 + Parquet 落盘。

    Args:
        version: 传给 m1 model_io.load_pickle; 默认 "current" 读 symlink
        booster: 直接注入已加载 booster (单测复用,跳过 pickle 反序列化)

    使用::

        s = LightGBMScorer()                    # 加载 data/models/lightgbm/current.pkl
        s.predict(factor_df)                    # pd.Series, name="ml_score"
        s.predict_and_save(factor_provider, tickers, start, end)  # Path
    """

    def __init__(
        self,
        version: str = "current",
        booster: Any = None,
    ) -> None:
        self._version = version
        self._booster = booster
        if booster is None:
            self._booster = load_pickle(MODEL_NAME, version=version)
        self._feature_names: list[str] = list(
            self._booster.feature_name() or []
        )

    @property
    def booster(self) -> Any:
        return self._booster

    @property
    def version(self) -> str:
        return self._version

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def predict(self, factor_df: pd.DataFrame) -> pd.Series:
        """对 factor_df 做预测。

        Args:
            factor_df: MultiIndex(date, ticker) × feature 列;列序由 booster
                       feature_name 自动对齐 (严格按 booster 顺序取子集)

        Returns:
            pd.Series, 同 X_df MultiIndex, dtype float, name="ml_score"。
            全特征 NaN 的行 → NaN (review C1: 统一 mock/real 语义;LightGBM
            Booster 原生会把 all-NaN 当 missing 返回非 NaN,本层显式 mask 保证
            "无特征 → 无预测" 的可组合语义)。
        """
        if factor_df is None:
            raise ValueError("factor_df is None")
        _validate_multiindex(factor_df)
        if factor_df.empty:
            raise ValueError("factor_df 为空")

        if not self._feature_names:
            # review H2: booster 无 feature_name 则缺失列序锚点,强制报错
            raise ValueError(
                "booster 无 feature_name(), 无法做列对齐; "
                "请用 m3 训练产物或在构造时显式注入 feature_names"
            )
        missing = [c for c in self._feature_names if c not in factor_df.columns]
        if missing:
            truncated = missing[:10]
            raise ValueError(
                f"factor_df 缺 booster 需要的特征列: {truncated}"
                + (f" ... ({len(missing)} total)" if len(missing) > 10 else "")
            )
        X_df = factor_df[self._feature_names]

        # review C1: 显式 mask all-NaN 行,保证输出 NaN 与 docstring 一致
        all_nan_mask = X_df.isna().all(axis=1)
        preds = self._booster.predict(X_df.fillna(0.0).to_numpy())
        preds = np.asarray(preds, dtype=float)
        if all_nan_mask.any():
            preds[all_nan_mask.to_numpy()] = float("nan")
        out = pd.Series(preds, index=X_df.index, dtype=float, name=ML_SCORE_COLUMN)
        return out

    def predict_and_save(
        self,
        factor_provider: Any,
        tickers: list[str],
        start: date,
        end: date,
        factor_names: list[str] | None = None,
        store: ParquetFactorStore | None = None,
    ) -> Path:
        """Pull factors → predict → merge-write data/factors/ml_score.parquet。

        Args:
            factor_provider: FactorProvider 实例
            tickers / start / end: 传 factor_provider.get_factors
            factor_names: 特征列表, 默认 = booster.feature_name()
            store: 注入 ParquetFactorStore (测试用); 默认指向 ML_SCORE_PARQUET 父目录

        Returns:
            落盘 parquet 文件路径
        """
        if not tickers:
            raise ValueError("tickers 不能为空")
        if factor_names is None:
            if not self._feature_names:
                raise ValueError(
                    "booster 无 feature_name, 必须显式传 factor_names"
                )
            factor_names = self._feature_names

        factor_df = factor_provider.get_factors(tickers, factor_names, start, end)
        if factor_df is None or factor_df.empty:
            raise ValueError(
                f"factor_provider.get_factors 返回空 (tickers={tickers}, "
                f"start={start}, end={end})"
            )
        preds = self.predict(factor_df).dropna()
        if preds.empty:
            raise ValueError("预测结果全为 NaN,不落盘")

        df = preds.to_frame()
        if store is None:
            store = ParquetFactorStore(data_path=ML_SCORE_PARQUET.parent)
        store.write_factors(ML_SCORE_GROUP, df)

        # review H1: 不再比较 Path 相等性 (绝对/相对路径语义差异);
        # 直接用 store 的 data_path 拼接 group 文件名,保证与 ParquetFactorStore
        # 内部的 _group_path() 一致
        out_path = store._data_path / f"{ML_SCORE_GROUP}.parquet"
        logger.info(
            "LightGBMScorer.predict_and_save: %d rows → %s", len(df), out_path
        )
        return out_path


# ---- D2: 独立 IC 评估 ----


def evaluate_ml_score(
    factor_provider: Any,
    market_data_provider: Any,
    universe: list[str],
    start: date,
    end: date,
    shift: int = 5,
    window: int = 252,
) -> dict[str, float]:
    """对 ml_score 因子计算 IC / ICIR (D2 决策: 独立于 get_ic_report,不改 factor-storage)。

    Args:
        factor_provider: FactorProvider 实例 (读 ml_score)
        market_data_provider: MarketDataProvider 实例 (读 OHLCV adj_close)
        universe: 股票代码列表 (非空)
        start / end: 日期范围
        shift: 前向收益天数,默认 5 (B3 label horizon)
        window: IC 汇总窗口,默认 252

    Returns:
        dict(ic_mean, ic_std, icir)

    Raises:
        ValueError: universe 为空 / 无 ml_score 因子 / prices 缺 adj_close
    """
    if not universe:
        raise ValueError("universe 不能为空")
    if shift < 1:
        raise ValueError(f"shift must be >= 1, got {shift}")

    factor_df = factor_provider.get_factors(
        universe, [ML_SCORE_COLUMN], start, end
    )
    if factor_df is None or factor_df.empty:
        raise ValueError(
            f"factor_provider 未返回 ml_score 数据 "
            f"(universe={universe}, range={start}..{end})"
        )
    if ML_SCORE_COLUMN not in factor_df.columns:
        raise ValueError(
            f"factor_provider 返回列缺 {ML_SCORE_COLUMN!r}: "
            f"{list(factor_df.columns)}"
        )

    prices = market_data_provider.get_daily_bars(
        universe,
        start,
        end + timedelta(days=shift * _CALENDAR_MULTIPLIER),
    )
    return compute_ic(
        factor_df[[ML_SCORE_COLUMN]],
        prices,
        shift=shift,
        window=window,
    )


# ---- 内部 ----


def _validate_multiindex(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f"factor_df 须 MultiIndex(date, ticker), 实得 {type(df.index).__name__}"
        )
    if set(df.index.names) < {"date", "ticker"}:
        raise ValueError(
            f"MultiIndex 须含 date+ticker, 实得 names={df.index.names}"
        )


__all__ = [
    "LightGBMScorer",
    "ML_SCORE_COLUMN",
    "ML_SCORE_GROUP",
    "evaluate_ml_score",
]
