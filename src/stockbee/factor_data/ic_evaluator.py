"""IC Evaluator — 因子 IC / ICIR 离线评估。

纯数值模块，无 provider 依赖。
计算因子值与前向收益的 Spearman rank 相关系数（Information Coefficient）。
"""

from __future__ import annotations

import math

import pandas as pd

_MIN_VALID = 20
_PRICE_COL = "adj_close"


def compute(
    factor_df: pd.DataFrame | pd.Series,
    prices_df: pd.DataFrame,
    shift: int = 1,
    window: int = 252,
) -> dict[str, float]:
    """计算因子的 IC / ICIR 统计。

    Args:
        factor_df: MultiIndex (date, ticker) 的单列 DataFrame 或 Series。
            多列 DataFrame → ValueError。
        prices_df: MultiIndex (date, ticker) 的 DataFrame，需含 adj_close 列。
        shift: 前向收益天数（factor at t vs return at t+shift）。
        window: 用于汇总统计的最近 IC 数量（默认 252 个交易日）。

    Returns:
        dict(ic_mean, ic_std, icir)。有效 IC 值不足 20 个时全部返回 NaN。
    """
    if shift < 1:
        raise ValueError(f"shift must be >= 1, got {shift}")
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    factor = _extract_factor(factor_df)
    if factor.index.duplicated().any():
        raise ValueError("factor_df has duplicate (date, ticker) rows")
    _validate_multiindex(prices_df, "prices_df")

    if _PRICE_COL not in prices_df.columns:
        raise ValueError(
            f"prices_df must contain '{_PRICE_COL}' column, "
            f"got columns: {list(prices_df.columns)}"
        )

    prices_sorted = prices_df.sort_index()
    _check_no_duplicate_index(prices_sorted, "prices_df")

    price_series = prices_sorted[_PRICE_COL]
    fwd_price = price_series.groupby(level="ticker").shift(-shift)
    fwd_ret = fwd_price / price_series - 1

    ic = _ic_series(factor.sort_index(), fwd_ret)

    # 窗口截取
    ic_windowed = ic.iloc[-window:] if len(ic) > window else ic
    valid = ic_windowed.dropna()

    if len(valid) < _MIN_VALID:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan")}

    ic_mean = float(valid.mean())
    ic_std = float(valid.std(ddof=1))
    icir = ic_mean / ic_std if ic_std != 0 else float("nan")
    return {"ic_mean": ic_mean, "ic_std": ic_std, "icir": icir}


def _ic_series(factor: pd.Series, fwd_ret: pd.Series) -> pd.Series:
    """逐日截面 Spearman rank IC。

    对每个 date，计算 factor 与 fwd_ret 在 tickers 间的 rank correlation。
    """
    combined = pd.DataFrame({"factor": factor, "fwd_ret": fwd_ret}).dropna()
    if combined.empty:
        return pd.Series(dtype=float)

    def _spearman(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return float("nan")
        return g["factor"].rank().corr(g["fwd_ret"].rank())

    return combined.groupby(level="date").apply(_spearman)


def _extract_factor(factor_df: pd.DataFrame | pd.Series) -> pd.Series:
    """将 factor_df 统一为 Series，校验 MultiIndex。"""
    if isinstance(factor_df, pd.Series):
        _validate_multiindex_series(factor_df, "factor_df")
        return factor_df
    _validate_multiindex(factor_df, "factor_df")
    if len(factor_df.columns) != 1:
        raise ValueError(
            f"factor_df must have exactly 1 column, got {len(factor_df.columns)}: "
            f"{list(factor_df.columns)}"
        )
    return factor_df.iloc[:, 0]


def _validate_multiindex(df: pd.DataFrame, name: str) -> None:
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(
            f"{name} must have MultiIndex (date, ticker), "
            f"got {type(df.index).__name__}"
        )
    if list(df.index.names) != ["date", "ticker"]:
        raise ValueError(
            f"{name} MultiIndex names must be ['date', 'ticker'], "
            f"got {list(df.index.names)}"
        )


def _validate_multiindex_series(s: pd.Series, name: str) -> None:
    if not isinstance(s.index, pd.MultiIndex):
        raise ValueError(
            f"{name} must have MultiIndex (date, ticker), "
            f"got {type(s.index).__name__}"
        )
    if list(s.index.names) != ["date", "ticker"]:
        raise ValueError(
            f"{name} MultiIndex names must be ['date', 'ticker'], "
            f"got {list(s.index.names)}"
        )


def _check_no_duplicate_index(df: pd.DataFrame, name: str) -> None:
    if df.index.duplicated().any():
        raise ValueError(f"{name} has duplicate (date, ticker) rows")
