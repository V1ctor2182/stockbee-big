"""LightGBM 训练标签工具。

B3 决策: label = mean(next_horizon_days_daily_returns)
  daily_ret[t] = (adj_close[t] - adj_close[t-1]) / adj_close[t-1]
  label[t] = mean(daily_ret[t+1], ..., daily_ret[t+horizon])

尾部 horizon 天无 label (NaN),训练前 drop。

决策来源: spec.md Decisions 表 B3 (2026-04-13)。
"""

from __future__ import annotations

import pandas as pd

DEFAULT_HORIZON = 5
LABEL_NAME = "label_fwd_5d"


def forward_return_5d(
    prices: pd.Series | pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
) -> pd.Series:
    """对 MultiIndex(date, ticker) 价格序列计算 B3 forward-return label。

    Args:
        prices: Series (MultiIndex date, ticker) 或 DataFrame (含 "adj_close" 列,
                同样的 MultiIndex)
        horizon: 前瞻天数,默认 5

    Returns:
        pd.Series,与输入同 MultiIndex,尾部最后 horizon 行按 ticker 为 NaN。
        名字 = LABEL_NAME ("label_fwd_5d")。

    Raises:
        TypeError: prices 既不是 Series 也不是 DataFrame
        ValueError: horizon < 1,或 prices 缺 MultiIndex (date, ticker),
                    或 DataFrame 缺 "adj_close" 列

    实现 (B3 公式):
        daily_ret = prices.groupby("ticker").pct_change()
        label = (daily_ret
                  .groupby("ticker")
                  .transform(lambda r: r.shift(-1).rolling(horizon).mean().shift(-(horizon-1))))
        # 等价: [t+1, t+horizon] horizon 天日收益率均值
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    series = _extract_adj_close(prices)
    _validate_multiindex(series.index)

    # sort_index 保证 groupby(level='ticker') 内部按日期升序
    series = series.sort_index()

    daily_ret = series.groupby(level="ticker", group_keys=False).pct_change()
    label = daily_ret.groupby(level="ticker", group_keys=False).transform(
        lambda ret: ret.shift(-1).rolling(horizon).mean().shift(-(horizon - 1))
    )
    label.name = LABEL_NAME
    return label


# ---- 内部辅助 ----


def _extract_adj_close(prices: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(prices, pd.Series):
        return prices.astype("float64")
    if isinstance(prices, pd.DataFrame):
        if "adj_close" not in prices.columns:
            raise ValueError(
                "DataFrame 必须含 'adj_close' 列 (找到列: "
                f"{list(prices.columns)})"
            )
        return prices["adj_close"].astype("float64")
    raise TypeError(
        f"prices 须为 Series 或 DataFrame, 实得 {type(prices).__name__}"
    )


def _validate_multiindex(idx: pd.Index) -> None:
    if not isinstance(idx, pd.MultiIndex):
        raise ValueError(
            "prices 索引须为 MultiIndex(date, ticker), "
            f"实得 {type(idx).__name__}"
        )
    names = [n for n in idx.names]
    if "ticker" not in names:
        raise ValueError(
            f"MultiIndex 须包含 'ticker' level, 实得 names={names}"
        )


__all__ = ["DEFAULT_HORIZON", "LABEL_NAME", "forward_return_5d"]
