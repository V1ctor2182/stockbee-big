"""LocalFactorProvider — 因子数据 Provider 实现。

整合 Alpha158 表达式引擎（动态计算）和 ParquetFactorStore（预计算读取），
实现 FactorProvider 接口的 get_factors / list_factors / get_ic_report。

MarketDataProvider 通过 setter 注入（表达式因子和 IC 报告需要 OHLCV 数据）。
IC 评估还需要通过 ic_universe setter 注入 ticker + 日期范围。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from stockbee.providers.base import ProviderConfig
from stockbee.providers.interfaces import FactorProvider, MarketDataProvider

from .alpha158 import Alpha158
from .expression_engine import Evaluator
from .ic_evaluator import compute as compute_ic
from .parquet_factor import ParquetFactorStore

logger = logging.getLogger(__name__)

# 交易日 → 日历日转换：保守 2x 覆盖周末 + 长假（春节/国庆）。
_CALENDAR_MULTIPLIER = 2
_IC_SHIFT = 1


@dataclass
class ICUniverse:
    """IC 评估所需的 ticker universe + 日期范围。"""

    tickers: list[str]
    start: date
    end: date


def _normalize_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保 MultiIndex date 级别是 datetime64，防止 date vs Timestamp 比较 TypeError。"""
    if df.empty or not isinstance(df.index, pd.MultiIndex):
        return df
    if "date" not in df.index.names:
        return df
    date_level = df.index.get_level_values("date")
    if not pd.api.types.is_datetime64_any_dtype(date_level):
        df = df.copy()
        # 按名字定位 level，不依赖 date 在 MultiIndex 的位置
        date_pos = df.index.names.index("date")
        df.index = df.index.set_levels(
            pd.to_datetime(df.index.levels[date_pos]), level="date",
        )
    return df


class LocalFactorProvider(FactorProvider):
    """因子数据 Provider — 整合表达式引擎 + 预计算存储。

    Expression factors: Alpha158 registry → Evaluator 动态计算
    Precomputed factors: ParquetFactorStore 读取

    名字冲突策略: expression 优先，同名 precomputed 被 shadow。
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        yaml_path = params.get("expression_config")
        data_path = params.get("precomputed_path", "data/factors")

        self._alpha158 = Alpha158(
            yaml_path=Path(yaml_path) if yaml_path else None,
        )
        self._parquet = ParquetFactorStore(data_path=data_path)
        self._data_path = Path(data_path)
        self._market_data: MarketDataProvider | None = None
        self._ic_universe: ICUniverse | None = None
        self._precomputed_index: dict[str, str] = {}  # col_name → group
        self._precomputed_index_built = False

    # ------ Setter injection ------

    @property
    def market_data(self) -> MarketDataProvider | None:
        return self._market_data

    @market_data.setter
    def market_data(self, provider: MarketDataProvider) -> None:
        self._market_data = provider

    @property
    def ic_universe(self) -> ICUniverse | None:
        return self._ic_universe

    @ic_universe.setter
    def ic_universe(self, universe: ICUniverse) -> None:
        self._ic_universe = universe

    # ------ Lifecycle ------

    def _do_initialize(self) -> None:
        self._ensure_precomputed_index()

    def _do_shutdown(self) -> None:
        pass

    # ------ FactorProvider interface ------

    def get_factors(
        self,
        tickers: list[str],
        factor_names: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """获取因子数据。

        Expression factors → Alpha158 + Evaluator 动态计算（需要 market_data）。
        Precomputed factors → ParquetFactorStore 读取。
        未知 factor name → raise ValueError。

        Returns:
            MultiIndex DataFrame (date, ticker) x factor_names。
            列顺序 = factor_names 的传入顺序。
            混合来源使用 outer join，缺失位置为 NaN。
        """
        if not factor_names:
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(index=idx)

        self._ensure_precomputed_index()

        # --- 分类 ---
        expression_names: list[str] = []
        precomputed_names: list[tuple[str, str]] = []  # (col_name, group)
        for name in factor_names:
            if name in self._alpha158:
                expression_names.append(name)
            elif name in self._precomputed_index:
                precomputed_names.append((name, self._precomputed_index[name]))
            else:
                raise ValueError(f"Unknown factor: {name!r}")

        results: dict[str, pd.Series] = {}

        # --- Expression factors: batched, single Evaluator ---
        if expression_names:
            if self._market_data is None:
                raise RuntimeError(
                    "MarketDataProvider not set — required for expression factors"
                )
            max_lb = max(
                self._alpha158.max_lookback(n) for n in expression_names
            )
            extended_start = start - timedelta(
                days=int(max_lb * _CALENDAR_MULTIPLIER),
            )
            ohlcv = self._market_data.get_daily_bars(
                tickers, extended_start, end,
            )
            ohlcv = _normalize_date_index(ohlcv)

            if ohlcv.empty:
                for name in expression_names:
                    results[name] = pd.Series(
                        dtype=float, name=name,
                        index=pd.MultiIndex.from_tuples(
                            [], names=["date", "ticker"],
                        ),
                    )
            else:
                ev = Evaluator(ohlcv)
                for name in expression_names:
                    raw = ev.evaluate(self._alpha158.get_expression(name))
                    # Scalar guard: constant-only expressions → broadcast
                    if not isinstance(raw, pd.Series):
                        raw = pd.Series(raw, index=ohlcv.index)
                    results[name] = raw.rename(name)

        # --- Precomputed factors: group-batched read ---
        if precomputed_names:
            groups: dict[str, list[str]] = {}
            for name, group in precomputed_names:
                groups.setdefault(group, []).append(name)

            for group, cols in groups.items():
                df = self._parquet.read_factors(
                    group, tickers=tickers, start=start, end=end,
                )
                for col in cols:
                    if col not in df.columns:
                        raise ValueError(
                            f"Factor {col!r} expected in group {group!r} "
                            f"but not found in data"
                        )
                    results[col] = df[col].rename(col)

        if not results:
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(index=idx, columns=factor_names)

        # --- Merge + trim + column order ---
        merged = pd.concat(results.values(), axis=1)

        # Trim expression factors' lookback overshoot to [start, end]
        dates = merged.index.get_level_values("date")
        mask = (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
        merged = merged[mask]

        merged = merged[factor_names]
        merged.sort_index(inplace=True)
        return merged

    def list_factors(self) -> list[dict[str, str]]:
        """列出所有可用因子及其类型。

        Expression factors 优先：同名 precomputed 被 shadow，不出现在列表中。
        """
        self._ensure_precomputed_index()
        result: list[dict[str, str]] = []
        expr_names = set(self._alpha158.list_factor_names())

        for name in self._alpha158.list_factor_names():
            result.append({"name": name, "type": "expression"})

        for col, group in sorted(self._precomputed_index.items()):
            if col not in expr_names:
                result.append(
                    {"name": col, "type": "precomputed", "group": group},
                )

        return result

    def get_ic_report(
        self,
        factor_name: str,
        window: int = 252,
    ) -> dict[str, float]:
        """返回因子的滚动 IC/ICIR 统计。

        需要 market_data 和 ic_universe 均已设置。
        shift 固定为 1（单日前向收益）。
        """
        if self._market_data is None:
            raise RuntimeError(
                "MarketDataProvider not set — required for IC report"
            )
        if self._ic_universe is None:
            raise RuntimeError(
                "IC universe not configured — "
                "set ic_universe before calling get_ic_report"
            )

        self._ensure_precomputed_index()
        u = self._ic_universe
        # 多取价格数据，供 compute_ic 计算前向收益（shift(-1) 需要 T+1 价格）
        # 用 5 天 buffer 覆盖周末 + 节假日（原 _IC_SHIFT * 2 = 2 天在周五结尾时不够）
        price_end = u.end + timedelta(days=max(_IC_SHIFT * _CALENDAR_MULTIPLIER, 5))

        if factor_name in self._alpha158:
            lookback = self._alpha158.max_lookback(factor_name)
            extended_start = u.start - timedelta(
                days=int(lookback * _CALENDAR_MULTIPLIER),
            )
            ohlcv = self._market_data.get_daily_bars(
                u.tickers, extended_start, price_end,
            )
            ohlcv = _normalize_date_index(ohlcv)

            if ohlcv.empty:
                return {
                    "ic_mean": float("nan"),
                    "ic_std": float("nan"),
                    "icir": float("nan"),
                }

            ev = Evaluator(ohlcv)
            raw = ev.evaluate(self._alpha158.get_expression(factor_name))
            if not isinstance(raw, pd.Series):
                raw = pd.Series(raw, index=ohlcv.index)
            # Trim factor to universe range (prices keep extended range)
            dates = raw.index.get_level_values("date")
            factor_series = raw[
                (dates >= pd.Timestamp(u.start))
                & (dates <= pd.Timestamp(u.end))
            ]
            prices = ohlcv
        elif factor_name in self._precomputed_index:
            group = self._precomputed_index[factor_name]
            df = self._parquet.read_factors(
                group, tickers=u.tickers, start=u.start, end=u.end,
            )
            if factor_name not in df.columns:
                raise ValueError(
                    f"Factor {factor_name!r} not in group {group!r}"
                )
            factor_series = df[factor_name]
            prices = self._market_data.get_daily_bars(
                u.tickers, u.start, price_end,
            )
            prices = _normalize_date_index(prices)
        else:
            raise ValueError(f"Unknown factor: {factor_name!r}")

        return compute_ic(factor_series, prices, shift=_IC_SHIFT, window=window)

    def refresh_precomputed_index(self) -> None:
        """强制重建 precomputed col → group 反向索引。

        Lazy build 后若有新 group 落盘（e.g. 同 provider 实例内先 get_factors 再
        write_factors 后又想读新 group），需要显式调用本方法，否则新 group 的
        列会被当作 "Unknown factor"。
        """
        self._precomputed_index = {}
        self._precomputed_index_built = False
        self._ensure_precomputed_index()

    # ------ Private helpers ------

    def _ensure_precomputed_index(self) -> None:
        """Lazy-build precomputed col → group reverse index."""
        if self._precomputed_index_built:
            return
        groups = self._parquet.list_precomputed_factors()
        if groups:
            self._precomputed_index = self._build_precomputed_index(groups)
        self._precomputed_index_built = True

    def _build_precomputed_index(
        self,
        groups: list[str],
    ) -> dict[str, str]:
        """Read Parquet schema metadata to build col_name → group mapping.

        只读 schema，不加载数据。Index 列 (date, ticker) 被排除。
        跨 group 的同名列会 log warning，后者覆盖前者。
        """
        mapping: dict[str, str] = {}
        for group in groups:
            path = self._data_path / f"{group}.parquet"
            if not path.exists():
                continue
            schema = pq.read_schema(path)
            for name in schema.names:
                if name in ("date", "ticker"):
                    continue
                if name in mapping:
                    logger.warning(
                        "Precomputed factor %r exists in both group %r and %r; "
                        "%r wins",
                        name, mapping[name], group, group,
                    )
                mapping[name] = group
        return mapping
