"""ParquetFactorStore — 预计算因子 Parquet 存储。

按 group 分文件存储预计算因子（fundamental / sentiment / ml_score 等）。
文件布局: data/factors/{group}.parquet
Index schema: MultiIndex (date, ticker)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class ParquetFactorStore:
    """预计算因子的 Parquet 持久化层。

    每个 group 一个文件：{data_path}/{group}.parquet
    文件 index: MultiIndex (date, ticker)
    列: 该 group 内的因子列（factor_name → float）

    设计约束：
    - 纯 I/O 层，不做列级 schema 验证
    - merge-write 用 reindex + loc 赋值（新数据完全覆盖重叠格，含 NaN）
    - data_path 目录在首次写入时创建（不在 __init__ 创建）
    - list_precomputed_factors() 返回 sorted，顺序确定
    - 写入使用 write-then-rename 保证原子性
    """

    def __init__(self, data_path: str | Path = "data/factors") -> None:
        self._data_path = Path(data_path)

    # ------ 公共接口 ------

    def read_factors(
        self,
        group: str,
        tickers: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """读取预计算因子。

        Args:
            group: 数据来源分组名（e.g. "fundamental", "sentiment"）
            tickers: 过滤的 ticker 列表。None 或 [] 均返回全部 tickers。
            start: 开始日期（含）。None 表示不过滤下界。
            end: 结束日期（含）。None 表示不过滤上界。

        Returns:
            MultiIndex DataFrame (date, ticker) x factors。
            group 文件不存在时返回空 MultiIndex DataFrame。
        """
        self._validate_group(group)
        path = self._group_path(group)

        if not path.exists():
            idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
            return pd.DataFrame(index=idx)

        df = pd.read_parquet(path)

        # pyarrow 可能将 datetime.date 级别 round-trip 为 object dtype，
        # 导致与 pd.Timestamp 比较时 TypeError。统一转为 datetime64。
        date_level = df.index.get_level_values("date")
        if not pd.api.types.is_datetime64_any_dtype(date_level):
            df.index = df.index.set_levels(
                pd.to_datetime(df.index.levels[0]), level="date"
            )

        # date 过滤（只在非 None 时过滤，避免 pd.Timestamp(None)=NaT 的静默空结果）
        if start is not None:
            mask = df.index.get_level_values("date") >= pd.Timestamp(start)
            df = df[mask]
        if end is not None:
            mask = df.index.get_level_values("date") <= pd.Timestamp(end)
            df = df[mask]

        # ticker 过滤：None = 不过滤；[] = 不过滤（与 None 等价）
        if tickers is not None and len(tickers) > 0:
            mask = df.index.get_level_values("ticker").isin(tickers)
            df = df[mask]

        return df

    def write_factors(self, group: str, df: pd.DataFrame) -> None:
        """写入/合并预计算因子到 Parquet。

        merge 策略：reindex 扩展 + .loc 直接赋值
        - 保留旧数据中已有但新数据没有的行/列
        - 新数据完全覆盖重叠的 (date, ticker, column) 格，包括 NaN

        写入使用 write-then-rename 保证单次写入的原子性。

        Args:
            group: 数据来源分组名
            df: MultiIndex (date, ticker) x factors DataFrame

        Raises:
            ValueError: df 为空（0行）
            ValueError: df 不是 MultiIndex 或 index names 不是 ["date", "ticker"]
            ValueError: group 名包含非法字符（路径穿越防护）
        """
        self._validate_group(group)
        self._validate_write_df(df)

        if df.empty:
            logger.debug("Skipping write of empty DataFrame for group %r", group)
            return

        self._data_path.mkdir(parents=True, exist_ok=True)
        path = self._group_path(group)

        if path.exists():
            existing = pd.read_parquet(path)
            # 扩展到两者的行/列 union
            all_idx = existing.index.union(df.index)
            all_cols = existing.columns.union(df.columns)
            combined = existing.reindex(index=all_idx, columns=all_cols)
            # 新数据完全覆盖：.loc 赋值是无条件的，NaN 也会写入
            combined.loc[df.index, df.columns] = df
            combined.sort_index(inplace=True)
            self._atomic_write(combined, path)
            logger.debug(
                "Updated factor group %r: %d rows, %d cols",
                group, len(combined), len(combined.columns),
            )
        else:
            df_out = df.copy()
            df_out.sort_index(inplace=True)
            self._atomic_write(df_out, path)
            logger.debug(
                "Created factor group %r: %d rows, %d cols",
                group, len(df_out), len(df_out.columns),
            )

    def list_precomputed_factors(self) -> list[str]:
        """返回所有已存储的 group 名列表（sorted，顺序确定）。

        Returns:
            Parquet 文件 stem 列表，e.g. ["fundamental", "ml_score", "sentiment"]
        """
        if not self._data_path.exists():
            return []
        return sorted(p.stem for p in self._data_path.glob("*.parquet"))

    # ------ 私有方法 ------

    def _group_path(self, group: str) -> Path:
        return self._data_path / f"{group}.parquet"

    def _atomic_write(self, df: pd.DataFrame, path: Path) -> None:
        """Write-then-rename 原子写入（POSIX rename 是原子操作）。"""
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp)
        tmp.rename(path)

    def _validate_group(self, group: str) -> None:
        """拒绝空 group 名和包含路径穿越字符的 group 名。"""
        if not group or "/" in group or "\\" in group or ".." in group:
            raise ValueError(f"Invalid group name: {group!r}")

    def _validate_write_df(self, df: pd.DataFrame) -> None:
        """确保写入的 df 有正确的 MultiIndex (date, ticker)。"""
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError(
                "df must have MultiIndex (date, ticker), "
                f"got {type(df.index).__name__}"
            )
        if list(df.index.names) != ["date", "ticker"]:
            raise ValueError(
                f"MultiIndex names must be ['date', 'ticker'], got {list(df.index.names)}"
            )
