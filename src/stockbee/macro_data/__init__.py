"""Macro Data 模块 — 19 个 FRED 宏观指标的存储、同步和 Z-score 计算。

实现 03-macro-data Room 的全部功能：
- Parquet 时间序列存储（回测）
- FRED API 实时拉取（实盘）
- 252 天滚动 Z-score 标准化
- Point-in-time 对齐（无前看偏差）
"""

from .indicators import FRED_INDICATORS, INDICATOR_GROUPS
from .parquet_macro import ParquetMacroProvider
from .fred_macro import FredMacroProvider
from .sync import MacroDataSyncer

__all__ = [
    "FRED_INDICATORS",
    "INDICATOR_GROUPS",
    "ParquetMacroProvider",
    "FredMacroProvider",
    "MacroDataSyncer",
]
