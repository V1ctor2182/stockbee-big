"""Macro Sources 模块 — 补充宏观数据源。

Polymarket 事件概率 + FRED 经济日历。
作为 MacroTiltEngine 的外生补充因子。
"""

from .polymarket import PolymarketFetcher
from .calendar import EconomicCalendar

__all__ = [
    "PolymarketFetcher",
    "EconomicCalendar",
]
