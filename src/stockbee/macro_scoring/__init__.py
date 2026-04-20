"""Macro Scoring 模块 — 日度看涨/看跌评分 + 经济周期分类。

5 种经济制度 (expansion/overheating/stagflation/recession/recovery)。
消费 MacroProvider Z-scores + Polymarket 概率悬崖 + 经济日历高波动日。

来源：Tech Design §3.5 MacroTiltEngine
"""

from .scorer import MacroScorer, MacroScore, RegimeType
from .llm_analyst import LLMMacroAnalyst, MacroInsight
from .provider import MacroScoringProvider
from .sector_tilts import SectorTilter, GICS_SECTORS

__all__ = [
    "MacroScorer",
    "MacroScore",
    "RegimeType",
    "LLMMacroAnalyst",
    "MacroInsight",
    "MacroScoringProvider",
    "SectorTilter",
    "GICS_SECTORS",
]
