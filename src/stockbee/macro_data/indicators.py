"""FRED 宏观指标定义 — 19 个指标覆盖 8 个经济维度。

来源：Tech Design §2.4 + §3.5, research-macro-features.md
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FredIndicator:
    """单个 FRED 指标的元数据。"""
    code: str           # FRED series ID
    name: str           # 中文名称
    dimension: str      # 经济维度
    frequency: str      # 发布频率: "daily" | "weekly" | "monthly" | "quarterly"
    unit: str           # 单位描述


# 19 个 FRED 指标完整定义
FRED_INDICATORS: dict[str, FredIndicator] = {
    # 利率 (3)
    "DFF": FredIndicator("DFF", "联邦基金利率", "利率", "daily", "percent"),
    "T10Y2Y": FredIndicator("T10Y2Y", "10Y-2Y期限差", "利率", "daily", "percent"),
    "DGS10": FredIndicator("DGS10", "10Y美债收益率", "利率", "daily", "percent"),

    # 就业 (3)
    "PAYEMS": FredIndicator("PAYEMS", "非农就业人数", "就业", "monthly", "thousands"),
    "UNRATE": FredIndicator("UNRATE", "失业率", "就业", "monthly", "percent"),
    "ICSA": FredIndicator("ICSA", "初请失业金人数", "就业", "weekly", "number"),

    # 物价 (2)
    "CPIAUCSL": FredIndicator("CPIAUCSL", "CPI", "物价", "monthly", "index"),
    "PPIFIS": FredIndicator("PPIFIS", "PPI", "物价", "monthly", "index"),

    # 增长 (2)
    "GDP": FredIndicator("GDP", "GDP", "增长", "quarterly", "billions"),
    "INDPRO": FredIndicator("INDPRO", "工业产值指数", "增长", "monthly", "index"),

    # 信用 (2)
    "BAMLH0A0HYM2": FredIndicator("BAMLH0A0HYM2", "高收益债利差(OAS)", "信用", "daily", "percent"),
    "DRTSCILM": FredIndicator("DRTSCILM", "银行信贷收紧比例", "信用", "quarterly", "percent"),

    # 流动性 (2)
    "M2SL": FredIndicator("M2SL", "M2货币供应量", "流动性", "monthly", "billions"),
    "VIXCLS": FredIndicator("VIXCLS", "VIX波动率指数", "流动性", "daily", "index"),

    # 汇率 (1)
    "DEXUSEU": FredIndicator("DEXUSEU", "USD/EUR汇率", "汇率", "daily", "exchange_rate"),

    # 商品 (2)
    "DCOILWTICO": FredIndicator("DCOILWTICO", "WTI原油价格", "商品", "daily", "dollars"),
    "PCOPPUSDM": FredIndicator("PCOPPUSDM", "铜价(全球)", "商品", "monthly", "dollars"),
}

# 按经济维度分组
INDICATOR_GROUPS: dict[str, list[str]] = {}
for code, ind in FRED_INDICATORS.items():
    INDICATOR_GROUPS.setdefault(ind.dimension, []).append(code)

# 所有指标代码列表
ALL_CODES: list[str] = list(FRED_INDICATORS.keys())
