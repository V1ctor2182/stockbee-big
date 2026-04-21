"""MacroScorer — 经济周期分类 + 日度 bull/bear 评分。

规则引擎实现（Phase 1）。基于 19 个 FRED 指标 Z-score 判断经济周期，
输出 macro_score [-1, +1] 供下游 sector/style tilt 使用。

来源：Tech Design §3.5, plan-milestones 2026-04-15
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class RegimeType(str, Enum):
    """5 种经济制度。"""
    EXPANSION = "expansion"
    OVERHEATING = "overheating"
    STAGFLATION = "stagflation"
    RECESSION = "recession"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class MacroScore:
    """日度宏观评分结果。"""
    score: float                          # [-1.0, +1.0]  负=看跌 正=看涨
    regime: RegimeType
    regime_confidence: dict[str, float]   # {regime_name: probability}
    scored_at: date
    signals: dict[str, float] = field(default_factory=dict)  # 诊断用: 各维度信号


# ---------------------------------------------------------------------------
# Regime classification rules
# ---------------------------------------------------------------------------
# 每条规则: (indicator_code, operator, threshold)
# 满足越多条 → confidence 越高

_REGIME_RULES: dict[RegimeType, list[tuple[str, str, float]]] = {
    # Expansion: GDP↑ + employment↑ + moderate inflation
    RegimeType.EXPANSION: [
        ("GDP", ">", 0.0),
        ("INDPRO", ">", 0.0),
        ("PAYEMS", ">", 0.0),
        ("UNRATE", "<", 0.0),
        ("CPIAUCSL", "<", 1.0),     # inflation below 1 std
        ("T10Y2Y", ">", 0.0),       # positive yield curve
    ],
    # Overheating: high inflation + rising rates
    RegimeType.OVERHEATING: [
        ("CPIAUCSL", ">", 1.0),
        ("PPIFIS", ">", 1.0),
        ("DFF", ">", 0.5),
        ("DGS10", ">", 0.5),
        ("GDP", ">", 0.0),          # still growing
        ("UNRATE", "<", 0.0),       # still low unemployment
    ],
    # Stagflation: high inflation + declining growth
    RegimeType.STAGFLATION: [
        ("CPIAUCSL", ">", 1.0),
        ("GDP", "<", 0.0),
        ("INDPRO", "<", 0.0),
        ("UNRATE", ">", 0.5),
        ("BAMLH0A0HYM2", ">", 0.5),  # widening credit spread
    ],
    # Recession: declining growth + rising unemployment
    RegimeType.RECESSION: [
        ("GDP", "<", -0.5),
        ("INDPRO", "<", -0.5),
        ("PAYEMS", "<", -0.5),
        ("UNRATE", ">", 1.0),
        ("ICSA", ">", 1.0),
        ("T10Y2Y", "<", 0.0),       # inverted yield curve
        ("BAMLH0A0HYM2", ">", 1.0), # wide credit spreads
    ],
    # Recovery: GDP bottoming + expectations rising
    RegimeType.RECOVERY: [
        ("GDP", ">", -0.5),         # GDP recovering (not deeply negative)
        ("GDP", "<", 0.5),          # but not yet strong
        ("INDPRO", ">", 0.0),
        ("UNRATE", ">", 0.0),       # unemployment still elevated
        ("VIXCLS", "<", 0.0),       # volatility declining
        ("T10Y2Y", ">", 0.0),       # yield curve normalizing
    ],
}


# ---------------------------------------------------------------------------
# Score weights by indicator dimension
# ---------------------------------------------------------------------------
# positive z_score on growth/employment → bullish (+)
# positive z_score on inflation/risk → bearish (-)

_SCORE_WEIGHTS: dict[str, float] = {
    # Growth (+)
    "GDP": 0.12,
    "INDPRO": 0.08,
    # Employment (+)
    "PAYEMS": 0.10,
    "UNRATE": -0.08,    # higher unemployment = bearish
    "ICSA": -0.05,      # higher claims = bearish
    # Inflation (-)
    "CPIAUCSL": -0.08,
    "PPIFIS": -0.05,
    # Rates (context-dependent, net negative for equities)
    "DFF": -0.06,
    "DGS10": -0.04,
    "T10Y2Y": 0.06,     # positive curve = healthy
    # Credit (-)
    "BAMLH0A0HYM2": -0.08,  # wider spread = risk
    "DRTSCILM": -0.04,
    # Liquidity
    "M2SL": 0.06,
    "VIXCLS": -0.06,    # higher VIX = bearish
    # FX / Commodities
    "DEXUSEU": -0.01,   # weak USD mildly negative
    "DCOILWTICO": -0.01,
    "PCOPPUSDM": 0.04,  # copper as growth proxy
}
# Weights sum ≈ 0 by design (balanced bull/bear in neutral regime).


# ---------------------------------------------------------------------------
# MacroScorer
# ---------------------------------------------------------------------------

class MacroScorer:
    """日度宏观评分引擎（规则引擎版）。

    Args:
        macro_provider: MacroProvider 实例 (get_latest_z_scores)
        polymarket: PolymarketFetcher 实例 (可选)
        calendar: EconomicCalendar 实例 (可选)
        propagation_weight: 知识图谱传播信号权重 (Phase 1 = 0.0)
    """

    def __init__(
        self,
        macro_provider,
        polymarket=None,
        calendar=None,
        *,
        propagation_weight: float = 0.0,
    ):
        self._macro = macro_provider
        self._polymarket = polymarket
        self._calendar = calendar
        self._propagation_weight = propagation_weight

    def score(self, as_of: date | None = None) -> MacroScore:
        """计算日度宏观评分。

        Args:
            as_of: 评分日期, None = today

        Returns:
            MacroScore with score, regime, confidence
        """
        scored_at = as_of or date.today()

        # 1. Get Z-scores from macro provider
        z_scores = self._macro.get_latest_z_scores()
        if not z_scores:
            logger.warning("Empty z_scores from macro_provider, returning neutral")
            return MacroScore(
                score=0.0,
                regime=RegimeType.EXPANSION,
                regime_confidence={r.value: 0.2 for r in RegimeType},
                scored_at=scored_at,
                signals={},
            )

        # 2. Classify regime
        regime, confidence = self._classify_regime(z_scores)

        # 3. Compute base score
        base_score = self._compute_base_score(z_scores)

        # 4. Apply optional adjustments
        score = base_score
        signals = {"base_score": base_score}

        poly_adj = self._apply_polymarket_adjustment()
        if poly_adj != 0.0:
            score += poly_adj
            signals["polymarket_adj"] = poly_adj

        # Clip to [-1, +1]
        score = max(-1.0, min(1.0, score))

        # 5. Dampen on high-volatility calendar days (less conviction)
        if self._is_high_volatility_day(scored_at):
            score *= 0.8
            signals["hv_dampening"] = 0.8

        return MacroScore(
            score=score,
            regime=regime,
            regime_confidence=confidence,
            scored_at=scored_at,
            signals=signals,
        )

    def _classify_regime(
        self, z_scores: dict[str, float]
    ) -> tuple[RegimeType, dict[str, float]]:
        """Classify economic regime based on Z-score rules.

        Returns:
            (best_regime, {regime_name: match_ratio})
        """
        match_counts: dict[RegimeType, float] = {}

        for regime, rules in _REGIME_RULES.items():
            matched = 0
            applicable = 0
            for code, op, threshold in rules:
                if code not in z_scores:
                    continue
                applicable += 1
                z = z_scores[code]
                if op == ">" and z > threshold:
                    matched += 1
                elif op == "<" and z < threshold:
                    matched += 1
            match_counts[regime] = matched / applicable if applicable > 0 else 0.0

        # Normalize to probabilities
        total = sum(match_counts.values())
        if total == 0:
            confidence = {r.value: 0.2 for r in RegimeType}
            return RegimeType.EXPANSION, confidence

        confidence = {r.value: v / total for r, v in match_counts.items()}
        best = max(match_counts, key=match_counts.get)  # type: ignore[arg-type]
        return best, confidence

    def _compute_base_score(self, z_scores: dict[str, float]) -> float:
        """Weighted aggregation of Z-scores into [-1, +1] score."""
        score = 0.0
        for code, weight in _SCORE_WEIGHTS.items():
            if code in z_scores:
                score += weight * z_scores[code]
        # Clip to [-1, +1]
        return max(-1.0, min(1.0, score))

    def _apply_polymarket_adjustment(self) -> float:
        """Adjust score based on Polymarket probability cliffs.

        Large probability swings on macro events signal market consensus shift.
        """
        if self._polymarket is None:
            return 0.0

        try:
            cliffs = self._polymarket.detect_cliffs()
        except Exception:
            logger.warning("Polymarket cliff detection failed, skipping", exc_info=True)
            return 0.0

        if not cliffs:
            return 0.0

        # Average cliff magnitude, capped at ±0.15 adjustment
        total_shift = 0.0
        valid_count = 0
        for event in cliffs:
            if event.previous_probability is None:
                continue
            shift = event.probability - event.previous_probability
            total_shift += shift
            valid_count += 1

        avg_shift = total_shift / valid_count if valid_count > 0 else 0.0
        # Positive cliff (probabilities rising) could be bearish (policy tightening)
        # or bullish — we use negative sign as heuristic (higher event probability = uncertainty)
        adjustment = -avg_shift * 0.3  # scale factor
        return max(-0.15, min(0.15, adjustment))

    def _is_high_volatility_day(self, d: date) -> bool:
        """Check if date is a high-volatility calendar day."""
        if self._calendar is None:
            return False
        try:
            return self._calendar.is_high_volatility_day(d)
        except Exception:
            logger.warning("Calendar check failed, skipping", exc_info=True)
            return False
