"""Tests for macro_scoring module — m1 MacroScorer.

Covers:
- Regime classification for all 5 regimes
- Base score computation + clipping
- Polymarket cliff adjustment
- Calendar high-volatility dampening
- Edge cases: empty z_scores, partial indicators, extreme values
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from unittest.mock import MagicMock

import pytest

from stockbee.macro_scoring.scorer import (
    MacroScore,
    MacroScorer,
    RegimeType,
    _REGIME_RULES,
    _SCORE_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures: mock providers
# ---------------------------------------------------------------------------

def _make_macro_provider(z_scores: dict[str, float]):
    """Create a mock MacroProvider with given z_scores."""
    provider = MagicMock()
    provider.get_latest_z_scores.return_value = z_scores
    return provider


@dataclass
class FakeMarketEvent:
    """Minimal MarketEvent for testing."""
    event_id: str = "test"
    question: str = "test"
    slug: str = "test"
    probability: float = 0.5
    previous_probability: float | None = 0.3
    volume: float = 0.0
    liquidity: float = 0.0
    is_cliff: bool = True
    fetched_at: str = "2026-04-15"


def _make_polymarket(cliffs: list[FakeMarketEvent] | None = None):
    """Create a mock PolymarketFetcher."""
    fetcher = MagicMock()
    fetcher.detect_cliffs.return_value = cliffs or []
    return fetcher


def _make_calendar(hv_days: set[date] | None = None):
    """Create a mock EconomicCalendar."""
    cal = MagicMock()
    hv = hv_days or set()
    cal.is_high_volatility_day.side_effect = lambda d: d in hv
    return cal


# ---------------------------------------------------------------------------
# Z-score profiles for each regime
# ---------------------------------------------------------------------------

EXPANSION_ZSCORES = {
    "GDP": 1.0, "INDPRO": 0.8, "PAYEMS": 1.0, "UNRATE": -0.5,
    "CPIAUCSL": 0.3, "T10Y2Y": 0.5, "DFF": 0.0, "DGS10": 0.0,
    "PPIFIS": 0.2, "ICSA": -0.3, "BAMLH0A0HYM2": -0.3,
    "DRTSCILM": -0.2, "M2SL": 0.5, "VIXCLS": -0.5,
    "DEXUSEU": 0.0, "DCOILWTICO": 0.0, "PCOPPUSDM": 0.5,
}

OVERHEATING_ZSCORES = {
    "GDP": 0.5, "INDPRO": 0.3, "PAYEMS": 0.5, "UNRATE": -0.5,
    "CPIAUCSL": 2.0, "PPIFIS": 1.5, "DFF": 1.5, "DGS10": 1.0,
    "T10Y2Y": -0.3, "ICSA": -0.2, "BAMLH0A0HYM2": 0.3,
    "DRTSCILM": 0.2, "M2SL": -0.3, "VIXCLS": 0.5,
    "DEXUSEU": 0.0, "DCOILWTICO": 1.0, "PCOPPUSDM": 0.5,
}

STAGFLATION_ZSCORES = {
    "GDP": -1.0, "INDPRO": -0.8, "PAYEMS": -0.3, "UNRATE": 1.0,
    "CPIAUCSL": 2.0, "PPIFIS": 1.5, "DFF": 0.5, "DGS10": 0.5,
    "T10Y2Y": -0.5, "ICSA": 0.8, "BAMLH0A0HYM2": 1.0,
    "DRTSCILM": 0.5, "M2SL": -0.5, "VIXCLS": 1.0,
    "DEXUSEU": 0.5, "DCOILWTICO": 1.5, "PCOPPUSDM": -0.5,
}

RECESSION_ZSCORES = {
    "GDP": -2.0, "INDPRO": -1.5, "PAYEMS": -1.5, "UNRATE": 2.0,
    "CPIAUCSL": -0.5, "PPIFIS": -0.5, "DFF": -1.0, "DGS10": -0.5,
    "T10Y2Y": -0.5, "ICSA": 2.0, "BAMLH0A0HYM2": 2.0,
    "DRTSCILM": 1.0, "M2SL": -1.0, "VIXCLS": 2.5,
    "DEXUSEU": -0.5, "DCOILWTICO": -1.0, "PCOPPUSDM": -1.0,
}

RECOVERY_ZSCORES = {
    "GDP": 0.2, "INDPRO": 0.3, "PAYEMS": 0.1, "UNRATE": 0.5,
    "CPIAUCSL": 0.0, "PPIFIS": 0.0, "DFF": -0.5, "DGS10": -0.3,
    "T10Y2Y": 0.5, "ICSA": 0.0, "BAMLH0A0HYM2": -0.3,
    "DRTSCILM": 0.0, "M2SL": 0.5, "VIXCLS": -0.5,
    "DEXUSEU": 0.0, "DCOILWTICO": 0.0, "PCOPPUSDM": 0.3,
}


# ===================================================================
# Regime classification tests
# ===================================================================

class TestRegimeClassification:
    """Test _classify_regime for all 5 regimes."""

    def test_expansion(self):
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        regime, conf = scorer._classify_regime(EXPANSION_ZSCORES)
        assert regime == RegimeType.EXPANSION
        assert conf["expansion"] == max(conf.values())

    def test_overheating(self):
        scorer = MacroScorer(_make_macro_provider(OVERHEATING_ZSCORES))
        regime, conf = scorer._classify_regime(OVERHEATING_ZSCORES)
        assert regime == RegimeType.OVERHEATING
        assert conf["overheating"] == max(conf.values())

    def test_stagflation(self):
        scorer = MacroScorer(_make_macro_provider(STAGFLATION_ZSCORES))
        regime, conf = scorer._classify_regime(STAGFLATION_ZSCORES)
        assert regime == RegimeType.STAGFLATION
        assert conf["stagflation"] == max(conf.values())

    def test_recession(self):
        scorer = MacroScorer(_make_macro_provider(RECESSION_ZSCORES))
        regime, conf = scorer._classify_regime(RECESSION_ZSCORES)
        assert regime == RegimeType.RECESSION
        assert conf["recession"] == max(conf.values())

    def test_recovery(self):
        scorer = MacroScorer(_make_macro_provider(RECOVERY_ZSCORES))
        regime, conf = scorer._classify_regime(RECOVERY_ZSCORES)
        assert regime == RegimeType.RECOVERY
        assert conf["recovery"] == max(conf.values())

    def test_confidence_sums_to_one(self):
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        _, conf = scorer._classify_regime(EXPANSION_ZSCORES)
        assert abs(sum(conf.values()) - 1.0) < 1e-9

    def test_all_regimes_present_in_confidence(self):
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        _, conf = scorer._classify_regime(EXPANSION_ZSCORES)
        for regime in RegimeType:
            assert regime.value in conf


# ===================================================================
# Base score tests
# ===================================================================

class TestBaseScore:
    """Test _compute_base_score."""

    def test_expansion_positive_score(self):
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        score = scorer._compute_base_score(EXPANSION_ZSCORES)
        assert score > 0.0, "Expansion should produce positive score"

    def test_recession_negative_score(self):
        scorer = MacroScorer(_make_macro_provider(RECESSION_ZSCORES))
        score = scorer._compute_base_score(RECESSION_ZSCORES)
        assert score < 0.0, "Recession should produce negative score"

    def test_score_clipped_to_range(self):
        # Extreme z-scores should still produce [-1, +1]
        extreme = {code: 3.0 for code in _SCORE_WEIGHTS}
        scorer = MacroScorer(_make_macro_provider(extreme))
        score = scorer._compute_base_score(extreme)
        assert -1.0 <= score <= 1.0

    def test_all_zero_zscores(self):
        zeros = {code: 0.0 for code in _SCORE_WEIGHTS}
        scorer = MacroScorer(_make_macro_provider(zeros))
        score = scorer._compute_base_score(zeros)
        assert score == 0.0, "All-zero z_scores should give 0.0 score"

    def test_partial_indicators(self):
        """Only a few indicators available — should still compute."""
        partial = {"GDP": 1.0, "UNRATE": -1.0}
        scorer = MacroScorer(_make_macro_provider(partial))
        score = scorer._compute_base_score(partial)
        expected = 0.12 * 1.0 + (-0.08) * (-1.0)  # GDP + UNRATE
        assert abs(score - expected) < 1e-9


# ===================================================================
# Full score() integration tests
# ===================================================================

class TestScoreIntegration:
    """Test the full score() pipeline."""

    def test_basic_scoring(self):
        provider = _make_macro_provider(EXPANSION_ZSCORES)
        scorer = MacroScorer(provider)
        result = scorer.score(as_of=date(2026, 4, 15))
        assert isinstance(result, MacroScore)
        assert -1.0 <= result.score <= 1.0
        assert result.regime in RegimeType
        assert result.scored_at == date(2026, 4, 15)
        assert "base_score" in result.signals

    def test_empty_zscores_returns_neutral(self):
        provider = _make_macro_provider({})
        scorer = MacroScorer(provider)
        result = scorer.score(as_of=date(2026, 4, 15))
        assert result.score == 0.0
        assert result.regime == RegimeType.EXPANSION
        # All regimes should have equal confidence
        for val in result.regime_confidence.values():
            assert abs(val - 0.2) < 1e-9

    def test_no_optional_providers(self):
        """Works fine without polymarket/calendar."""
        provider = _make_macro_provider(RECESSION_ZSCORES)
        scorer = MacroScorer(provider)
        result = scorer.score()
        assert result.score < 0.0
        assert "polymarket_adj" not in result.signals

    def test_score_uses_today_when_no_date(self):
        provider = _make_macro_provider(EXPANSION_ZSCORES)
        scorer = MacroScorer(provider)
        result = scorer.score()
        assert result.scored_at == date.today()


# ===================================================================
# Polymarket adjustment tests
# ===================================================================

class TestPolymarketAdjustment:
    """Test _apply_polymarket_adjustment."""

    def test_no_cliffs_no_adjustment(self):
        poly = _make_polymarket([])
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        adj = scorer._apply_polymarket_adjustment()
        assert adj == 0.0

    def test_positive_cliff_negative_adjustment(self):
        """Rising probability → uncertainty → slightly bearish."""
        cliff = FakeMarketEvent(probability=0.8, previous_probability=0.3)
        poly = _make_polymarket([cliff])
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        adj = scorer._apply_polymarket_adjustment()
        assert adj < 0.0

    def test_negative_cliff_positive_adjustment(self):
        """Falling probability → reduced uncertainty → slightly bullish."""
        cliff = FakeMarketEvent(probability=0.2, previous_probability=0.8)
        poly = _make_polymarket([cliff])
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        adj = scorer._apply_polymarket_adjustment()
        assert adj > 0.0

    def test_adjustment_capped_at_015(self):
        """Adjustment should not exceed ±0.15."""
        cliff = FakeMarketEvent(probability=1.0, previous_probability=0.0)
        poly = _make_polymarket([cliff])
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        adj = scorer._apply_polymarket_adjustment()
        assert -0.15 <= adj <= 0.15

    def test_none_previous_probability_skipped(self):
        """First-ever fetch has no previous — should not crash."""
        cliff = FakeMarketEvent(probability=0.5, previous_probability=None)
        poly = _make_polymarket([cliff])
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        adj = scorer._apply_polymarket_adjustment()
        assert adj == 0.0  # no valid shifts

    def test_polymarket_exception_returns_zero(self):
        """If polymarket raises, gracefully return 0."""
        poly = MagicMock()
        poly.detect_cliffs.side_effect = RuntimeError("API down")
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        adj = scorer._apply_polymarket_adjustment()
        assert adj == 0.0

    def test_mixed_none_and_valid_cliffs(self):
        """Mix of valid and None previous_probability cliffs."""
        cliffs = [
            FakeMarketEvent(probability=0.8, previous_probability=0.3),  # valid: +0.5
            FakeMarketEvent(probability=0.5, previous_probability=None),  # skipped
            FakeMarketEvent(probability=0.2, previous_probability=0.6),  # valid: -0.4
        ]
        poly = _make_polymarket(cliffs)
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        adj = scorer._apply_polymarket_adjustment()
        # avg_shift = (0.5 + (-0.4)) / 2 = 0.05, adjustment = -0.05 * 0.3 = -0.015
        expected = -(0.05) * 0.3
        assert abs(adj - expected) < 1e-9

    def test_polymarket_adjustment_in_full_score(self):
        """Verify polymarket_adj appears in signals when active."""
        cliff = FakeMarketEvent(probability=0.8, previous_probability=0.3)
        poly = _make_polymarket([cliff])
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), polymarket=poly)
        result = scorer.score(as_of=date(2026, 4, 15))
        assert "polymarket_adj" in result.signals


# ===================================================================
# Calendar dampening tests
# ===================================================================

class TestCalendarDampening:
    """Test high-volatility day dampening."""

    def test_hv_day_dampens_score(self):
        hv_date = date(2026, 4, 15)
        cal = _make_calendar({hv_date})
        provider = _make_macro_provider(EXPANSION_ZSCORES)
        scorer = MacroScorer(provider, calendar=cal)

        # Score without calendar
        scorer_no_cal = MacroScorer(provider)
        base_result = scorer_no_cal.score(as_of=hv_date)

        # Score with calendar on HV day
        result = scorer.score(as_of=hv_date)
        assert abs(result.score) < abs(base_result.score) or base_result.score == 0.0
        if base_result.score != 0.0:
            assert abs(result.score - base_result.score * 0.8) < 1e-9

    def test_non_hv_day_no_dampening(self):
        hv_date = date(2026, 4, 15)
        normal_date = date(2026, 4, 16)
        cal = _make_calendar({hv_date})
        provider = _make_macro_provider(EXPANSION_ZSCORES)

        scorer = MacroScorer(provider, calendar=cal)
        scorer_no_cal = MacroScorer(provider)

        result = scorer.score(as_of=normal_date)
        base_result = scorer_no_cal.score(as_of=normal_date)
        assert abs(result.score - base_result.score) < 1e-9

    def test_calendar_exception_no_dampening(self):
        cal = MagicMock()
        cal.is_high_volatility_day.side_effect = RuntimeError("DB error")
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), calendar=cal)
        result = scorer.score(as_of=date(2026, 4, 15))
        # Should succeed without dampening
        assert isinstance(result, MacroScore)

    def test_hv_dampening_signal_in_result(self):
        hv_date = date(2026, 4, 15)
        cal = _make_calendar({hv_date})
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES), calendar=cal)
        result = scorer.score(as_of=hv_date)
        assert "hv_dampening" in result.signals


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_extreme_positive_zscores(self):
        extreme = {code: 3.0 for code in _SCORE_WEIGHTS}
        scorer = MacroScorer(_make_macro_provider(extreme))
        result = scorer.score(as_of=date(2026, 4, 15))
        assert -1.0 <= result.score <= 1.0

    def test_all_extreme_negative_zscores(self):
        extreme = {code: -3.0 for code in _SCORE_WEIGHTS}
        scorer = MacroScorer(_make_macro_provider(extreme))
        result = scorer.score(as_of=date(2026, 4, 15))
        assert -1.0 <= result.score <= 1.0

    def test_single_indicator(self):
        """Only one indicator available."""
        single = {"GDP": 2.0}
        scorer = MacroScorer(_make_macro_provider(single))
        result = scorer.score(as_of=date(2026, 4, 15))
        assert isinstance(result, MacroScore)
        assert result.score > 0.0  # GDP positive → bullish

    def test_unknown_indicator_ignored(self):
        """Z-scores with unknown codes should not crash."""
        z = {"GDP": 1.0, "UNKNOWN_CODE": 5.0}
        scorer = MacroScorer(_make_macro_provider(z))
        result = scorer.score(as_of=date(2026, 4, 15))
        assert isinstance(result, MacroScore)

    def test_regime_rules_cover_all_types(self):
        """All RegimeType values should have rules."""
        for regime in RegimeType:
            assert regime in _REGIME_RULES, f"Missing rules for {regime}"

    def test_propagation_weight_stored(self):
        """propagation_weight should be stored for Phase 2."""
        scorer = MacroScorer(
            _make_macro_provider({}), propagation_weight=0.4
        )
        assert scorer._propagation_weight == 0.4

    def test_score_weights_roughly_balanced(self):
        """Weights should approximately sum to 0 (balanced bull/bear)."""
        total = sum(_SCORE_WEIGHTS.values())
        assert abs(total) < 0.15, f"Weights sum to {total}, should be ~0"

    def test_frozen_macro_score(self):
        """MacroScore is immutable."""
        result = MacroScore(
            score=0.5,
            regime=RegimeType.EXPANSION,
            regime_confidence={"expansion": 1.0},
            scored_at=date(2026, 4, 15),
        )
        with pytest.raises(AttributeError):
            result.score = 0.0  # type: ignore[misc]
