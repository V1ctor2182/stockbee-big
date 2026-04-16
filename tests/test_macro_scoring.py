"""Tests for macro_scoring module — m1 MacroScorer + m2 LLMMacroAnalyst.

Covers:
- m1: Regime classification for all 5 regimes
- m1: Base score computation + clipping
- m1: Polymarket cliff adjustment
- m1: Calendar high-volatility dampening
- m1: Edge cases: empty z_scores, partial indicators, extreme values
- m2: Prompt building, LLM response parsing, SQLite caching, error handling
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
from stockbee.macro_scoring.llm_analyst import (
    LLMMacroAnalyst,
    MacroInsight,
    _SYSTEM_PROMPT,
)
from stockbee.macro_scoring.provider import MacroScoringProvider
from stockbee.providers.base import ProviderConfig


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


# ===================================================================
# m2: LLMMacroAnalyst tests
# ===================================================================

SAMPLE_LLM_RESPONSE = """\
REGIME: expansion
OUTLOOK_SCORE: 0.65
RISK_FACTORS:
- Rising inflation expectations could force Fed to hold rates longer
- Credit spreads widening modestly signals caution
- Geopolitical tensions in trade policy
REASONING:
The US economy remains in an expansion phase with solid GDP growth and improving \
employment metrics. The labor market is tight with unemployment below the historical average.

However, inflation remains sticky above the Fed's 2% target, which may delay rate cuts. \
Credit conditions are tightening modestly but not at alarming levels. Overall, the macro \
environment supports a mildly bullish equity stance with sector rotation toward cyclicals.
"""


@dataclass
class FakeReleaseEvent:
    """Minimal ReleaseEvent for testing."""
    release_id: int = 1
    release_name: str = "CPI"
    release_date: str = "2026-04-20"
    high_volatility: bool = True


def _make_router(content: str = SAMPLE_LLM_RESPONSE, raise_error: bool = False):
    """Create a mock LLMRouter."""
    router = MagicMock()
    if raise_error:
        router.route.side_effect = RuntimeError("LLM unavailable")
    else:
        resp = MagicMock()
        resp.content = content
        resp.model_used = "anthropic/claude-opus-4-6"
        router.route.return_value = resp
    return router


class TestPromptBuilding:
    """Test _build_prompt."""

    def test_basic_prompt_has_zscores(self):
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        prompt = analyst._build_prompt(EXPANSION_ZSCORES, date(2026, 4, 15))
        assert "2026-04-15" in prompt
        assert "Z-Scores" in prompt
        assert "GDP" in prompt

    def test_prompt_with_polymarket(self):
        events = [FakeMarketEvent(
            question="Fed rate cut in May?", probability=0.35,
            previous_probability=0.2, is_cliff=True
        )]
        poly = MagicMock()
        poly.get_latest_events.return_value = events
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES), polymarket=poly
        )
        prompt = analyst._build_prompt(EXPANSION_ZSCORES, date(2026, 4, 15))
        assert "Polymarket" in prompt
        assert "Fed rate cut" in prompt
        assert "CLIFF" in prompt

    def test_prompt_with_calendar(self):
        cal = MagicMock()
        cal.get_events.return_value = [
            FakeReleaseEvent(release_name="CPI", release_date="2026-04-20"),
        ]
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES), calendar=cal
        )
        prompt = analyst._build_prompt(EXPANSION_ZSCORES, date(2026, 4, 15))
        assert "High-Volatility" in prompt
        assert "CPI" in prompt

    def test_prompt_without_optional_sources(self):
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        prompt = analyst._build_prompt(EXPANSION_ZSCORES, date(2026, 4, 15))
        assert "Polymarket" not in prompt
        assert "High-Volatility" not in prompt

    def test_prompt_polymarket_exception_graceful(self):
        poly = MagicMock()
        poly.get_latest_events.side_effect = RuntimeError("API down")
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES), polymarket=poly
        )
        prompt = analyst._build_prompt(EXPANSION_ZSCORES, date(2026, 4, 15))
        # Should not crash, polymarket section omitted
        assert "Z-Scores" in prompt

    def test_prompt_calendar_exception_graceful(self):
        cal = MagicMock()
        cal.get_events.side_effect = RuntimeError("DB error")
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES), calendar=cal
        )
        prompt = analyst._build_prompt(EXPANSION_ZSCORES, date(2026, 4, 15))
        assert "Z-Scores" in prompt


class TestResponseParsing:
    """Test _parse_response."""

    def test_parse_full_response(self):
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        insight = analyst._parse_response(
            SAMPLE_LLM_RESPONSE, "claude-opus-4-6", date(2026, 4, 15)
        )
        assert insight is not None
        assert insight.regime_assessment == "expansion"
        assert abs(insight.outlook_score - 0.65) < 1e-9
        assert len(insight.risk_factors) == 3
        assert "inflation" in insight.risk_factors[0].lower()
        assert "expansion" in insight.reasoning.lower()
        assert insight.model_used == "claude-opus-4-6"
        assert insight.analyzed_at == date(2026, 4, 15)

    def test_parse_empty_response(self):
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        assert analyst._parse_response("", "test", date(2026, 4, 15)) is None
        assert analyst._parse_response("   ", "test", date(2026, 4, 15)) is None

    def test_parse_malformed_outlook_defaults_to_zero(self):
        bad_response = "REGIME: expansion\nOUTLOOK_SCORE: not-a-number\nRISK_FACTORS:\n- test\nREASONING:\ntest"
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        insight = analyst._parse_response(bad_response, "test", date(2026, 4, 15))
        assert insight is not None
        assert insight.outlook_score == 0.0

    def test_parse_outlook_clipped(self):
        response = "REGIME: recession\nOUTLOOK_SCORE: -5.0\nRISK_FACTORS:\n- doom\nREASONING:\nbad"
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        insight = analyst._parse_response(response, "test", date(2026, 4, 15))
        assert insight is not None
        assert insight.outlook_score == -1.0

    def test_parse_missing_risk_factors(self):
        response = "REGIME: expansion\nOUTLOOK_SCORE: 0.5\nREASONING:\nall good"
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        insight = analyst._parse_response(response, "test", date(2026, 4, 15))
        assert insight is not None
        assert insight.risk_factors == []

    def test_parse_missing_reasoning(self):
        response = "REGIME: expansion\nOUTLOOK_SCORE: 0.5\nRISK_FACTORS:\n- test"
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES)
        )
        insight = analyst._parse_response(response, "test", date(2026, 4, 15))
        assert insight is not None
        assert insight.reasoning == ""


class TestLLMAnalystIntegration:
    """Test full analyze() flow with mock LLM."""

    def test_analyze_calls_router(self, tmp_path):
        router = _make_router()
        analyst = LLMMacroAnalyst(
            router, _make_macro_provider(EXPANSION_ZSCORES), cache_dir=tmp_path
        )
        insight = analyst.analyze(as_of=date(2026, 4, 15))
        assert insight is not None
        assert insight.regime_assessment == "expansion"
        router.route.assert_called_once()

    def test_analyze_empty_zscores_returns_none(self, tmp_path):
        router = _make_router()
        analyst = LLMMacroAnalyst(
            router, _make_macro_provider({}), cache_dir=tmp_path
        )
        result = analyst.analyze(as_of=date(2026, 4, 15))
        assert result is None
        router.route.assert_not_called()

    def test_analyze_llm_failure_returns_none(self, tmp_path):
        router = _make_router(raise_error=True)
        analyst = LLMMacroAnalyst(
            router, _make_macro_provider(EXPANSION_ZSCORES), cache_dir=tmp_path
        )
        result = analyst.analyze(as_of=date(2026, 4, 15))
        assert result is None

    def test_analyze_caches_result(self, tmp_path):
        router = _make_router()
        analyst = LLMMacroAnalyst(
            router, _make_macro_provider(EXPANSION_ZSCORES), cache_dir=tmp_path
        )
        # First call → LLM
        insight1 = analyst.analyze(as_of=date(2026, 4, 15))
        assert insight1 is not None
        assert router.route.call_count == 1

        # Second call same week → cache
        insight2 = analyst.analyze(as_of=date(2026, 4, 18))
        assert insight2 is not None
        assert router.route.call_count == 1  # not called again
        assert insight2.regime_assessment == insight1.regime_assessment

    def test_analyze_cache_expired(self, tmp_path):
        router = _make_router()
        analyst = LLMMacroAnalyst(
            router, _make_macro_provider(EXPANSION_ZSCORES), cache_dir=tmp_path
        )
        # First call
        analyst.analyze(as_of=date(2026, 4, 8))
        assert router.route.call_count == 1

        # 8 days later → cache expired → new LLM call
        analyst.analyze(as_of=date(2026, 4, 16))
        assert router.route.call_count == 2

    def test_close_db(self, tmp_path):
        analyst = LLMMacroAnalyst(
            _make_router(), _make_macro_provider(EXPANSION_ZSCORES), cache_dir=tmp_path
        )
        analyst.analyze(as_of=date(2026, 4, 15))
        analyst.close()
        assert analyst._db is None

    def test_frozen_macro_insight(self):
        insight = MacroInsight(
            regime_assessment="expansion",
            risk_factors=["test"],
            outlook_score=0.5,
            reasoning="test",
            model_used="test",
            analyzed_at=date(2026, 4, 15),
        )
        with pytest.raises(AttributeError):
            insight.outlook_score = 0.0  # type: ignore[misc]


class TestExtractHelpers:
    """Test static parsing helpers."""

    def test_extract_field(self):
        text = "REGIME: expansion\nOUTLOOK_SCORE: 0.5"
        assert LLMMacroAnalyst._extract_field(text, "REGIME", "x") == "expansion"
        assert LLMMacroAnalyst._extract_field(text, "OUTLOOK_SCORE", "0") == "0.5"
        assert LLMMacroAnalyst._extract_field(text, "MISSING", "default") == "default"

    def test_extract_list(self):
        text = "RISK_FACTORS:\n- item one\n- item two\n- item three\nREASONING:\nfoo"
        items = LLMMacroAnalyst._extract_list(text, "RISK_FACTORS")
        assert items == ["item one", "item two", "item three"]

    def test_extract_list_missing(self):
        assert LLMMacroAnalyst._extract_list("no list here", "RISK_FACTORS") == []

    def test_extract_section(self):
        text = "REASONING:\nline one\nline two\n\nline three"
        result = LLMMacroAnalyst._extract_section(text, "REASONING")
        assert "line one" in result
        assert "line three" in result

    def test_extract_section_missing(self):
        assert LLMMacroAnalyst._extract_section("no section", "REASONING") == ""


# ===================================================================
# m3: MacroScoringProvider tests
# ===================================================================

def _make_provider(tmp_path, mode="backtest") -> MacroScoringProvider:
    """Create a MacroScoringProvider with tmp SQLite."""
    config = ProviderConfig(
        implementation="MacroScoringProvider",
        params={"db_path": str(tmp_path / "macro_scores.db"), "mode": mode},
    )
    provider = MacroScoringProvider(config)
    provider.initialize()
    return provider


class TestProviderLifecycle:
    """Test initialize / shutdown / health_check."""

    def test_initialize_creates_db(self, tmp_path):
        provider = _make_provider(tmp_path)
        assert provider.is_initialized
        assert provider.health_check()
        assert (tmp_path / "macro_scores.db").exists()

    def test_shutdown(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.shutdown()
        assert not provider.is_initialized

    def test_double_initialize(self, tmp_path):
        provider = _make_provider(tmp_path)
        provider.initialize()  # should be idempotent
        assert provider.is_initialized


class TestScoreAndStore:
    """Test score_and_store + retrieval."""

    def test_score_and_store_backtest(self, tmp_path):
        provider = _make_provider(tmp_path)
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        provider.set_scorer(scorer)

        result = provider.score_and_store(as_of=date(2026, 4, 15))
        assert result is not None
        assert result.score > 0.0
        assert result.regime == RegimeType.EXPANSION

        # Verify persisted
        score = provider.get_macro_score(date(2026, 4, 15))
        assert score is not None
        assert abs(score - result.score) < 1e-9

        regime = provider.get_regime(date(2026, 4, 15))
        assert regime == "expansion"

    def test_score_and_store_no_scorer(self, tmp_path):
        provider = _make_provider(tmp_path)
        result = provider.score_and_store(as_of=date(2026, 4, 15))
        assert result is None

    def test_score_and_store_replaces_existing(self, tmp_path):
        provider = _make_provider(tmp_path)
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        provider.set_scorer(scorer)

        provider.score_and_store(as_of=date(2026, 4, 15))
        # Score again same date — should replace, not error
        result2 = provider.score_and_store(as_of=date(2026, 4, 15))
        assert result2 is not None

    def test_score_and_store_live_with_llm(self, tmp_path):
        provider = _make_provider(tmp_path, mode="live")
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        provider.set_scorer(scorer)

        router = _make_router()
        analyst = LLMMacroAnalyst(
            router, _make_macro_provider(EXPANSION_ZSCORES),
            cache_dir=tmp_path / "llm_cache",
        )
        provider.set_analyst(analyst)

        result = provider.score_and_store(as_of=date(2026, 4, 15))
        assert result is not None
        assert "llm_blend" in result.signals
        assert "llm_outlook" in result.signals
        router.route.assert_called_once()

    def test_score_and_store_live_llm_failure(self, tmp_path):
        provider = _make_provider(tmp_path, mode="live")
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        provider.set_scorer(scorer)

        router = _make_router(raise_error=True)
        analyst = LLMMacroAnalyst(
            router, _make_macro_provider(EXPANSION_ZSCORES),
            cache_dir=tmp_path / "llm_cache",
        )
        provider.set_analyst(analyst)

        # Should fallback to pure rule-based score
        result = provider.score_and_store(as_of=date(2026, 4, 15))
        assert result is not None
        assert "llm_blend" not in result.signals


class TestGetHistory:
    """Test get_history."""

    def test_history_multiple_dates(self, tmp_path):
        provider = _make_provider(tmp_path)
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        provider.set_scorer(scorer)

        for day in range(10, 16):
            provider.score_and_store(as_of=date(2026, 4, day))

        df = provider.get_history(date(2026, 4, 10), date(2026, 4, 15))
        assert len(df) == 6
        assert list(df.columns) == ["date", "score", "regime", "llm_outlook"]
        assert all(df["regime"] == "expansion")

    def test_history_empty_range(self, tmp_path):
        provider = _make_provider(tmp_path)
        df = provider.get_history(date(2026, 1, 1), date(2026, 1, 31))
        assert len(df) == 0
        assert list(df.columns) == ["date", "score", "regime", "llm_outlook"]

    def test_history_partial_range(self, tmp_path):
        provider = _make_provider(tmp_path)
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        provider.set_scorer(scorer)

        provider.score_and_store(as_of=date(2026, 4, 12))
        provider.score_and_store(as_of=date(2026, 4, 14))

        df = provider.get_history(date(2026, 4, 10), date(2026, 4, 15))
        assert len(df) == 2


class TestProviderQuery:
    """Test get_macro_score / get_regime edge cases."""

    def test_query_missing_date(self, tmp_path):
        provider = _make_provider(tmp_path)
        assert provider.get_macro_score(date(2026, 4, 15)) is None
        assert provider.get_regime(date(2026, 4, 15)) is None

    def test_query_after_shutdown_and_reinit(self, tmp_path):
        provider = _make_provider(tmp_path)
        scorer = MacroScorer(_make_macro_provider(EXPANSION_ZSCORES))
        provider.set_scorer(scorer)
        provider.score_and_store(as_of=date(2026, 4, 15))
        provider.shutdown()

        # Re-initialize — data should persist
        provider.initialize()
        score = provider.get_macro_score(date(2026, 4, 15))
        assert score is not None

    def test_recession_regime_stored(self, tmp_path):
        provider = _make_provider(tmp_path)
        scorer = MacroScorer(_make_macro_provider(RECESSION_ZSCORES))
        provider.set_scorer(scorer)
        provider.score_and_store(as_of=date(2026, 4, 15))
        assert provider.get_regime(date(2026, 4, 15)) == "recession"
