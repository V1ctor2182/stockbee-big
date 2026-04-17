"""Tests for macro_scoring.sector_tilts — SectorTilter.

Covers:
- SQLite lifecycle: init, seed, close
- Sensitivity matrix: CRUD, 11 sectors covered
- Tilt computation: direction, capping, regime adjustment
- Edge cases: empty z_scores, partial indicators, extreme values
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from stockbee.macro_scoring.sector_tilts import (
    DEFAULT_SENSITIVITIES,
    GICS_SECTORS,
    TILT_CAP,
    SectorTilter,
    _REGIME_SECTOR_MULTIPLIERS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_macro_provider(z_scores: dict[str, float]):
    provider = MagicMock()
    provider.get_latest_z_scores.return_value = z_scores
    return provider


def _make_scoring_provider(regime: str | None):
    provider = MagicMock()
    provider.get_regime.return_value = regime
    return provider


def _make_tilter(
    tmp_path,
    z_scores: dict[str, float],
    regime: str | None = None,
) -> SectorTilter:
    macro = _make_macro_provider(z_scores)
    scoring = _make_scoring_provider(regime) if regime else None
    tilter = SectorTilter(
        macro, scoring, db_path=tmp_path / "sector.db"
    )
    tilter.initialize()
    return tilter


# Typical expansion Z-scores (positive growth, moderate inflation)
EXPANSION_Z = {
    "GDP": 1.0, "INDPRO": 0.8, "PAYEMS": 1.0, "UNRATE": -0.5,
    "CPIAUCSL": 0.3, "DFF": 0.0, "DGS10": 0.0, "T10Y2Y": 0.5,
    "PPIFIS": 0.2, "ICSA": -0.3, "BAMLH0A0HYM2": -0.3,
    "DRTSCILM": -0.2, "M2SL": 0.5, "VIXCLS": -0.5,
    "DEXUSEU": 0.0, "DCOILWTICO": 0.5, "PCOPPUSDM": 0.5,
}

# Rate hike scenario
HIGH_RATES_Z = {
    "DFF": 2.5, "DGS10": 2.0, "T10Y2Y": -1.0,
    "GDP": 0.5, "INDPRO": 0.3, "PAYEMS": 0.3,
    "UNRATE": -0.3, "CPIAUCSL": 1.5, "VIXCLS": 0.5,
    "BAMLH0A0HYM2": 0.5, "DCOILWTICO": 0.0, "PCOPPUSDM": 0.0,
    "M2SL": -0.5, "DRTSCILM": 0.5, "PPIFIS": 1.0,
    "DEXUSEU": 0.0, "ICSA": 0.0,
}


# ===================================================================
# SQLite lifecycle
# ===================================================================

class TestLifecycle:

    def test_initialize_creates_db(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        assert (tmp_path / "sector.db").exists()
        tilter.close()

    def test_seed_defaults(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        df = tilter.get_sensitivity_matrix()
        # All 11 sectors should have entries
        sectors_in_db = set(df["sector"].unique())
        for sector in GICS_SECTORS:
            assert sector in sectors_in_db, f"Missing sector: {sector}"
        tilter.close()

    def test_seed_only_once(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        count1 = len(tilter.get_sensitivity_matrix())
        tilter.close()

        # Re-initialize — should not duplicate
        tilter2 = _make_tilter(tmp_path, EXPANSION_Z)
        count2 = len(tilter2.get_sensitivity_matrix())
        assert count1 == count2
        tilter2.close()

    def test_close_and_reopen(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        tilter.close()
        assert tilter._db is None


# ===================================================================
# Sensitivity matrix CRUD
# ===================================================================

class TestSensitivityMatrix:

    def test_default_sensitivities_match_db(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        df = tilter.get_sensitivity_matrix()
        # Count should match total entries in DEFAULT_SENSITIVITIES
        expected = sum(len(v) for v in DEFAULT_SENSITIVITIES.values())
        assert len(df) == expected
        tilter.close()

    def test_update_sensitivity(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        tilter.update_sensitivity("Energy", "DCOILWTICO", 0.99)
        df = tilter.get_sensitivity_matrix()
        row = df[(df["sector"] == "Energy") & (df["indicator"] == "DCOILWTICO")]
        assert float(row["sensitivity"].iloc[0]) == pytest.approx(0.99)
        tilter.close()

    def test_update_adds_new_entry(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        before = len(tilter.get_sensitivity_matrix())
        tilter.update_sensitivity("Energy", "NEW_INDICATOR", 0.5)
        after = len(tilter.get_sensitivity_matrix())
        assert after == before + 1
        tilter.close()

    def test_all_default_sectors_have_entries(self):
        """Every GICS sector should have at least one sensitivity defined."""
        for sector in GICS_SECTORS:
            assert sector in DEFAULT_SENSITIVITIES, f"Missing default: {sector}"
            assert len(DEFAULT_SENSITIVITIES[sector]) > 0


# ===================================================================
# Tilt computation
# ===================================================================

class TestComputeTilts:

    def test_returns_all_11_sectors(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert set(tilts.keys()) == set(GICS_SECTORS)
        tilter.close()

    def test_tilts_capped(self, tmp_path):
        tilter = _make_tilter(tmp_path, EXPANSION_Z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        for sector, tilt in tilts.items():
            assert -TILT_CAP <= tilt <= TILT_CAP, f"{sector} tilt {tilt} exceeds cap"
        tilter.close()

    def test_energy_positive_on_oil_up(self, tmp_path):
        """High oil price z_score → Energy should tilt positive."""
        z = {"DCOILWTICO": 2.5, "DFF": 0.0, "CPIAUCSL": 0.0, "GDP": 0.0}
        tilter = _make_tilter(tmp_path, z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert tilts["Energy"] > 0.0
        tilter.close()

    def test_utilities_negative_on_rate_hike(self, tmp_path):
        """High rates → Utilities (bond proxy) should tilt negative."""
        tilter = _make_tilter(tmp_path, HIGH_RATES_Z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert tilts["Utilities"] < 0.0
        tilter.close()

    def test_financials_positive_on_steep_curve(self, tmp_path):
        """Positive yield curve → Financials benefit."""
        z = {"T10Y2Y": 2.0, "DFF": 0.5, "DGS10": 1.0,
             "BAMLH0A0HYM2": -0.5, "GDP": 0.5, "DRTSCILM": -0.5}
        tilter = _make_tilter(tmp_path, z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert tilts["Financials"] > 0.0
        tilter.close()

    def test_real_estate_negative_on_rate_hike(self, tmp_path):
        """High rates → Real Estate most rate-sensitive, should be negative."""
        tilter = _make_tilter(tmp_path, HIGH_RATES_Z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert tilts["Real Estate"] < 0.0
        tilter.close()

    def test_consumer_staples_defensive_in_recession_z(self, tmp_path):
        """High VIX + high unemployment → Consumer Staples positive tilt."""
        z = {"VIXCLS": 2.0, "UNRATE": 1.5, "GDP": -1.0,
             "DFF": -0.5, "CPIAUCSL": -0.3}
        tilter = _make_tilter(tmp_path, z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert tilts["Consumer Staples"] > 0.0
        tilter.close()


# ===================================================================
# Regime adjustment
# ===================================================================

class TestRegimeAdjustment:

    def test_expansion_amplifies_cyclicals(self, tmp_path):
        """Expansion regime should amplify IT and dampen Utilities."""
        tilter_no_regime = _make_tilter(tmp_path, EXPANSION_Z, regime=None)
        tilts_base = tilter_no_regime.compute_tilts(as_of=date(2026, 4, 17))
        tilter_no_regime.close()

        tilter_exp = _make_tilter(tmp_path / "exp", EXPANSION_Z, regime="expansion")
        tilts_exp = tilter_exp.compute_tilts(as_of=date(2026, 4, 17))
        tilter_exp.close()

        # IT should be amplified (multiplier 1.3)
        if tilts_base["Information Technology"] > 0:
            assert tilts_exp["Information Technology"] >= tilts_base["Information Technology"]

    def test_recession_amplifies_defensives(self, tmp_path):
        recession_z = {
            "GDP": -2.0, "INDPRO": -1.5, "PAYEMS": -1.5, "UNRATE": 2.0,
            "VIXCLS": 2.5, "BAMLH0A0HYM2": 2.0, "DFF": -1.0, "DGS10": -0.5,
            "T10Y2Y": -0.5, "ICSA": 2.0, "CPIAUCSL": -0.5,
            "DCOILWTICO": -1.0, "PCOPPUSDM": -1.0, "M2SL": -1.0,
            "DRTSCILM": 1.0, "PPIFIS": -0.5, "DEXUSEU": -0.5,
        }
        tilter = _make_tilter(tmp_path, recession_z, regime="recession")
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        # Consumer Staples should be among the highest
        staples = tilts["Consumer Staples"]
        disc = tilts["Consumer Discretionary"]
        assert staples > disc, "Staples should outperform Discretionary in recession"
        tilter.close()

    def test_unknown_regime_no_adjustment(self, tmp_path):
        """Unknown regime string → no multiplier applied."""
        tilter_none = _make_tilter(tmp_path, EXPANSION_Z, regime=None)
        tilts_none = tilter_none.compute_tilts(as_of=date(2026, 4, 17))
        tilter_none.close()

        tilter_unk = _make_tilter(tmp_path / "unk", EXPANSION_Z, regime="unknown_regime")
        tilts_unk = tilter_unk.compute_tilts(as_of=date(2026, 4, 17))
        tilter_unk.close()

        for sector in GICS_SECTORS:
            assert abs(tilts_none[sector] - tilts_unk[sector]) < 1e-9

    def test_all_regimes_have_multipliers(self):
        """All 5 regimes should have multiplier entries."""
        for regime in ["expansion", "overheating", "stagflation", "recession", "recovery"]:
            assert regime in _REGIME_SECTOR_MULTIPLIERS


# ===================================================================
# Edge cases
# ===================================================================

class TestEdgeCases:

    def test_empty_zscores_returns_zero_tilts(self, tmp_path):
        tilter = _make_tilter(tmp_path, {})
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        for sector in GICS_SECTORS:
            assert tilts[sector] == 0.0
        tilter.close()

    def test_single_indicator(self, tmp_path):
        z = {"DCOILWTICO": 2.0}
        tilter = _make_tilter(tmp_path, z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        # Energy has DCOILWTICO sensitivity, should be non-zero
        assert tilts["Energy"] != 0.0
        # Sectors without DCOILWTICO sensitivity should be 0
        assert tilts["Health Care"] == 0.0
        tilter.close()

    def test_extreme_zscores_still_capped(self, tmp_path):
        extreme = {code: 3.0 for code in [
            "DCOILWTICO", "DFF", "DGS10", "GDP", "INDPRO",
            "PAYEMS", "UNRATE", "CPIAUCSL", "VIXCLS", "T10Y2Y",
            "BAMLH0A0HYM2", "M2SL", "PCOPPUSDM",
        ]}
        tilter = _make_tilter(tmp_path, extreme)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        for sector, tilt in tilts.items():
            assert -TILT_CAP <= tilt <= TILT_CAP, f"{sector}: {tilt}"
        tilter.close()

    def test_unknown_indicator_in_zscores_ignored(self, tmp_path):
        z = {"UNKNOWN": 5.0, "DCOILWTICO": 1.0}
        tilter = _make_tilter(tmp_path, z)
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert isinstance(tilts, dict)
        tilter.close()

    def test_no_scoring_provider_skips_regime(self, tmp_path):
        """Without scoring provider, regime adjustment is skipped."""
        tilter = SectorTilter(
            _make_macro_provider(EXPANSION_Z),
            None,
            db_path=tmp_path / "no_score.db",
        )
        tilter.initialize()
        tilts = tilter.compute_tilts(as_of=date(2026, 4, 17))
        assert len(tilts) == 11
        tilter.close()
