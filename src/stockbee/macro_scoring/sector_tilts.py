"""SectorTilter — 11 GICS 行业宏观敏感度 tilt。

基于宏观指标 Z-scores 和行业敏感度矩阵计算各行业的权重调整 (±3%)。
敏感度矩阵存储在 SQLite，支持运行时调参。

来源：Tech Design §3.5 sector_tilt, plan-milestones 2026-04-17
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

TILT_CAP = 0.03  # ±3% max tilt per sector


# ---------------------------------------------------------------------------
# 11 GICS Sectors
# ---------------------------------------------------------------------------

GICS_SECTORS: list[str] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Health Care",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Utilities",
    "Real Estate",
]


# ---------------------------------------------------------------------------
# Default sensitivity matrix
# ---------------------------------------------------------------------------
# sector → {indicator: sensitivity}
# Positive sensitivity = sector benefits when indicator Z-score is high.
# Negative sensitivity = sector suffers when indicator Z-score is high.
#
# Sources: Tech Design §3.5, standard macro-sector relationships.
# Only include indicators with meaningful sensitivity (|s| >= 0.1).

DEFAULT_SENSITIVITIES: dict[str, dict[str, float]] = {
    "Energy": {
        "DCOILWTICO": 0.8,   # oil price is primary driver
        "DFF": -0.3,          # higher rates → capex pressure
        "CPIAUCSL": 0.2,      # inflation hedge
        "GDP": 0.3,           # cyclical
    },
    "Materials": {
        "PCOPPUSDM": 0.6,     # commodity proxy
        "INDPRO": 0.5,        # industrial demand
        "GDP": 0.4,           # cyclical
        "CPIAUCSL": 0.2,      # input costs pass-through
        "DFF": -0.2,
    },
    "Industrials": {
        "INDPRO": 0.6,
        "GDP": 0.5,
        "PAYEMS": 0.3,
        "DFF": -0.2,
        "BAMLH0A0HYM2": -0.3,  # credit-sensitive capex
    },
    "Consumer Discretionary": {
        "PAYEMS": 0.5,        # employment → spending
        "UNRATE": -0.5,       # unemployment → less spending
        "GDP": 0.4,
        "DFF": -0.3,          # rate-sensitive (auto, housing)
        "VIXCLS": -0.2,       # fear → less discretionary spend
    },
    "Consumer Staples": {
        "UNRATE": 0.2,        # defensive: benefits in downturn
        "VIXCLS": 0.3,        # flight to safety
        "GDP": -0.1,          # counter-cyclical
        "DFF": -0.1,
        "CPIAUCSL": -0.2,     # margin pressure from inflation
    },
    "Health Care": {
        "VIXCLS": 0.2,        # defensive
        "GDP": -0.1,          # counter-cyclical
        "DFF": -0.2,
        "UNRATE": 0.1,        # mild defensive
    },
    "Financials": {
        "DFF": 0.5,           # higher rates → net interest margin
        "DGS10": 0.4,
        "T10Y2Y": 0.6,        # steep curve → profitability
        "BAMLH0A0HYM2": -0.5, # credit spreads → loan losses
        "GDP": 0.3,
        "DRTSCILM": -0.3,     # tightening standards → less lending
    },
    "Information Technology": {
        "DFF": -0.5,          # growth stocks rate-sensitive
        "DGS10": -0.4,
        "GDP": 0.3,
        "VIXCLS": -0.3,       # risk-on sector
        "INDPRO": 0.2,
    },
    "Communication Services": {
        "DFF": -0.4,
        "DGS10": -0.3,
        "GDP": 0.2,
        "VIXCLS": -0.2,
    },
    "Utilities": {
        "DFF": -0.6,          # bond proxy, very rate-sensitive
        "DGS10": -0.5,
        "VIXCLS": 0.3,        # defensive / flight to safety
        "GDP": -0.1,
        "T10Y2Y": -0.2,
    },
    "Real Estate": {
        "DFF": -0.7,          # most rate-sensitive sector
        "DGS10": -0.5,
        "GDP": 0.2,
        "UNRATE": -0.2,
        "M2SL": 0.3,          # liquidity → property values
    },
}


# ---------------------------------------------------------------------------
# Regime adjustments
# ---------------------------------------------------------------------------
# Additional tilt multiplier per regime. 1.0 = no change, >1 = amplify, <1 = dampen.

_REGIME_SECTOR_MULTIPLIERS: dict[str, dict[str, float]] = {
    "expansion": {
        "Information Technology": 1.3,
        "Consumer Discretionary": 1.2,
        "Industrials": 1.2,
        "Utilities": 0.7,
        "Consumer Staples": 0.8,
    },
    "overheating": {
        "Energy": 1.3,
        "Materials": 1.2,
        "Information Technology": 0.7,
        "Real Estate": 0.7,
    },
    "stagflation": {
        "Consumer Staples": 1.3,
        "Health Care": 1.3,
        "Utilities": 1.2,
        "Consumer Discretionary": 0.6,
        "Information Technology": 0.7,
    },
    "recession": {
        "Consumer Staples": 1.4,
        "Health Care": 1.3,
        "Utilities": 1.3,
        "Consumer Discretionary": 0.5,
        "Industrials": 0.6,
        "Financials": 0.7,
    },
    "recovery": {
        "Consumer Discretionary": 1.3,
        "Industrials": 1.3,
        "Financials": 1.2,
        "Materials": 1.2,
        "Utilities": 0.7,
        "Consumer Staples": 0.8,
    },
}


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sector_sensitivities (
    sector      TEXT NOT NULL,
    indicator   TEXT NOT NULL,
    sensitivity REAL NOT NULL,
    PRIMARY KEY (sector, indicator)
);
"""


# ---------------------------------------------------------------------------
# SectorTilter
# ---------------------------------------------------------------------------

class SectorTilter:
    """11 GICS 行业宏观敏感度 tilt 引擎。

    Args:
        macro_provider: MacroProvider (get_latest_z_scores)
        macro_scoring_provider: MacroScoringProvider (get_regime) — optional
        db_path: SQLite 数据库路径 (默认 data/sector_sensitivities.db)
    """

    def __init__(
        self,
        macro_provider,
        macro_scoring_provider=None,
        *,
        db_path: str | Path = "data/sector_sensitivities.db",
    ):
        self._macro = macro_provider
        self._scoring = macro_scoring_provider
        self._db_path = Path(db_path)
        self._db: sqlite3.Connection | None = None

    def initialize(self) -> None:
        """Initialize SQLite and seed defaults if empty."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute(_SCHEMA)
        self._db.commit()
        # Seed if empty
        count = self._db.execute(
            "SELECT COUNT(*) FROM sector_sensitivities"
        ).fetchone()[0]
        if count == 0:
            self._seed_defaults()

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute_tilts(self, as_of: date | None = None) -> dict[str, float]:
        """Compute sector tilts based on current macro Z-scores.

        Returns:
            {sector_name: tilt_weight} where tilt is in [-0.03, +0.03]
        """
        z_scores = self._macro.get_latest_z_scores()
        if not z_scores:
            logger.warning("Empty z_scores, returning zero tilts")
            return {s: 0.0 for s in GICS_SECTORS}

        # Load sensitivity matrix
        matrix = self._load_matrix()

        # Compute raw tilts
        tilts = self._compute_raw_tilts(z_scores, matrix)

        # Apply regime adjustment
        regime = self._get_regime(as_of)
        if regime is not None:
            tilts = self._apply_regime_adjustment(tilts, regime)

        # Cap to ±TILT_CAP
        return {s: max(-TILT_CAP, min(TILT_CAP, t)) for s, t in tilts.items()}

    def get_sensitivity_matrix(self) -> pd.DataFrame:
        """Return the full sensitivity matrix as a DataFrame.

        Returns:
            DataFrame with columns: sector, indicator, sensitivity
        """
        assert self._db is not None
        rows = self._db.execute(
            "SELECT sector, indicator, sensitivity FROM sector_sensitivities "
            "ORDER BY sector, indicator"
        ).fetchall()
        return pd.DataFrame(rows, columns=["sector", "indicator", "sensitivity"])

    def update_sensitivity(
        self, sector: str, indicator: str, sensitivity: float
    ) -> None:
        """Update a single sensitivity value."""
        assert self._db is not None
        self._db.execute(
            "INSERT OR REPLACE INTO sector_sensitivities "
            "(sector, indicator, sensitivity) VALUES (?, ?, ?)",
            (sector, indicator, sensitivity),
        )
        self._db.commit()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_matrix(self) -> dict[str, dict[str, float]]:
        """Load sensitivity matrix from SQLite into nested dict."""
        assert self._db is not None
        rows = self._db.execute(
            "SELECT sector, indicator, sensitivity FROM sector_sensitivities"
        ).fetchall()
        matrix: dict[str, dict[str, float]] = {}
        for sector, indicator, sensitivity in rows:
            matrix.setdefault(sector, {})[indicator] = sensitivity
        return matrix

    def _compute_raw_tilts(
        self,
        z_scores: dict[str, float],
        matrix: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        """Weighted sum of z_scores × sensitivities per sector."""
        tilts: dict[str, float] = {}
        for sector in GICS_SECTORS:
            sensitivities = matrix.get(sector, {})
            if not sensitivities:
                tilts[sector] = 0.0
                continue
            total = 0.0
            weight_sum = 0.0
            for indicator, sensitivity in sensitivities.items():
                if indicator in z_scores:
                    total += z_scores[indicator] * sensitivity
                    weight_sum += abs(sensitivity)
            # Normalize by total weight to keep scale consistent
            tilts[sector] = total / weight_sum if weight_sum > 0 else 0.0
        return tilts

    def _apply_regime_adjustment(
        self, tilts: dict[str, float], regime: str
    ) -> dict[str, float]:
        """Apply regime-specific multipliers to tilts."""
        multipliers = _REGIME_SECTOR_MULTIPLIERS.get(regime, {})
        return {
            sector: tilt * multipliers.get(sector, 1.0)
            for sector, tilt in tilts.items()
        }

    def _get_regime(self, as_of: date | None) -> str | None:
        """Get current regime from MacroScoringProvider."""
        if self._scoring is None:
            return None
        as_of = as_of or date.today()
        return self._scoring.get_regime(as_of)

    def _seed_defaults(self) -> None:
        """Seed the sensitivity matrix with default values."""
        assert self._db is not None
        rows = [
            (sector, indicator, sensitivity)
            for sector, indicators in DEFAULT_SENSITIVITIES.items()
            for indicator, sensitivity in indicators.items()
        ]
        self._db.executemany(
            "INSERT INTO sector_sensitivities (sector, indicator, sensitivity) "
            "VALUES (?, ?, ?)",
            rows,
        )
        self._db.commit()
        logger.info("Seeded %d sensitivity entries", len(rows))
