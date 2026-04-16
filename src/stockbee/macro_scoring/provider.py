"""MacroScoringProvider — 日度宏观评分 Provider + SQLite 持久化。

统一 Provider 接口，供下游 sector/style tilt 消费。
回测模式: 纯规则引擎 (无 LLM)。
实盘模式: 规则引擎 + LLM weekly overlay。

来源：Tech Design §3.5, plan-milestones 2026-04-15
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

from stockbee.providers.base import BaseProvider, ProviderConfig

from .llm_analyst import LLMMacroAnalyst
from .scorer import MacroScore, MacroScorer, RegimeType

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS macro_scores (
    scored_date    TEXT PRIMARY KEY,
    score          REAL NOT NULL,
    regime         TEXT NOT NULL,
    confidence_json TEXT NOT NULL,
    llm_outlook    REAL,
    llm_insight_json TEXT,
    signals_json   TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


class MacroScoringProvider(BaseProvider):
    """日度宏观评分 Provider。

    Args (via ProviderConfig.params):
        db_path: SQLite 数据库路径 (默认 data/macro_scores.db)
        mode: "backtest" | "live" (默认 backtest)
    """

    def __init__(self, config: ProviderConfig | None = None) -> None:
        super().__init__(config)
        params = self._config.params
        self._db_path = Path(params.get("db_path", "data/macro_scores.db"))
        self._mode = params.get("mode", "backtest")
        self._db: sqlite3.Connection | None = None
        self._scorer: MacroScorer | None = None
        self._analyst: LLMMacroAnalyst | None = None

    def set_scorer(self, scorer: MacroScorer) -> None:
        """注入 MacroScorer 实例。"""
        self._scorer = scorer

    def set_analyst(self, analyst: LLMMacroAnalyst) -> None:
        """注入 LLMMacroAnalyst 实例 (实盘模式)。"""
        self._analyst = analyst

    def _do_initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(_SCHEMA)
        self._db.commit()

    def _do_shutdown(self) -> None:
        if self._analyst is not None:
            self._analyst.close()
        if self._db is not None:
            self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_macro_score(self, as_of: date) -> float | None:
        """Get macro score for a given date. Returns None if no data."""
        row = self._query_row(as_of)
        return row[0] if row else None

    def get_regime(self, as_of: date) -> str | None:
        """Get economic regime for a given date. Returns None if no data."""
        row = self._query_row(as_of)
        return row[1] if row else None

    def get_history(self, start: date, end: date) -> pd.DataFrame:
        """Get score history as DataFrame.

        Returns:
            DataFrame with columns: date, score, regime, llm_outlook
            Empty DataFrame if no data in range.
        """
        assert self._db is not None
        rows = self._db.execute(
            "SELECT scored_date, score, regime, llm_outlook "
            "FROM macro_scores WHERE scored_date >= ? AND scored_date <= ? "
            "ORDER BY scored_date",
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        if not rows:
            return pd.DataFrame(columns=["date", "score", "regime", "llm_outlook"])
        df = pd.DataFrame(rows, columns=["date", "score", "regime", "llm_outlook"])
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def score_and_store(self, as_of: date | None = None) -> MacroScore | None:
        """Score the given date and persist to SQLite.

        In live mode, also fetches weekly LLM insight and blends outlook.
        Returns None if scorer is not set.
        """
        if self._scorer is None:
            logger.warning("No scorer set, cannot score")
            return None

        as_of = as_of or date.today()
        result = self._scorer.score(as_of)

        # Optional LLM overlay (live mode only)
        llm_outlook: float | None = None
        llm_insight_json: str | None = None
        if self._mode == "live" and self._analyst is not None:
            insight = self._analyst.analyze(as_of)
            if insight is not None:
                llm_outlook = insight.outlook_score
                llm_insight_json = json.dumps({
                    "regime_assessment": insight.regime_assessment,
                    "risk_factors": insight.risk_factors,
                    "reasoning": insight.reasoning[:500],
                })
                # Blend: 0.7 rule-based + 0.3 LLM outlook
                result = MacroScore(
                    score=max(-1.0, min(1.0, result.score * 0.7 + llm_outlook * 0.3)),
                    regime=result.regime,
                    regime_confidence=result.regime_confidence,
                    scored_at=result.scored_at,
                    signals={**result.signals, "llm_blend": 0.3, "llm_outlook": llm_outlook},
                )

        self._store(result, llm_outlook, llm_insight_json)
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _query_row(self, as_of: date) -> tuple | None:
        """Query (score, regime) for a date."""
        assert self._db is not None
        return self._db.execute(
            "SELECT score, regime FROM macro_scores WHERE scored_date = ?",
            (as_of.isoformat(),),
        ).fetchone()

    def _store(
        self,
        result: MacroScore,
        llm_outlook: float | None,
        llm_insight_json: str | None,
    ) -> None:
        """Persist a MacroScore to SQLite."""
        assert self._db is not None
        self._db.execute(
            "INSERT OR REPLACE INTO macro_scores "
            "(scored_date, score, regime, confidence_json, llm_outlook, llm_insight_json, signals_json) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                result.scored_at.isoformat(),
                result.score,
                result.regime.value,
                json.dumps(result.regime_confidence),
                llm_outlook,
                llm_insight_json,
                json.dumps(result.signals),
            ),
        )
        self._db.commit()
