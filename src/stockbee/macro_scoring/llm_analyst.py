"""LLMMacroAnalyst — 周度 LLM 宏观深度分析。

使用 LLMRouter (TaskType.MACRO_ANALYSIS → Claude Opus) 对当前宏观环境
做深度解读。结果缓存 7 天，避免重复调用。

来源：Tech Design §3.5, plan-milestones 2026-04-15
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_TTL_DAYS = 7


@dataclass(frozen=True)
class MacroInsight:
    """LLM 周度宏观分析结果。"""
    regime_assessment: str          # LLM 对当前经济周期的判断
    risk_factors: list[str]         # 主要风险因素
    outlook_score: float            # LLM 给出的 outlook [-1.0, +1.0]
    reasoning: str                  # 详细分析推理
    model_used: str                 # 使用的模型
    analyzed_at: date               # 分析日期


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a senior macro economist analyzing US economic conditions for a \
quantitative equity trading system. Your analysis directly influences \
portfolio tilt decisions.

Respond in the following format (plain text, not JSON):

REGIME: <one of: expansion, overheating, stagflation, recession, recovery>
OUTLOOK_SCORE: <float from -1.0 (very bearish) to +1.0 (very bullish)>
RISK_FACTORS:
- <risk factor 1>
- <risk factor 2>
- <risk factor 3>
REASONING:
<2-3 paragraph analysis of current macro environment, key drivers, and outlook>
"""


class LLMMacroAnalyst:
    """周度 LLM 宏观分析引擎。

    Args:
        router: LLMRouter 实例
        macro_provider: MacroProvider 实例 (get_latest_z_scores)
        polymarket: PolymarketFetcher 实例 (可选)
        calendar: EconomicCalendar 实例 (可选)
        cache_dir: SQLite 缓存目录 (默认 data/)
    """

    def __init__(
        self,
        router,
        macro_provider,
        polymarket=None,
        calendar=None,
        *,
        cache_dir: str | Path = "data",
    ):
        self._router = router
        self._macro = macro_provider
        self._polymarket = polymarket
        self._calendar = calendar
        self._cache_dir = Path(cache_dir)
        self._db: sqlite3.Connection | None = None

    def _ensure_db(self) -> sqlite3.Connection:
        """Lazy-init SQLite cache."""
        if self._db is not None:
            return self._db
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = self._cache_dir / "macro_llm_cache.db"
        self._db = sqlite3.connect(str(db_path))
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS macro_insights (
                analyzed_date TEXT PRIMARY KEY,
                insight_json  TEXT NOT NULL,
                model_used    TEXT NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now'))
            )"""
        )
        self._db.commit()
        return self._db

    def analyze(self, as_of: date | None = None) -> MacroInsight | None:
        """Run weekly macro analysis.

        Returns cached result if within TTL, otherwise calls LLM.
        Returns None if z_scores are empty or LLM fails.
        """
        as_of = as_of or date.today()

        # Check cache first
        cached = self._get_cached(as_of)
        if cached is not None:
            return cached

        # Get z_scores
        z_scores = self._macro.get_latest_z_scores()
        if not z_scores:
            logger.warning("Empty z_scores, skipping LLM analysis")
            return None

        # Build prompt
        prompt = self._build_prompt(z_scores, as_of)

        # Call LLM
        try:
            from stockbee.llm_routing import TaskType

            response = self._router.route(
                TaskType.MACRO_ANALYSIS,
                prompt,
                system_prompt=_SYSTEM_PROMPT,
            )
        except Exception:
            logger.warning("LLM macro analysis failed", exc_info=True)
            return None

        # Parse response
        insight = self._parse_response(response.content, response.model_used, as_of)
        if insight is not None:
            self._save_cache(insight)

        return insight

    def _build_prompt(self, z_scores: dict[str, float], as_of: date) -> str:
        """Build the analysis prompt with current macro data."""
        lines = [f"Date: {as_of.isoformat()}", "", "## Current FRED Z-Scores (252-day rolling)", ""]

        # Group z_scores by dimension
        try:
            from stockbee.macro_data.indicators import FRED_INDICATORS
            for code, z in sorted(z_scores.items()):
                ind = FRED_INDICATORS.get(code)
                label = f"{ind.name} ({ind.dimension})" if ind else code
                direction = "↑" if z > 0.5 else "↓" if z < -0.5 else "→"
                lines.append(f"- {label}: z={z:+.2f} {direction}")
        except ImportError:
            for code, z in sorted(z_scores.items()):
                lines.append(f"- {code}: z={z:+.2f}")

        # Polymarket events
        if self._polymarket is not None:
            try:
                events = self._polymarket.get_latest_events(limit=10)
                if events:
                    lines.extend(["", "## Polymarket Macro Events", ""])
                    for e in events[:10]:
                        cliff_tag = " ⚠️CLIFF" if e.is_cliff else ""
                        lines.append(f"- {e.question}: {e.probability:.0%}{cliff_tag}")
            except Exception:
                logger.debug("Polymarket events fetch failed for prompt", exc_info=True)

        # Upcoming high-volatility events
        if self._calendar is not None:
            try:
                upcoming = self._calendar.get_events(
                    start=as_of, end=as_of + timedelta(days=14), high_volatility_only=True
                )
                if upcoming:
                    lines.extend(["", "## Upcoming High-Volatility Events (14 days)", ""])
                    for ev in upcoming[:5]:
                        lines.append(f"- {ev.release_name}: {ev.release_date}")
            except Exception:
                logger.debug("Calendar events fetch failed for prompt", exc_info=True)

        lines.extend([
            "",
            "## Task",
            "Analyze the current US macro environment. Classify the economic regime, "
            "identify top risk factors, and provide an outlook score.",
        ])

        return "\n".join(lines)

    def _parse_response(
        self, content: str, model_used: str, as_of: date
    ) -> MacroInsight | None:
        """Parse LLM text response into MacroInsight."""
        if not content or not content.strip():
            logger.warning("Empty LLM response")
            return None

        # Extract fields
        regime = self._extract_field(content, "REGIME", "unknown")
        outlook_str = self._extract_field(content, "OUTLOOK_SCORE", "0.0")
        reasoning = self._extract_section(content, "REASONING")

        # Parse outlook_score
        try:
            outlook_score = float(outlook_str)
            outlook_score = max(-1.0, min(1.0, outlook_score))
        except (ValueError, TypeError):
            logger.warning(f"Could not parse outlook_score: {outlook_str!r}, defaulting to 0.0")
            outlook_score = 0.0

        # Parse risk factors
        risk_factors = self._extract_list(content, "RISK_FACTORS")

        return MacroInsight(
            regime_assessment=regime,
            risk_factors=risk_factors,
            outlook_score=outlook_score,
            reasoning=reasoning,
            model_used=model_used,
            analyzed_at=as_of,
        )

    @staticmethod
    def _extract_field(text: str, field_name: str, default: str) -> str:
        """Extract a single-line field value like 'FIELD: value'."""
        pattern = rf"^{field_name}:\s*(.+)$"
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        return match.group(1).strip() if match else default

    @staticmethod
    def _extract_section(text: str, section_name: str) -> str:
        """Extract a multi-line section after 'SECTION_NAME:'."""
        pattern = rf"^{section_name}:\s*\n(.*?)(?=\n[A-Z_]+:|\Z)"
        match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_list(text: str, section_name: str) -> list[str]:
        """Extract a bulleted list after 'SECTION_NAME:'."""
        pattern = rf"^{section_name}:\s*\n((?:- .+\n?)+)"
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if not match:
            return []
        items = re.findall(r"^- (.+)$", match.group(1), re.MULTILINE)
        return [item.strip() for item in items]

    # ---------------------------------------------------------------------------
    # Cache
    # ---------------------------------------------------------------------------

    def _get_cached(self, as_of: date) -> MacroInsight | None:
        """Return cached insight if within TTL."""
        db = self._ensure_db()
        cutoff = (as_of - timedelta(days=CACHE_TTL_DAYS)).isoformat()
        row = db.execute(
            "SELECT insight_json, model_used, analyzed_date FROM macro_insights "
            "WHERE analyzed_date > ? ORDER BY analyzed_date DESC LIMIT 1",
            (cutoff,),
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return MacroInsight(
            regime_assessment=data["regime_assessment"],
            risk_factors=data["risk_factors"],
            outlook_score=data["outlook_score"],
            reasoning=data["reasoning"],
            model_used=row[1],
            analyzed_at=date.fromisoformat(row[2]),
        )

    def _save_cache(self, insight: MacroInsight) -> None:
        """Save insight to cache."""
        db = self._ensure_db()
        data = {
            "regime_assessment": insight.regime_assessment,
            "risk_factors": insight.risk_factors,
            "outlook_score": insight.outlook_score,
            "reasoning": insight.reasoning,
        }
        db.execute(
            "INSERT OR REPLACE INTO macro_insights (analyzed_date, insight_json, model_used) "
            "VALUES (?, ?, ?)",
            (insight.analyzed_at.isoformat(), json.dumps(data), insight.model_used),
        )
        db.commit()

    def close(self) -> None:
        """Close the cache database."""
        if self._db is not None:
            self._db.close()
            self._db = None
