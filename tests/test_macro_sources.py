# tests/test_macro_sources.py
"""Macro Sources 模块测试。
# PYTHONPATH=src .venv/bin/python -m pytest tests/test_macro_sources.py -v

测试对象：PolymarketFetcher (SQLite 存储 + cliff 检测)、EconomicCalendar (SQLite 存储 + 查询)。
不测实际 API 调用（需要网络），只测本地逻辑。
"""

from datetime import date, timedelta
from unittest.mock import patch, MagicMock
import json

import pytest

from stockbee.macro_sources.polymarket import (
    PolymarketFetcher, MarketEvent, DEFAULT_CLIFF_THRESHOLD,
)
from stockbee.macro_sources.calendar import (
    EconomicCalendar, ReleaseEvent, HIGH_VOLATILITY_RELEASES,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def polymarket(tmp_path) -> PolymarketFetcher:
    fetcher = PolymarketFetcher(
        db_path=str(tmp_path / "macro_sources.db"),
        cliff_threshold=0.15,
    )
    fetcher.initialize()
    return fetcher


@pytest.fixture
def calendar(tmp_path) -> EconomicCalendar:
    cal = EconomicCalendar(
        api_key="test_key",
        db_path=str(tmp_path / "macro_sources.db"),
    )
    cal.initialize()
    return cal


@pytest.fixture
def sample_events() -> list[MarketEvent]:
    return [
        MarketEvent(
            event_id="evt1", question="Fed rate cut Q2 2026?",
            slug="fed-rate-cut-q2-2026", probability=0.65,
            previous_probability=0.45, volume=5000000.0,
            liquidity=200000.0, is_cliff=True,
            fetched_at="2026-04-06T00:00:00Z",
        ),
        MarketEvent(
            event_id="evt2", question="US recession 2026?",
            slug="us-recession-2026", probability=0.30,
            previous_probability=0.28, volume=3000000.0,
            liquidity=150000.0, is_cliff=False,
            fetched_at="2026-04-06T00:00:00Z",
        ),
        MarketEvent(
            event_id="evt3", question="Trump tariff increase?",
            slug="trump-tariff-increase", probability=0.80,
            previous_probability=None, volume=8000000.0,
            liquidity=500000.0, is_cliff=False,
            fetched_at="2026-04-06T00:00:00Z",
        ),
    ]


# =========================================================================
# PolymarketFetcher Tests
# =========================================================================

class TestPolymarketFetcher:

    def test_save_and_read_events(self, polymarket, sample_events):
        count = polymarket.save_events(sample_events)
        assert count == 3

        latest = polymarket.get_latest_events()
        assert len(latest) == 3
        assert latest[0]["event_id"] in ["evt1", "evt2", "evt3"]

    def test_cliff_detection(self, polymarket, sample_events):
        cliffs = polymarket.detect_cliffs(sample_events)
        assert len(cliffs) == 1
        assert cliffs[0].event_id == "evt1"
        assert cliffs[0].probability == 0.65
        assert cliffs[0].previous_probability == 0.45

    def test_no_cliff_when_no_previous(self, polymarket):
        event = MarketEvent(
            event_id="new", question="New event?", slug="new",
            probability=0.50, previous_probability=None,
            volume=1000.0, liquidity=100.0, is_cliff=False,
            fetched_at="2026-04-06T00:00:00Z",
        )
        assert not polymarket._is_cliff(event.probability, event.previous_probability)

    def test_cliff_threshold(self, polymarket):
        # 14% change — not a cliff (threshold is 15%)
        assert not polymarket._is_cliff(0.50, 0.36)
        # 15% change — is a cliff
        assert polymarket._is_cliff(0.50, 0.35)
        # 20% change — is a cliff
        assert polymarket._is_cliff(0.70, 0.50)

    def test_last_saved_probability_from_db(self, polymarket, sample_events):
        polymarket.save_events(sample_events)

        # Method returns the most recent row — after save that equals the value
        # we just wrote. The name reflects "latest saved", not an abstract
        # "previous" (regression test for the rename / semantic clarification).
        prev = polymarket._get_last_saved_probability("evt1")
        assert prev == 0.65

    def test_last_saved_probability_missing(self, polymarket):
        prev = polymarket._get_last_saved_probability("nonexistent")
        assert prev is None

    def test_extract_probability_binary_yes_first(self, polymarket):
        market = {
            "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps(["0.72", "0.28"]),
        }
        assert polymarket._extract_probability(market) == 0.72

    def test_extract_probability_binary_yes_second(self, polymarket):
        # Some markets list "No" first — probability must track the Yes index
        # rather than blindly returning outcomePrices[0].
        market = {
            "outcomes": json.dumps(["No", "Yes"]),
            "outcomePrices": json.dumps(["0.28", "0.72"]),
        }
        assert polymarket._extract_probability(market) == 0.72

    def test_extract_probability_skip_multi_outcome(self, polymarket):
        # Non-binary market (e.g. "Fed May decision" with hold/-25/-50 bp
        # outcomes) has no single event probability — must return None instead
        # of the meaningless outcomePrices[0].
        market = {
            "outcomes": json.dumps(["Hold", "-25bp", "-50bp"]),
            "outcomePrices": json.dumps(["0.30", "0.50", "0.20"]),
        }
        assert polymarket._extract_probability(market) is None

    def test_extract_probability_skip_non_yes_binary(self, polymarket):
        # Two outcomes but neither is a Yes-like label → skip.
        market = {
            "outcomes": json.dumps(["Trump", "Biden"]),
            "outcomePrices": json.dumps(["0.55", "0.45"]),
        }
        assert polymarket._extract_probability(market) is None

    def test_extract_probability_missing_outcomes_field(self, polymarket):
        # outcomePrices alone is no longer enough — bestAsk fallback was a bug.
        market = {"outcomePrices": json.dumps(["0.72", "0.28"])}
        assert polymarket._extract_probability(market) is None

    def test_extract_probability_malformed_json(self, polymarket):
        market = {
            "outcomes": "not json",
            "outcomePrices": "[0.72, 0.28]",
        }
        assert polymarket._extract_probability(market) is None

    def test_extract_probability_none(self, polymarket):
        market = {}
        assert polymarket._extract_probability(market) is None

    def test_empty_save(self, polymarket):
        count = polymarket.save_events([])
        assert count == 0

    def test_shutdown_and_reinit(self, polymarket, sample_events):
        polymarket.save_events(sample_events)
        polymarket.shutdown()

        polymarket.initialize()
        latest = polymarket.get_latest_events()
        assert len(latest) == 3

    def test_multiple_fetches_track_history(self, polymarket):
        """同一事件多次抓取，保留历史。"""
        e1 = MarketEvent(
            event_id="evt1", question="Q?", slug="q",
            probability=0.50, previous_probability=None,
            volume=1000.0, liquidity=100.0, is_cliff=False,
            fetched_at="2026-04-06T00:00:00Z",
        )
        polymarket.save_events([e1])

        e2 = MarketEvent(
            event_id="evt1", question="Q?", slug="q",
            probability=0.70, previous_probability=0.50,
            volume=1000.0, liquidity=100.0, is_cliff=True,
            fetched_at="2026-04-07T00:00:00Z",
        )
        polymarket.save_events([e2])

        # Latest should return the most recent fetch
        latest = polymarket.get_latest_events()
        assert len(latest) == 1
        assert latest[0]["probability"] == 0.70


# =========================================================================
# EconomicCalendar Tests
# =========================================================================

class TestEconomicCalendar:

    def test_high_volatility_releases_defined(self):
        assert len(HIGH_VOLATILITY_RELEASES) == 8
        assert 50 in HIGH_VOLATILITY_RELEASES  # NFP
        assert 10 in HIGH_VOLATILITY_RELEASES  # CPI
        assert 53 in HIGH_VOLATILITY_RELEASES  # GDP

    def test_save_and_query_events(self, calendar):
        events = [
            ReleaseEvent(release_id=50, release_name="Employment Situation",
                         release_date=date(2026, 5, 1), high_volatility=True),
            ReleaseEvent(release_id=10, release_name="CPI",
                         release_date=date(2026, 5, 15), high_volatility=True),
            ReleaseEvent(release_id=999, release_name="Some Minor Release",
                         release_date=date(2026, 5, 10), high_volatility=False),
        ]
        count = calendar._save_events(events)
        assert count == 3

        result = calendar.get_events(start=date(2026, 5, 1), end=date(2026, 5, 31))
        assert len(result) == 3

    def test_high_volatility_filter(self, calendar):
        events = [
            ReleaseEvent(50, "NFP", date(2026, 5, 1), True),
            ReleaseEvent(999, "Minor", date(2026, 5, 2), False),
        ]
        calendar._save_events(events)

        hv = calendar.get_events(high_volatility_only=True)
        assert len(hv) == 1
        assert hv[0].release_name == "NFP"

    def test_is_high_volatility_day(self, calendar):
        events = [
            ReleaseEvent(50, "NFP", date(2026, 5, 1), True),
            ReleaseEvent(999, "Minor", date(2026, 5, 2), False),
        ]
        calendar._save_events(events)

        assert calendar.is_high_volatility_day(date(2026, 5, 1)) is True
        assert calendar.is_high_volatility_day(date(2026, 5, 2)) is False
        assert calendar.is_high_volatility_day(date(2026, 5, 3)) is False

    def test_get_next_high_volatility(self, calendar):
        events = [
            ReleaseEvent(50, "NFP", date(2026, 5, 1), True),
            ReleaseEvent(10, "CPI", date(2026, 5, 15), True),
        ]
        calendar._save_events(events)

        nxt = calendar.get_next_high_volatility(after=date(2026, 5, 2))
        assert nxt is not None
        assert nxt.release_name == "CPI"
        assert nxt.release_date == date(2026, 5, 15)

    def test_no_next_high_volatility(self, calendar):
        nxt = calendar.get_next_high_volatility(after=date(2026, 12, 31))
        assert nxt is None

    def test_parse_release_dates(self, calendar):
        raw = [
            {"release_id": 50, "release_name": "Employment Situation", "date": "2026-05-01"},
            {"release_id": 10, "release_name": "CPI", "date": "2026-05-15"},
            {"release_id": 999, "release_name": "Unknown", "date": "2026-05-10"},
        ]
        events = calendar._parse_release_dates(raw)
        assert len(events) == 3
        assert events[0].high_volatility is True
        assert events[2].high_volatility is False

    def test_sync_without_api_key(self, tmp_path):
        cal = EconomicCalendar(api_key="", db_path=str(tmp_path / "test.db"))
        cal.initialize()
        count = cal.sync()
        assert count == 0  # should skip gracefully in default (dev) mode

    def test_sync_without_api_key_strict_raises(self, tmp_path):
        # Production callers pass strict=True so a missing key fails loudly
        # instead of silently returning "zero events".
        cal = EconomicCalendar(api_key="", db_path=str(tmp_path / "test.db"))
        cal.initialize()
        with pytest.raises(RuntimeError, match="No FRED API key"):
            cal.sync(strict=True)

    def test_date_range_query(self, calendar):
        events = [
            ReleaseEvent(50, "NFP", date(2026, 4, 1), True),
            ReleaseEvent(10, "CPI", date(2026, 5, 15), True),
            ReleaseEvent(53, "GDP", date(2026, 6, 30), True),
        ]
        calendar._save_events(events)

        may = calendar.get_events(start=date(2026, 5, 1), end=date(2026, 5, 31))
        assert len(may) == 1
        assert may[0].release_name == "CPI"

    def test_shutdown_and_reinit(self, calendar):
        events = [ReleaseEvent(50, "NFP", date(2026, 5, 1), True)]
        calendar._save_events(events)
        calendar.shutdown()

        calendar.initialize()
        result = calendar.get_events()
        assert len(result) == 1


# =========================================================================
# PolymarketFetcher — keyword regex + API error path + same-microsecond
# collision regression tests.
# =========================================================================

class TestMacroKeywordRegex:
    """Guard the MACRO_KEYWORDS matcher against substring false positives and
    word-boundary false negatives."""

    def _match(self, polymarket, question, description=""):
        return polymarket._is_macro_related(
            {"question": question, "description": description}
        )

    def test_matches_fed_with_punctuation(self, polymarket):
        # The old `"fed "` substring missed these; \bfed\b catches them.
        assert self._match(polymarket, "Fed's decision in May?")
        assert self._match(polymarket, "Will the Fed cut rates?")
        assert self._match(polymarket, "Fed, acting alone, is unlikely")

    def test_does_not_match_fedex(self, polymarket):
        # Substring match would have been fine here (the old code used "fed "
        # with trailing space), but the new regex with \b must keep this
        # negative.
        assert not self._match(polymarket, "Will FedEx stock rally?")

    def test_matches_description_field(self, polymarket):
        assert self._match(
            polymarket,
            question="Will ABC happen in 2026?",
            description="Event depends on FOMC decision in June.",
        )

    def test_case_insensitive(self, polymarket):
        assert self._match(polymarket, "CPI PRINT ABOVE 3%?")
        assert self._match(polymarket, "cpi print above 3%?")

    def test_cpi_word_boundary_not_substring(self, polymarket):
        # "cpix" (fictional ticker) must not be treated as macro.
        assert not self._match(polymarket, "Will CPIX stock break 100?")

    def test_missing_keyword(self, polymarket):
        assert not self._match(polymarket, "Who will win the Super Bowl?")


class TestPolymarketCallApi:
    """Exercise the network error handling paths without hitting real API."""

    def test_url_error_returns_empty(self, polymarket, caplog):
        import urllib.error
        with patch("stockbee.macro_sources.polymarket.urlopen") as mock_open:
            mock_open.side_effect = urllib.error.URLError("network down")
            result = polymarket._call_api("/markets", params={"limit": 10})
        assert result == []
        # Error log must NOT leak the full URL (only the path prefix).
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "gamma-api.polymarket.com" not in messages
        assert "/markets" in messages

    def test_json_decode_error_returns_empty(self, polymarket):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda *a: None
        with patch("stockbee.macro_sources.polymarket.urlopen", return_value=mock_resp):
            result = polymarket._call_api("/markets")
        assert result == []

    def test_envelope_data_key(self, polymarket):
        # API may return either a bare list or {"data": [...]}.
        envelope = {"data": [{"id": "x", "question": "Fed cut?"}]}
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(envelope).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda *a: None
        with patch("stockbee.macro_sources.polymarket.urlopen", return_value=mock_resp):
            result = polymarket._call_api("/markets")
        assert result == [{"id": "x", "question": "Fed cut?"}]


class TestPolymarketFetchedAtCollision:
    """Same-microsecond / same-second fetches must not silently overwrite
    history rows."""

    def _make_event(self, prob, fetched_at):
        return MarketEvent(
            event_id="evt-collide", question="Fed cut?",
            slug="fed-cut", probability=prob,
            previous_probability=None, volume=1000.0,
            liquidity=100.0, is_cliff=False, fetched_at=fetched_at,
        )

    def test_distinct_microsecond_preserves_history(self, polymarket):
        # Two rows with different microsecond suffixes must both survive.
        e1 = self._make_event(0.50, "2026-04-10T12:00:00.100000+00:00")
        e2 = self._make_event(0.70, "2026-04-10T12:00:00.200000+00:00")
        assert polymarket.save_events([e1]) == 1
        assert polymarket.save_events([e2]) == 1

        conn = polymarket._conn
        rows = conn.execute(
            "SELECT probability FROM polymarket_events WHERE event_id = ? ORDER BY fetched_at",
            ("evt-collide",),
        ).fetchall()
        assert [r[0] for r in rows] == [0.50, 0.70]

    def test_identical_fetched_at_is_idempotent_not_destructive(self, polymarket):
        # Re-saving the *same* (event_id, fetched_at) row must not clobber it
        # with a newer probability — INSERT OR IGNORE preserves the original.
        ts = "2026-04-10T12:00:00.000000+00:00"
        e_first = self._make_event(0.50, ts)
        e_retry = self._make_event(0.99, ts)  # same timestamp, different value

        assert polymarket.save_events([e_first]) == 1
        # Retry at same timestamp: row exists, INSERT OR IGNORE no-ops.
        polymarket.save_events([e_retry])

        conn = polymarket._conn
        rows = conn.execute(
            "SELECT probability FROM polymarket_events WHERE event_id = ?",
            ("evt-collide",),
        ).fetchall()
        assert len(rows) == 1
        # Original value wins (not silently overwritten by retry).
        assert rows[0][0] == 0.50

    def test_fetch_macro_events_uses_microsecond_precision(self):
        # The real fetch path stamps events with microsecond-precision ISO
        # strings so back-to-back fetches in the same second never collide.
        from stockbee.macro_sources import polymarket as pmod

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([
            {
                "id": "e1",
                "question": "Will the Fed cut rates?",
                "description": "",
                "slug": "fed-cut",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["0.60", "0.40"]),
                "volume": 1000,
                "liquidity": 100,
            }
        ]).encode()
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda *a: None

        fetcher = pmod.PolymarketFetcher(db_path=":memory:")
        fetcher.initialize()
        try:
            with patch("stockbee.macro_sources.polymarket.urlopen", return_value=mock_resp):
                events = fetcher.fetch_macro_events(limit=5, fetch_size=5)
            assert len(events) == 1
            # Microsecond precision → 6 fractional digits after the dot.
            ts = events[0].fetched_at
            assert "." in ts, f"expected microsecond suffix, got {ts}"
            frac = ts.split(".", 1)[1]
            # Strip tz suffix (+00:00) before counting fractional digits.
            frac_digits = frac.split("+", 1)[0].split("Z", 1)[0]
            assert len(frac_digits) == 6
        finally:
            fetcher.shutdown()
