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

    def test_previous_probability_from_db(self, polymarket, sample_events):
        polymarket.save_events(sample_events)

        prev = polymarket._get_previous_probability("evt1")
        assert prev == 0.65  # last saved probability

    def test_previous_probability_missing(self, polymarket):
        prev = polymarket._get_previous_probability("nonexistent")
        assert prev is None

    def test_extract_probability(self, polymarket):
        market = {"outcomePrices": json.dumps([0.72, 0.28])}
        assert polymarket._extract_probability(market) == 0.72

    def test_extract_probability_fallback(self, polymarket):
        market = {"bestAsk": 0.55}
        assert polymarket._extract_probability(market) == 0.55

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
        assert count == 0  # should skip gracefully

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
