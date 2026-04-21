"""m2b LocalSentimentProvider + g2 P3 重构测试。

覆盖:
- score_texts 透传 scorer (identity 断言)
- get_ticker_sentiment:
    * 空结果 → zeros + count=0
    * 加权均值公式手算
    * lookback_days 过滤
    * 多 ticker 共享新闻
    * NaN confidence 跳过
    * Σconf=0 → zeros
- backfill:
    * 只写 finbert_confidence IS NULL
    * 不覆盖 sentiment_score (g2 写)
    * since 过滤
    * limit 分页
    * 空 DB
- g2 重构后与 m2b 共享同一 FinBERTScorer singleton
- register_default: ProviderRegistry 注册 + 实例创建
- PROVIDER 初始化校验
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from stockbee.news_data.news_store import SqliteNewsProvider
from stockbee.providers.base import ProviderConfig
from stockbee.providers.registry import ProviderRegistry
from stockbee.small_models import finbert_scorer as fbs
from stockbee.small_models.finbert_scorer import (
    FinBERTScorer,
    get_default_scorer,
    reset_default_scorer,
)
from stockbee.small_models.local_sentiment_provider import LocalSentimentProvider


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_default_scorer()
    yield
    reset_default_scorer()


@pytest.fixture
def news_store_with_rows(tmp_path):
    """带 3 条新闻的 news_store: 2 AAPL 近期 + 1 MSFT 近期 (都有 finbert_*)。"""
    cfg = ProviderConfig(
        implementation="test", params={"db_path": str(tmp_path / "news.db")}
    )
    store = SqliteNewsProvider(cfg)
    store.initialize()
    base = datetime.now(timezone.utc) - timedelta(days=1)
    events = [
        {
            "headline": "AAPL beats earnings",
            "source": "reuters",
            "timestamp": base.isoformat(),
            "tickers": ["AAPL"],
            "sentiment_score": 0.9,
            "finbert_negative": 0.05,
            "finbert_neutral": 0.05,
            "finbert_confidence": 0.9,
        },
        {
            "headline": "AAPL new product launch",
            "source": "bloomberg",
            "timestamp": (base + timedelta(hours=1)).isoformat(),
            "tickers": ["AAPL"],
            "sentiment_score": 0.7,
            "finbert_negative": 0.10,
            "finbert_neutral": 0.20,
            "finbert_confidence": 0.7,
        },
        {
            "headline": "MSFT cloud growth",
            "source": "reuters",
            "timestamp": (base + timedelta(hours=2)).isoformat(),
            "tickers": ["MSFT"],
            "sentiment_score": 0.8,
            "finbert_negative": 0.10,
            "finbert_neutral": 0.10,
            "finbert_confidence": 0.8,
        },
    ]
    store.insert_news_batch(events)
    yield store
    store.shutdown()


@pytest.fixture
def provider(news_store_with_rows):
    """注入 mock FinBERTScorer + news_store 的已初始化 provider。"""
    scorer = FinBERTScorer(device="mock")
    p = LocalSentimentProvider(
        scorer=scorer, news_store=news_store_with_rows
    )
    p.initialize()
    yield p
    p.shutdown()


# ============================================================================
# score_texts 透传
# ============================================================================


class TestScoreTextsPassthrough:
    def test_passthrough_identity(self, provider):
        """provider.score_texts 和注入的 scorer.score_texts 结果一致。"""
        texts = ["earnings beat estimates", "company announces layoffs"]
        from_provider = provider.score_texts(texts)
        from_scorer = provider._scorer.score_texts(texts)
        assert from_provider == from_scorer

    def test_shared_scorer_default_singleton(self, news_store_with_rows):
        """默认不注入 scorer → 走 m2a get_default_scorer singleton。"""
        reset_default_scorer()
        p = LocalSentimentProvider(news_store=news_store_with_rows)
        p.initialize()
        assert p._scorer is get_default_scorer()
        p.shutdown()

    def test_require_initialized(self, news_store_with_rows):
        """未 initialize 时 score_texts raise。"""
        p = LocalSentimentProvider(
            scorer=FinBERTScorer(device="mock"), news_store=news_store_with_rows
        )
        with pytest.raises(RuntimeError, match="not initialized"):
            p.score_texts(["x"])


# ============================================================================
# get_ticker_sentiment
# ============================================================================


class TestGetTickerSentiment:
    def test_empty_result_zeros(self, provider):
        out = provider.get_ticker_sentiment("NOSUCH", lookback_days=7)
        assert out == {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "confidence": 0.0,
            "count": 0.0,
        }

    def test_weighted_mean_formula(self, provider):
        """手算 AAPL 的加权均值:
        row1: pos=0.9 conf=0.9 → contrib 0.81 to positive
        row2: pos=0.7 conf=0.7 → contrib 0.49
        sum_conf = 1.6
        weighted_positive = 1.3 / 1.6 = 0.8125

        confidence (H1): max(聚合后 positive/negative/neutral) = 0.8125
        """
        out = provider.get_ticker_sentiment("AAPL", lookback_days=7)
        assert out["count"] == 2.0
        # positive: (0.9*0.9 + 0.7*0.7) / (0.9 + 0.7) = 1.3 / 1.6 = 0.8125
        assert out["positive"] == pytest.approx(0.8125, abs=1e-4)
        # negative: (0.05*0.9 + 0.10*0.7) / 1.6 = 0.115 / 1.6 = 0.071875
        assert out["negative"] == pytest.approx(0.071875, abs=1e-4)
        # neutral: (0.05*0.9 + 0.20*0.7) / 1.6 = 0.185 / 1.6 = 0.115625
        assert out["neutral"] == pytest.approx(0.115625, abs=1e-4)
        # confidence = max(0.8125, 0.0719, 0.1156) = 0.8125 (与 FinBERTScorer 单行
        # confidence = max 三项的语义对齐)
        assert out["confidence"] == pytest.approx(0.8125, abs=1e-4)

    def test_lookback_filter_excludes_old(self, provider, news_store_with_rows):
        """插一条 100 天前的 AAPL, lookback=7 应排除。"""
        old = datetime.now(timezone.utc) - timedelta(days=100)
        news_store_with_rows.insert_news(
            headline="AAPL ancient news",
            source="reuters",
            timestamp=old.isoformat(),
            tickers=["AAPL"],
            sentiment_score=0.3,
            finbert_negative=0.5,
            finbert_neutral=0.2,
            finbert_confidence=0.5,
        )
        out = provider.get_ticker_sentiment("AAPL", lookback_days=7)
        # 仍然只算近期 2 条
        assert out["count"] == 2.0

    def test_multi_ticker_shared_news(self, news_store_with_rows):
        """一条新闻带多个 ticker → 对每个 ticker 都可查到。"""
        shared_time = datetime.now(timezone.utc) - timedelta(hours=1)
        news_store_with_rows.insert_news(
            headline="Both AAPL and MSFT up on AI boom",
            source="reuters",
            timestamp=shared_time.isoformat(),
            tickers=["AAPL", "MSFT"],
            sentiment_score=0.75,
            finbert_negative=0.1,
            finbert_neutral=0.15,
            finbert_confidence=0.75,
        )
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=news_store_with_rows)
        p.initialize()
        aapl = p.get_ticker_sentiment("AAPL", lookback_days=7)
        msft = p.get_ticker_sentiment("MSFT", lookback_days=7)
        # AAPL: 原 2 条 + 新 1 条 = 3
        assert aapl["count"] == 3.0
        # MSFT: 原 1 条 + 新 1 条 = 2
        assert msft["count"] == 2.0
        p.shutdown()

    def test_nan_confidence_skipped(self, news_store_with_rows):
        """finbert_confidence NaN / NULL 行被跳过。"""
        bad_time = datetime.now(timezone.utc) - timedelta(hours=3)
        conn = news_store_with_rows._conn
        news_store_with_rows.insert_news(
            headline="AAPL with null conf",
            source="reuters",
            timestamp=bad_time.isoformat(),
            tickers=["AAPL"],
            sentiment_score=0.5,
            # 不传 finbert_* → 三列都是 NULL
        )
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=news_store_with_rows)
        p.initialize()
        out = p.get_ticker_sentiment("AAPL", lookback_days=7)
        # 依然 2 条 (null conf 被 SQL WHERE IS NOT NULL 过滤)
        assert out["count"] == 2.0
        p.shutdown()

    def test_zero_confidence_rows_return_zeros(self, tmp_path):
        """所有 AAPL 行 conf=0 → Σconf=0 → zeros 完整 dict + count=0 (不 raise)。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        store.insert_news(
            headline="AAPL zero conf",
            source="reuters",
            timestamp=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
            tickers=["AAPL"],
            sentiment_score=0.5,
            finbert_negative=0.3,
            finbert_neutral=0.3,
            finbert_confidence=0.0,
        )
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=store)
        p.initialize()
        out = p.get_ticker_sentiment("AAPL", lookback_days=7)
        assert out == {
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 0.0,
            "confidence": 0.0,
            "count": 0.0,
        }
        p.shutdown()
        store.shutdown()

    def test_invalid_ticker(self, provider):
        with pytest.raises(ValueError):
            provider.get_ticker_sentiment("", lookback_days=7)
        with pytest.raises(ValueError):
            provider.get_ticker_sentiment(None)  # type: ignore[arg-type]

    def test_invalid_lookback(self, provider):
        with pytest.raises(ValueError):
            provider.get_ticker_sentiment("AAPL", lookback_days=-1)

    def test_ticker_case_insensitive(self, provider):
        """lowercase "aapl" 应匹配 uppercase "AAPL" 的新闻 (大写规范)。"""
        assert provider.get_ticker_sentiment("aapl")["count"] == 2.0


# ============================================================================
# backfill
# ============================================================================


class TestBackfill:
    def test_only_writes_null_rows(self, tmp_path):
        """只更新 finbert_confidence IS NULL; 已有值不覆盖。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        # 2 条需回填 + 1 条已有值
        store.insert_news(
            headline="needs backfill 1",
            source="reuters",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
        )
        store.insert_news(
            headline="needs backfill 2",
            source="bloomberg",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
        )
        # 这条预置值
        store.insert_news(
            headline="already scored",
            source="reuters",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            sentiment_score=0.5,
            finbert_negative=0.1,
            finbert_neutral=0.2,
            finbert_confidence=0.7,
        )
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=store)
        p.initialize()

        updated = p.backfill()
        assert updated == 2

        # 已有值未动
        row = store._conn.execute(
            "SELECT finbert_negative, finbert_neutral, finbert_confidence "
            "FROM news_events WHERE headline = 'already scored'"
        ).fetchone()
        assert row == (0.1, 0.2, 0.7)
        p.shutdown()
        store.shutdown()

    def test_does_not_touch_sentiment_score(self, tmp_path):
        """backfill 只写 3 个 finbert_* 列, sentiment_score 不改。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        store.insert_news(
            headline="beats earnings big",
            source="reuters",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            sentiment_score=0.42,  # 预置的 g2 所写 sentiment_score
        )
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=store)
        p.initialize()
        p.backfill()
        row = store._conn.execute(
            "SELECT sentiment_score, finbert_confidence FROM news_events WHERE id=1"
        ).fetchone()
        assert row[0] == 0.42  # 原值保留
        assert row[1] is not None  # backfill 已写
        p.shutdown()
        store.shutdown()

    def test_since_filter(self, tmp_path):
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        now = datetime.now(timezone.utc)
        store.insert_news(
            headline="old news",
            source="reuters",
            timestamp=(now - timedelta(days=100)).isoformat(),
            tickers=["AAPL"],
        )
        store.insert_news(
            headline="recent news",
            source="bloomberg",
            timestamp=now.isoformat(),
            tickers=["AAPL"],
        )
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=store)
        p.initialize()
        since = now - timedelta(days=30)
        updated = p.backfill(since=since)
        assert updated == 1  # 只有 recent
        p.shutdown()
        store.shutdown()

    def test_limit_pagination(self, tmp_path):
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        now = datetime.now(timezone.utc)
        for i in range(5):
            store.insert_news(
                headline=f"news {i}",
                source=f"src{i}",
                timestamp=(now + timedelta(minutes=i)).isoformat(),
                tickers=["AAPL"],
            )
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=store)
        p.initialize()
        u1 = p.backfill(limit=2)
        u2 = p.backfill(limit=2)
        u3 = p.backfill(limit=2)
        assert u1 + u2 + u3 == 5
        assert u1 == 2 and u2 == 2 and u3 == 1
        p.shutdown()
        store.shutdown()

    def test_empty_db(self, tmp_path):
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "empty.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        scorer = FinBERTScorer(device="mock")
        p = LocalSentimentProvider(scorer=scorer, news_store=store)
        p.initialize()
        assert p.backfill() == 0
        p.shutdown()
        store.shutdown()

    def test_invalid_limit(self, provider):
        with pytest.raises(ValueError):
            provider.backfill(limit=0)

    def test_require_news_store(self):
        """没注入 news_store 调用 backfill → raise。"""
        p = LocalSentimentProvider(scorer=FinBERTScorer(device="mock"))
        p.initialize()
        with pytest.raises(RuntimeError, match="news_store"):
            p.backfill()


# ============================================================================
# P3: g2 和 m2b 共享同一 FinBERTScorer singleton
# ============================================================================


class TestP3SharedSingleton:
    def test_g2_and_provider_use_same_scorer(self, news_store_with_rows):
        """g2_classifier + LocalSentimentProvider 都经 get_default_scorer() → 同一实例。"""
        from stockbee.news_data.g2_classifier import G2Classifier, G2Config

        reset_default_scorer()
        # provider 默认注入 singleton
        p = LocalSentimentProvider(news_store=news_store_with_rows)
        p.initialize()
        # g2 也取同一 singleton
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._ensure_scorer()
        assert p._scorer is g2._scorer, (
            "P3 重构后 g2 和 LocalSentimentProvider 必须复用 FinBERTScorer singleton"
        )
        p.shutdown()


# ============================================================================
# sync.py _run_g2 writes finbert_* (CRITICAL review fix)
# ============================================================================


class TestSyncPipelineWritesFinBERTColumns:
    """review CRITICAL: _run_g2 必须把 G2Result.finbert_negative/neutral/confidence
    持久化到 DB,否则 LocalSentimentProvider.get_ticker_sentiment 永远读不到数据。"""

    def test_run_g2_writes_finbert_columns_to_db(self, tmp_path):
        from stockbee.news_data.g2_classifier import G2Classifier, G2Config
        from stockbee.news_data.sync import NewsDataSyncer

        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        # 插 1 条,g_level=1
        news_id = store.insert_news(
            headline="Apple beats earnings by wide margin",
            source="reuters",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            g_level=1,
        )
        assert news_id is not None

        g2 = G2Classifier(G2Config(use_finbert=True))
        # 注入 mock scorer (复用 test_news_data 的辅助不够, 本地简化)
        class _S:
            def score_texts(self, texts, batch_size=32):
                return [
                    {
                        "positive": 0.88, "negative": 0.05,
                        "neutral": 0.07, "confidence": 0.88,
                    }
                    for _ in texts
                ]
        g2._scorer = _S()
        g2._scorer_available = True

        # 构造 syncer 的最小依赖 (sources + g3 可为空 [])
        syncer = NewsDataSyncer(sources=[], store=store, g1=None, g2=g2, g3=None)
        event = {"headline": "Apple beats earnings by wide margin", "snippet": None}
        syncer._run_g2([(event, news_id)])

        row = store._conn.execute(
            "SELECT finbert_negative, finbert_neutral, finbert_confidence "
            "FROM news_events WHERE id = ?",
            (news_id,),
        ).fetchone()
        assert row is not None
        # 三列都非 NULL,和 mock scorer 一致
        assert row[0] == pytest.approx(0.05, abs=1e-4)
        assert row[1] == pytest.approx(0.07, abs=1e-4)
        assert row[2] == pytest.approx(0.88, abs=1e-4)
        store.shutdown()

    def test_run_g2_rule_fallback_does_not_write_finbert(self, tmp_path):
        """use_finbert=False / scorer 不可用 → G2Result.used_finbert=False,
        sync 传 finbert_confidence=None → 列保持 NULL。"""
        from stockbee.news_data.g2_classifier import G2Classifier, G2Config
        from stockbee.news_data.sync import NewsDataSyncer

        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        news_id = store.insert_news(
            headline="Generic headline",
            source="reuters",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            g_level=1,
        )
        g2 = G2Classifier(G2Config(use_finbert=False))
        syncer = NewsDataSyncer(sources=[], store=store, g1=None, g2=g2, g3=None)
        event = {"headline": "Generic headline", "snippet": None}
        syncer._run_g2([(event, news_id)])

        row = store._conn.execute(
            "SELECT finbert_negative, finbert_neutral, finbert_confidence "
            "FROM news_events WHERE id = ?",
            (news_id,),
        ).fetchone()
        assert row == (None, None, None)
        store.shutdown()


# ============================================================================
# register_default
# ============================================================================


class TestRegisterDefault:
    def test_registers_class_and_instance(self, news_store_with_rows):
        reg = ProviderRegistry()
        instance = LocalSentimentProvider.register_default(
            reg, news_store=news_store_with_rows
        )
        assert isinstance(instance, LocalSentimentProvider)
        # 类已注册
        assert "LocalSentimentProvider" in reg._classes
        # 实例按 "SentimentProvider" 抽象名注册
        assert reg._instances.get("SentimentProvider") is instance
        # 实例已初始化
        assert instance.is_initialized
        # review H3: cache 被注入 (与 registry.create() 行为一致)
        assert instance.cache is reg.cache
        instance.shutdown()

    def test_registry_get(self, news_store_with_rows):
        reg = ProviderRegistry()
        LocalSentimentProvider.register_default(reg, news_store=news_store_with_rows)
        fetched = reg.get("SentimentProvider")
        assert isinstance(fetched, LocalSentimentProvider)
        fetched.shutdown()
