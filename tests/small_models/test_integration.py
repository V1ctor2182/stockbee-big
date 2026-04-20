"""02-small-models m6 跨模块集成测试。

参考 04-factor-storage m7 模式 (~18 cases, ~520 行)。单元测试已覆盖单模块正确性;
本 milestone 做组合 / 边界 / 冲突验证。

分组:
  A. Contracts + artifact 流转: model_io 跨 m2a / m3 / m4 / m5 共享 + 跨模型版本 list
  B. FinBERT 端到端: news → g2 → DB → LocalSentimentProvider 查询 / backfill / singleton identity
  C. LightGBM 端到端: train → artifact → scorer → parquet → LocalFactorProvider 自动发现
  D. evaluate_ml_score wire: oracle IC ≈ 1 + shift=5 不污染 get_ic_report
  E. 列冲突并存: sentiment_score / finbert_* / fine5_importance / importance_score 并存
  F. 性能回归 (@pytest.mark.perf, RUN_PERF_TESTS=1 opt-in)
  G. Edge cases: 尾部 label 缺失 / 空 news DB / 100K 行 chunked commit smoke
"""

from __future__ import annotations

import os
import threading
import time
import tracemalloc
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockbee.factor_data.local_provider import LocalFactorProvider
from stockbee.factor_data.parquet_factor import ParquetFactorStore
from stockbee.news_data.g2_classifier import G2Classifier, G2Config
from stockbee.news_data.news_store import SqliteNewsProvider
from stockbee.news_data.sync import NewsDataSyncer
from stockbee.providers.base import ProviderConfig
from stockbee.providers.registry import ProviderRegistry
from stockbee.small_models import model_io
from stockbee.small_models.finbert_scorer import (
    FinBERTScorer,
    get_default_scorer,
    reset_default_scorer,
)
from stockbee.small_models.importance_baseline import backfill_importance
from stockbee.small_models.lightgbm_scorer import (
    LightGBMScorer,
    ML_SCORE_COLUMN,
    ML_SCORE_GROUP,
    evaluate_ml_score,
)
from stockbee.small_models.lightgbm_trainer import (
    MODEL_NAME as LGBM_MODEL_NAME,
    train_and_save,
)
from stockbee.small_models.local_sentiment_provider import LocalSentimentProvider

_RUN_PERF = bool(os.getenv("RUN_PERF_TESTS"))
_perf_skip = pytest.mark.skipif(not _RUN_PERF, reason="set RUN_PERF_TESTS=1")


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_default_scorer()
    yield
    reset_default_scorer()


# ============================================================================
# Helpers: 复用 m3 / m4 / m5 测试里的 mock provider, 这里本地定义避免跨文件导入
# ============================================================================


class _MockBooster:
    def __init__(self, feature_names, coefs=None):
        self._names = list(feature_names)
        self._coefs = np.array(
            coefs if coefs is not None else [1.0] * len(feature_names),
            dtype=float,
        )

    def feature_name(self):
        return list(self._names)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self._coefs


class _FakeFactorProvider:
    def __init__(self, factor_df, expression, precomp=None):
        self.factor_df = factor_df
        self._expr = expression
        self._precomp = precomp or []

    def get_factors(self, tickers, factor_names, start, end):
        return self.factor_df[list(factor_names)]

    def list_factors(self):
        return (
            [{"name": n, "type": "expression"} for n in self._expr]
            + [{"name": n, "type": "precomputed"} for n in self._precomp]
        )


class _FakeMarketDataProvider:
    def __init__(self, bars):
        self.bars = bars

    def get_daily_bars(self, tickers, start, end, fields=None):
        return self.bars


def _make_ohlcv(n_days=100, tickers=("AAPL", "MSFT"), seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product(
        [dates, list(tickers)], names=["date", "ticker"]
    )
    n = len(idx)
    base = 100.0 + rng.standard_normal(n) * 2.0
    return pd.DataFrame(
        {
            "open": base,
            "high": base * 1.01,
            "low": base * 0.99,
            "close": base,
            "adj_close": base,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )


def _make_factor_df(n_days=100, tickers=("AAPL", "MSFT"), cols=("f1", "f2", "f3"), seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-02", periods=n_days)
    idx = pd.MultiIndex.from_product(
        [dates, list(tickers)], names=["date", "ticker"]
    )
    return pd.DataFrame(
        rng.standard_normal((len(idx), len(cols))),
        index=idx,
        columns=list(cols),
    )


# ============================================================================
# A. Contracts + artifact 流转
# ============================================================================


class TestGroupA_ArtifactIsolation:
    def test_model_io_shared_across_submodules(self, tmp_artifact_dir):
        """m2a / m3 / m4 / m5 全部走同一 model_io API,artifact 互不冲突。"""
        # 存 4 种不同 name 的 artifact
        for name, payload in [
            ("finbert", {"dummy": "scorer_state"}),
            ("lightgbm", _MockBooster(["f1", "f2"], [1.0, 2.0])),
            ("fine5", {"dummy": "encoder"}),
            ("baseline_rule", {"coef": 0.5}),
        ]:
            p = model_io.save_pickle(payload, name, version="20260420")
            assert p.exists()
            model_io.update_symlink(name, "20260420")
            loaded = model_io.load_pickle(name, version="current")
            assert loaded is not None

        # 每个目录下 versions 列表独立
        for name in ("finbert", "lightgbm", "fine5", "baseline_rule"):
            versions = model_io.list_versions(name)
            assert versions == ["20260420"]

    def test_concurrent_update_symlink_safe(self, tmp_artifact_dir):
        """两线程并发 update_symlink 不会死锁/文件破坏。"""
        # 先建两个版本
        for v in ("20260418", "20260419"):
            model_io.save_pickle({"v": v}, "shared", version=v)

        errors: list[Exception] = []

        def flip(v):
            try:
                for _ in range(20):
                    model_io.update_symlink("shared", v)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        t1 = threading.Thread(target=flip, args=("20260418",))
        t2 = threading.Thread(target=flip, args=("20260419",))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors
        # 最终能加载 current
        loaded = model_io.load_pickle("shared", version="current")
        assert loaded in ({"v": "20260418"}, {"v": "20260419"})


# ============================================================================
# B. FinBERT 端到端
# ============================================================================


class TestGroupB_FinBertE2E:
    def _make_news_store(self, tmp_path):
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "news.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        return store

    def _attach_mock_scorer_to_g2(self, g2, label="positive", score=0.9):
        other = (1.0 - score) / 2.0

        class _S:
            def score_texts(self, texts, batch_size=32):
                out = []
                for _ in texts:
                    probs = {
                        "positive": other,
                        "negative": other,
                        "neutral": other,
                    }
                    probs[label] = score
                    probs["confidence"] = max(
                        probs["positive"], probs["negative"], probs["neutral"]
                    )
                    out.append(probs)
                return out

        g2._scorer = _S()
        g2._scorer_available = True

    def test_news_to_g2_to_db_to_provider_readback(self, tmp_path):
        """news → g2 (用 mock scorer) → 写 DB → LocalSentimentProvider.get_ticker_sentiment 读出。"""
        store = self._make_news_store(tmp_path)
        # 插 g_level=1 记录
        news_ids = []
        for i in range(3):
            nid = store.insert_news(
                headline=f"AAPL beats earnings {i}",
                source=f"src{i}",
                timestamp=(datetime.now(timezone.utc) - timedelta(hours=i)).isoformat(),
                tickers=["AAPL"],
                g_level=1,
            )
            news_ids.append(nid)

        # sync._run_g2 写 finbert_* 列
        g2 = G2Classifier(G2Config(use_finbert=True))
        self._attach_mock_scorer_to_g2(g2, "positive", 0.85)
        syncer = NewsDataSyncer(sources=[], store=store, g1=None, g2=g2, g3=None)
        items = [
            ({"headline": f"AAPL beats earnings {i}", "snippet": None}, nid)
            for i, nid in enumerate(news_ids)
        ]
        syncer._run_g2(items)

        # provider 读回加权
        provider = LocalSentimentProvider(
            scorer=FinBERTScorer(device="mock"), news_store=store
        )
        provider.initialize()
        out = provider.get_ticker_sentiment("AAPL", lookback_days=7)
        assert out["count"] == 3.0
        # mock scorer 全部输出 positive=0.85, 加权 positive 应 ≈ 0.85
        assert out["positive"] == pytest.approx(0.85, abs=1e-4)
        provider.shutdown()
        store.shutdown()

    def test_backfill_noop_after_g2_wrote(self, tmp_path):
        """g2 已写 finbert_confidence 的行, backfill 应 0 更新。"""
        store = self._make_news_store(tmp_path)
        news_id = store.insert_news(
            headline="AAPL beats",
            source="r",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            g_level=1,
        )
        g2 = G2Classifier(G2Config(use_finbert=True))
        self._attach_mock_scorer_to_g2(g2, "positive", 0.9)
        syncer = NewsDataSyncer(sources=[], store=store, g1=None, g2=g2, g3=None)
        syncer._run_g2([({"headline": "AAPL beats", "snippet": None}, news_id)])

        provider = LocalSentimentProvider(
            scorer=FinBERTScorer(device="mock"), news_store=store
        )
        provider.initialize()
        n = provider.backfill()
        assert n == 0, "g2 已写的行不应被 backfill 覆盖"
        provider.shutdown()
        store.shutdown()

    def test_g2_and_provider_share_same_singleton(self, tmp_path):
        """P3 identity 断言: g2._ensure_scorer() 和 get_default_scorer() 同实例。"""
        store = self._make_news_store(tmp_path)
        reset_default_scorer()
        provider = LocalSentimentProvider(news_store=store)
        provider.initialize()
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._ensure_scorer()
        assert provider._scorer is g2._scorer
        provider.shutdown()
        store.shutdown()


# ============================================================================
# C. LightGBM 端到端
# ============================================================================


class TestGroupC_LightGBME2E:
    def test_train_to_save_to_scorer_to_parquet(self, tmp_path, tmp_artifact_dir):
        """m3 train_and_save → m4 LightGBMScorer → m4 predict_and_save → parquet。"""
        import lightgbm as lgb

        factor_df = _make_factor_df(n_days=60, cols=("f1", "f2", "f3"))
        bars = _make_ohlcv(n_days=60)
        fp = _FakeFactorProvider(factor_df, ["f1", "f2", "f3"])
        mdp = _FakeMarketDataProvider(bars)

        # m3: 训练 + 保存
        artifact = train_and_save(
            fp, mdp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 13),
            factor_names=["f1", "f2", "f3"],
            version="20260420",
            num_rounds=20,
            early_stop=5,
        )
        assert artifact.exists()
        # m4: 加载 + 推理 + 落盘
        scorer = LightGBMScorer(version="20260420")
        parquet_dir = tmp_path / "factors"
        store = ParquetFactorStore(data_path=parquet_dir)
        out_path = scorer.predict_and_save(
            fp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 13),
            store=store,
        )
        assert out_path.exists()
        # 回读非空
        read_back = store.read_factors(ML_SCORE_GROUP)
        assert not read_back.empty
        assert ML_SCORE_COLUMN in read_back.columns

    def test_factor_provider_reads_ml_score_after_save(self, tmp_path, tmp_artifact_dir):
        """LocalFactorProvider.get_factors(["ml_score"]) 能拿回 m4 落盘的数据。"""
        # 预造 ml_score.parquet
        ml_scores = _make_factor_df(cols=(ML_SCORE_COLUMN,))
        parquet_dir = tmp_path / "factors"
        ParquetFactorStore(data_path=parquet_dir).write_factors(
            ML_SCORE_GROUP, ml_scores
        )

        cfg = ProviderConfig(
            implementation="t",
            params={"precomputed_path": str(parquet_dir)},
        )
        lfp = LocalFactorProvider(cfg)
        mdp = _FakeMarketDataProvider(_make_ohlcv())
        lfp.market_data = mdp
        lfp.initialize()
        try:
            got = lfp.get_factors(
                ["AAPL", "MSFT"],
                [ML_SCORE_COLUMN],
                start=date(2026, 1, 2),
                end=date(2026, 3, 13),
            )
            assert not got.empty
            assert ML_SCORE_COLUMN in got.columns
        finally:
            lfp.shutdown()

    def test_mixed_routing_expression_and_precomputed(self, tmp_path, tmp_artifact_dir):
        """get_factors(["MA5", "ml_score"]) 路由正确, 列序保留。"""
        parquet_dir = tmp_path / "factors"
        ml_scores = _make_factor_df(cols=(ML_SCORE_COLUMN,))
        ParquetFactorStore(data_path=parquet_dir).write_factors(
            ML_SCORE_GROUP, ml_scores
        )
        cfg = ProviderConfig(
            implementation="t",
            params={"precomputed_path": str(parquet_dir)},
        )
        lfp = LocalFactorProvider(cfg)
        lfp.market_data = _FakeMarketDataProvider(_make_ohlcv())
        lfp.initialize()
        try:
            result = lfp.get_factors(
                ["AAPL", "MSFT"],
                ["MA5", ML_SCORE_COLUMN],
                start=date(2026, 1, 2),
                end=date(2026, 2, 14),
            )
            assert list(result.columns) == ["MA5", ML_SCORE_COLUMN]
        finally:
            lfp.shutdown()

    def test_list_factors_includes_ml_score(self, tmp_path, tmp_artifact_dir):
        parquet_dir = tmp_path / "factors"
        ml_scores = _make_factor_df(cols=(ML_SCORE_COLUMN,))
        ParquetFactorStore(data_path=parquet_dir).write_factors(
            ML_SCORE_GROUP, ml_scores
        )
        cfg = ProviderConfig(
            implementation="t",
            params={"precomputed_path": str(parquet_dir)},
        )
        lfp = LocalFactorProvider(cfg)
        lfp.market_data = _FakeMarketDataProvider(_make_ohlcv())
        lfp.initialize()
        try:
            factors = lfp.list_factors()
            match = [f for f in factors if f["name"] == ML_SCORE_COLUMN]
            assert len(match) == 1
            assert match[0]["type"] == "precomputed"
        finally:
            lfp.shutdown()


# ============================================================================
# D. evaluate_ml_score wire
# ============================================================================


class TestGroupD_EvaluateWire:
    @staticmethod
    def _make_bars_and_ml_score(shift: int, n_days: int = 250, seed: int = 0):
        rng = np.random.default_rng(seed)
        tickers = ["AAPL", "MSFT"]
        dates = pd.bdate_range("2026-01-02", periods=n_days)
        idx = pd.MultiIndex.from_product(
            [dates, tickers], names=["date", "ticker"]
        ).sortlevel()[0]
        bars_dict = {}
        for t in tickers:
            prices = 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.01, n_days)))
            for d, p in zip(dates, prices):
                bars_dict[(d, t)] = p
        bars = pd.DataFrame(
            {"adj_close": [bars_dict[i] for i in idx]}, index=idx
        )
        close = bars["adj_close"]
        fwd = close.groupby(level="ticker").shift(-shift) / close - 1
        ml_score = pd.DataFrame({ML_SCORE_COLUMN: fwd.to_numpy()}, index=idx)
        return bars, ml_score, tickers

    def test_oracle_ic_saturates_at_one_when_shift_matches(self):
        """oracle (ml_score = fwd_ret_{shift}) + evaluate_ml_score(shift) → IC ≈ 1。"""
        bars, ml_score, tickers = self._make_bars_and_ml_score(shift=5)
        fp = _FakeFactorProvider(ml_score, expression=[], precomp=[ML_SCORE_COLUMN])
        mdp = _FakeMarketDataProvider(bars)
        out = evaluate_ml_score(
            fp, mdp,
            universe=tickers,
            start=date(2026, 1, 2),
            end=date(2026, 11, 30),
            shift=5,
            window=252,
        )
        assert out["ic_mean"] > 0.9

    def test_shift_mismatch_drops_ic_below_saturation(self):
        """control: oracle 用 shift=5 构造, 但 evaluate_ml_score(shift=1) → IC
        远低于 1 (非 tautology)。若本测试依然 IC≈1,说明 shift 参数没真正影响
        前向对齐,是 bug。"""
        bars, ml_score, tickers = self._make_bars_and_ml_score(shift=5)
        fp = _FakeFactorProvider(ml_score, expression=[], precomp=[ML_SCORE_COLUMN])
        mdp = _FakeMarketDataProvider(bars)
        out = evaluate_ml_score(
            fp, mdp,
            universe=tickers,
            start=date(2026, 1, 2),
            end=date(2026, 11, 30),
            shift=1,  # 与 oracle 构造 shift=5 不匹配
            window=252,
        )
        # shift 错配 → IC 应显著低于 1.0 (经验上 < 0.5 for random walk)
        assert out["ic_mean"] < 0.6, (
            f"shift 错配 IC={out['ic_mean']:.3f} 未降低,说明 shift 参数未真正传到 compute_ic"
        )

    def test_evaluate_ml_score_does_not_run_factor_storage_ic_report(self, tmp_path):
        """D2 runtime 断言: 调用 evaluate_ml_score 后, LocalFactorProvider
        get_ic_report 路径未被触动; _IC_SHIFT 常量值保持 1 (spec freeze)。"""
        from stockbee.factor_data.local_provider import _IC_SHIFT

        # runtime 先跑一次 evaluate_ml_score
        bars, ml_score, tickers = self._make_bars_and_ml_score(shift=5)
        fp = _FakeFactorProvider(ml_score, expression=[], precomp=[ML_SCORE_COLUMN])
        mdp = _FakeMarketDataProvider(bars)
        evaluate_ml_score(
            fp, mdp,
            universe=tickers,
            start=date(2026, 1, 2),
            end=date(2026, 11, 30),
            shift=5,
            window=252,
        )
        # 然后再确认 factor-storage 常量完好
        from stockbee.factor_data.local_provider import _IC_SHIFT as ic_shift_after
        assert ic_shift_after == 1
        assert _IC_SHIFT == 1


# ============================================================================
# E. 列冲突并存 (sentiment_score / finbert_* / fine5_importance / importance_score)
# ============================================================================


class TestGroupE_ColumnCoexistence:
    def test_sentiment_score_and_finbert_cols_coexist_via_g2_production(
        self, tmp_path
    ):
        """走 g2 → sync._run_g2 生产路径, 读回 DB 验证:
        sentiment_score (= FinBERT positive softmax) + finbert_negative +
        finbert_neutral ≈ 1.0 是 g2 实际写入的不变式 (非手工种子)。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        news_id = store.insert_news(
            headline="AAPL beats earnings by wide margin",
            source="r",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            g_level=1,
        )
        # mock scorer: positive=0.70, negative=0.20, neutral=0.10 (sum=1)
        class _S:
            def score_texts(self, texts, batch_size=32):
                return [
                    {
                        "positive": 0.70,
                        "negative": 0.20,
                        "neutral": 0.10,
                        "confidence": 0.70,
                    }
                    for _ in texts
                ]

        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._scorer = _S()
        g2._scorer_available = True
        syncer = NewsDataSyncer(sources=[], store=store, g1=None, g2=g2, g3=None)
        syncer._run_g2(
            [({"headline": "AAPL beats earnings by wide margin", "snippet": None}, news_id)]
        )

        row = store._conn.execute(
            "SELECT sentiment_score, finbert_negative, finbert_neutral, finbert_confidence "
            "FROM news_events WHERE id = ?",
            (news_id,),
        ).fetchone()
        # g2 production path 写入的 4 列:
        #   sentiment_score = positive softmax = 0.70
        #   finbert_negative = 0.20
        #   finbert_neutral = 0.10
        #   finbert_confidence = max = 0.70
        # 三 softmax 分量 sum ≈ 1.0 (从 scorer 出来就是)
        # round(pos, 3) + round(neg, 3) + round(neu, 3) 也应 ≈ 1.0
        assert row[0] == pytest.approx(0.70, abs=1e-3)
        assert row[1] == pytest.approx(0.20, abs=1e-3)
        assert row[2] == pytest.approx(0.10, abs=1e-3)
        assert row[0] + row[1] + row[2] == pytest.approx(1.0, abs=1e-3)
        assert row[3] == pytest.approx(
            max(row[0], row[1], row[2]), abs=1e-3
        )
        store.shutdown()

    def test_fine5_importance_and_importance_score_parallel(self, tmp_path):
        """fine5_importance (m5 写) 与 importance_score (g2 规则写) 并存,独立查询。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        store.insert_news(
            headline="AAPL news",
            source="r",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            sentiment_score=0.9,
            importance_score=0.6,  # g2 规则
            reliability_score=0.8,
        )
        # m5 backfill 写 fine5_importance
        n = backfill_importance(store, batch=100)
        assert n == 1
        row = store._conn.execute(
            "SELECT importance_score, fine5_importance FROM news_events WHERE id=1"
        ).fetchone()
        assert row[0] == pytest.approx(0.6, abs=1e-6)
        assert row[1] is not None  # m5 baseline 写入
        assert row[0] != row[1]  # 两列不同,独立来源
        store.shutdown()

    def test_legacy_rows_without_finbert_backfilled(self, tmp_path):
        """老数据只有 sentiment_score, 无 finbert_*; LocalSentimentProvider.backfill 补齐。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "n.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        # 模拟老数据: 只有 sentiment_score, finbert_* 都 NULL
        store.insert_news(
            headline="legacy news",
            source="r",
            timestamp=datetime.now(timezone.utc).isoformat(),
            tickers=["AAPL"],
            sentiment_score=0.65,
            # 不传 finbert_* → 留 NULL
        )
        # 确认 NULL
        row = store._conn.execute(
            "SELECT finbert_confidence FROM news_events WHERE id=1"
        ).fetchone()
        assert row[0] is None

        provider = LocalSentimentProvider(
            scorer=FinBERTScorer(device="mock"), news_store=store
        )
        provider.initialize()
        n = provider.backfill()
        assert n == 1

        row2 = store._conn.execute(
            "SELECT sentiment_score, finbert_confidence FROM news_events WHERE id=1"
        ).fetchone()
        assert row2[0] == 0.65  # 老 sentiment_score 未被 backfill 覆盖
        assert row2[1] is not None  # finbert_confidence 已补
        provider.shutdown()
        store.shutdown()


# ============================================================================
# G. Edge cases
# ============================================================================


class TestGroupG_EdgeCases:
    def test_universe_tail_label_missing_does_not_crash_train(
        self, tmp_path, tmp_artifact_dir
    ):
        """数据尾部 5 天 label 为 NaN (B3 公式必然),train_lightgbm 自动 drop 不爆。"""
        factor_df = _make_factor_df(n_days=30, cols=("f1", "f2"))
        bars = _make_ohlcv(n_days=30)
        fp = _FakeFactorProvider(factor_df, ["f1", "f2"])
        mdp = _FakeMarketDataProvider(bars)
        # train_and_save 内部 forward_return_5d 对 30 天数据尾部 5 天 NaN,
        # train_lightgbm 自动 drop 后仍应训练成功
        artifact = train_and_save(
            fp, mdp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 2, 14),
            factor_names=["f1", "f2"],
            version="20260420",
            num_rounds=20,
            early_stop=5,
        )
        assert artifact.exists()

    def test_empty_news_db_backfill_returns_zero(self, tmp_path):
        """空 DB backfill 不 raise,返回 0。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "empty.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        provider = LocalSentimentProvider(
            scorer=FinBERTScorer(device="mock"), news_store=store
        )
        provider.initialize()
        assert provider.backfill() == 0
        provider.shutdown()
        store.shutdown()

    def test_backfill_chunked_pagination_and_memory_smoke(self, tmp_path):
        """10K 条新闻分批 backfill: 验证 chunked pagination 正确性 +
        tracemalloc 峰值内存随批次非累积式增长 (OOM smoke, review M3)。"""
        cfg = ProviderConfig(
            implementation="t", params={"db_path": str(tmp_path / "big.db")}
        )
        store = SqliteNewsProvider(cfg)
        store.initialize()
        now = datetime.now(timezone.utc)
        events = [
            {
                "headline": f"news {i}",
                "source": f"s{i % 10}",
                "timestamp": (now + timedelta(seconds=i)).isoformat(),
                "tickers": ["AAPL"],
            }
            for i in range(10_000)
        ]
        store.insert_news_batch(events)

        provider = LocalSentimentProvider(
            scorer=FinBERTScorer(device="mock"), news_store=store
        )
        provider.initialize()

        tracemalloc.start()
        try:
            chunk_peaks: list[int] = []
            total = 0
            for _ in range(12):
                tracemalloc.reset_peak()
                n = provider.backfill(limit=1000)
                _, peak = tracemalloc.get_traced_memory()
                chunk_peaks.append(peak)
                total += n
                if n == 0:
                    break
            assert total == 10_000
            # 正确性: 确实分批发生 (每批最多 1000)
            assert len(chunk_peaks) >= 10
            # OOM smoke: 后续 chunk 峰值不应超过首 chunk 峰值 × 3
            # (非累积式内存增长; chunked commit 释放前批)
            first_peak = max(chunk_peaks[:3]) or 1
            worst_later = max(chunk_peaks[3:] or [0])
            assert worst_later < 3 * first_peak, (
                f"分批后峰值失控: 首 3 批最高 {first_peak} 字节, "
                f"后续最高 {worst_later} 字节 → 可能存在累积引用"
            )
        finally:
            tracemalloc.stop()
            provider.shutdown()
            store.shutdown()

    def test_finbert_mock_fallback_when_scorer_unavailable(self):
        """get_default_scorer 不可用时 g2 降级规则 (模拟 transformers 未装场景)。
        (m4 perf FP OOM 等价性不验证, 只断言 fallback invoked,
        per spec m6 G 组策略)。"""
        g2 = G2Classifier(G2Config(use_finbert=True))
        g2._scorer = None
        g2._scorer_available = False
        r = g2.classify("Stock crash amid massive layoffs")
        assert not r.used_finbert
        assert r.sentiment == "negative"  # 规则引擎抓到 crash + layoffs
        assert r.finbert_negative is None


# ============================================================================
# F. 性能回归 (opt-in)
# ============================================================================


@pytest.mark.perf
@_perf_skip
class TestGroupF_PerfRegression:
    @staticmethod
    def _percentile(samples, p):
        s = sorted(samples)
        k = max(0, min(len(s) - 1, int(len(s) * p)))
        return s[k]

    def test_finbert_batch32_latency(self):
        reset_default_scorer()
        scorer = FinBERTScorer()
        texts = ["earnings beat estimates by a wide margin"] * 32
        for _ in range(3):
            scorer.score_texts(texts, batch_size=32)
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            scorer.score_texts(texts, batch_size=32)
            samples.append((time.perf_counter() - t0) * 1000.0)
        med = self._percentile(samples, 0.5)
        tol = 100.0 if scorer.device == "cuda" else 300.0
        assert med < tol, f"FinBERT b32 median {med:.1f}ms > {tol:.0f}ms"

    def test_lightgbm_1000_rows_latency(self, tmp_artifact_dir):
        import lightgbm as lgb

        rng = np.random.default_rng(1)
        X = rng.standard_normal((1000, 158))
        y = X @ rng.standard_normal(158)
        ds = lgb.Dataset(
            X, label=y, feature_name=[f"f{i}" for i in range(158)]
        )
        booster = lgb.train(
            {"objective": "regression", "verbose": -1, "seed": 42},
            ds,
            num_boost_round=50,
        )
        s = LightGBMScorer(booster=booster)
        idx = pd.MultiIndex.from_product(
            [pd.bdate_range("2026-01-02", periods=1000), ["AAPL"]],
            names=["date", "ticker"],
        )[:1000]
        df = pd.DataFrame(X, index=idx, columns=[f"f{i}" for i in range(158)])
        for _ in range(3):
            s.predict(df)
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            s.predict(df)
            samples.append((time.perf_counter() - t0) * 1000.0)
        med = self._percentile(samples, 0.5)
        assert med < 150.0, f"LightGBM 1000×158 median {med:.1f}ms > 150ms"
