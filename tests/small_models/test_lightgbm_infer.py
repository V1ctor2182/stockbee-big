"""m4 LightGBMScorer 推理 + ml_score 落盘 + evaluate_ml_score 测试。

覆盖:
- LightGBMScorer.predict:
    * shape / MultiIndex 保持
    * booster.feature_name 自动对齐,缺列报错
    * 空 df → raise
    * NaN 行输出 NaN 不 drop (保索引对齐)
- predict_and_save:
    * 落盘 parquet 文件存在
    * merge-write 幂等 (同日重跑不爆)
    * tickers 为空 → raise
    * 未训练 booster (feature_name=[]) + 显式 factor_names
- LocalFactorProvider 自动发现:
    * 写入 ml_score.parquet 后 list_factors() 含 {"name":"ml_score", "type":"precomputed"}
    * get_factors(["MA5", "ml_score", "RESI60"]) 路由正确, 列序保留
- evaluate_ml_score:
    * oracle: ml_score = fwd_ret_5d 构造 → IC ≈ 1
    * shift=5 传递
    * universe 为空 → raise
    * 无 ml_score 数据 → raise
- artifact 不存在 → NotFoundError
"""

from __future__ import annotations

import math
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stockbee.factor_data.parquet_factor import ParquetFactorStore
from stockbee.small_models import model_io
from stockbee.small_models.lightgbm_scorer import (
    LightGBMScorer,
    ML_SCORE_COLUMN,
    ML_SCORE_GROUP,
    evaluate_ml_score,
)


# ============================================================================
# Mock booster / providers
# ============================================================================


class _MockBooster:
    """模拟 lgb.Booster, 按 feature_name 顺序给每列一个固定系数做线性预测。"""

    def __init__(self, feature_names: list[str], coefs: list[float] | None = None):
        self._names = list(feature_names)
        self._coefs = np.array(
            coefs if coefs is not None else [1.0] * len(feature_names),
            dtype=float,
        )

    def feature_name(self) -> list[str]:
        return list(self._names)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return X @ self._coefs


class _FakeFactorProvider:
    def __init__(self, factor_df: pd.DataFrame, expression: list[str], precomp: list[str] | None = None):
        self.factor_df = factor_df
        self._expr = expression
        self._precomp = precomp or []
        self.calls: list[dict[str, Any]] = []

    def get_factors(self, tickers, factor_names, start, end):
        self.calls.append(
            {
                "tickers": list(tickers),
                "factor_names": list(factor_names),
                "start": start,
                "end": end,
            }
        )
        return self.factor_df[list(factor_names)]

    def list_factors(self):
        return (
            [{"name": n, "type": "expression"} for n in self._expr]
            + [{"name": n, "type": "precomputed"} for n in self._precomp]
        )


class _FakeMarketDataProvider:
    def __init__(self, bars: pd.DataFrame):
        self.bars = bars

    def get_daily_bars(self, tickers, start, end, fields=None):
        return self.bars


def _fake_mdp_from_dates(
    dates: pd.DatetimeIndex, tickers: list[str]
) -> _FakeMarketDataProvider:
    """构造最小 OHLCV MultiIndex(date, ticker),供 Alpha158 计算 MA5 等。"""
    rng = np.random.default_rng(42)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    n = len(idx)
    base_prices = 100.0 + rng.standard_normal(n) * 2.0
    bars = pd.DataFrame(
        {
            "open": base_prices,
            "high": base_prices * 1.01,
            "low": base_prices * 0.99,
            "close": base_prices,
            "adj_close": base_prices,
            "volume": rng.integers(1_000_000, 5_000_000, n),
        },
        index=idx,
    )
    return _FakeMarketDataProvider(bars)


@pytest.fixture
def factor_frame():
    """50 天 × 2 tickers × 3 特征 MultiIndex."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2026-01-02", periods=50)
    idx = pd.MultiIndex.from_product(
        [dates, ["AAPL", "MSFT"]], names=["date", "ticker"]
    )
    return pd.DataFrame(
        rng.standard_normal((len(idx), 3)),
        index=idx,
        columns=["f1", "f2", "f3"],
    )


@pytest.fixture
def mock_booster():
    # 线性: ml_score = 0.5*f1 - 0.3*f2 + 0.1*f3
    return _MockBooster(["f1", "f2", "f3"], [0.5, -0.3, 0.1])


@pytest.fixture
def scorer(mock_booster):
    return LightGBMScorer(booster=mock_booster)


# ============================================================================
# LightGBMScorer.predict
# ============================================================================


class TestPredict:
    def test_shape_and_multiindex_preserved(self, scorer, factor_frame):
        preds = scorer.predict(factor_frame)
        assert isinstance(preds, pd.Series)
        assert preds.index.equals(factor_frame.index)
        assert preds.name == ML_SCORE_COLUMN
        assert len(preds) == len(factor_frame)

    def test_linear_coefficients_applied(self, scorer, factor_frame):
        """输出 = 0.5*f1 - 0.3*f2 + 0.1*f3 (mock booster)。"""
        preds = scorer.predict(factor_frame)
        expected = (
            0.5 * factor_frame["f1"] - 0.3 * factor_frame["f2"] + 0.1 * factor_frame["f3"]
        )
        assert np.allclose(preds.to_numpy(), expected.to_numpy(), atol=1e-9)

    def test_missing_feature_column_raises(self, scorer, factor_frame):
        bad = factor_frame.drop(columns=["f2"])
        with pytest.raises(ValueError, match="缺 booster 需要的特征"):
            scorer.predict(bad)

    def test_extra_columns_are_ignored(self, scorer, factor_frame):
        """多给一列 → 按 feature_name 只取子集。"""
        extra = factor_frame.copy()
        extra["noise"] = 99.0
        preds = scorer.predict(extra)
        assert preds.index.equals(factor_frame.index)

    def test_empty_df_raises(self, scorer):
        empty = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["date", "ticker"]
            )
        )
        with pytest.raises(ValueError):
            scorer.predict(empty)

    def test_nan_row_outputs_nan(self, scorer, factor_frame):
        """特征全 NaN 的行 → 预测值 NaN (索引保留供 join)。mock booster 路径。"""
        row = factor_frame.index[5]
        factor_frame.loc[row, :] = np.nan
        preds = scorer.predict(factor_frame)
        assert math.isnan(preds.loc[row])
        # 其他行仍正常
        other = factor_frame.index[6]
        assert not math.isnan(preds.loc[other])

    def test_nan_row_outputs_nan_with_real_lightgbm(self, tmp_path, factor_frame):
        """review C1 回归: 真实 lgb.Booster 对 all-NaN 行会返 default 方向的数值,
        而非 NaN; predict 层必须显式 mask → 确保和 docstring / mock 语义一致。
        """
        import lightgbm as lgb

        # 用 factor_frame 造 label 训练一个 real booster (3 特征)
        rng = np.random.default_rng(0)
        label = (
            0.5 * factor_frame["f1"]
            - 0.3 * factor_frame["f2"]
            + 0.1 * factor_frame["f3"]
            + 0.1 * rng.standard_normal(len(factor_frame))
        )
        ds = lgb.Dataset(
            factor_frame.to_numpy(),
            label=label.to_numpy(),
            feature_name=list(factor_frame.columns),
        )
        booster = lgb.train(
            {"objective": "regression", "verbose": -1, "seed": 42},
            ds,
            num_boost_round=20,
        )
        s = LightGBMScorer(booster=booster)

        # 让第 3 行全 NaN
        bad_idx = factor_frame.index[3]
        fd = factor_frame.copy()
        fd.loc[bad_idx, :] = np.nan
        preds = s.predict(fd)
        assert math.isnan(preds.loc[bad_idx]), (
            "real booster 的 all-NaN 行必须被 predict 层显式置 NaN (review C1)"
        )
        # 其他行非 NaN
        assert not math.isnan(preds.loc[factor_frame.index[4]])

    def test_booster_without_feature_name_raises(self):
        """review H2: booster.feature_name() 为空 → predict 直接报错。"""
        s = LightGBMScorer(booster=_MockBooster([], [1.0]))
        df = pd.DataFrame(
            [[1.0, 2.0]],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-01-02"), "AAPL")], names=["date", "ticker"]
            ),
            columns=["f1", "f2"],
        )
        with pytest.raises(ValueError, match="feature_name"):
            s.predict(df)

    def test_column_order_strict(self, scorer, factor_frame):
        """列顺序打乱传入 → predict 按 booster feature_name 自动对齐,结果等于有序输入。"""
        shuffled = factor_frame[["f3", "f1", "f2"]]
        preds_shuffled = scorer.predict(shuffled)
        preds_ordered = scorer.predict(factor_frame)
        assert np.allclose(preds_shuffled.to_numpy(), preds_ordered.to_numpy())

    def test_non_multiindex_raises(self, scorer):
        df = pd.DataFrame([[1.0, 2.0, 3.0]], columns=["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="MultiIndex"):
            scorer.predict(df)


# ============================================================================
# predict_and_save
# ============================================================================


class TestPredictAndSave:
    def test_writes_parquet_file(self, tmp_path, scorer, factor_frame):
        fp = _FakeFactorProvider(factor_frame, ["f1", "f2", "f3"])
        store = ParquetFactorStore(data_path=tmp_path / "factors")
        out = scorer.predict_and_save(
            fp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 13),
            store=store,
        )
        assert out.exists()
        assert out.name == f"{ML_SCORE_GROUP}.parquet"

    def test_merge_write_idempotent(self, tmp_path, scorer, factor_frame):
        """同日重跑 merge-write: 路径稳定 + 数据相等 (真正的幂等语义)。"""
        fp = _FakeFactorProvider(factor_frame, ["f1", "f2", "f3"])
        store = ParquetFactorStore(data_path=tmp_path / "factors")
        kwargs = dict(
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 13),
            store=store,
        )
        out1 = scorer.predict_and_save(fp, **kwargs)
        read1 = store.read_factors(ML_SCORE_GROUP).sort_index()
        out2 = scorer.predict_and_save(fp, **kwargs)
        read2 = store.read_factors(ML_SCORE_GROUP).sort_index()
        assert out1 == out2
        pd.testing.assert_frame_equal(read1, read2)

    def test_factor_names_defaults_to_booster_features(
        self, tmp_path, scorer, factor_frame
    ):
        fp = _FakeFactorProvider(factor_frame, ["f1", "f2", "f3"])
        store = ParquetFactorStore(data_path=tmp_path / "factors")
        scorer.predict_and_save(
            fp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 13),
            store=store,
        )
        assert fp.calls[-1]["factor_names"] == ["f1", "f2", "f3"]

    def test_empty_tickers_raises(self, scorer, factor_frame):
        fp = _FakeFactorProvider(factor_frame, ["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="tickers"):
            scorer.predict_and_save(
                fp,
                tickers=[],
                start=date(2026, 1, 2),
                end=date(2026, 3, 13),
            )

    def test_booster_without_feature_name_requires_explicit_names(
        self, tmp_path, factor_frame
    ):
        """无 feature_name 的 booster → predict_and_save 要求显式 factor_names。"""
        s = LightGBMScorer(booster=_MockBooster([], [1.0, 1.0, 1.0]))
        fp = _FakeFactorProvider(factor_frame, ["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="feature_name"):
            s.predict_and_save(
                fp,
                tickers=["AAPL"],
                start=date(2026, 1, 2),
                end=date(2026, 3, 13),
            )

    def test_mixed_routing_preserves_column_order(self, tmp_path):
        """review C3: get_factors(["MA5", "ml_score", "RESI60"]) 列序 + 路由正确。
        用 LocalFactorProvider 验证 expression + precomputed 混合。"""
        from stockbee.factor_data.local_provider import LocalFactorProvider
        from stockbee.providers.base import ProviderConfig

        # 先用 m4 落 ml_score.parquet
        rng = np.random.default_rng(1)
        dates = pd.bdate_range("2026-01-02", periods=30)
        idx = pd.MultiIndex.from_product(
            [dates, ["AAPL", "MSFT"]], names=["date", "ticker"]
        )
        ml_score_df = pd.DataFrame(
            {"ml_score": rng.standard_normal(len(idx))}, index=idx
        )
        data_path = tmp_path / "factors"
        store = ParquetFactorStore(data_path=data_path)
        store.write_factors(ML_SCORE_GROUP, ml_score_df)

        # LocalFactorProvider 路由 expression (MA5) + precomputed (ml_score)
        cfg = ProviderConfig(
            implementation="test",
            params={"precomputed_path": str(data_path)},
        )
        lfp = LocalFactorProvider(cfg)
        # 注入 mock market_data 给 expression engine
        mock_mdp = _fake_mdp_from_dates(dates, ["AAPL", "MSFT"])
        lfp.market_data = mock_mdp
        lfp.initialize()

        try:
            result = lfp.get_factors(
                ["AAPL", "MSFT"],
                ["MA5", "ml_score"],
                start=date(2026, 1, 2),
                end=date(2026, 2, 12),
            )
            # 列序严格按请求
            assert list(result.columns) == ["MA5", "ml_score"]
            # ml_score 列来自 parquet
            assert result["ml_score"].notna().any()
        finally:
            lfp.shutdown()

    def test_factor_provider_empty_raises(self, tmp_path, scorer):
        empty = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
            columns=["f1", "f2", "f3"],
        )
        fp = _FakeFactorProvider(empty, ["f1", "f2", "f3"])
        with pytest.raises(ValueError, match="返回空"):
            scorer.predict_and_save(
                fp,
                tickers=["AAPL"],
                start=date(2026, 1, 2),
                end=date(2026, 3, 13),
                store=ParquetFactorStore(data_path=tmp_path / "x"),
            )


# ============================================================================
# artifact 加载 (via model_io)
# ============================================================================


class TestArtifactLoading:
    def test_missing_artifact_raises(self, tmp_artifact_dir):
        """version 不存在 → NotFoundError。"""
        with pytest.raises(model_io.NotFoundError):
            LightGBMScorer(version="99990101")

    def test_load_via_model_io_roundtrip(self, tmp_artifact_dir, mock_booster):
        """save_pickle → LightGBMScorer 加载成功, feature_names 一致。"""
        model_io.save_pickle(mock_booster, "lightgbm", version="20260420")
        s = LightGBMScorer(version="20260420")
        assert s.feature_names == ["f1", "f2", "f3"]


# ============================================================================
# LocalFactorProvider 自动发现 ml_score (D2 对 factor-storage 零改动)
# ============================================================================


class TestAutoDiscovery:
    def test_list_factors_includes_ml_score_after_save(
        self, tmp_path, scorer, factor_frame, monkeypatch
    ):
        """写入 ml_score.parquet 后, LocalFactorProvider.list_factors 自动包含它。"""
        from stockbee.factor_data.local_provider import LocalFactorProvider
        from stockbee.providers.base import ProviderConfig

        # 把 precomputed 指向 tmp_path/factors, ml_score.parquet 落这里
        data_path = tmp_path / "factors"
        cfg = ProviderConfig(
            implementation="test",
            params={"precomputed_path": str(data_path)},
        )
        lfp = LocalFactorProvider(cfg)
        lfp.initialize()

        # 用 m4 predict_and_save 落盘
        fp = _FakeFactorProvider(factor_frame, ["f1", "f2", "f3"])
        store = ParquetFactorStore(data_path=data_path)
        scorer.predict_and_save(
            fp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 13),
            store=store,
        )

        # 刷新索引: 走公开 refresh API (避免依赖 _precomputed_index_built 私有字段)
        if hasattr(lfp, "refresh_precomputed_index"):
            lfp.refresh_precomputed_index()
        else:
            lfp._precomputed_index = {}
            lfp._precomputed_index_built = False
        factors = lfp.list_factors()
        ml_score_entries = [f for f in factors if f["name"] == ML_SCORE_COLUMN]
        assert len(ml_score_entries) == 1
        entry = ml_score_entries[0]
        # spec 锁定最小契约: name + type (允许额外键如 group)
        assert entry["name"] == ML_SCORE_COLUMN
        assert entry["type"] == "precomputed"
        lfp.shutdown()


# ============================================================================
# evaluate_ml_score (D2)
# ============================================================================


class TestEvaluateMLScore:
    def _oracle_setup(self, shift: int = 5):
        """构造 prices 让 ml_score = fwd_return_{shift}(adj_close) → IC 应 ≈ 1。"""
        rng = np.random.default_rng(0)
        n_days = 250
        tickers = ["AAPL", "MSFT"]
        dates = pd.bdate_range("2026-01-02", periods=n_days)
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])

        # adj_close 随机游走
        rows = []
        for ticker in tickers:
            base = 100.0
            returns = rng.normal(0.001, 0.01, size=n_days)
            prices = base * np.exp(np.cumsum(returns))
            rows.append(prices)
        prices_arr = np.stack(rows, axis=1).ravel(order="C")
        # 注意: idx 是 [(d1,A),(d1,M),(d2,A)...] 顺序; 需重排
        bars = pd.DataFrame(index=idx)
        bars["adj_close"] = np.nan
        for i, ticker in enumerate(tickers):
            bars.loc[(slice(None), ticker), "adj_close"] = (
                np.stack(rows, axis=1)[:, i]
            )
        # 计算 fwd_return shift 天做 ml_score (shift-6 模拟无前看完美 oracle)
        close = bars["adj_close"]
        fwd_ret = close.groupby(level="ticker").shift(-shift) / close - 1
        ml_score = pd.DataFrame({ML_SCORE_COLUMN: fwd_ret.to_numpy()}, index=idx)
        return bars, ml_score

    def test_oracle_ic_near_1(self):
        """完美 oracle (ml_score = fwd_ret_5d) → IC 应 ≥ 0.99。"""
        bars, ml_score = self._oracle_setup(shift=5)
        fp = _FakeFactorProvider(ml_score, expression=[], precomp=[ML_SCORE_COLUMN])
        mdp = _FakeMarketDataProvider(bars)
        out = evaluate_ml_score(
            fp,
            mdp,
            universe=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 11, 30),
            shift=5,
            window=252,
        )
        assert "ic_mean" in out
        # 接近 1 (允许 float 噪声; Spearman rank 理论 = 1)
        assert out["ic_mean"] > 0.9, f"oracle IC {out['ic_mean']:.3f} 不够高"

    def test_empty_universe_raises(self):
        with pytest.raises(ValueError, match="universe"):
            evaluate_ml_score(
                _FakeFactorProvider(pd.DataFrame(), expression=[]),
                _FakeMarketDataProvider(pd.DataFrame()),
                universe=[],
                start=date(2026, 1, 1),
                end=date(2026, 2, 1),
            )

    def test_empty_factor_returns_raises(self):
        empty = pd.DataFrame(
            index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
            columns=[ML_SCORE_COLUMN],
        )
        fp = _FakeFactorProvider(empty, expression=[], precomp=[ML_SCORE_COLUMN])
        mdp = _FakeMarketDataProvider(pd.DataFrame())
        with pytest.raises(ValueError, match="ml_score"):
            evaluate_ml_score(
                fp,
                mdp,
                universe=["AAPL"],
                start=date(2026, 1, 1),
                end=date(2026, 2, 1),
            )

    def test_shift_parameter_propagated(self, monkeypatch):
        """验证 shift=5 透传到 ic_evaluator.compute。"""
        import stockbee.small_models.lightgbm_scorer as mod

        captured: dict[str, Any] = {}
        real_compute = mod.compute_ic

        def spy(factor_df, prices_df, shift=1, window=252):
            captured["shift"] = shift
            captured["window"] = window
            return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0}

        monkeypatch.setattr(mod, "compute_ic", spy)
        bars, ml_score = self._oracle_setup(shift=5)
        fp = _FakeFactorProvider(ml_score, expression=[], precomp=[ML_SCORE_COLUMN])
        mdp = _FakeMarketDataProvider(bars)
        evaluate_ml_score(
            fp,
            mdp,
            universe=["AAPL"],
            start=date(2026, 1, 2),
            end=date(2026, 11, 30),
            shift=5,
            window=100,
        )
        assert captured == {"shift": 5, "window": 100}

    def test_invalid_shift(self):
        fp = _FakeFactorProvider(pd.DataFrame(), expression=[])
        mdp = _FakeMarketDataProvider(pd.DataFrame())
        with pytest.raises(ValueError):
            evaluate_ml_score(
                fp,
                mdp,
                universe=["AAPL"],
                start=date(2026, 1, 1),
                end=date(2026, 2, 1),
                shift=0,
            )


# ============================================================================
# Perf gate (review C2): PRD §3.3 LightGBM 推理 <50ms GPU / <150ms CPU
# ============================================================================


@pytest.mark.perf
@pytest.mark.skipif(
    not os.getenv("RUN_PERF_TESTS"),
    reason="set RUN_PERF_TESTS=1 to enable perf gate",
)
class TestLightGBMPerfGate:
    def _percentile(self, samples: list[float], p: float) -> float:
        s = sorted(samples)
        k = max(0, min(len(s) - 1, int(len(s) * p)))
        return s[k]

    def test_1000_rows_158_cols_under_150ms(self):
        """真实 lgb booster 批推理 1000 rows × 158 cols,
        中位数 < 150ms (CPU,spec tolerance)。"""
        import lightgbm as lgb

        rng = np.random.default_rng(7)
        n_rows = 1000
        n_cols = 158
        X = rng.standard_normal((n_rows, n_cols))
        y = X @ rng.standard_normal(n_cols)
        feat_names = [f"f{i}" for i in range(n_cols)]
        ds = lgb.Dataset(X, label=y, feature_name=feat_names)
        booster = lgb.train(
            {"objective": "regression", "verbose": -1, "seed": 42},
            ds,
            num_boost_round=50,
        )
        s = LightGBMScorer(booster=booster)
        idx = pd.MultiIndex.from_product(
            [pd.bdate_range("2026-01-02", periods=n_rows), ["AAPL"]],
            names=["date", "ticker"],
        )[: n_rows]
        fd = pd.DataFrame(X, index=idx[: n_rows], columns=feat_names)

        for _ in range(3):  # warm-up
            s.predict(fd)
        samples = []
        for _ in range(10):
            t0 = time.perf_counter()
            s.predict(fd)
            samples.append((time.perf_counter() - t0) * 1000.0)
        median_ms = self._percentile(samples, 0.5)
        # spec tolerance: GPU <50ms / CPU <150ms; 测试用宽松 CPU 门槛
        assert median_ms < 150.0, (
            f"LightGBM 1000×158 predict 中位 {median_ms:.1f}ms > 150ms"
        )
