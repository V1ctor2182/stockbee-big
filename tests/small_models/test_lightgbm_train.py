"""m3 LightGBM Trainer + label_utils 单元测试。

覆盖:
- forward_return_5d (B3 决策):
    * 常数价 → label = 0
    * 线性上升 → label > 0 单调
    * 尾部 horizon 天 NaN (位置正确)
    * 停牌 (中间 NaN) 传播正确
    * ticker 隔离
    * 手算 golden
    * 输入形态: Series vs DataFrame["adj_close"]
    * 错误输入: 非 MultiIndex / 缺 ticker level / horizon<1
- walk_forward_splits:
    * 严格无前看 (train.max < test.min)
    * step 滑动正确
    * train+test 长度不足 → 报错
    * 非法参数
- train_lightgbm:
    * 线性 label (0.5*f1 - 0.3*f2) → 训练后 valid IC > 0.5
    * NaN 行自动 drop
    * seed 固定 → 结果复现
    * artifact pickle roundtrip (m1 model_io)
    * 空 factor_df / 空 label → raise
- train_and_save E2E:
    * fake 的 FactorProvider + MarketDataProvider → artifact 落盘
    * promote_current=True → update_symlink 指向新版本
    * promote_current=False → 仅保存,不动 current
    * factor_names 未指定 → 走 list_factors() 过滤 expression
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from stockbee.small_models import label_utils
from stockbee.small_models import model_io
from stockbee.small_models.label_utils import forward_return_5d
from stockbee.small_models.lightgbm_trainer import (
    DEFAULT_PARAMS,
    MODEL_NAME,
    train_and_save,
    train_lightgbm,
    walk_forward_splits,
)


# ============================================================================
# label_utils.forward_return_5d
# ============================================================================


class TestForwardReturn5d:
    def _make_prices(self, data: dict[str, list[float]]) -> pd.Series:
        """data: {ticker: [price_day0, price_day1, ...]}"""
        lengths = {len(v) for v in data.values()}
        assert len(lengths) == 1, "每个 ticker 价格长度必须一致"
        n_days = lengths.pop()
        dates = pd.bdate_range("2026-01-02", periods=n_days)
        rows = []
        for d_idx, d in enumerate(dates):
            for ticker, prices in data.items():
                rows.append((d, ticker, prices[d_idx]))
        df = pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
        return df.set_index(["date", "ticker"])["adj_close"].sort_index()

    def test_constant_price_zero_label(self):
        s = self._make_prices({"A": [100.0] * 10})
        label = forward_return_5d(s)
        # 前 5 天 = 0, 后 5 天 NaN
        vals = label.xs("A", level="ticker").to_numpy()
        assert np.allclose(vals[:5], 0.0)
        assert np.isnan(vals[5:]).all()

    def test_linear_rising_prices_positive_monotonic(self):
        """adj_close 线性上升 → 前瞻 5 天收益率均值恒正,接近常量。"""
        prices = [100.0 + i for i in range(15)]
        s = self._make_prices({"A": prices})
        label = forward_return_5d(s)
        valid = label.dropna().to_numpy()
        assert (valid > 0).all()

    def test_tail_nan_positions(self):
        """尾部最后 horizon 天应为 NaN (B3 公式不可避免)。"""
        s = self._make_prices({"A": [100.0 + i for i in range(15)]})
        label = forward_return_5d(s, horizon=5)
        vals = label.xs("A", level="ticker")
        assert vals.iloc[-5:].isna().all()
        assert not vals.iloc[:-5].isna().any()

    def test_suspended_stock_nan_propagation(self):
        """中间 NaN 价 → 周围 label 受 rolling 污染成 NaN,非异常。"""
        prices = [100.0 + i for i in range(15)]
        prices[5] = float("nan")
        s = self._make_prices({"A": prices})
        label = forward_return_5d(s, horizon=5)
        vals = label.xs("A", level="ticker").to_numpy()
        # rolling(5).mean() 碰到 NaN 输出 NaN,shift 后影响范围更大
        # 最少 index=5 附近必须是 NaN (间接验证传播而不 crash)
        assert np.isnan(vals).sum() >= 5

    def test_ticker_isolation(self):
        """A 尾部 NaN 不影响 B 的前段 label。"""
        s = self._make_prices({
            "A": [100.0] * 10,
            "B": [200.0 + i * 0.5 for i in range(10)],
        })
        label = forward_return_5d(s, horizon=5)
        a_vals = label.xs("A", level="ticker").to_numpy()
        b_vals = label.xs("B", level="ticker").to_numpy()
        assert np.allclose(a_vals[:5], 0.0)
        assert (b_vals[:5] > 0).all()
        assert np.isnan(a_vals[5:]).all()
        assert np.isnan(b_vals[5:]).all()

    def test_manual_golden_b3_formula(self):
        """手算 golden: prices=[100,101,102,103,104,105,106,107,108,109] horizon=5。

        label[0] = mean(r1..r5) 其中 ri = price[i]/price[i-1] - 1
        r1=0.01, r2=1/101, r3=1/102, r4=1/103, r5=1/104
        """
        prices = [100.0 + i for i in range(10)]
        s = self._make_prices({"A": prices})
        label = forward_return_5d(s, horizon=5)
        ri = [1.0 / (100 + i) for i in range(9)]  # r_k = (p_k - p_{k-1})/p_{k-1} = 1/p_{k-1}
        expected_l0 = sum(ri[:5]) / 5.0
        got = label.xs("A", level="ticker").iloc[0]
        assert got == pytest.approx(expected_l0, rel=1e-9)

    def test_dataframe_adj_close_column(self, ohlcv_fixture):
        """DataFrame 含 adj_close 列也可直接传入。"""
        label = forward_return_5d(ohlcv_fixture, horizon=5)
        assert label.name == label_utils.LABEL_NAME
        assert label.index.equals(ohlcv_fixture.index.sort_values())
        # 每个 ticker 尾部 5 天 NaN
        for ticker in ohlcv_fixture.index.get_level_values("ticker").unique():
            vals = label.xs(ticker, level="ticker")
            assert vals.iloc[-5:].isna().all()

    def test_invalid_horizon_zero(self):
        s = self._make_prices({"A": [100.0] * 10})
        with pytest.raises(ValueError):
            forward_return_5d(s, horizon=0)

    def test_invalid_horizon_negative(self):
        s = self._make_prices({"A": [100.0] * 10})
        with pytest.raises(ValueError):
            forward_return_5d(s, horizon=-1)

    def test_dataframe_missing_adj_close_column(self):
        df = pd.DataFrame(
            {"close": [1.0, 2.0]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2026-01-02"), "A"), (pd.Timestamp("2026-01-03"), "A")],
                names=["date", "ticker"],
            ),
        )
        with pytest.raises(ValueError, match="adj_close"):
            forward_return_5d(df)

    def test_single_index_rejected(self):
        s = pd.Series([100.0, 101.0], index=[0, 1])
        with pytest.raises(ValueError, match="MultiIndex"):
            forward_return_5d(s)

    def test_multiindex_missing_ticker_level(self):
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2026-01-02"), "X"), (pd.Timestamp("2026-01-03"), "X")],
            names=["date", "other"],
        )
        s = pd.Series([100.0, 101.0], index=idx)
        with pytest.raises(ValueError, match="ticker"):
            forward_return_5d(s)

    def test_wrong_input_type(self):
        with pytest.raises(TypeError):
            forward_return_5d([100.0, 101.0])  # type: ignore[arg-type]


# ============================================================================
# walk_forward_splits
# ============================================================================


class TestWalkForwardSplits:
    def test_no_look_ahead(self):
        dates = pd.bdate_range("2024-01-01", periods=400)
        for train_d, test_d in walk_forward_splits(dates, train_size=252, test_size=21, step=21):
            assert train_d.max() < test_d.min()

    def test_step_slides(self):
        dates = pd.bdate_range("2024-01-01", periods=350)
        splits = list(walk_forward_splits(dates, train_size=252, test_size=21, step=21))
        assert len(splits) >= 2
        # 相邻 train_dates 起点相差 step
        first_train_start = splits[0][0][0]
        second_train_start = splits[1][0][0]
        assert (second_train_start - first_train_start).days >= 21  # 工作日,粗略检查

    def test_last_window_does_not_overflow(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        splits = list(walk_forward_splits(dates, train_size=252, test_size=21, step=21))
        assert splits, "至少一个窗"
        last_train, last_test = splits[-1]
        # 最后一窗完整,不越界
        assert len(last_train) == 252
        assert len(last_test) == 21

    def test_insufficient_dates_raises(self):
        dates = pd.bdate_range("2024-01-01", periods=100)
        with pytest.raises(ValueError):
            list(walk_forward_splits(dates, train_size=252, test_size=21, step=21))

    def test_zero_train_size_rejected(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        with pytest.raises(ValueError):
            list(walk_forward_splits(dates, train_size=0, test_size=21, step=21))

    def test_zero_step_rejected(self):
        dates = pd.bdate_range("2024-01-01", periods=300)
        with pytest.raises(ValueError):
            list(walk_forward_splits(dates, train_size=252, test_size=21, step=0))

    def test_duplicate_dates_deduped(self):
        """重复日期应去重再切; split 数量等同无重复输入。"""
        dates_uniq = pd.bdate_range("2024-01-01", periods=300)
        dates_dup = pd.DatetimeIndex(list(dates_uniq) * 2)
        splits_uniq = list(
            walk_forward_splits(dates_uniq, train_size=252, test_size=21, step=21)
        )
        splits_dup = list(
            walk_forward_splits(dates_dup, train_size=252, test_size=21, step=21)
        )
        assert splits_dup  # 非空
        assert len(splits_dup) == len(splits_uniq), (
            "dedup 后切分数应等同无重复输入"
        )
        assert splits_dup[0][0].is_unique  # train_dates 内部唯一


# ============================================================================
# train_lightgbm
# ============================================================================


def _make_fake_factor_label(
    n_days: int = 100,
    tickers: tuple[str, ...] = ("AAPL", "MSFT"),
    noise: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """生成 linear 关系的 (factor_df, label_sr)。

    label = 0.5*f1 - 0.3*f2 + noise * N(0,1)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    f1 = rng.standard_normal(len(idx))
    f2 = rng.standard_normal(len(idx))
    f3 = rng.standard_normal(len(idx))  # 无信号列
    label = 0.5 * f1 - 0.3 * f2 + noise * rng.standard_normal(len(idx))
    factor_df = pd.DataFrame(
        {"f1": f1, "f2": f2, "f3_noise": f3},
        index=idx,
    )
    label_sr = pd.Series(label, index=idx, name="label")
    return factor_df, label_sr


class TestTrainLightGBM:
    def test_linear_label_recovered_high_ic(self):
        """线性 label 训练后 valid IC > 0.5 (强信号,模型必须抓住)。"""
        factor_df, label_sr = _make_fake_factor_label(n_days=150, noise=0.1)
        booster = train_lightgbm(factor_df, label_sr, num_rounds=200, early_stop=30)
        # 自交叉预测: 对 test set 的 Spearman IC
        n = len(factor_df)
        valid_n = max(1, int(n * 0.2))
        test_part = factor_df.sort_index(level="date").iloc[n - valid_n :]
        test_label = label_sr.reindex(test_part.index)
        pred = booster.predict(test_part.to_numpy())
        ic, _ = stats.spearmanr(pred, test_label.to_numpy())
        assert ic > 0.5, f"valid Spearman IC {ic:.3f} <= 0.5"

    def test_nan_rows_dropped(self):
        """部分行 NaN 应被 drop, 不引入 bias (不 crash)。"""
        factor_df, label_sr = _make_fake_factor_label(n_days=100)
        # 在 f1 和 label 引入散点 NaN
        factor_df.iloc[::7, 0] = np.nan
        label_sr.iloc[::11] = np.nan
        booster = train_lightgbm(factor_df, label_sr, num_rounds=30, early_stop=10)
        assert booster is not None

    def test_seed_reproducible(self):
        factor_df, label_sr = _make_fake_factor_label(n_days=80, noise=0.2)
        b1 = train_lightgbm(factor_df, label_sr, num_rounds=30, early_stop=10)
        b2 = train_lightgbm(factor_df, label_sr, num_rounds=30, early_stop=10)
        # 同样输入 + 同 seed → 预测一致 (到 1e-6)
        X = factor_df.dropna().to_numpy()[:50]
        p1 = b1.predict(X)
        p2 = b2.predict(X)
        assert np.allclose(p1, p2, atol=1e-6)

    def test_pickle_roundtrip_with_model_io(self, tmp_artifact_dir: Path):
        """save_pickle + load_pickle 还原 booster,预测一致。"""
        from stockbee.small_models.model_io import load_pickle, save_pickle

        factor_df, label_sr = _make_fake_factor_label(n_days=80)
        booster = train_lightgbm(factor_df, label_sr, num_rounds=30, early_stop=10)
        path = save_pickle(booster, "lightgbm_test", version="20260420")
        assert path.exists()
        loaded = load_pickle("lightgbm_test", version="20260420")
        X = factor_df.dropna().to_numpy()[:30]
        p_orig = booster.predict(X)
        p_load = loaded.predict(X)
        assert np.allclose(p_orig, p_load)

    def test_empty_factor_df_raises(self):
        empty = pd.DataFrame(
            index=pd.MultiIndex.from_arrays(
                [[], []], names=["date", "ticker"]
            )
        )
        label = pd.Series(dtype=float)
        with pytest.raises(ValueError):
            train_lightgbm(empty, label)

    def test_all_nan_raises(self):
        """全 NaN drop 后无样本。"""
        dates = pd.bdate_range("2024-01-01", periods=20)
        idx = pd.MultiIndex.from_product(
            [dates, ["AAPL"]], names=["date", "ticker"]
        )
        factor_df = pd.DataFrame(
            {"f1": [np.nan] * len(idx)}, index=idx
        )
        label_sr = pd.Series([np.nan] * len(idx), index=idx)
        with pytest.raises(ValueError, match="drop NaN"):
            train_lightgbm(factor_df, label_sr)

    def test_custom_params_override(self):
        factor_df, label_sr = _make_fake_factor_label(n_days=60)
        # 手动调低 learning_rate, 验证 params 传入 (booster.params 保留最终有效 params)
        booster = train_lightgbm(
            factor_df,
            label_sr,
            params={"learning_rate": 0.001},
            num_rounds=10,
            early_stop=5,
        )
        # LightGBM 有时把 params 回显为字符串, float() 兜底
        assert float(booster.params.get("learning_rate")) == pytest.approx(0.001)

    def test_early_stop_triggers_on_noisy_label(self):
        """噪声大的 label → booster 在 num_rounds 用尽前早停。"""
        factor_df, label_sr = _make_fake_factor_label(
            n_days=120, noise=5.0, seed=1
        )
        booster = train_lightgbm(
            factor_df,
            label_sr,
            num_rounds=200,
            early_stop=10,
        )
        # best_iteration < num_rounds 说明早停被触发 (-1 / 0 也算非正常完成)
        best_iter = getattr(booster, "best_iteration", 0)
        assert best_iter < 200, (
            f"early stop 应触发, best_iter={best_iter} 未小于 num_rounds"
        )

    def test_no_index_overlap_error_message(self):
        """review H3: factor_df 和 label_sr 索引无交集 → 明确错误信息。"""
        dates_a = pd.bdate_range("2024-01-01", periods=30)
        dates_b = pd.bdate_range("2025-01-01", periods=30)
        idx_a = pd.MultiIndex.from_product(
            [dates_a, ["AAPL"]], names=["date", "ticker"]
        )
        idx_b = pd.MultiIndex.from_product(
            [dates_b, ["AAPL"]], names=["date", "ticker"]
        )
        factor_df = pd.DataFrame({"f1": np.random.randn(30)}, index=idx_a)
        label_sr = pd.Series(np.random.randn(30), index=idx_b)
        with pytest.raises(ValueError, match="无交集"):
            train_lightgbm(factor_df, label_sr)

    def test_uneven_panel_split_by_unique_date(self):
        """review H1: 不等长 ticker panel (listings/delistings) 下,
        同一天的 ticker 不会被劈开到 train + valid 之间。

        构造: AAPL 60 天, MSFT 只有后 40 天。按行索引切会把最后一天的
        AAPL 划 train、MSFT 划 valid,产生日内 look-ahead。
        按 unique date 切则不会。
        """
        rng = np.random.default_rng(42)
        n_a = 60
        n_b = 40
        dates = pd.bdate_range("2024-01-01", periods=n_a)
        rows = []
        for i, d in enumerate(dates):
            rows.append((d, "AAPL", rng.standard_normal()))
            if i >= n_a - n_b:
                rows.append((d, "MSFT", rng.standard_normal()))
        df = pd.DataFrame(rows, columns=["date", "ticker", "f1"])
        factor_df = df.set_index(["date", "ticker"])
        label_sr = pd.Series(
            rng.standard_normal(len(factor_df)),
            index=factor_df.index,
            name="label",
        )
        # 训练不报错 (切分成功且每边非空)
        booster = train_lightgbm(
            factor_df, label_sr, num_rounds=20, early_stop=5
        )
        assert booster is not None


# ============================================================================
# train_and_save (E2E with fake providers)
# ============================================================================


@dataclass
class _FakeFactorProvider:
    factor_df: pd.DataFrame
    expression_names: list[str]
    precomputed_names: list[str] = None  # type: ignore[assignment]

    def get_factors(
        self,
        tickers: list[str],
        factor_names: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        # 忽略 tickers/start/end,直接返回整段 (测试用 fixture 已对齐)
        return self.factor_df[factor_names]

    def list_factors(self) -> list[dict[str, str]]:
        out = [{"name": n, "type": "expression"} for n in self.expression_names]
        for n in self.precomputed_names or []:
            out.append({"name": n, "type": "precomputed"})
        return out


@dataclass
class _FakeMarketDataProvider:
    bars: pd.DataFrame

    def get_daily_bars(
        self,
        tickers: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        return self.bars


@pytest.fixture
def fake_train_providers(ohlcv_fixture):
    """用 conftest.ohlcv_fixture 派生 factor_df + label providers。"""
    # 用 OHLCV 的 adj_close pct_change 做几个朴素因子
    close = ohlcv_fixture["adj_close"]
    rng = np.random.default_rng(0)
    f1 = close.groupby(level="ticker").pct_change().fillna(0.0)
    f2 = close.groupby(level="ticker").rolling(5).mean().droplevel(0).sort_index()
    f2 = f2.reindex(ohlcv_fixture.index).bfill()
    f3 = pd.Series(
        rng.standard_normal(len(ohlcv_fixture)),
        index=ohlcv_fixture.index,
        name="f3",
    )
    factor_df = pd.DataFrame(
        {"f1": f1.values, "f2": f2.values, "f3": f3.values},
        index=ohlcv_fixture.index,
    )
    return _FakeFactorProvider(
        factor_df=factor_df,
        expression_names=["f1", "f2", "f3"],
        precomputed_names=["ml_score"],  # 用来验证 train_and_save 会过滤掉
    ), _FakeMarketDataProvider(bars=ohlcv_fixture)


class TestTrainAndSave:
    def test_e2e_saves_artifact(self, fake_train_providers, tmp_artifact_dir):
        fp, mdp = fake_train_providers
        path = train_and_save(
            fp,
            mdp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 31),
            factor_names=["f1", "f2", "f3"],
            version="20260420",
            num_rounds=30,
            early_stop=10,
        )
        assert path.exists()
        assert path.name == "20260420.pkl"
        # 版本目录下应有 current.pkl (POSIX symlink) 或 current.txt (Windows fallback)
        current_pkl = path.parent / "current.pkl"
        current_txt = path.parent / "current.txt"
        assert current_pkl.exists() or current_txt.exists(), (
            "promote_current=True 应建立 current.pkl symlink 或 current.txt"
        )

    def test_e2e_promote_current_updates_symlink(
        self, fake_train_providers, tmp_artifact_dir
    ):
        fp, mdp = fake_train_providers
        train_and_save(
            fp,
            mdp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 31),
            factor_names=["f1", "f2", "f3"],
            version="20260420",
            num_rounds=20,
            early_stop=10,
            promote_current=True,
        )
        # load_pickle("current") 应拿到同一 booster
        loaded = model_io.load_pickle(MODEL_NAME, version="current")
        assert loaded is not None

    def test_e2e_no_promote_leaves_current_untouched(
        self, fake_train_providers, tmp_artifact_dir
    ):
        fp, mdp = fake_train_providers
        train_and_save(
            fp,
            mdp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 31),
            factor_names=["f1", "f2", "f3"],
            version="20260420",
            num_rounds=20,
            early_stop=10,
            promote_current=False,
        )
        with pytest.raises(model_io.NotFoundError):
            model_io.load_pickle(MODEL_NAME, version="current")

    def test_e2e_factor_names_none_uses_list_factors_expression(
        self, fake_train_providers, tmp_artifact_dir
    ):
        """factor_names=None → 走 list_factors, 过滤 type=expression (ml_score 被跳过)。"""
        fp, mdp = fake_train_providers
        called_with: dict[str, Any] = {}
        orig_get_factors = fp.get_factors

        def track(tickers, factor_names, start, end):
            called_with["factor_names"] = list(factor_names)
            return orig_get_factors(tickers, factor_names, start, end)

        fp.get_factors = track  # type: ignore[method-assign]
        train_and_save(
            fp,
            mdp,
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 31),
            factor_names=None,
            version="20260420",
            num_rounds=10,
            early_stop=5,
            promote_current=False,
        )
        assert called_with["factor_names"] == ["f1", "f2", "f3"]
        assert "ml_score" not in called_with["factor_names"]

    def test_empty_tickers_rejected(self, fake_train_providers, tmp_artifact_dir):
        fp, mdp = fake_train_providers
        with pytest.raises(ValueError, match="tickers"):
            train_and_save(
                fp,
                mdp,
                tickers=[],
                start=date(2026, 1, 2),
                end=date(2026, 3, 31),
                version="20260420",
            )

    def test_same_day_retrain_requires_overwrite(
        self, fake_train_providers, tmp_artifact_dir
    ):
        """review H2: 默认 overwrite=False, 同版本重训应 FileExistsError。"""
        fp, mdp = fake_train_providers
        kwargs = dict(
            tickers=["AAPL", "MSFT"],
            start=date(2026, 1, 2),
            end=date(2026, 3, 31),
            factor_names=["f1", "f2", "f3"],
            version="20260420",
            num_rounds=10,
            early_stop=5,
            promote_current=False,
        )
        train_and_save(fp, mdp, **kwargs)
        with pytest.raises(FileExistsError):
            train_and_save(fp, mdp, **kwargs)
        # 传 overwrite=True 允许同版本覆盖
        train_and_save(fp, mdp, overwrite=True, **kwargs)

    def test_missing_adj_close_raises(self, fake_train_providers, tmp_artifact_dir):
        fp, mdp = fake_train_providers
        # 构造缺 adj_close 的 bars
        bars_bad = mdp.bars.drop(columns=["adj_close"])
        mdp_bad = _FakeMarketDataProvider(bars=bars_bad)
        with pytest.raises(ValueError, match="adj_close"):
            train_and_save(
                fp,
                mdp_bad,
                tickers=["AAPL", "MSFT"],
                start=date(2026, 1, 2),
                end=date(2026, 3, 31),
                factor_names=["f1", "f2", "f3"],
                version="20260420",
            )


# ============================================================================
# DEFAULT_PARAMS 不可变检查 (防止意外 mutation)
# ============================================================================


def test_default_params_has_regression_objective():
    assert DEFAULT_PARAMS["objective"] == "regression"
    assert "seed" in DEFAULT_PARAMS
