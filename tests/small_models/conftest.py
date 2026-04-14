"""02-small-models 共享 pytest fixtures。

决策 OQ2: 放在 tests/small_models/ 下,pytest 自动发现,m2-m6 测试文件同目录零样板使用。

提供 fixture:
- ohlcv_fixture        : 2 tickers × 60 天 OHLCV (adj_close / open / high / low / close / volume)
- alpha158_fixture     : 5 因子 × 50 天 × 2 tickers MultiIndex(date, ticker)
- news_fixture         : 20 条假新闻 (含 ticker 关联 + timestamp)
- embeddings_fixture   : 假 embedding matrix (N, D), D 参数化默认 1024 (e5-large-v2)
- finbert_golden       : 10 条已知情感 (4 正 / 4 负 / 2 中性)
- tmp_artifact_dir     : 把 small_models.paths.MODEL_ROOT 重定向到 pytest tmp_path,
                         隔离 data/models/ 不污染真实目录
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stockbee.small_models import paths as small_model_paths

_TICKERS = ["AAPL", "MSFT"]


@pytest.fixture
def ohlcv_fixture() -> pd.DataFrame:
    """60 个交易日 × 2 tickers 的 OHLCV。MultiIndex (date, ticker)。

    adj_close 走随机游走 (seed 固定),其他列机械派生;无缺失,便于 label_utils 测试。
    """
    rng = np.random.default_rng(42)
    n_days = 60
    start = pd.Timestamp("2026-01-02")
    dates = pd.bdate_range(start, periods=n_days)
    rows = []
    for ticker in _TICKERS:
        base = 100.0 + (10.0 if ticker == "AAPL" else 0.0)
        returns = rng.normal(0.0005, 0.015, size=n_days)
        prices = base * np.exp(np.cumsum(returns))
        for d, px in zip(dates, prices):
            open_ = px * (1 + rng.normal(0, 0.002))
            high = max(open_, px) * (1 + abs(rng.normal(0, 0.003)))
            low = min(open_, px) * (1 - abs(rng.normal(0, 0.003)))
            close = px
            adj_close = px
            volume = rng.integers(1_000_000, 5_000_000)
            rows.append((d, ticker, open_, high, low, close, adj_close, volume))
    df = pd.DataFrame(
        rows,
        columns=["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"],
    )
    return df.set_index(["date", "ticker"]).sort_index()


@pytest.fixture
def alpha158_fixture() -> pd.DataFrame:
    """5 因子 × 50 天 × 2 tickers。MultiIndex (date, ticker) × 因子名列。"""
    rng = np.random.default_rng(7)
    n_days = 50
    dates = pd.bdate_range("2026-01-02", periods=n_days)
    factor_cols = ["MA5", "MA10", "STD20", "RSQR20", "KMID"]
    idx = pd.MultiIndex.from_product([dates, _TICKERS], names=["date", "ticker"])
    data = rng.standard_normal((len(idx), len(factor_cols)))
    return pd.DataFrame(data, index=idx, columns=factor_cols)


@pytest.fixture
def news_fixture() -> pd.DataFrame:
    """20 条假新闻,覆盖多 ticker + 时间范围。"""
    rows = []
    base_ts = datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
    headlines = [
        ("AAPL beats Q1 earnings by 12%", ["AAPL"]),
        ("Apple cuts iPhone production 10%", ["AAPL"]),
        ("MSFT cloud revenue grows 35%", ["MSFT"]),
        ("Microsoft announces layoffs", ["MSFT"]),
        ("AAPL announces new M4 chip", ["AAPL"]),
        ("Apple services revenue flat", ["AAPL"]),
        ("MSFT wins DoD contract worth $10B", ["MSFT"]),
        ("Microsoft Azure outage affects thousands", ["MSFT"]),
        ("AAPL supplier warns on demand", ["AAPL"]),
        ("Apple hit with EU antitrust fine", ["AAPL"]),
        ("Microsoft acquires AI startup", ["MSFT"]),
        ("MSFT stock hits all-time high", ["MSFT"]),
        ("Tech sector mixed on inflation data", ["AAPL", "MSFT"]),
        ("AAPL pauses Vision Pro production", ["AAPL"]),
        ("Microsoft Copilot adoption accelerates", ["MSFT"]),
        ("Apple beats analyst estimates", ["AAPL"]),
        ("MSFT gaming division slows", ["MSFT"]),
        ("AI boom lifts AAPL and MSFT", ["AAPL", "MSFT"]),
        ("Apple App Store policy under scrutiny", ["AAPL"]),
        ("Microsoft cybersecurity breach disclosed", ["MSFT"]),
    ]
    for i, (headline, tickers) in enumerate(headlines):
        ts = base_ts + timedelta(hours=i * 6)
        rows.append({
            "id": i + 1,
            "timestamp": ts.isoformat(),
            "source": "reuters" if i % 2 == 0 else "bloomberg",
            "headline": headline,
            "tickers": tickers,
            "g_level": 2,
        })
    return pd.DataFrame(rows)


@pytest.fixture(params=[1024, 768])
def embeddings_fixture(request) -> tuple[np.ndarray, int]:
    """假 embedding 矩阵 (N, D),D 参数化 (e5-large-v2=1024, e5-base=768)。

    行范数归一化,便于 cosine 测试。返回 (matrix, dim)。
    """
    dim = request.param
    rng = np.random.default_rng(dim)  # 不同 dim 不同种子便于调试
    n = 12
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms, dim


@pytest.fixture
def finbert_golden() -> list[dict]:
    """10 条金融新闻 + 已知情感 (4 正 / 4 负 / 2 中性),供 m2a 性能/准确性验证。"""
    return [
        {"text": "Company beats earnings expectations, raises full-year guidance", "label": "positive"},
        {"text": "Stock hits new 52-week high on record quarterly profit", "label": "positive"},
        {"text": "Dividend increased 15% as free cash flow grows", "label": "positive"},
        {"text": "Analysts upgrade rating to buy after strong results", "label": "positive"},
        {"text": "Company slashes guidance amid demand collapse", "label": "negative"},
        {"text": "Regulator fines firm $500M for antitrust violations", "label": "negative"},
        {"text": "CFO resigns amid accounting investigation", "label": "negative"},
        {"text": "Layoffs announced, 10% of workforce affected", "label": "negative"},
        {"text": "Company to report earnings next Tuesday", "label": "neutral"},
        {"text": "Board approves routine share buyback program", "label": "neutral"},
    ]


@pytest.fixture
def tmp_artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 small_models.paths.MODEL_ROOT 重定向到 tmp_path/models,隔离真实 data/models。

    monkeypatch 覆盖模块级 MODEL_ROOT 以及 model_io 里引用的 MODEL_ROOT
    (模块在 import 时复制了常量引用,需同步覆盖)。
    """
    from stockbee.small_models import model_io as _model_io

    new_root = tmp_path / "models"
    new_root.mkdir()
    monkeypatch.setattr(small_model_paths, "MODEL_ROOT", new_root)
    monkeypatch.setattr(_model_io, "MODEL_ROOT", new_root)
    return new_root
