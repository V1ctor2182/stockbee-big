# 小模型体系 (02-small-models)

## Intent

**FinBERT 情绪 + Fin-E5 重要性 + LightGBM 收益预测**

所有模型本地部署，GPU 加速推理（CPU fallback 必须可跑通测试）。
FinBERT <100ms / LightGBM <50ms 为 PRD 硬约束。

来源：PRD §3.3, Tech Design §3.3
研究状态：研究完成 (新闻小模型.md, 量化小模型.md)

**Scope 决策（见 Decisions）**：本 room 只做推理 + 预训练集成，不做 fine-tune / 周度 refit / XGBoost 因子融合。
**架构决策（见 Decisions）**：P3 复用 FinBERTScorer singleton（消除 g2 重复推理）；D2 独立 evaluate_ml_score（不破坏 factor-storage 已稳定的 get_ic_report）。

## Constraints

- **FinBERT 情绪准确率 > 80%** — 来源: PRD §3.3（预训练 ProsusAI/finbert 基线 ~87%）
- **LightGBM 预测 IC > 0.03** — 来源: PRD §3.3
- **FinBERT 推理 < 100ms**（batch=32, len=128）— 来源: Tech Design §3.3
- **LightGBM 推理 < 50ms**（1000 rows × 158 cols）— 来源: Tech Design §3.3

## Decisions

| 决策 | 结论 | 日期 |
|------|------|------|
| **A1→P3** FinBERT 部署策略 | g2_classifier 从私有 FinBERT 实例重构为复用 m2a `FinBERTScorer` singleton（消除重复推理）。新列 `finbert_negative` / `finbert_neutral` / `finbert_confidence` 由 g2 写；`sentiment_score` 保留 = FinBERT positive softmax（g2 原语义）| 2026-04-13 |
| **B3** LightGBM label 公式 | `mean(next_5_days_daily_returns)`，基于 `adj_close` 日收益率均值（贴 PRD 原文"未来 5 日平均收益率"）| 2026-04-13 |
| **D2** m4 IC 评估策略 | 新增独立函数 `evaluate_ml_score(shift=5)`，**不修改** `LocalFactorProvider.get_ic_report`（后者 factor-storage 已 freeze，`_IC_SHIFT=1` 硬编码不动）| 2026-04-13 |
| model_io 归属 | m1 交付，供 m2a / m3 / m4 / m5 共用；不放 m2a（否则违反并行开发拓扑）| 2026-04-13 |
| Fin-E5 scope | 降级为 embedding infra + cosine 去重 + rule baseline importance；精准 importance fine-tune 推迟到 `04-alpha-mining` | 2026-04-13 |
| Fin-E5 fallback 模型 | `intfloat/e5-large-v2`（HF 无 canonical `fin-e5`）| 2026-04-13 |
| Fin-E5 embedding dim 参数化 | 从 `scorer.model.config.hidden_size` 动态取（e5-large-v2=1024, base=768），**不硬编码**；测试 fixture 参数化 | 2026-04-13 |
| cosine_dedup 语义 | union-find 传递闭包（A~B, B~C → A/B/C 同 cluster，即使 A-C 直接距离超过 threshold）| 2026-04-13 |
| XGBoost 因子融合 | 推迟到 `04-alpha-mining`，本 room 不做 | 2026-04-13 |
| 周度 refit | 推迟到 Phase 2；本 room 不做 CLI / APScheduler 调度 | 2026-04-13 |
| FinBERT/Fin-E5 fine-tune | 推迟到 Phase 2；本 room 仅加载预训练权重 | 2026-04-13 |
| 模型 artifact 路径 | `data/models/{name}/{YYYYMMDD}.pkl` + `data/models/{name}/current.pkl` (POSIX symlink，Windows fallback = copy) | 2026-04-13 |
| ml_score 落盘 | 单 Parquet 文件 `data/factors/ml_score.parquet`（group 名 = `ml_score`，走 ParquetFactorStore merge-write）| 2026-04-13 |
| ml_score 自动发现 | `LocalFactorProvider._build_precomputed_index` 已泛化读 parquet schema（line 324-358），**无需白名单注册**；m4 非测试代码对 factor-storage 零改动 | 2026-04-13 |
| `list_factors()` 对 `ml_score` 交付 | m4 显式断言 `list_factors()` 返回中包含 `{name: "ml_score", type: "precomputed"}` | 2026-04-13 |
| 预训练模型版本 | FinBERT: `ProsusAI/finbert`；Fin-E5: `intfloat/e5-large-v2` | 2026-04-13 |
| GPU 可选 | `torch.cuda.is_available()` 自动选择；CPU fallback 必须可跑测试（CI 环境无 GPU） | 2026-04-13 |
| 性能基准 tolerance | FinBERT: GPU <100ms / CPU <300ms；LightGBM: GPU <50ms / CPU <150ms | 2026-04-13 |
| 性能测试隔离 | `@pytest.mark.perf` marker，默认 CI skip；本地/夜间手动跑；采样 3 warm-up + 10 中位数 | 2026-04-13 |
| GPU OOM fallback 测试策略 | 仅断言 fallback path invoked；**不验证** CPU 推理等价性（mock `torch.cuda.OutOfMemoryError` 不可信）| 2026-04-13 |
| SentimentProvider 归属 | `LocalSentimentProvider` 放 `small_models/`（与 scorer 共位，不放 `news_data/`）| 2026-04-13 |
| Fin-E5 无 Provider | Fin-E5 不建 Provider 接口，作 `small_models/` 内部模块，下游直接 import | 2026-04-13 |
| `get_ticker_sentiment` 加权公式 | `Σ(conf_i × prob_i) / Σ conf_i`，权重 = `finbert_confidence`（max softmax）；Σconf=0 → 返回 zeros + count=0 | 2026-04-13 |
| `__init__.py` 增量导出约定 | 每个 milestone 向 `src/stockbee/small_models/__init__.py` 增量 re-export 本 milestone 对外符号（参考 factor-storage）| 2026-04-13 |
| Provider Registry 注册位置 | m2b 交付时由 `LocalSentimentProvider.register_default()` 注册到 `PROVIDER_REGISTRY`（触发点在 `__init__.py` 或显式 setup）| 2026-04-13 |
| 测试 mock 策略 | 单元测试 mock HF 模型加载（避免每次 pytest 下 ~500MB 权重）；只在本地手动验收 + m6 F 组性能测试加载真实权重 | 2026-04-13 |
| Milestone 方案 | 7 milestones（m1 含 model_io、m2a/m2b FinBERT 分层、m3/m4 训推分拆、m5 Fin-E5、m6 集成测试）；经两轮 plan-eng subagent review 定稿 | 2026-04-13 |
| **OQ1** `save_pickle` 同版本重复写入 | 默认 raise `FileExistsError`，参数 `overwrite: bool = False` 显式允许覆盖 | 2026-04-13 |
| **OQ2** fixtures 导入方式 | `tests/small_models/conftest.py`（pytest 自动发现子目录 conftest），m2-m6 测试文件落在 `tests/small_models/` 下零样板。progress.yaml 原 `tests/conftest_small_models.py` 同步调整 | 2026-04-13 |
| **OQ3** `update_symlink` Windows fallback | 写 `data/models/{name}/current.txt` 明文记录版本号；`load_pickle("current")` 先查 symlink，若缺再读 `current.txt` → 版本文件。POSIX 下 symlink 为准，切换版本时清理旧 `current.txt` | 2026-04-13 |
| **OQ4** `save_pickle` 是否自动 update_symlink | **否**。训练完显式 `update_symlink(name, version)`，保留"先保存再评审再上线"流程 | 2026-04-13 |

## Contracts

### SentimentProvider 实现 (`providers/interfaces.py:281`)
- `LocalSentimentProvider.score_texts(texts) → list[dict]`
  - 返回 `[{"positive", "negative", "neutral", "confidence"}]`
  - 三项 softmax，sum ≈ 1；confidence = max(三项)
- `LocalSentimentProvider.get_ticker_sentiment(ticker, lookback_days=7) → dict`
  - JOIN `news_events` × `news_tickers`，窗口 = `[now - lookback_days, now]`
  - 权重 = `finbert_confidence`；公式 `Σ(conf × prob) / Σconf`
  - 返回 `{"positive", "negative", "neutral", "confidence", "count"}`
  - 无匹配 / Σconf=0 → 全 0 + count=0（不 raise）
- `LocalSentimentProvider.backfill(since: date | None = None, limit: int = 1000) → int`
  - 只更新 `finbert_confidence IS NULL` 的行
  - 写 `finbert_negative` / `finbert_neutral` / `finbert_confidence`；**不触** `sentiment_score`
- `LocalSentimentProvider.register_default()` → 注册到 `PROVIDER_REGISTRY`

### FactorProvider 扩展：`ml_score` 作 precomputed factor
- 请求：`FactorProvider.get_factors(tickers, ["ml_score"], start, end)` → MultiIndex (date, ticker) × [ml_score]
- 落盘：`data/factors/ml_score.parquet`，主键 `(date, ticker)`，merge-write 幂等
- 混合：`get_factors(tickers, ["MA5", "ml_score", "RESI60"])` 由 `LocalFactorProvider` 路由，expression + precomputed 合并，列序按请求保留
- `list_factors()` 返回中包含 `{"name": "ml_score", "type": "precomputed"}`（m4 断言交付）
- **自动发现**：`LocalFactorProvider._build_precomputed_index` 已泛化，m4 对 factor-storage 代码零改动

### 独立 IC 评估 (D2)
```python
def evaluate_ml_score(
    factor_provider, market_data_provider,
    universe, start, end,
    shift: int = 5, window: int = 252,
) -> dict[str, float]:
    """m4 独立交付，不改 get_ic_report。
    走 ic_evaluator.compute(ml_score, prices, shift=5, window=window)
    返回 {ic_mean, ic_std, icir}
    """
```

### LightGBM Label 公式（B3）
```python
daily_ret = adj_close.groupby(level="ticker").pct_change()
label = (
    daily_ret.groupby(level="ticker")
    .shift(-1)
    .rolling(5).mean()
    .shift(-(5 - 1))
)
# 等价：取 [t+1, t+5] 五天日收益率的均值
# 尾部最后 5 天 NaN，训练时 drop
```

### FinBERT Scorer (m2a)
- `FinBERTScorer(device=None, model_name="ProsusAI/finbert")`
- `FinBERTScorer.score_texts(texts, batch_size=32) → list[dict]`
- max_length=128, truncation=True
- 输出 key：`positive`, `negative`, `neutral`, `confidence`（= max 三项）
- `get_default_scorer()` → singleton，供 g2 + m2b 共享

### Fin-E5 Scorer (m5)
- `FinE5Scorer(device=None, model_name="intfloat/e5-large-v2")`
- `encode(texts, batch_size=16, max_length=512) → np.ndarray (N, hidden_size)`
- `cosine_sim(a, b=None) → np.ndarray`
- `cosine_dedup(embeds, threshold=0.95) → list[int]`（union-find，传递闭包）

### Importance Baseline (m5)
- `baseline_importance(news_df) → Series`，值域 `[0, 1]`
- 公式：`min(count_30d/10, 1.0) × |sentiment_score - 0.5| × 2 × clip(reliability_score, 0, 1)`
- 写列：`fine5_importance`（保留 g2 写的 `importance_score` 并存）

### Model Artifact IO (m1)
- `artifact_path(name, version="current") → Path`
- `save_pickle(obj, name, version=None) → Path`（version 默认 = today）
- `load_pickle(name, version="current") → object`
- `update_symlink(name, version) → None`
- `list_versions(name) → list[str]`（descending date）

### SQLite Schema 扩展（m1 交付）
```sql
-- 老库自动升级：_do_initialize 用 PRAGMA table_info 探测 + 条件 ALTER
ALTER TABLE news_events ADD COLUMN finbert_negative REAL;
ALTER TABLE news_events ADD COLUMN finbert_neutral REAL;
ALTER TABLE news_events ADD COLUMN finbert_confidence REAL;
ALTER TABLE news_events ADD COLUMN fine5_importance REAL;
-- sentiment_score 保留 = FinBERT positive softmax (g2 已写)
-- importance_score 保留 = g2 规则评分（与 fine5_importance 并存，下游自选）
```

### 依赖拓扑
```
m1 (contracts + fixtures + model_io + schema migration)
 ├─→ m2a (FinBERT scorer + <100ms gate) ──→ m2b (Provider + g2 refactor + DB 回写)
 ├─→ m3 (label_utils + LightGBM trainer) ────→ m4 (推理 + ml_score + evaluate_ml_score)
 └─→ m5 (Fin-E5 + baseline importance)
       ↓ (m2b / m4 / m5 全完成后)
      m6 (跨模块集成测试 + 性能回归)
```

m2a / m3 / m5 在 m1 完成后可**并行**开发。

---
_所有 spec 状态: active (room 完成后升级, 2026-04-20)_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
