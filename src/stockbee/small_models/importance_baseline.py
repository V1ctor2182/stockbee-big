"""Rule-baseline importance 评分 + SQLite 回写。

决策 (见 spec.md):
- Fin-E5 fine-tune importance 推迟到 04-alpha-mining;本 room 只交付 baseline 公式
- 公式: min(count_30d/10, 1.0) × |sentiment_score - 0.5| × 2 × clip(reliability, 0, 1)
  * count_30d 单调饱和: 10 条以上新闻 per ticker-30d 已足够 → factor=1
  * |sentiment - 0.5| × 2: 情绪越极端越重要 ([0, 1])
  * reliability clip: 来源信任度 ([0, 1])
- 值域: [0, 1]
- 回写目标列: fine5_importance (m1 schema 预留);不动 g2 写的 importance_score

count_30d = 该新闻时间窗内 (t-30d, t) 与本条新闻至少共享一个 ticker 的其它新闻数。
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REQUIRED_COLS = ("count_30d", "sentiment_score", "reliability_score")
_COUNT_SATURATION = 10
DEFAULT_COUNT_WINDOW_DAYS = 30


def baseline_importance(news_df: pd.DataFrame) -> pd.Series:
    """对 news DataFrame 批量计算 baseline importance。

    Args:
        news_df: 含 count_30d / sentiment_score / reliability_score 三列

    Returns:
        pd.Series,值域 [0, 1],索引同 news_df。NaN 输入 → NaN 输出 (不 raise)。
    """
    missing = [c for c in _REQUIRED_COLS if c not in news_df.columns]
    if missing:
        raise ValueError(
            f"news_df 缺列: {missing}; 需要 {list(_REQUIRED_COLS)}"
        )

    count = news_df["count_30d"].astype("float64")
    sentiment = news_df["sentiment_score"].astype("float64")
    reliability = news_df["reliability_score"].astype("float64")

    count_factor = np.minimum(count / _COUNT_SATURATION, 1.0)
    sentiment_factor = np.abs(sentiment - 0.5) * 2.0
    reliability_factor = np.clip(reliability, 0.0, 1.0)
    score = count_factor * sentiment_factor * reliability_factor
    out = pd.Series(score, index=news_df.index, name="fine5_importance")
    out = out.clip(lower=0.0, upper=1.0)  # 防止浮点误差越界
    return out


def backfill_importance(
    news_store: Any,
    since: pd.Timestamp | str | None = None,
    batch: int = 1000,
    window_days: int = DEFAULT_COUNT_WINDOW_DAYS,
) -> int:
    """对 news_events 中 fine5_importance IS NULL 的行批量计算并回写 baseline。

    只更新 NULL 行,不覆盖任何已有非空值 (幂等)。需要 sentiment_score / reliability_score
    存在 (NULL 行跳过);count_30d 由 SQL 按 ticker 联表聚合。

    Args:
        news_store: SqliteNewsProvider 实例,已 initialize (内部 _conn 可用)
        since: 仅处理 timestamp >= since 的行;None = 全量
        batch: 单次 UPDATE 的最大行数
        window_days: count_30d 的窗口长度 (默认 30)

    Returns:
        实际写入的行数 (=被更新的 id 数)。
    """
    if batch < 1:
        raise ValueError(f"batch must be >= 1, got {batch}")
    if window_days < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")

    conn = getattr(news_store, "_conn", None)
    if conn is None:
        raise RuntimeError(
            "news_store._conn is None; 请先 news_store.initialize()"
        )

    # 1. 拉取待 backfill 的 candidate
    where_since = ""
    params: list[Any] = []
    if since is not None:
        where_since = " AND e.timestamp >= ?"
        params.append(_normalize_since(since))
    sql = f"""
        SELECT e.id, e.timestamp, e.sentiment_score, e.reliability_score
        FROM news_events e
        WHERE e.fine5_importance IS NULL
              AND e.sentiment_score IS NOT NULL
              AND e.reliability_score IS NOT NULL
              {where_since}
        ORDER BY e.timestamp
        LIMIT ?
    """
    params.append(batch)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return 0

    # 2. 每行算 count_30d: 与本条共享至少一个 ticker, timestamp ∈ [t-window, t) 的其它 news
    #    使用 correlated subquery,避免一次性拉全库
    ids = [r[0] for r in rows]
    counts: dict[int, int] = {}
    for row_id, ts, _sent, _rel in rows:
        if not ts:
            logger.warning("backfill: row_id=%s 缺 timestamp,count_30d=0", row_id)
            counts[row_id] = 0
            continue
        try:
            t_end = pd.Timestamp(ts)
        except (ValueError, TypeError):
            logger.warning(
                "backfill: row_id=%s timestamp=%r 无法解析,count_30d=0",
                row_id,
                ts,
            )
            counts[row_id] = 0
            continue
        t_start = (t_end - timedelta(days=window_days)).isoformat()
        cnt_sql = """
            SELECT COUNT(DISTINCT e2.id)
            FROM news_events e2
            JOIN news_tickers nt2 ON nt2.news_id = e2.id
            WHERE e2.id != ?
              AND e2.timestamp >= ?
              AND e2.timestamp < ?
              AND nt2.ticker IN (
                  SELECT nt.ticker FROM news_tickers nt WHERE nt.news_id = ?
              )
        """
        cnt = conn.execute(
            cnt_sql, (row_id, t_start, t_end.isoformat(), row_id)
        ).fetchone()
        counts[row_id] = int(cnt[0]) if cnt and cnt[0] is not None else 0

    # 3. 组装 DataFrame → 公式
    df = pd.DataFrame(
        {
            "id": ids,
            "count_30d": [counts.get(i, 0) for i in ids],
            "sentiment_score": [r[2] for r in rows],
            "reliability_score": [r[3] for r in rows],
        }
    )
    df["fine5_importance"] = baseline_importance(df).to_numpy()

    # 4. 回写 (同批次一个 transaction)
    updated = 0
    with news_store._cursor() as cur:
        for _, rr in df.iterrows():
            if pd.isna(rr["fine5_importance"]):
                continue
            cur.execute(
                "UPDATE news_events SET fine5_importance = ? "
                "WHERE id = ? AND fine5_importance IS NULL",
                (float(rr["fine5_importance"]), int(rr["id"])),
            )
            updated += cur.rowcount
    logger.info(
        "baseline_importance backfill: %d candidates, %d updated "
        "(window=%d days)",
        len(rows),
        updated,
        window_days,
    )
    return updated


def _normalize_since(since: Any) -> str:
    """归一化 since → UTC ISO 8601。

    review H1: DB 里 timestamp 统一 +00:00,非 UTC 时区的输入经 tz_convert 转成
    +00:00 才能和 DB 字符串做字典序比较。
    """
    from datetime import timezone

    ts = pd.Timestamp(since)
    if ts.tz is None:
        ts = ts.tz_localize(timezone.utc)
    else:
        ts = ts.tz_convert(timezone.utc)
    return ts.isoformat()


__all__ = [
    "DEFAULT_COUNT_WINDOW_DAYS",
    "backfill_importance",
    "baseline_importance",
]
