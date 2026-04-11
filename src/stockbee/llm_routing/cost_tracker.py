"""CostTracker — LLM 调用成本追踪 + 响应缓存。

按任务类型累计 token 用量和成本。预算检查同时支持任务级和总预算。
响应缓存带 TTL，key 包含 (task_type, prompt, system_prompt, output_schema, model)
以防止 prompt 模板迭代后命中旧答案。
SQLite 存储，线程安全（check_same_thread=False + lock）。

来源：Tech Design §3.2
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .task_config import DEFAULT_TASK_CONFIGS, TOTAL_MONTHLY_BUDGET, TaskConfig, TaskType

logger = logging.getLogger(__name__)

# Default cache TTL. 24h is long enough that intra-day retries stay free
# but short enough that stale market-moving news won't leak across days.
DEFAULT_CACHE_TTL_HOURS = 24

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type       TEXT NOT NULL,
    model           TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    input_tokens    INTEGER DEFAULT 0,
    output_tokens   INTEGER DEFAULT 0,
    cost            REAL DEFAULT 0.0,
    from_fallback   INTEGER DEFAULT 0,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_llm_calls_month
    ON llm_calls(created_at);

CREATE INDEX IF NOT EXISTS idx_llm_calls_task
    ON llm_calls(task_type);

CREATE TABLE IF NOT EXISTS llm_cache (
    prompt_hash     TEXT PRIMARY KEY,
    task_type       TEXT NOT NULL,
    response_json   TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
"""


def _utcnow() -> datetime:
    """Indirection for tests to monkey-patch."""
    return datetime.now(timezone.utc)


class CostTracker:
    """LLM 成本追踪 + 响应缓存。

    使用方式：
        tracker = CostTracker(db_path="data/llm_costs.db")
        tracker.initialize()

        # 检查预算（总预算 / 任务级预算）
        if tracker.is_over_budget():
            ...
        if tracker.is_over_budget(task_type=TaskType.G1_FILTER):
            ...

        # 记录调用
        tracker.record_call(task_type, model, prompt, content, ...)

        # 查缓存（默认 24h TTL）
        cached = tracker.get_cached(task_type, prompt)
        cached = tracker.get_cached(task_type, prompt, max_age_hours=1)

        # 月度报告
        report = tracker.monthly_report()
    """

    def __init__(
        self,
        db_path: str | Path = "data/llm_costs.db",
        monthly_budget: float = TOTAL_MONTHLY_BUDGET,
        task_configs: dict[TaskType, TaskConfig] | None = None,
        default_cache_ttl_hours: float = DEFAULT_CACHE_TTL_HOURS,
    ) -> None:
        self._db_path = Path(db_path)
        self._monthly_budget = monthly_budget
        self._task_configs = task_configs or DEFAULT_TASK_CONFIGS
        self._default_cache_ttl_hours = default_cache_ttl_hours
        self._conn: sqlite3.Connection | None = None
        # SQLite connection shared across threads. Serialise writes with a
        # lock — WAL alone doesn't satisfy Python's check_same_thread guard,
        # and even with check_same_thread=False we want to serialise writes
        # so we don't race on the sqlite3 cursor object.
        self._lock = threading.Lock()

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
        )
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
        logger.info(
            "CostTracker ready: %s (total_budget=$%.2f/mo, default_ttl=%.1fh)",
            self._db_path, self._monthly_budget, self._default_cache_ttl_hours,
        )

    def shutdown(self) -> None:
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record_call(
        self,
        task_type: TaskType,
        model: str,
        prompt: str,
        response_content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: float = 0.0,
        from_fallback: bool = False,
        *,
        system_prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
    ) -> None:
        """记录一次 LLM 调用。同时更新响应缓存。

        cache key 维度包含 (task_type, prompt, system_prompt, output_schema, model)，
        所以升级 system prompt / 换 schema / 换 model 都会让缓存 miss 到新 entry。
        """
        if not self._conn:
            return

        now = _utcnow().isoformat(timespec="microseconds")
        prompt_hash = self._hash_prompt(
            task_type, prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            model=model,
        )

        with self._lock:
            self._conn.execute(
                """INSERT INTO llm_calls
                   (task_type, model, prompt_hash, input_tokens, output_tokens,
                    cost, from_fallback, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_type.value, model, prompt_hash, input_tokens, output_tokens,
                 cost, int(from_fallback), now),
            )
            self._conn.execute(
                """INSERT OR REPLACE INTO llm_cache
                   (prompt_hash, task_type, response_json, created_at)
                   VALUES (?, ?, ?, ?)""",
                (prompt_hash, task_type.value,
                 json.dumps({"content": response_content}), now),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def get_cached(
        self,
        task_type: TaskType,
        prompt: str,
        *,
        system_prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        model: str | None = None,
        max_age_hours: float | None = None,
    ) -> str | None:
        """查询缓存。超过 ``max_age_hours`` 的条目视作 miss。

        ``model`` 参数必须传 —— 否则同一 prompt 在不同 model 下会串缓存。
        为了向后兼容，如果 caller 没传 model，会从该 task 的默认配置读取
        primary model 做 fallback。
        """
        if not self._conn:
            return None

        if model is None:
            cfg = self._task_configs.get(task_type)
            model = cfg.model if cfg else ""

        prompt_hash = self._hash_prompt(
            task_type, prompt,
            system_prompt=system_prompt,
            output_schema=output_schema,
            model=model,
        )

        with self._lock:
            cur = self._conn.execute(
                "SELECT response_json, created_at FROM llm_cache WHERE prompt_hash = ?",
                (prompt_hash,),
            )
            row = cur.fetchone()

        if not row:
            return None

        response_json, created_at_str = row

        ttl_hours = max_age_hours if max_age_hours is not None else self._default_cache_ttl_hours
        if ttl_hours is not None and ttl_hours >= 0:
            try:
                created_at = datetime.fromisoformat(created_at_str)
            except ValueError:
                return None
            age = _utcnow() - created_at
            if age > timedelta(hours=ttl_hours):
                return None

        try:
            return json.loads(response_json)["content"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

    def is_over_budget(
        self,
        month: str | None = None,
        task_type: TaskType | None = None,
    ) -> bool:
        """检查当月是否超预算。

        - 不传 ``task_type`` → 检查 *总* 预算
        - 传了 ``task_type`` → 检查该任务自己的 ``TaskConfig.monthly_budget``；
          若 TaskConfig 里 monthly_budget==0 视作"未设置任务级预算"，直接返回 False
        """
        if task_type is not None:
            cfg = self._task_configs.get(task_type)
            cap = cfg.monthly_budget if cfg else 0.0
            if cap <= 0:
                return False
            return self.monthly_spent(month, task_type=task_type) >= cap

        return self.monthly_spent(month) >= self._monthly_budget

    def monthly_spent(
        self,
        month: str | None = None,
        task_type: TaskType | None = None,
    ) -> float:
        """获取某月份的累计花费。月份默认是 UTC 当月（避免 UTC/本地时区错账）。"""
        if not self._conn:
            return 0.0

        month = month or self._current_month_utc()
        query = "SELECT COALESCE(SUM(cost), 0) FROM llm_calls WHERE created_at LIKE ?"
        params: list[Any] = [f"{month}%"]
        if task_type is not None:
            query += " AND task_type = ?"
            params.append(task_type.value)

        with self._lock:
            cur = self._conn.execute(query, params)
            row = cur.fetchone()
        return float(row[0]) if row else 0.0

    def monthly_report(self, month: str | None = None) -> dict[str, Any]:
        """月度成本报告，按任务类型分组。月份默认是 UTC 当月。"""
        if not self._conn:
            return {}

        month = month or self._current_month_utc()
        with self._lock:
            cur = self._conn.execute(
                """SELECT task_type,
                          COUNT(*) as calls,
                          SUM(input_tokens) as total_input,
                          SUM(output_tokens) as total_output,
                          SUM(cost) as total_cost
                   FROM llm_calls
                   WHERE created_at LIKE ?
                   GROUP BY task_type""",
                (f"{month}%",),
            )
            rows = cur.fetchall()

        breakdown: dict[str, Any] = {}
        total_cost = 0.0
        total_calls = 0

        for row in rows:
            task_type, calls, inp, out, cost = row
            cfg_cap = 0.0
            try:
                cfg_cap = self._task_configs[TaskType(task_type)].monthly_budget
            except (KeyError, ValueError):
                pass
            breakdown[task_type] = {
                "calls": calls,
                "input_tokens": inp or 0,
                "output_tokens": out or 0,
                "cost": round(cost or 0.0, 6),
                "task_budget": cfg_cap,
                "task_over_budget": (cfg_cap > 0 and (cost or 0.0) >= cfg_cap),
            }
            total_cost += cost or 0.0
            total_calls += calls

        return {
            "month": month,
            "total_cost": round(total_cost, 4),
            "total_calls": total_calls,
            "budget": self._monthly_budget,
            "remaining": round(self._monthly_budget - total_cost, 4),
            "over_budget": total_cost >= self._monthly_budget,
            "breakdown": breakdown,
        }

    def cache_size(self) -> int:
        """返回缓存条目数。"""
        if not self._conn:
            return 0
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) FROM llm_cache")
            row = cur.fetchone()
        return row[0] if row else 0

    def clear_expired_cache(self, max_age_hours: float | None = None) -> int:
        """Delete expired cache entries. Returns deleted count."""
        if not self._conn:
            return 0
        ttl_hours = max_age_hours if max_age_hours is not None else self._default_cache_ttl_hours
        cutoff = (_utcnow() - timedelta(hours=ttl_hours)).isoformat(timespec="microseconds")
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM llm_cache WHERE created_at < ?",
                (cutoff,),
            )
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _current_month_utc() -> str:
        return _utcnow().strftime("%Y-%m")

    @staticmethod
    def _hash_prompt(
        task_type: TaskType,
        prompt: str,
        *,
        system_prompt: str | None = None,
        output_schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> str:
        """生成缓存 key 的确定性 hash。

        维度：(task_type, prompt, system_prompt, canonical(output_schema), model)
        """
        schema_canonical = (
            json.dumps(output_schema, sort_keys=True, ensure_ascii=False)
            if output_schema is not None else ""
        )
        key = "|".join([
            task_type.value,
            model or "",
            system_prompt or "",
            schema_canonical,
            prompt,
        ])
        return hashlib.sha256(key.encode()).hexdigest()[:32]
