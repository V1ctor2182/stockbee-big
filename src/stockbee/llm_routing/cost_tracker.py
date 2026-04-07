"""CostTracker — LLM 调用成本追踪 + 响应缓存。

按任务类型累计 token 用量和成本。月度预算检查（$15 上限）。
响应缓存：同一 prompt hash 复用上次成功结果。
SQLite 存储历史调用记录。

来源：Tech Design §3.2
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .task_config import TaskType, TOTAL_MONTHLY_BUDGET

logger = logging.getLogger(__name__)

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


class CostTracker:
    """LLM 成本追踪 + 响应缓存。

    使用方式：
        tracker = CostTracker(db_path="data/llm_costs.db")
        tracker.initialize()

        # 检查预算
        if tracker.is_over_budget():
            # 降级到缓存

        # 记录调用
        tracker.record_call(response)

        # 查缓存
        cached = tracker.get_cached(task_type, prompt)

        # 月度报告
        report = tracker.monthly_report()
    """

    def __init__(
        self,
        db_path: str | Path = "data/llm_costs.db",
        monthly_budget: float = TOTAL_MONTHLY_BUDGET,
    ) -> None:
        self._db_path = Path(db_path)
        self._monthly_budget = monthly_budget
        self._conn: sqlite3.Connection | None = None

    def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        logger.info("CostTracker ready: %s (budget=$%.0f/mo)", self._db_path, self._monthly_budget)

    def shutdown(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

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
    ) -> None:
        """记录一次 LLM 调用。同时更新缓存。"""
        if not self._conn:
            return

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prompt_hash = self._hash_prompt(task_type, prompt)

        self._conn.execute(
            """INSERT INTO llm_calls
               (task_type, model, prompt_hash, input_tokens, output_tokens,
                cost, from_fallback, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_type.value, model, prompt_hash, input_tokens, output_tokens,
             cost, int(from_fallback), now),
        )

        # 更新缓存
        self._conn.execute(
            """INSERT OR REPLACE INTO llm_cache
               (prompt_hash, task_type, response_json, created_at)
               VALUES (?, ?, ?, ?)""",
            (prompt_hash, task_type.value, json.dumps({"content": response_content}), now),
        )
        self._conn.commit()

    def get_cached(self, task_type: TaskType, prompt: str) -> str | None:
        """查询缓存，返回上次成功响应内容，无缓存返回 None。"""
        if not self._conn:
            return None

        prompt_hash = self._hash_prompt(task_type, prompt)
        cur = self._conn.execute(
            "SELECT response_json FROM llm_cache WHERE prompt_hash = ?",
            (prompt_hash,),
        )
        row = cur.fetchone()
        if row:
            try:
                return json.loads(row[0])["content"]
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    def is_over_budget(self, month: str | None = None) -> bool:
        """检查当月是否超预算。"""
        spent = self.monthly_spent(month)
        return spent >= self._monthly_budget

    def monthly_spent(self, month: str | None = None) -> float:
        """获取当月累计花费。"""
        if not self._conn:
            return 0.0

        month = month or date.today().strftime("%Y-%m")
        cur = self._conn.execute(
            "SELECT COALESCE(SUM(cost), 0) FROM llm_calls WHERE created_at LIKE ?",
            (f"{month}%",),
        )
        return cur.fetchone()[0]

    def monthly_report(self, month: str | None = None) -> dict[str, Any]:
        """月度成本报告，按任务类型分组。"""
        if not self._conn:
            return {}

        month = month or date.today().strftime("%Y-%m")
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

        breakdown: dict[str, Any] = {}
        total_cost = 0.0
        total_calls = 0

        for row in cur.fetchall():
            task_type, calls, inp, out, cost = row
            breakdown[task_type] = {
                "calls": calls,
                "input_tokens": inp or 0,
                "output_tokens": out or 0,
                "cost": cost or 0.0,
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
        cur = self._conn.execute("SELECT COUNT(*) FROM llm_cache")
        return cur.fetchone()[0]

    @staticmethod
    def _hash_prompt(task_type: TaskType, prompt: str) -> str:
        """生成 prompt 的确定性 hash。"""
        key = f"{task_type.value}:{prompt}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
