"""m1 contract / model_io / fixtures / schema migration 合约测试。

覆盖:
  - paths: MODEL_ROOT / ML_SCORE_PARQUET 常量格式
  - model_io: save/load roundtrip, overwrite, current symlink 覆盖,
              versions 排序,NotFoundError,InvalidArtifactError,
              POSIX 下 update_symlink 清理老的 current.txt,
              Windows fallback 走 current.txt 解析
  - fixtures: shape / dtype / index 断言
  - news_events schema 迁移: 4 新列存在,老库升级,幂等,老数据不丢
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX symlink path only"
)

from stockbee.news_data.news_store import SqliteNewsProvider
from stockbee.providers.base import ProviderConfig
from stockbee.small_models import (
    InvalidArtifactError,
    ML_SCORE_PARQUET,
    MODEL_ROOT,
    NotFoundError,
    artifact_path,
    list_versions,
    load_pickle,
    save_pickle,
    update_symlink,
)
from stockbee.small_models import model_io as _model_io
from stockbee.small_models import paths as small_model_paths


# ------------------------------ paths ----------------------------------------


class TestPaths:
    def test_model_root_points_to_data_models(self):
        assert MODEL_ROOT == Path("data/models")

    def test_ml_score_parquet_under_data_factors(self):
        assert ML_SCORE_PARQUET == Path("data/factors/ml_score.parquet")
        assert ML_SCORE_PARQUET.suffix == ".parquet"


# ------------------------------ model_io -------------------------------------


class TestArtifactPath:
    def test_current_default(self, tmp_artifact_dir):
        p = artifact_path("lightgbm")
        assert p == tmp_artifact_dir / "lightgbm" / "current.pkl"

    def test_versioned(self, tmp_artifact_dir):
        p = artifact_path("finbert", "20260413")
        assert p == tmp_artifact_dir / "finbert" / "20260413.pkl"

    def test_rejects_bad_version(self, tmp_artifact_dir):
        with pytest.raises(ValueError):
            artifact_path("finbert", "v1")

    def test_rejects_path_traversal(self, tmp_artifact_dir):
        for bad in ["../etc", "", "..", ".", ".hidden", "a/b", "a\\b", "a\x00b"]:
            with pytest.raises(ValueError):
                artifact_path(bad)

    def test_rejects_non_str_name(self, tmp_artifact_dir):
        with pytest.raises(ValueError):
            artifact_path(None)  # type: ignore[arg-type]

    def test_rejects_invalid_calendar_date(self, tmp_artifact_dir):
        for bad in ["20260230", "20260001", "20261332", "00000000"]:
            with pytest.raises(ValueError, match="valid calendar date|YYYYMMDD"):
                artifact_path("m", bad)

    def test_rejects_non_str_version(self, tmp_artifact_dir):
        with pytest.raises(TypeError):
            artifact_path("m", 20260413)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            artifact_path("m", True)  # type: ignore[arg-type]


class TestSaveLoadPickle:
    def test_roundtrip(self, tmp_artifact_dir):
        obj = {"weights": [1, 2, 3], "bias": 0.5}
        target = save_pickle(obj, "lightgbm", version="20260413")
        assert target.exists()
        assert target.name == "20260413.pkl"
        loaded = load_pickle("lightgbm", "20260413")
        assert loaded == obj

    def test_default_version_is_today(self, tmp_artifact_dir):
        target = save_pickle({"x": 1}, "finbert")
        assert target.stem == date.today().strftime("%Y%m%d")

    def test_save_current_literal_rejected(self, tmp_artifact_dir):
        with pytest.raises(ValueError):
            save_pickle({"x": 1}, "finbert", version="current")

    def test_default_rejects_overwrite(self, tmp_artifact_dir):
        save_pickle({"x": 1}, "model", version="20260413")
        with pytest.raises(FileExistsError):
            save_pickle({"x": 2}, "model", version="20260413")

    def test_overwrite_true(self, tmp_artifact_dir):
        save_pickle({"x": 1}, "model", version="20260413")
        save_pickle({"x": 2}, "model", version="20260413", overwrite=True)
        assert load_pickle("model", "20260413") == {"x": 2}

    def test_concurrent_save_race_no_silent_overwrite(
        self, tmp_artifact_dir, monkeypatch
    ):
        """Codex P2: 并发 save_pickle(overwrite=False) 必须原子。

        模拟两进程 race: A 和 B 同时 save 同 (name, version),B 在 A 的 os.link
        调用之前抢先把目标文件建好。A 的 os.link 应 raise FileExistsError,
        而不是靠 pre-check + os.replace 的非原子组合静默覆盖。
        """
        name, version = "model", "20260413"
        target = artifact_path(name, version)
        target.parent.mkdir(parents=True, exist_ok=True)
        real_link = os.link

        def racing_link(src, dst, *a, **kw):
            # 模拟对方进程抢在我们 link 前写了目标文件
            if not Path(dst).exists():
                Path(dst).write_bytes(b"peer wrote first")
            return real_link(src, dst, *a, **kw)

        monkeypatch.setattr(os, "link", racing_link)
        with pytest.raises(FileExistsError, match="already exists"):
            save_pickle({"mine": True}, name, version=version)
        # 对方内容不应被我们覆盖
        assert target.read_bytes() == b"peer wrote first"

    def test_load_missing_raises_notfound(self, tmp_artifact_dir):
        with pytest.raises(NotFoundError):
            load_pickle("nowhere", "20260101")

    def test_load_current_missing_raises_notfound(self, tmp_artifact_dir):
        (tmp_artifact_dir / "finbert").mkdir()
        with pytest.raises(NotFoundError):
            load_pickle("finbert", "current")

    def test_corrupt_pickle_raises(self, tmp_artifact_dir):
        (tmp_artifact_dir / "mdl").mkdir()
        bad = tmp_artifact_dir / "mdl" / "20260413.pkl"
        bad.write_bytes(b"not a pickle")
        with pytest.raises(InvalidArtifactError):
            load_pickle("mdl", "20260413")

    def test_save_creates_nested_dirs(self, tmp_artifact_dir):
        assert not (tmp_artifact_dir / "brand_new").exists()
        save_pickle([1, 2], "brand_new", version="20260413")
        assert (tmp_artifact_dir / "brand_new" / "20260413.pkl").exists()


class TestUpdateSymlink:
    @_posix_only
    def test_symlink_points_to_version(self, tmp_artifact_dir):
        save_pickle({"v": 1}, "finbert", version="20260101")
        update_symlink("finbert", "20260101")
        current = tmp_artifact_dir / "finbert" / "current.pkl"
        assert current.exists()
        assert load_pickle("finbert", "current") == {"v": 1}

    def test_symlink_switches_between_versions(self, tmp_artifact_dir):
        save_pickle({"v": 1}, "m", version="20260101")
        save_pickle({"v": 2}, "m", version="20260202")
        update_symlink("m", "20260101")
        assert load_pickle("m", "current") == {"v": 1}
        update_symlink("m", "20260202")
        assert load_pickle("m", "current") == {"v": 2}

    def test_missing_version_raises_notfound(self, tmp_artifact_dir):
        (tmp_artifact_dir / "m").mkdir()
        with pytest.raises(NotFoundError):
            update_symlink("m", "20991231")

    def test_rejects_current_literal(self, tmp_artifact_dir):
        with pytest.raises(ValueError):
            update_symlink("m", "current")

    @_posix_only
    def test_posix_clears_old_current_txt(self, tmp_artifact_dir):
        """POSIX 下若残留 current.txt (Windows 迁回场景),update_symlink 应清理。"""
        save_pickle({"v": 1}, "x", version="20260101")
        (tmp_artifact_dir / "x" / "current.txt").write_text("99999999")
        update_symlink("x", "20260101")
        assert not (tmp_artifact_dir / "x" / "current.txt").exists()
        assert load_pickle("x", "current") == {"v": 1}

    def test_osrror_fallback_clears_stale_symlink(
        self, tmp_artifact_dir, monkeypatch
    ):
        """C1: POSIX symlink_to 抛 OSError 时,必须清理旧 symlink 再写 current.txt,
        否则下次 load_pickle 读到 stale artifact。"""
        save_pickle({"v": "old"}, "m", version="20260101")
        save_pickle({"v": "new"}, "m", version="20260202")
        update_symlink("m", "20260101")  # 建立初始 symlink -> 20260101
        assert load_pickle("m", "current") == {"v": "old"}

        # 第二次切换:模拟 symlink_to 抛 OSError,强制走回退
        from stockbee.small_models import model_io as mi
        orig_symlink_to = Path.symlink_to

        def boom(self, target, *a, **kw):
            raise OSError("simulated fs restriction")

        monkeypatch.setattr(Path, "symlink_to", boom)
        update_symlink("m", "20260202")

        # 旧 symlink 必须已被清理,load 走 current.txt 得到新版本
        current_link = tmp_artifact_dir / "m" / "current.pkl"
        current_txt = tmp_artifact_dir / "m" / "current.txt"
        assert not current_link.exists() and not current_link.is_symlink()
        assert current_txt.exists() and current_txt.read_text().strip() == "20260202"
        monkeypatch.setattr(Path, "symlink_to", orig_symlink_to)
        assert load_pickle("m", "current") == {"v": "new"}

    @_posix_only
    def test_dangling_symlink_falls_back_to_current_txt(
        self, tmp_artifact_dir
    ):
        """H1: symlink target 被外部删除后,load_pickle 应降级到 current.txt。"""
        save_pickle({"v": 1}, "m", version="20260101")
        save_pickle({"v": 2}, "m", version="20260202")
        update_symlink("m", "20260101")
        # 手写 current.txt 指向 20260202,然后删 20260101 的文件,让 symlink 悬挂
        (tmp_artifact_dir / "m" / "current.txt").write_text("20260202")
        (tmp_artifact_dir / "m" / "20260101.pkl").unlink()
        assert load_pickle("m", "current") == {"v": 2}

    @_posix_only
    def test_dangling_symlink_without_txt_raises_invalid(
        self, tmp_artifact_dir
    ):
        save_pickle({"v": 1}, "m", version="20260101")
        update_symlink("m", "20260101")
        (tmp_artifact_dir / "m" / "20260101.pkl").unlink()
        with pytest.raises(InvalidArtifactError, match="dangling"):
            load_pickle("m", "current")


class TestWindowsFallback:
    """OQ3: current.txt 回退路径,模拟无 symlink 环境。"""

    def test_load_current_reads_current_txt(self, tmp_artifact_dir, monkeypatch):
        save_pickle({"v": 7}, "m", version="20260413")
        monkeypatch.setattr(_model_io, "_supports_symlink", lambda: False)
        update_symlink("m", "20260413")
        current_txt = tmp_artifact_dir / "m" / "current.txt"
        assert current_txt.exists()
        assert current_txt.read_text().strip() == "20260413"
        assert not (tmp_artifact_dir / "m" / "current.pkl").exists()
        assert load_pickle("m", "current") == {"v": 7}

    def test_corrupt_current_txt_raises(self, tmp_artifact_dir, monkeypatch):
        (tmp_artifact_dir / "m").mkdir()
        (tmp_artifact_dir / "m" / "current.txt").write_text("not-a-version")
        monkeypatch.setattr(_model_io, "_supports_symlink", lambda: False)
        with pytest.raises(InvalidArtifactError):
            load_pickle("m", "current")


class TestListVersions:
    def test_empty_when_missing(self, tmp_artifact_dir):
        assert list_versions("nothing") == []

    def test_descending_order(self, tmp_artifact_dir):
        for v in ["20260101", "20260505", "20260303"]:
            save_pickle({"x": v}, "lgbm", version=v)
        assert list_versions("lgbm") == ["20260505", "20260303", "20260101"]

    def test_excludes_current_symlink(self, tmp_artifact_dir):
        save_pickle({"x": 1}, "m", version="20260101")
        update_symlink("m", "20260101")
        assert list_versions("m") == ["20260101"]

    def test_ignores_non_matching_files(self, tmp_artifact_dir):
        save_pickle({"x": 1}, "m", version="20260101")
        (tmp_artifact_dir / "m" / "notes.txt").write_text("hi")
        (tmp_artifact_dir / "m" / "bad_name.pkl").write_bytes(b"")
        assert list_versions("m") == ["20260101"]


# ------------------------------ fixtures -------------------------------------


class TestFixturesShape:
    def test_ohlcv(self, ohlcv_fixture):
        df = ohlcv_fixture
        assert list(df.index.names) == ["date", "ticker"]
        assert set(df.columns) == {
            "open", "high", "low", "close", "adj_close", "volume",
        }
        assert len(df) == 60 * 2
        assert df["adj_close"].notna().all()
        assert (df["high"] >= df["low"]).all()

    def test_alpha158(self, alpha158_fixture):
        df = alpha158_fixture
        assert list(df.index.names) == ["date", "ticker"]
        assert df.shape == (50 * 2, 5)
        assert {"MA5", "MA10", "STD20", "RSQR20", "KMID"} == set(df.columns)

    def test_news(self, news_fixture):
        assert len(news_fixture) == 20
        assert {"timestamp", "headline", "tickers", "g_level"}.issubset(news_fixture.columns)
        assert all(isinstance(t, list) for t in news_fixture["tickers"])

    def test_embeddings(self, embeddings_fixture):
        mat, dim = embeddings_fixture
        assert mat.shape == (12, dim)
        assert mat.dtype == np.float32
        norms = np.linalg.norm(mat, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_finbert_golden(self, finbert_golden):
        assert len(finbert_golden) == 10
        labels = [r["label"] for r in finbert_golden]
        assert labels.count("positive") == 4
        assert labels.count("negative") == 4
        assert labels.count("neutral") == 2

    def test_tmp_artifact_dir_isolates(self, tmp_artifact_dir):
        assert tmp_artifact_dir.exists()
        # import 级别的常量也被 patch 了
        assert _model_io.MODEL_ROOT == tmp_artifact_dir
        assert small_model_paths.MODEL_ROOT == tmp_artifact_dir


# ------------------------------ schema migration -----------------------------


_EXPECTED_NEW_COLS = {
    "finbert_negative",
    "finbert_neutral",
    "finbert_confidence",
    "fine5_importance",
}


def _open_provider(db_path: Path) -> SqliteNewsProvider:
    p = SqliteNewsProvider(ProviderConfig(
        implementation="SqliteNewsProvider",
        params={"db_path": str(db_path)},
    ))
    p._do_initialize()
    return p


def _columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(news_events)").fetchall()}


class TestSchemaMigration:
    def test_not_initialized_raises(self, tmp_path):
        """L2: _migrate_schema 在未 initialize 时必须 raise,不静默 no-op。"""
        p = SqliteNewsProvider(ProviderConfig(
            implementation="SqliteNewsProvider",
            params={"db_path": str(tmp_path / "x.db")},
        ))
        with pytest.raises(RuntimeError, match="not initialized"):
            p._migrate_schema()

    def test_concurrent_alter_race_tolerated(self, tmp_path, monkeypatch):
        """H4: A 进程 PRAGMA 判断列缺失后,B 进程已抢先 ALTER;A 的 ALTER 抛
        duplicate column OperationalError。_migrate_schema 应在重新探测后 skip 不崩。

        用 monkeypatch 让 _columns 第一次返回老 set(列缺失),让 _migrate_schema
        进入 ALTER 分支;peer 已实际加了列 → ALTER 抛 duplicate column;
        第二次 _columns 返回真实 set → 识别为"对方已加" → 成功。
        """
        db = tmp_path / "race.db"
        seed = sqlite3.connect(str(db))
        seed.executescript("""
            CREATE TABLE news_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, source TEXT NOT NULL, source_url TEXT,
                headline TEXT NOT NULL, snippet TEXT,
                sentiment_score REAL, importance_score REAL, reliability_score REAL,
                g_level INTEGER NOT NULL DEFAULT 0, analysis TEXT, created_at TEXT NOT NULL
            );
            CREATE TABLE news_tickers (
                news_id INTEGER NOT NULL, ticker TEXT NOT NULL,
                PRIMARY KEY (news_id, ticker)
            );
            CREATE TABLE g3_daily_counts (date TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0);
        """)
        # peer 已加 finbert_negative(模拟对方进程抢先)
        seed.execute("ALTER TABLE news_events ADD COLUMN finbert_negative REAL")
        seed.commit()
        seed.close()

        p = SqliteNewsProvider(ProviderConfig(
            implementation="SqliteNewsProvider",
            params={"db_path": str(db)},
        ))
        p._conn = sqlite3.connect(str(db))

        real_columns = p._columns
        call_n = {"n": 0}
        empty_legacy_cols = {
            "id", "timestamp", "source", "source_url", "headline", "snippet",
            "sentiment_score", "importance_score", "reliability_score",
            "g_level", "analysis", "created_at",
        }

        def faked(*a, **kw):
            call_n["n"] += 1
            # 第一次(探测阶段):撒谎说 finbert_negative 没有 → 触发 ALTER → duplicate column
            if call_n["n"] == 1:
                return empty_legacy_cols
            # 之后走真实探测(识别对方已加 + 后续列)
            return real_columns(*a, **kw)

        monkeypatch.setattr(p, "_columns", faked)
        try:
            p._migrate_schema()  # 不应 crash
        finally:
            monkeypatch.undo()
            cols = _columns(p._conn)
            assert _EXPECTED_NEW_COLS.issubset(cols)
            p._do_shutdown()

    def test_fresh_db_has_new_columns(self, tmp_path):
        p = _open_provider(tmp_path / "fresh.db")
        try:
            assert _EXPECTED_NEW_COLS.issubset(_columns(p._conn))
        finally:
            p._do_shutdown()

    def test_migration_is_idempotent(self, tmp_path):
        p = _open_provider(tmp_path / "idem.db")
        try:
            p._migrate_schema()
            p._migrate_schema()
            cols = _columns(p._conn)
            assert sum(1 for c in cols if c == "finbert_negative") == 1
        finally:
            p._do_shutdown()

    def test_reopen_does_not_break(self, tmp_path):
        db = tmp_path / "reopen.db"
        p1 = _open_provider(db)
        p1.insert_news(
            headline="seed row",
            source="reuters",
            timestamp="2026-03-01T00:00:00+00:00",
        )
        p1._do_shutdown()
        p2 = _open_provider(db)
        try:
            assert p2._count_all() == 1
            assert _EXPECTED_NEW_COLS.issubset(_columns(p2._conn))
        finally:
            p2._do_shutdown()

    def test_legacy_db_auto_upgrades_without_data_loss(self, tmp_path):
        """模拟老库 (缺 4 新列) 开库后自动扩列,旧数据完整。"""
        db = tmp_path / "legacy.db"
        legacy_sql = """
            CREATE TABLE news_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT,
                headline TEXT NOT NULL,
                snippet TEXT,
                sentiment_score REAL,
                importance_score REAL,
                reliability_score REAL,
                g_level INTEGER NOT NULL DEFAULT 0,
                analysis TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE news_tickers (
                news_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                PRIMARY KEY (news_id, ticker)
            );
            CREATE TABLE g3_daily_counts (date TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0);
        """
        conn = sqlite3.connect(str(db))
        conn.executescript(legacy_sql)
        conn.execute(
            """INSERT INTO news_events
               (timestamp, source, headline, sentiment_score, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            ("2026-03-01T00:00:00+00:00", "reuters", "legacy headline", 0.8, "2026-03-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        p = _open_provider(db)
        try:
            cols = _columns(p._conn)
            assert _EXPECTED_NEW_COLS.issubset(cols)
            cur = p._conn.execute(
                "SELECT headline, sentiment_score, finbert_negative FROM news_events"
            )
            row = cur.fetchone()
            assert row[0] == "legacy headline"
            assert row[1] == pytest.approx(0.8)
            assert row[2] is None  # 新列默认 NULL
        finally:
            p._do_shutdown()

    def test_new_columns_are_real_type(self, tmp_path):
        p = _open_provider(tmp_path / "types.db")
        try:
            info = p._conn.execute("PRAGMA table_info(news_events)").fetchall()
            types = {row[1]: row[2] for row in info}
            for col in _EXPECTED_NEW_COLS:
                assert types[col].upper() == "REAL", f"{col} type is {types[col]}"
        finally:
            p._do_shutdown()
