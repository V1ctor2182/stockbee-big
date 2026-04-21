"""通用模型 artifact IO。

目录结构::

    data/models/{name}/{YYYYMMDD}.pkl      # 版本化 artifact
    data/models/{name}/current.pkl         # POSIX symlink -> 版本文件
    data/models/{name}/current.txt         # Windows fallback,明文记录当前版本号

m2a / m3 / m4 / m5 全部走 save_pickle / load_pickle / update_symlink。

版本号约定: 传入 str(形如 20260413) 或 None 取 today().strftime("%Y%m%d")。

**安全约束**:`load_pickle` 反序列化会执行任意 `__reduce__`,仅用于受信任的
本地训练产物。不要读外网或 user-uploaded pickle。
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any

from . import paths as _paths

logger = logging.getLogger(__name__)

_VERSION_RE = re.compile(r"^\d{8}$")
_CURRENT = "current"
_PICKLE_SUFFIX = ".pkl"
_WINDOWS_CURRENT_TXT = "current.txt"

# 向后兼容 re-export(tmp_artifact_dir 测试 fixture 直接 monkeypatch 这个名字)。
# 实际读取走 _root() 以支持 paths.MODEL_ROOT 运行时被替换。
MODEL_ROOT = _paths.MODEL_ROOT


class NotFoundError(FileNotFoundError):
    """请求的 artifact 不存在。"""


class InvalidArtifactError(ValueError):
    """artifact 文件损坏、格式错误或 current.txt 内容非法。"""


def _root() -> Path:
    """返回当前 MODEL_ROOT。优先读本模块 MODEL_ROOT(兼容老 monkeypatch 写法),
    其次 paths.MODEL_ROOT(新代码推荐改这里)。两者一致时无差别。"""
    return Path(globals().get("MODEL_ROOT") or _paths.MODEL_ROOT)


def _model_dir(name: str) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid artifact name: {name!r}")
    if "/" in name or "\\" in name or name in (".", "..") or name.startswith("."):
        raise ValueError(f"Invalid artifact name: {name!r}")
    if "\x00" in name:
        raise ValueError(f"Invalid artifact name: {name!r}")
    return _root() / name


def _resolve_version(version: str | None) -> str:
    if version is None:
        return date.today().strftime("%Y%m%d")
    if not isinstance(version, str):
        raise TypeError(
            f"version must be str or None, got {type(version).__name__}"
        )
    if version == _CURRENT:
        return _CURRENT
    if not _VERSION_RE.match(version):
        raise ValueError(
            f"version must be 'current' or 8-digit YYYYMMDD, got {version!r}"
        )
    try:
        datetime.strptime(version, "%Y%m%d")
    except ValueError as exc:
        raise ValueError(
            f"version {version!r} is not a valid calendar date"
        ) from exc
    return version


def artifact_path(name: str, version: str = "current") -> Path:
    """返回 artifact 文件路径。

    version='current' → 返回 current.pkl 路径 (POSIX 下为 symlink)。
    version='YYYYMMDD' → 返回版本文件路径。
    不校验文件是否存在。
    """
    version = _resolve_version(version)
    return _model_dir(name) / f"{version}{_PICKLE_SUFFIX}"


def save_pickle(
    obj: Any,
    name: str,
    version: str | None = None,
    overwrite: bool = False,
) -> Path:
    """将 obj pickle 到 data/models/{name}/{version}.pkl (原子写)。

    version=None 默认取 today (YYYYMMDD)。
    overwrite=False (默认) 若同版本文件已存在则 raise FileExistsError;
    m3 重训等显式覆盖场景传 overwrite=True。

    实现为 tmp + rename,崩溃不会留半写文件;Windows 下同样可覆盖读中的文件。

    **不自动** update current symlink — 训练完需显式 `update_symlink(name, version)`。
    """
    version_str = _resolve_version(version)
    if version_str == _CURRENT:
        raise ValueError("save_pickle version 不能是 'current',必须是 YYYYMMDD")

    target = _model_dir(name) / f"{version_str}{_PICKLE_SUFFIX}"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f"{version_str}.pkl.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    try:
        with tmp.open("wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        if overwrite:
            os.replace(tmp, target)
        else:
            # 原子"存在即失败"建链。os.link 在目标存在时抛 FileExistsError,
            # 消除 check-then-replace 的 TOCTOU — 两进程并发 save 同版本,
            # 只有一个能成功;另一个拿到 FileExistsError 而不是静默覆盖。
            try:
                os.link(tmp, target)
            except FileExistsError:
                raise FileExistsError(
                    f"artifact already exists: {target}; pass overwrite=True to replace"
                )
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    logger.debug("save_pickle: %s -> %s", name, target)
    return target


def load_pickle(name: str, version: str = "current") -> Any:
    """加载 artifact。

    version='current' POSIX 读 symlink;若 symlink 悬挂则降级到 current.txt;
    都缺则 NotFoundError。pickle 损坏 → InvalidArtifactError。

    **安全**:仅用于受信任的本地产物,见模块 docstring。
    """
    version_str = _resolve_version(version)
    model_dir = _model_dir(name)

    if version_str == _CURRENT:
        target = _resolve_current_file(model_dir, name)
    else:
        target = model_dir / f"{version_str}{_PICKLE_SUFFIX}"

    if not target.exists():
        raise NotFoundError(f"artifact not found: {target}")
    try:
        with target.open("rb") as f:
            return pickle.load(f)
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as exc:
        raise InvalidArtifactError(f"corrupt artifact at {target}: {exc}") from exc


def update_symlink(name: str, version: str) -> None:
    """将 {name}/current.pkl 指向 {name}/{version}.pkl。

    POSIX: 原子替换 symlink (tmp + rename);完成后清理残留 current.txt。
    Windows / 无 symlink 权限: 写 current.txt 记录版本号;同时清理残留 current.pkl
    避免旧 symlink 被误读。OSError 触发从 POSIX 降级到 txt 时同样清理。
    """
    version_str = _resolve_version(version)
    if version_str == _CURRENT:
        raise ValueError("update_symlink version 不能是 'current'")

    model_dir = _model_dir(name)
    source = model_dir / f"{version_str}{_PICKLE_SUFFIX}"
    if not source.exists():
        raise NotFoundError(f"cannot point current to missing version: {source}")

    current_link = model_dir / f"{_CURRENT}{_PICKLE_SUFFIX}"
    current_txt = model_dir / _WINDOWS_CURRENT_TXT

    if _supports_symlink():
        tmp = model_dir / f"{_CURRENT}.pkl.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        try:
            tmp.symlink_to(source.name)
            os.replace(tmp, current_link)
        except OSError:
            _unlink_silent(tmp)
            _unlink_silent(current_link)  # C1: 清理旧 symlink/文件防读到 stale
            _write_current_txt(current_txt, version_str)
            return
        _unlink_silent(current_txt)
    else:
        _unlink_silent(current_link)
        _write_current_txt(current_txt, version_str)

    logger.debug("update_symlink: %s/current -> %s", name, source.name)


def list_versions(name: str) -> list[str]:
    """返回 name 的所有版本号 (YYYYMMDD),按降序排列。current.pkl 不计入。"""
    model_dir = _model_dir(name)
    if not model_dir.exists():
        return []
    out: list[str] = []
    for p in model_dir.iterdir():
        if p.suffix != _PICKLE_SUFFIX or p.is_symlink():
            continue
        stem = p.stem
        if _VERSION_RE.match(stem):
            out.append(stem)
    return sorted(out, reverse=True)


# ------ 内部辅助 ------


def _supports_symlink() -> bool:
    """POSIX 默认支持;Windows 视权限。"""
    return sys.platform != "win32" and hasattr(os, "symlink")


def _unlink_silent(p: Path) -> None:
    """删文件或 symlink,不存在时静默返回。兼容悬挂 symlink(exists()=False)。"""
    try:
        p.unlink(missing_ok=True)
    except IsADirectoryError:
        raise
    except OSError:
        # 悬挂 symlink: unlink 走链接自己,若未能删再 fallback to os.remove on link path
        if p.is_symlink():
            os.unlink(p)


def _write_current_txt(path: Path, version: str) -> None:
    tmp = path.with_suffix(
        path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    )
    tmp.write_text(version, encoding="utf-8")
    os.replace(tmp, path)


def _resolve_current_file(model_dir: Path, name: str) -> Path:
    """load_pickle 'current' 路径解析。

    顺序:
      1. symlink (POSIX):若存在且未悬挂 → 直读
      2. 悬挂 symlink 或无 symlink → 查 current.txt 拼版本文件
      3. 都缺 → NotFoundError
    """
    current_link = model_dir / f"{_CURRENT}{_PICKLE_SUFFIX}"
    if current_link.exists():
        return current_link
    # symlink 悬挂:降级查 current.txt(H1)
    current_txt = model_dir / _WINDOWS_CURRENT_TXT
    if current_txt.exists():
        version = current_txt.read_text(encoding="utf-8").strip()
        if not _VERSION_RE.match(version):
            raise InvalidArtifactError(
                f"{current_txt} content {version!r} is not YYYYMMDD"
            )
        return model_dir / f"{version}{_PICKLE_SUFFIX}"
    if current_link.is_symlink():
        raise InvalidArtifactError(
            f"dangling current symlink at {current_link} -> "
            f"{os.readlink(current_link)!r}"
        )
    raise NotFoundError(f"no current artifact for {name} in {model_dir}")


__all__ = [
    "InvalidArtifactError",
    "MODEL_ROOT",
    "NotFoundError",
    "artifact_path",
    "list_versions",
    "load_pickle",
    "save_pickle",
    "update_symlink",
]
