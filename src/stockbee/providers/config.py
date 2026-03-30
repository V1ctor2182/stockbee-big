"""ConfigLoader — YAML 配置加载器。

读取 config/providers-{env}.yaml，解析环境变量，
生成 ProviderConfig 字典供 Registry 批量创建。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .base import ProviderConfig, ProviderEnv

logger = logging.getLogger(__name__)

# 匹配 ${ENV_VAR} 或 ${ENV_VAR:-default}
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")


def _substitute_env_vars(value: str) -> str:
    """替换字符串中的 ${ENV_VAR} 和 ${ENV_VAR:-default}。"""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(var_name)
        if env_value is not None:
            return env_value
        if default is not None:
            return default
        logger.warning("Environment variable not set: %s", var_name)
        return match.group(0)  # 保留原文

    return _ENV_VAR_PATTERN.sub(_replace, value)


def _process_values(obj: Any) -> Any:
    """递归替换字典/列表中所有字符串的环境变量。"""
    if isinstance(obj, str):
        return _substitute_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _process_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_process_values(item) for item in obj]
    return obj


def load_provider_configs(
    config_path: str | Path,
) -> dict[str, ProviderConfig]:
    """从 YAML 文件加载 Provider 配置。

    YAML 格式：
        MarketDataProvider:
          implementation: ParquetMarketData
          path: /data/ohlcv/
          fallback: AlpacaMarketData

        NewsProvider:
          implementation: SqliteNewsProvider
          db_path: ${DATA_DIR:-data}/news.db

    Returns:
        {provider_name: ProviderConfig} 字典
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config format in {config_path}: expected dict")

    raw = _process_values(raw)

    configs: dict[str, ProviderConfig] = {}
    for provider_name, settings in raw.items():
        if not isinstance(settings, dict):
            continue
        implementation = settings.pop("implementation", None)
        if not implementation:
            logger.warning(
                "Skipping %s: missing 'implementation' field", provider_name
            )
            continue
        fallback = settings.pop("fallback", None)
        configs[provider_name] = ProviderConfig(
            implementation=implementation,
            params=settings,
            fallback=fallback,
        )

    logger.info(
        "Loaded %d provider configs from %s", len(configs), config_path.name
    )
    return configs


def load_env_config(
    config_dir: str | Path = "config",
    env: ProviderEnv = ProviderEnv.BACKTEST,
) -> dict[str, ProviderConfig]:
    """按环境加载配置文件。

    查找顺序：
    1. config/providers-{env}.yaml
    2. config/providers.yaml (fallback)
    """
    config_dir = Path(config_dir)
    env_path = config_dir / f"providers-{env.value}.yaml"
    default_path = config_dir / "providers.yaml"

    if env_path.exists():
        return load_provider_configs(env_path)
    if default_path.exists():
        logger.info(
            "Env config %s not found, falling back to %s",
            env_path.name,
            default_path.name,
        )
        return load_provider_configs(default_path)

    raise FileNotFoundError(
        f"No config found: tried {env_path} and {default_path}"
    )
