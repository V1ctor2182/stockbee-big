"""Alpha158 因子注册表 — YAML 加载 + AST cache + max_lookback。

加载 config/factors-alpha158.yaml，展开 rolling 模板的笛卡尔积，
构建 name → expression 映射。不负责求值（m6 LocalFactorProvider 的工作）。

公开 API：
    Alpha158(yaml_path=None)
    .get_expression(name) -> Node      # 解析并缓存 AST
    .get_expression_str(name) -> str   # 原始表达式字符串
    .max_lookback(name) -> int         # AST.lookback()
    .list_factor_names() -> list[str]  # 158 个名字，固定顺序
    len(alpha158) -> int               # 158
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import yaml

from .expression_engine import Node, parse


_DEFAULT_YAML = Path(__file__).resolve().parent.parent.parent.parent / "config" / "factors-alpha158.yaml"


class Alpha158:
    """Alpha158 因子注册表。

    延迟解析：get_expression() 首次调用时 parse + cache，
    避免初始化时一次性解析全部 158 个表达式。
    """

    def __init__(self, yaml_path: Path | None = None) -> None:
        path = yaml_path or _DEFAULT_YAML
        with open(path) as f:
            cfg = yaml.safe_load(f)

        self._factors: OrderedDict[str, str] = OrderedDict()
        self._ast_cache: dict[str, Node] = {}

        groups = cfg["groups"]

        for entry in groups.get("kbar", []):
            self._factors[entry["name"]] = entry["expr"]

        for entry in groups.get("price", []):
            self._factors[entry["name"]] = entry["expr"]

        rolling = groups.get("rolling", {})
        windows = rolling.get("windows", [])
        for op in rolling.get("operators", []):
            for w in windows:
                name = op["name_template"].replace("{w}", str(w))
                expr = op["expr_template"].replace("{w}", str(w))
                self._factors[name] = expr

    def get_expression(self, name: str) -> Node:
        """返回 name 对应的 AST（缓存）。未知 name → KeyError。"""
        if name not in self._factors:
            raise KeyError(f"未知因子 {name!r}，共 {len(self._factors)} 个因子可用")
        if name not in self._ast_cache:
            self._ast_cache[name] = parse(self._factors[name])
        return self._ast_cache[name]

    def get_expression_str(self, name: str) -> str:
        """返回 name 对应的原始表达式字符串。未知 name → KeyError。"""
        if name not in self._factors:
            raise KeyError(f"未知因子 {name!r}")
        return self._factors[name]

    def max_lookback(self, name: str) -> int:
        """返回该因子所需的最大历史回溯窗口（交易日数）。"""
        return self.get_expression(name).lookback()

    def list_factor_names(self) -> list[str]:
        """全部因子名，顺序固定（KBAR → price → rolling 按 operator × window 展开）。"""
        return list(self._factors.keys())

    def __len__(self) -> int:
        return len(self._factors)

    def __contains__(self, name: str) -> bool:
        return name in self._factors
