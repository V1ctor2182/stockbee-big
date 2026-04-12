"""因子数据模块（factor_data）。

包含：
- expression_engine: 表达式 DSL 的 Tokenizer / Parser / AST（m1a）
                     + Evaluator + 基础/高级/TS 函数（m1b/m2/m3）
- alpha158: Alpha158 因子注册表 + max_lookback（m3）
- parquet_factor: 预计算因子 Parquet 存储（m4）
"""

from .alpha158 import Alpha158
from .expression_engine import (
    Evaluator,
    ExpressionError,
    Node,
    evaluate,
    parse,
    tokenize,
)
from .parquet_factor import ParquetFactorStore

__all__ = [
    "Alpha158",
    "Evaluator",
    "ExpressionError",
    "Node",
    "ParquetFactorStore",
    "evaluate",
    "parse",
    "tokenize",
]
