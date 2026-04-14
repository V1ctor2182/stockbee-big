"""因子数据模块（factor_data）。

包含：
- expression_engine: 表达式 DSL 的 Tokenizer / Parser / AST（m1a）
                     + Evaluator + 基础/高级/TS 函数（m1b/m2/m3）
- alpha158: Alpha158 因子注册表 + max_lookback（m3）
- parquet_factor: 预计算因子 Parquet 存储（m4）
- ic_evaluator: IC / ICIR 离线评估（m5）
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
from .ic_evaluator import compute as compute_ic
from .local_provider import ICUniverse, LocalFactorProvider
from .parquet_factor import ParquetFactorStore

__all__ = [
    "Alpha158",
    "Evaluator",
    "ExpressionError",
    "ICUniverse",
    "LocalFactorProvider",
    "Node",
    "ParquetFactorStore",
    "compute_ic",
    "evaluate",
    "parse",
    "tokenize",
]
