"""Alpha158 表达式 DSL — Tokenizer / Parser / AST。

仅负责解析和结构分析（m1a）；求值逻辑在 m1b（Evaluator）中实现。

公开 API：
    parse(src: str) -> Node          -- 解析表达式，返回 AST 根节点
    tokenize(src: str) -> list[Token]  -- 暴露给测试和调试
    Node                             -- AST 基类
    ExpressionError                  -- 所有解析 / 结构错误的基类

支持语法（Alpha158 子集）：
    变量    : $close $open $high $low $volume $vwap（及短别名 c o h l v，大小写不敏感）
    数字    : 整数（Python int）/ 浮点（Python float）
    运算符  : + - * /  以及比较 < > <= >= == !=（用于 IF 条件）
    函数调用: MA($close, 20)、REF($close, 5)、IF($close > $open, 1, 0) …
    括号 / 一元负

不支持（m1a 显式排除）：
    ** 幂运算、位运算 & | ^ ~ ！、逻辑 and/or/not
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator

# ---------------------------------------------------------------------------
# 错误类
# ---------------------------------------------------------------------------

class ExpressionError(Exception):
    """解析 / 结构错误基类。"""


class TokenizerError(ExpressionError):
    """Tokenizer 层错误（非法字符等）。"""

    def __init__(self, msg: str, pos: int) -> None:
        super().__init__(f"{msg} (pos {pos})")
        self.pos = pos


class ParseError(ExpressionError):
    """Parser 层错误（语法 / 参数个数等）。"""

    def __init__(self, msg: str, pos: int | None = None) -> None:
        location = f" (pos {pos})" if pos is not None else ""
        super().__init__(f"{msg}{location}")
        self.pos = pos


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------

class TokenType(Enum):
    NUMBER   = auto()   # 123 / 4.5
    VAR      = auto()   # $close 或短别名 c/o/h/l/v
    IDENT    = auto()   # 函数名 MA / REF …
    LPAREN   = auto()   # (
    RPAREN   = auto()   # )
    COMMA    = auto()   # ,
    OP       = auto()   # + - * / < > <= >= == !=
    EOF      = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: Any       # str / int / float
    pos: int         # 在源字符串中的起始字节偏移


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# 规范变量名映射（输入不区分大小写）
_VAR_CANON: dict[str, str] = {
    "$close": "CLOSE",  "$open": "OPEN",  "$high": "HIGH",
    "$low":   "LOW",    "$volume": "VOLUME", "$vwap": "VWAP",
    "c": "CLOSE", "o": "OPEN", "h": "HIGH",
    "l": "LOW",   "v": "VOLUME",
}

# Token 正则表（顺序重要：长 token 优先）
_TOKEN_RE = re.compile(
    r"""
    (?P<SKIP>[ \t\r\n]+)              |  # 空白，跳过
    (?P<NUMBER>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)  |  # 数字
    (?P<VAR_LONG>\$[A-Za-z][A-Za-z0-9_]*)  |  # $close $open …
    (?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)      |  # 函数名或短别名
    (?P<OP2><=|>=|==|!=)              |  # 双字符运算符（先匹配）
    (?P<OP1>[+\-*/,()<>])                   # 单字符运算符/标点
    """,
    re.VERBOSE,
)

# 仅包含短别名的单字母集合（大小写不敏感）
_SHORT_ALIAS = frozenset("cohlv")


def tokenize(src: str) -> list[Token]:
    """将表达式字符串切分为 Token 列表。最后追加 EOF。"""
    tokens: list[Token] = []
    pos = 0
    length = len(src)

    while pos < length:
        m = _TOKEN_RE.match(src, pos)
        if m is None:
            raise TokenizerError(f"非法字符 {src[pos]!r}", pos)

        kind = m.lastgroup
        raw = m.group()
        start = pos
        pos = m.end()

        if kind == "SKIP":
            continue

        if kind == "NUMBER":
            # 保留原生类型：整数 vs 浮点
            val: int | float = int(raw) if "." not in raw and "e" not in raw.lower() else float(raw)
            tokens.append(Token(TokenType.NUMBER, val, start))

        elif kind == "VAR_LONG":
            canon = _VAR_CANON.get(raw.lower())
            if canon is None:
                raise TokenizerError(f"未知变量 {raw!r}", start)
            tokens.append(Token(TokenType.VAR, canon, start))

        elif kind == "IDENT":
            # 先检查是否是短别名（单字母且在集合中）
            if len(raw) == 1 and raw.lower() in _SHORT_ALIAS:
                canon = _VAR_CANON[raw.lower()]
                tokens.append(Token(TokenType.VAR, canon, start))
            else:
                # 函数名：规范化为大写
                tokens.append(Token(TokenType.IDENT, raw.upper(), start))

        elif kind in ("OP2", "OP1"):
            if raw == "(":
                tokens.append(Token(TokenType.LPAREN, raw, start))
            elif raw == ")":
                tokens.append(Token(TokenType.RPAREN, raw, start))
            elif raw == ",":
                tokens.append(Token(TokenType.COMMA, raw, start))
            else:
                tokens.append(Token(TokenType.OP, raw, start))

    tokens.append(Token(TokenType.EOF, None, pos))
    return tokens


# ---------------------------------------------------------------------------
# AST 节点
# ---------------------------------------------------------------------------

class Node(ABC):
    """AST 节点基类。所有子类为 frozen dataclass，自带 __eq__/__repr__。"""

    def walk(self) -> Iterator["Node"]:
        """前序 DFS 遍历：先 yield self，再递归 children（从左到右）。"""
        yield self
        for child in self._children():
            yield from child.walk()

    @abstractmethod
    def lookback(self) -> int:
        """返回该子树所需的最大历史回溯窗口（交易日数）。"""
        ...

    def _children(self) -> list["Node"]:
        """子节点列表，供 walk() 递归使用。子类按需重写。"""
        return []


@dataclass(frozen=True)
class Constant(Node):
    """数字常量节点。value 保留 Python 原生 int 或 float。"""
    value: int | float

    def lookback(self) -> int:
        return 0


@dataclass(frozen=True)
class Variable(Node):
    """变量节点。name 为规范大写形式（CLOSE/OPEN/HIGH/LOW/VOLUME/VWAP）。"""
    name: str   # 例如 "CLOSE"

    def lookback(self) -> int:
        return 0


@dataclass(frozen=True)
class UnaryOp(Node):
    """一元运算节点（目前只支持 -）。"""
    op: str
    operand: Node

    def lookback(self) -> int:
        return self.operand.lookback()

    def _children(self) -> list[Node]:
        return [self.operand]


@dataclass(frozen=True)
class BinaryOp(Node):
    """二元运算节点。

    lookback = max(left, right)：二元运算逐点对齐两序列，
    所需历史为两侧的最大值，而非相加。
    """
    op: str
    left: Node
    right: Node

    def lookback(self) -> int:
        return max(self.left.lookback(), self.right.lookback())

    def _children(self) -> list[Node]:
        return [self.left, self.right]


@dataclass(frozen=True)
class FunctionCall(Node):
    """函数调用节点。name 为规范大写形式。"""
    name: str
    args: tuple[Node, ...]   # frozen=True 要求可哈希，用 tuple

    def lookback(self) -> int:
        spec = _get_spec(self.name)   # 未知函数抛 ExpressionError

        win_idx = spec.rolling_arg_index

        # MAX/MIN 重载：检查 arg[win_idx] 是否为 int ≥ 2
        if spec.overloaded_max_min:
            win_node = self.args[win_idx] if win_idx < len(self.args) else None
            is_rolling = (
                isinstance(win_node, Constant)
                and isinstance(win_node.value, int)
                and win_node.value >= 2
            )
            if not is_rolling:
                return max(arg.lookback() for arg in self.args)
            # 走滚动分支
            win = int(win_node.value)  # type: ignore[union-attr]
            data_args = [a for i, a in enumerate(self.args) if i != win_idx]
            return win + max((a.lookback() for a in data_args), default=0)

        # QUANTILE 横截面：第二参数是分位数 q，不是滚动窗口
        if spec.cross_section:
            # lookback = 第一个数据参数的 lookback（其余参数忽略）
            return self.args[0].lookback() if self.args else 0

        # 纯 element-wise（无 rolling_arg_index）
        if win_idx is None:
            return max((arg.lookback() for arg in self.args), default=0)

        # 严格滚动函数：窗口 parse 阶段已校验为 Constant int ≥ 1
        win_node = self.args[win_idx]
        assert isinstance(win_node, Constant) and isinstance(win_node.value, int)
        win = win_node.value
        data_args = [a for i, a in enumerate(self.args) if i != win_idx]
        return win + max((a.lookback() for a in data_args), default=0)

    def _children(self) -> list[Node]:
        return list(self.args)


# ---------------------------------------------------------------------------
# FunctionSpec 注册表（m1a：仅元数据，无实现）
# ---------------------------------------------------------------------------

@dataclass
class FunctionSpec:
    """函数元数据。

    Attributes:
        name:               规范大写函数名
        min_arity:          最少参数个数
        max_arity:          最多参数个数（None = 不限）
        rolling_arg_index:  窗口参数的下标（None 表示无滚动窗口）
        overloaded_max_min: True 表示 MAX/MIN 重载（看 arg[rolling_arg_index] 类型决定）
        cross_section:      True 表示横截面语义（QUANTILE），lookback 不加窗口
        impl:               函数实现（m1b/m2 通过 register_impl 填入）
    """
    name: str
    min_arity: int
    max_arity: int | None
    rolling_arg_index: int | None = None
    overloaded_max_min: bool = False
    cross_section: bool = False
    impl: Callable[..., Any] | None = field(default=None, compare=False)


# 注册表本体
_REGISTRY: dict[str, FunctionSpec] = {}


def _reg(
    name: str,
    min_arity: int,
    max_arity: int | None,
    rolling_arg_index: int | None = None,
    overloaded_max_min: bool = False,
    cross_section: bool = False,
) -> None:
    _REGISTRY[name] = FunctionSpec(
        name=name,
        min_arity=min_arity,
        max_arity=max_arity,
        rolling_arg_index=rolling_arg_index,
        overloaded_max_min=overloaded_max_min,
        cross_section=cross_section,
    )


# --- 滚动函数：rolling_arg_index=1（arg0=data, arg1=window） ---
for _n in ("REF", "DELAY", "MA", "MEAN", "STD", "SUM", "DELTA",
           "EMA", "SLOPE", "RSQUARE", "RESI", "IDXMAX", "IDXMIN"):
    _reg(_n, min_arity=2, max_arity=2, rolling_arg_index=1)

# --- CORR：arg0=data1, arg1=data2, arg2=window ---
_reg("CORR", min_arity=3, max_arity=3, rolling_arg_index=2)

# --- element-wise 函数 ---
_reg("ABS",  min_arity=1, max_arity=1)
_reg("LOG",  min_arity=1, max_arity=1)
_reg("SIGN", min_arity=1, max_arity=1)
_reg("RANK", min_arity=1, max_arity=1, cross_section=True)   # 横截面，按日做 cross-section rank

# --- IF：3 参数，全部是数据参数 ---
_reg("IF", min_arity=3, max_arity=3)

# --- MAX/MIN 重载 ---
_reg("MAX", min_arity=2, max_arity=2, rolling_arg_index=1, overloaded_max_min=True)
_reg("MIN", min_arity=2, max_arity=2, rolling_arg_index=1, overloaded_max_min=True)

# --- QUANTILE 横截面：(data, q)，q 是分位数 0~1 的浮点，非窗口 ---
_reg("QUANTILE", min_arity=2, max_arity=2, cross_section=True)


def _get_spec(name: str) -> FunctionSpec:
    """获取函数规格，未知函数抛 ExpressionError。"""
    spec = _REGISTRY.get(name)
    if spec is None:
        raise ExpressionError(f"未知函数 {name!r}，支持的函数：{sorted(_REGISTRY)}")
    return spec


def register_impl(name: str, fn: Callable[..., Any]) -> None:
    """注册函数实现（由 m1b / m2 调用）。"""
    spec = _get_spec(name)
    # FunctionSpec 是普通 dataclass（非 frozen），可以直接赋值
    spec.impl = fn


def _validate_function_args(call: "FunctionCall", pos: int) -> None:
    """在 parse 阶段对 FunctionCall 做结构性 + 函数特化校验。

    不合规的 AST 直接 raise ParseError，不允许流转到 lookback / evaluator。
    校验项：
      1. 严格滚动函数的窗口位必须是正整数 Constant（MAX/MIN 重载不校验）
      2. QUANTILE 的分位数 q 必须是 [0,1] 范围内的数字 Constant
    """
    spec = _REGISTRY[call.name]

    # --- 1. 严格滚动函数：窗口必须是正整数常量 ---
    if spec.rolling_arg_index is not None and not spec.overloaded_max_min:
        win_node = call.args[spec.rolling_arg_index]
        if not (
            isinstance(win_node, Constant)
            and isinstance(win_node.value, int)
            and win_node.value >= 1
        ):
            raise ParseError(
                f"函数 {call.name} 的窗口参数（位置 {spec.rolling_arg_index}）"
                f"必须是 ≥1 的整数常量，实际为 {win_node!r}",
                pos,
            )

    # --- 2. QUANTILE q 参数：必须是 [0,1] 范围内的数字常量 ---
    if call.name == "QUANTILE":
        q_node = call.args[1]
        if not isinstance(q_node, Constant) or not isinstance(q_node.value, (int, float)):
            raise ParseError(
                f"QUANTILE 的第二参数 q 必须是数字常量，实际为 {q_node!r}",
                pos,
            )
        if not (0 <= q_node.value <= 1):
            raise ParseError(
                f"QUANTILE 的分位数 q 必须在 [0,1] 范围内，实际为 {q_node.value}",
                pos,
            )


# ---------------------------------------------------------------------------
# Parser（递归下降）
# ---------------------------------------------------------------------------
# 优先级（从低到高）：
#   比较：< > <= >= == !=
#   加减：+ -
#   乘除：* /
#   一元负：-
#   基础：数字 / 变量 / ( expr ) / 函数调用

_COMPARE_OPS = frozenset({"<", ">", "<=", ">=", "==", "!="})
_ADDITIVE_OPS = frozenset({"+", "-"})
_MULTIPLICATIVE_OPS = frozenset({"*", "/"})


class _Parser:
    """内部 Parser 类。每次 parse() 调用创建新实例。"""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # --- 游标辅助 ---

    def _peek(self) -> Token:
        return self._tokens[self._pos]

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, ttype: TokenType, value: Any = None) -> Token:
        tok = self._peek()
        if tok.type != ttype:
            raise ParseError(
                f"期望 {ttype.name}，实际得到 {tok.type.name} {tok.value!r}",
                tok.pos,
            )
        if value is not None and tok.value != value:
            raise ParseError(
                f"期望 {value!r}，实际得到 {tok.value!r}",
                tok.pos,
            )
        return self._advance()

    # --- 递归下降 ---

    def parse(self) -> Node:
        node = self._comparison()
        if self._peek().type != TokenType.EOF:
            tok = self._peek()
            raise ParseError(f"多余的字符 {tok.value!r}", tok.pos)
        return node

    def _comparison(self) -> Node:
        left = self._additive()
        if self._peek().type == TokenType.OP and self._peek().value in _COMPARE_OPS:
            op = self._advance().value
            right = self._additive()
            # 显式拒绝链式比较（0 < x < 1 这种），避免静默左结合误读
            if self._peek().type == TokenType.OP and self._peek().value in _COMPARE_OPS:
                tok = self._peek()
                raise ParseError(
                    "不支持链式比较（如 a < b < c），请拆成显式的 IF/AND 组合",
                    tok.pos,
                )
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def _additive(self) -> Node:
        left = self._multiplicative()
        while self._peek().type == TokenType.OP and self._peek().value in _ADDITIVE_OPS:
            op = self._advance().value
            right = self._multiplicative()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def _multiplicative(self) -> Node:
        left = self._unary()
        while (
            self._peek().type == TokenType.OP
            and self._peek().value in _MULTIPLICATIVE_OPS
        ):
            op = self._advance().value
            right = self._unary()
            left = BinaryOp(op=op, left=left, right=right)
        return left

    def _unary(self) -> Node:
        tok = self._peek()
        if tok.type == TokenType.OP and tok.value == "-":
            self._advance()
            operand = self._unary()   # 右结合
            return UnaryOp(op="-", operand=operand)
        return self._primary()

    def _primary(self) -> Node:
        tok = self._peek()

        # 数字常量
        if tok.type == TokenType.NUMBER:
            self._advance()
            return Constant(value=tok.value)

        # 变量
        if tok.type == TokenType.VAR:
            self._advance()
            return Variable(name=tok.value)

        # 括号表达式
        if tok.type == TokenType.LPAREN:
            self._advance()
            node = self._comparison()
            self._expect(TokenType.RPAREN)
            return node

        # 函数调用（IDENT 后紧跟 LPAREN）
        if tok.type == TokenType.IDENT:
            name = tok.value
            self._advance()

            # 校验函数名
            spec = _get_spec(name)   # 未知 → ParseError（ExpressionError 子类）

            self._expect(TokenType.LPAREN)
            args: list[Node] = []
            if self._peek().type != TokenType.RPAREN:
                args.append(self._comparison())
                while self._peek().type == TokenType.COMMA:
                    self._advance()
                    args.append(self._comparison())
            self._expect(TokenType.RPAREN)

            # 检查 arity
            n = len(args)
            if n < spec.min_arity or (spec.max_arity is not None and n > spec.max_arity):
                expected = (
                    f"{spec.min_arity}"
                    if spec.min_arity == spec.max_arity
                    else f"{spec.min_arity}~{spec.max_arity}"
                )
                raise ParseError(
                    f"函数 {name} 期望 {expected} 个参数，实际得到 {n} 个",
                    tok.pos,
                )

            call = FunctionCall(name=name, args=tuple(args))
            _validate_function_args(call, tok.pos)
            return call

        raise ParseError(
            f"意外的 token {tok.type.name} {tok.value!r}",
            tok.pos,
        )


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def parse(src: str) -> Node:
    """解析 Alpha158 表达式字符串，返回 AST 根节点。

    Args:
        src: 表达式字符串，例如 "MA(REF($close,5),20)"

    Returns:
        AST 根节点（Node 子类实例）

    Raises:
        TokenizerError: 非法字符
        ParseError:     语法错误 / 未知函数 / 参数个数错
        ExpressionError: 其他结构错误
    """
    tokens = tokenize(src)
    return _Parser(tokens).parse()
