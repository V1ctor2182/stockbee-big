# tests/test_factor_data.py
"""m1a — Tokenizer / Parser / AST 单元测试。

不跑 OHLCV 数据，只验证：
  - Token 流正确性
  - AST 结构形状
  - lookback() 数值
  - walk() DFS 顺序

运行方式（仓库根目录）：
    pytest tests/test_factor_data.py -v
"""

import pytest

from stockbee.factor_data.expression_engine import (
    BinaryOp,
    Constant,
    ExpressionError,
    FunctionCall,
    Node,
    ParseError,
    Token,
    TokenType,
    TokenizerError,
    UnaryOp,
    Variable,
    _REGISTRY,
    parse,
    tokenize,
)


# ===========================================================================
# Tokenizer（6 个 case）
# ===========================================================================

class TestTokenizer:
    def test_integer_and_float_distinct(self):
        """整数和浮点保留原生 Python 类型。"""
        toks = tokenize("10 2.5")
        assert toks[0].type == TokenType.NUMBER
        assert isinstance(toks[0].value, int) and toks[0].value == 10
        assert toks[1].type == TokenType.NUMBER
        assert isinstance(toks[1].value, float) and toks[1].value == 2.5

    def test_var_long_and_short_alias_normalize(self):
        """$close / $Close / c / C 全部归一化为 CLOSE；v -> VOLUME。"""
        for expr in ("$close", "$Close", "$CLOSE"):
            toks = tokenize(expr)
            assert toks[0].type == TokenType.VAR
            assert toks[0].value == "CLOSE"

        toks_c = tokenize("c")
        assert toks_c[0].type == TokenType.VAR and toks_c[0].value == "CLOSE"

        toks_C = tokenize("C")
        assert toks_C[0].type == TokenType.VAR and toks_C[0].value == "CLOSE"

        toks_v = tokenize("v")
        assert toks_v[0].type == TokenType.VAR and toks_v[0].value == "VOLUME"

    def test_operators_parens_comma(self):
        """运算符、括号、逗号均被正确 Token 化。"""
        toks = tokenize("(a+b, c<=d)")
        types = [t.type for t in toks if t.type != TokenType.EOF]
        assert TokenType.LPAREN in types
        assert TokenType.RPAREN in types
        assert TokenType.COMMA in types
        assert TokenType.OP in types

    def test_whitespace_skipped(self):
        """空白字符不产生 Token。"""
        toks = tokenize("  MA ( $close ,  20 )  ")
        types = [t.type for t in toks]
        assert TokenType.IDENT == types[0]
        assert TokenType.EOF == types[-1]
        # 没有 SKIP 类型的 Token（空白已丢弃）
        assert all(t.type != TokenType.NUMBER or t.value in (20,) for t in toks
                   if t.type == TokenType.NUMBER)

    def test_illegal_char_raises(self):
        """非法字符抛 TokenizerError。"""
        with pytest.raises(TokenizerError):
            tokenize("$close @ 1")

    def test_dollar_alone_raises(self):
        """单独的 $ 或 $123 等非法变量抛 TokenizerError。"""
        with pytest.raises(TokenizerError):
            tokenize("$123")

        with pytest.raises(TokenizerError):
            tokenize("$ ")


# ===========================================================================
# Parser（10 个 case）
# ===========================================================================

class TestParser:
    def test_basic_binop(self):
        """$close + 1 生成 BinaryOp(+, Variable(CLOSE), Constant(1))。"""
        node = parse("$close + 1")
        assert isinstance(node, BinaryOp)
        assert node.op == "+"
        assert isinstance(node.left, Variable) and node.left.name == "CLOSE"
        assert isinstance(node.right, Constant) and node.right.value == 1

    def test_operator_precedence(self):
        """a + b * c 中乘法优先：BinaryOp(+, a, BinaryOp(*, b, c))。"""
        node = parse("$close + $open * $high")
        assert isinstance(node, BinaryOp) and node.op == "+"
        assert isinstance(node.right, BinaryOp) and node.right.op == "*"

    def test_paren_overrides_precedence(self):
        """(a + b) * c：括号改变优先级。"""
        node = parse("($close + $open) * $high")
        assert isinstance(node, BinaryOp) and node.op == "*"
        assert isinstance(node.left, BinaryOp) and node.left.op == "+"

    def test_unary_minus(self):
        """一元负 -$close 和 -MA($close, 20) 均正确解析。"""
        n1 = parse("-$close")
        assert isinstance(n1, UnaryOp) and n1.op == "-"
        assert isinstance(n1.operand, Variable)

        n2 = parse("-MA($close, 20)")
        assert isinstance(n2, UnaryOp) and n2.op == "-"
        assert isinstance(n2.operand, FunctionCall) and n2.operand.name == "MA"

    def test_nested_function_call(self):
        """MA(REF($close,5),20) 解析为嵌套 FunctionCall。"""
        node = parse("MA(REF($close,5),20)")
        assert isinstance(node, FunctionCall) and node.name == "MA"
        inner = node.args[0]
        assert isinstance(inner, FunctionCall) and inner.name == "REF"
        assert isinstance(inner.args[0], Variable) and inner.args[0].name == "CLOSE"
        assert isinstance(inner.args[1], Constant) and inner.args[1].value == 5
        assert isinstance(node.args[1], Constant) and node.args[1].value == 20

    def test_function_case_insensitive(self):
        """ma($close, 20) 与 MA($close, 20) 解析结果相同。"""
        n1 = parse("MA($close, 20)")
        n2 = parse("ma($close, 20)")
        assert n1 == n2

    def test_unknown_function_raises(self):
        """未知函数名抛 ExpressionError（ParseError 是其子类）。"""
        with pytest.raises(ExpressionError):
            parse("UNKNOWN($close, 10)")

    def test_wrong_arity_raises(self):
        """参数个数不对抛 ParseError。"""
        with pytest.raises(ParseError):
            parse("MA($close)")        # 少一个参数
        with pytest.raises(ParseError):
            parse("ABS($close, $open)")  # 多一个参数

    def test_trailing_garbage_raises(self):
        """表达式后有多余字符抛 ParseError。"""
        with pytest.raises(ParseError):
            parse("$close + 1 garbage")

    def test_if_with_comparison(self):
        """IF($close > $open, 1, 0) 完整解析。"""
        node = parse("IF($close > $open, 1, 0)")
        assert isinstance(node, FunctionCall) and node.name == "IF"
        cond = node.args[0]
        assert isinstance(cond, BinaryOp) and cond.op == ">"
        assert isinstance(node.args[1], Constant) and node.args[1].value == 1
        assert isinstance(node.args[2], Constant) and node.args[2].value == 0

    def test_chained_comparison_rejected(self):
        """链式比较（0 < $close < 1）必须被显式拒绝，防止静默左结合误读。"""
        with pytest.raises(ParseError):
            parse("IF(0 < $close < 1, 1, 0)")

    def test_non_constant_rolling_window_raises_at_parse(self):
        """严格滚动函数的窗口必须是正整数 Constant，parse 阶段就 raise。"""
        # 窗口是 Variable
        with pytest.raises(ParseError):
            parse("MA($close, $open)")
        # 窗口是嵌套表达式
        with pytest.raises(ParseError):
            parse("MA($close, REF($open, 5))")
        # 窗口是 float
        with pytest.raises(ParseError):
            parse("MA($close, 1.5)")
        # 窗口是 0（< 1）
        with pytest.raises(ParseError):
            parse("MA($close, 0)")
        # 负窗口（UnaryOp 非 Constant）
        with pytest.raises(ParseError):
            parse("MA($close, -5)")

    def test_quantile_q_must_be_constant(self):
        """QUANTILE 的 q 参数必须是数字 Constant，非常量直接 ParseError。"""
        with pytest.raises(ParseError):
            parse("QUANTILE($close, $open)")
        with pytest.raises(ParseError):
            parse("QUANTILE($close, REF($open, 5))")

    def test_quantile_q_out_of_range(self):
        """QUANTILE 的 q 参数必须在 [0,1]，否则 ParseError。"""
        with pytest.raises(ParseError):
            parse("QUANTILE($close, 1.5)")
        with pytest.raises(ParseError):
            parse("QUANTILE($close, 2)")

    def test_quantile_q_boundary_valid(self):
        """QUANTILE 的 q 边界值 0 和 1 都合法。"""
        assert parse("QUANTILE($close, 0)").lookback() == 0
        assert parse("QUANTILE($close, 1)").lookback() == 0
        assert parse("QUANTILE($close, 0.5)").lookback() == 0


# ===========================================================================
# AST walk / lookback（16 个 case）
# ===========================================================================

class TestASTWalk:
    def test_walk_preorder_dfs(self):
        """walk() 前序 DFS：先 FunctionCall，再 REF，再 Variable，再 Constant（窗口）。"""
        # MA(REF($close,5),20) 的前序：MA → REF → CLOSE → 5 → 20
        node = parse("MA(REF($close,5),20)")
        walked = list(node.walk())
        names = []
        for n in walked:
            if isinstance(n, FunctionCall):
                names.append(n.name)
            elif isinstance(n, Variable):
                names.append(n.name)
            elif isinstance(n, Constant):
                names.append(n.value)
        assert names == ["MA", "REF", "CLOSE", 5, 20]


class TestLookback:
    def test_primitive_zero(self):
        """常量和变量的 lookback 为 0。"""
        assert parse("$close").lookback() == 0
        assert parse("42").lookback() == 0
        assert parse("3.14").lookback() == 0

    def test_ma_single(self):
        """MA($close, 20) → 20。"""
        assert parse("MA($close, 20)").lookback() == 20

    def test_ref_single(self):
        """REF($close, 5) → 5。"""
        assert parse("REF($close, 5)").lookback() == 5

    def test_nested_ma_ref(self):
        """MA(REF($close,5),20) → 25（核心用例）。"""
        assert parse("MA(REF($close,5),20)").lookback() == 25

    def test_triple_nested(self):
        """STD(MA(REF($close,5),20),10) → 35。"""
        assert parse("STD(MA(REF($close,5),20),10)").lookback() == 35

    def test_binop_inside_rolling(self):
        """MA($close + $open, 20) → 20（binop 内部 lookback=0，不相加）。"""
        assert parse("MA($close + $open, 20)").lookback() == 20

    def test_corr_three_args(self):
        """CORR($close, $open, 10) → 10。"""
        assert parse("CORR($close, $open, 10)").lookback() == 10

    def test_corr_nested_first_arg(self):
        """CORR(REF($close,5), $volume, 10) → 15（5 + 10）。"""
        assert parse("CORR(REF($close,5), $volume, 10)").lookback() == 15

    def test_max_rolling(self):
        """MAX($close, 10) → 10（第二参数 int ≥ 2 → 滚动）。"""
        assert parse("MAX($close, 10)").lookback() == 10

    def test_max_elementwise(self):
        """MAX($close, $open) → 0（第二参数是 Variable → element-wise）。"""
        assert parse("MAX($close, $open)").lookback() == 0

    def test_max_float_elementwise(self):
        """MAX($close, 1.5) → 0（float 字面量，非 int → element-wise）。"""
        assert parse("MAX($close, 1.5)").lookback() == 0

    def test_binop_max_not_sum(self):
        """REF($close,5) + MA($close,20) → 20（binop 取 max，不相加）。"""
        assert parse("REF($close,5) + MA($close,20)").lookback() == 20

    def test_rank_zero(self):
        """RANK($close) → 0（横截面，无时序窗口）。"""
        assert parse("RANK($close)").lookback() == 0

    def test_quantile_cross_section(self):
        """QUANTILE($close, 0.5) → 0（横截面，q 不是窗口）。"""
        assert parse("QUANTILE($close, 0.5)").lookback() == 0

    def test_if_max_of_all_args(self):
        """IF($close > REF($close,5), MA($close,20), 0) → 20。"""
        assert parse("IF($close > REF($close,5), MA($close,20), 0)").lookback() == 20



# ===========================================================================
# FunctionSpec contract（保证 m2 Evaluator 能依赖 metadata 做派发）
# ===========================================================================

class TestFunctionSpecContract:
    def test_rank_is_cross_section(self):
        """RANK 必须带 cross_section=True，m2 Evaluator 才能走 groupby(date) 路径。"""
        assert _REGISTRY["RANK"].cross_section is True

    def test_quantile_is_cross_section(self):
        """QUANTILE 必须带 cross_section=True。"""
        assert _REGISTRY["QUANTILE"].cross_section is True

    def test_rolling_functions_not_cross_section(self):
        """时序滚动函数不应被标记为 cross_section。"""
        for name in ("MA", "REF", "STD", "SUM", "DELTA", "EMA",
                     "SLOPE", "RSQUARE", "RESI", "CORR",
                     "IDXMAX", "IDXMIN"):
            assert _REGISTRY[name].cross_section is False, (
                f"{name} 不应被标记为 cross_section"
            )
