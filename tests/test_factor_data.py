# tests/test_factor_data.py
"""m1a — Tokenizer / Parser / AST 单元测试
m1b — Evaluator + 12 基础函数数值测试。

m1a 不跑 OHLCV 数据，只验证：
  - Token 流正确性
  - AST 结构形状
  - lookback() 数值
  - walk() DFS 顺序

m1b 造 2 tickers 合成 OHLCV，验证：
  - 变量映射（CLOSE→adj_close，OPEN 非复权等）
  - 每个基础函数数值正确
  - ticker 隔离（rolling/shift 不跨 ticker 串数据）
  - MAX/MIN overload 两条分支

运行方式（仓库根目录）：
    pytest tests/test_factor_data.py -v
"""

import numpy as np
import pandas as pd
import pytest

from stockbee.factor_data.expression_engine import (
    BinaryOp,
    Constant,
    Evaluator,
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
    evaluate,
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


# ===========================================================================
# m1b — Evaluator + 基础函数（含合成 OHLCV 数值验证）
# ===========================================================================
#
# Panel 设计（2 tickers × 10 天，日期对齐）：
#   AAA: close = 1..10 递增
#   BBB: close = 20..11 递减
#   adj_close ≠ close（验证 CLOSE→adj_close 映射）：adj_close = close * 2
#   open = close - 0.5（用于 MAX/MIN element-wise 和非复权验证）
#   volume = 100..1000（AAA），200..2000（BBB）
# ---------------------------------------------------------------------------

_DATES = pd.date_range("2024-01-01", periods=10, freq="D")


def _make_panel(include_vwap: bool = False) -> pd.DataFrame:
    """造 2 tickers × 10 天合成 OHLCV。索引 (date, ticker)。"""
    rows = []
    for ticker, close_series in (
        ("AAA", list(range(1, 11))),              # 1..10
        ("BBB", list(range(20, 10, -1))),         # 20..11
    ):
        for i, date in enumerate(_DATES):
            close = close_series[i]
            row = {
                "date":      date,
                "ticker":    ticker,
                "open":      close - 0.5,          # 非复权，MAX elem 测试用
                "high":      close + 0.5,
                "low":       close - 1.0,
                "close":     close,                # 非复权原价
                "adj_close": close * 2.0,          # 复权价 = 2 × close
                "volume":    (100 if ticker == "AAA" else 200) * (i + 1),
            }
            if include_vwap:
                row["vwap"] = close + 0.1
            rows.append(row)

    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
    return df


@pytest.fixture
def panel() -> pd.DataFrame:
    return _make_panel()


@pytest.fixture
def panel_with_vwap() -> pd.DataFrame:
    return _make_panel(include_vwap=True)


def _aaa(s: pd.Series) -> pd.Series:
    """从 (date, ticker) 索引 Series 中取 AAA 子序列，按日期排序。"""
    return s.xs("AAA", level="ticker").sort_index()


def _bbb(s: pd.Series) -> pd.Series:
    return s.xs("BBB", level="ticker").sort_index()


# ---------------------------------------------------------------------------
# TestEvaluator — 基础求值 + 变量映射
# ---------------------------------------------------------------------------

class TestEvaluator:
    def test_constant_only_returns_scalar(self, panel):
        """Constant root → 返回原生 Python 标量。"""
        assert evaluate(parse("42"), panel) == 42
        assert evaluate(parse("3.14"), panel) == pytest.approx(3.14)

    def test_close_maps_to_adj_close(self, panel):
        """$close → panel['adj_close'] 而非 panel['close']。"""
        result = evaluate(parse("$close"), panel)
        expected = panel["adj_close"]
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_names=False
        )
        # 显式验证 AAA 第一行是 2.0（= 1 * 2），不是 1.0
        assert _aaa(result).iloc[0] == 2.0

    def test_open_is_not_adjusted(self, panel):
        """$open → panel['open']，不被当作 adj_open。"""
        result = evaluate(parse("$open"), panel)
        pd.testing.assert_series_equal(
            result.sort_index(), panel["open"].sort_index(), check_names=False
        )

    def test_high_low_volume_direct(self, panel):
        """HIGH/LOW/VOLUME 直连对应列。"""
        pd.testing.assert_series_equal(
            evaluate(parse("$high"), panel).sort_index(),
            panel["high"].sort_index(),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            evaluate(parse("$low"), panel).sort_index(),
            panel["low"].sort_index(),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            evaluate(parse("$volume"), panel).sort_index(),
            panel["volume"].sort_index(),
            check_names=False,
        )

    def test_vwap_when_present(self, panel_with_vwap):
        """带 vwap 列的 panel 可以引用 $vwap。"""
        result = evaluate(parse("$vwap"), panel_with_vwap)
        pd.testing.assert_series_equal(
            result.sort_index(),
            panel_with_vwap["vwap"].sort_index(),
            check_names=False,
        )

    def test_vwap_missing_raises(self, panel):
        """panel 无 vwap 列时引用 $vwap → ExpressionError。"""
        with pytest.raises(ExpressionError, match="vwap"):
            evaluate(parse("$vwap"), panel)

    def test_binop_arithmetic(self, panel):
        """$close + $open：基于复权 close + 非复权 open。"""
        result = evaluate(parse("$close + $open"), panel)
        expected = panel["adj_close"] + panel["open"]
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_names=False
        )

    def test_binop_with_constant(self, panel):
        """$close * 2 → adj_close * 2。"""
        result = evaluate(parse("$close * 2"), panel)
        expected = panel["adj_close"] * 2
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_names=False
        )

    def test_unary_negation(self, panel):
        """-$close → -adj_close。"""
        result = evaluate(parse("-$close"), panel)
        expected = -panel["adj_close"]
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_names=False
        )

    def test_evaluator_reuse_across_asts(self, panel):
        """同一 Evaluator 实例可连续求值多个 AST。"""
        ev = Evaluator(panel)
        r1 = ev.evaluate(parse("$close"))
        r2 = ev.evaluate(parse("MA($close, 3)"))
        assert isinstance(r1, pd.Series)
        assert isinstance(r2, pd.Series)
        # 第二次 evaluate 不影响第一次结果
        pd.testing.assert_series_equal(
            r1.sort_index(), panel["adj_close"].sort_index(), check_names=False
        )


# ---------------------------------------------------------------------------
# TestBasicFunctions — 12 个函数的数值正确性
# ---------------------------------------------------------------------------

class TestBasicFunctions:
    def test_ref_shift_positive_n(self, panel):
        """REF($close, 1)：每个 ticker 内沿时间 shift(1)，首行 NaN。"""
        result = evaluate(parse("REF($close, 1)"), panel)
        aaa = _aaa(result)
        assert pd.isna(aaa.iloc[0])
        # AAA: adj_close = 2,4,6,...；REF(1) 第 2 行 = 2（第 1 行 adj_close）
        assert aaa.iloc[1] == 2.0
        assert aaa.iloc[9] == 18.0   # 第 10 行 = 第 9 行 adj_close = 9*2

    def test_delay_equals_ref(self, panel):
        """DELAY 和 REF 是别名，结果必须完全相等。"""
        r_ref = evaluate(parse("REF($close, 3)"), panel)
        r_delay = evaluate(parse("DELAY($close, 3)"), panel)
        pd.testing.assert_series_equal(r_ref, r_delay)

    def test_ma_rolling_3(self, panel):
        """MA($close, 3) 每个 ticker 手算 + 前 2 行 NaN。"""
        result = evaluate(parse("MA($close, 3)"), panel)
        aaa = _aaa(result)
        # adj_close AAA = 2,4,6,8,10,12,14,16,18,20
        # MA(3) 第 3 行 = (2+4+6)/3 = 4
        assert pd.isna(aaa.iloc[0])
        assert pd.isna(aaa.iloc[1])
        assert aaa.iloc[2] == pytest.approx(4.0)
        assert aaa.iloc[3] == pytest.approx(6.0)
        assert aaa.iloc[9] == pytest.approx(18.0)  # (16+18+20)/3

        bbb = _bbb(result)
        # adj_close BBB = 40,38,36,34,32,30,28,26,24,22
        # MA(3) 第 3 行 = (40+38+36)/3 = 38
        assert bbb.iloc[2] == pytest.approx(38.0)
        assert bbb.iloc[9] == pytest.approx(24.0)  # (26+24+22)/3

    def test_mean_equals_ma(self, panel):
        """MEAN 是 MA 的别名。"""
        r_ma = evaluate(parse("MA($close, 5)"), panel)
        r_mean = evaluate(parse("MEAN($close, 5)"), panel)
        pd.testing.assert_series_equal(r_ma, r_mean)

    def test_std_ddof1(self, panel):
        """STD($close, 3) 用 ddof=1 样本方差。"""
        result = evaluate(parse("STD($close, 3)"), panel)
        aaa = _aaa(result)
        # adj_close AAA 第 1-3 行 = 2,4,6；样本 std = 2.0
        assert pd.isna(aaa.iloc[0])
        assert pd.isna(aaa.iloc[1])
        assert aaa.iloc[2] == pytest.approx(2.0)
        # numpy std ddof=1 验证
        expected = float(np.std([2.0, 4.0, 6.0], ddof=1))
        assert aaa.iloc[2] == pytest.approx(expected)

    def test_sum_rolling(self, panel):
        """SUM($volume, 2)：滚动 2 日和。"""
        result = evaluate(parse("SUM($volume, 2)"), panel)
        aaa = _aaa(result)
        # volume AAA = 100,200,300,400,...,1000
        assert pd.isna(aaa.iloc[0])
        assert aaa.iloc[1] == 300.0  # 100+200
        assert aaa.iloc[9] == 1900.0  # 900+1000

    def test_delta_equals_minus_ref(self, panel):
        """DELTA(x, n) = x - REF(x, n)。"""
        r_delta = evaluate(parse("DELTA($close, 2)"), panel)
        r_manual = evaluate(parse("$close - REF($close, 2)"), panel)
        pd.testing.assert_series_equal(r_delta, r_manual)
        # 具体数值：AAA adj_close 差 2 行 = 每个 4.0（等差 2×2）
        aaa = _aaa(r_delta)
        assert pd.isna(aaa.iloc[0])
        assert pd.isna(aaa.iloc[1])
        assert aaa.iloc[2] == pytest.approx(4.0)

    def test_abs_of_negative(self, panel):
        """ABS(-$close) == $close。"""
        r_abs = evaluate(parse("ABS(-$close)"), panel)
        r_close = evaluate(parse("$close"), panel)
        pd.testing.assert_series_equal(r_abs, r_close, check_names=False)
        assert (r_abs >= 0).all()

    def test_log_positive(self, panel):
        """LOG($close) vs np.log(adj_close)。"""
        result = evaluate(parse("LOG($close)"), panel)
        expected = pd.Series(
            np.log(panel["adj_close"].to_numpy()),
            index=panel.index,
        )
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_names=False
        )

    def test_sign_delta(self, panel):
        """SIGN($close - REF($close, 1)) 对递增 AAA 应全为 +1（除首行 NaN）。"""
        result = evaluate(parse("SIGN($close - REF($close, 1))"), panel)
        aaa = _aaa(result)
        assert pd.isna(aaa.iloc[0])
        # AAA adj_close 递增 2→4→6...，diff 恒正 → sign = 1
        assert (aaa.iloc[1:] == 1.0).all()

        bbb = _bbb(result)
        # BBB adj_close 递减 40→38→36...，diff 恒负 → sign = -1
        assert pd.isna(bbb.iloc[0])
        assert (bbb.iloc[1:] == -1.0).all()

    def test_max_rolling_branch(self, panel):
        """MAX($close, 3)：arg[1]=int≥2 → rolling max。"""
        result = evaluate(parse("MAX($close, 3)"), panel)
        aaa = _aaa(result)
        # AAA 递增 → rolling max 就是窗口末端
        assert pd.isna(aaa.iloc[0])
        assert pd.isna(aaa.iloc[1])
        assert aaa.iloc[2] == pytest.approx(6.0)    # max(2,4,6)
        assert aaa.iloc[9] == pytest.approx(20.0)   # max(16,18,20)

        bbb = _bbb(result)
        # BBB 递减 → rolling max 就是窗口起点
        assert bbb.iloc[2] == pytest.approx(40.0)   # max(40,38,36)
        assert bbb.iloc[9] == pytest.approx(26.0)   # max(26,24,22)

    def test_max_elementwise_branch_two_series(self, panel):
        """MAX($close, $open)：arg[1] 是 Variable → element-wise max。

        adj_close 始终 > open（AAA: 2..20 vs 0.5..9.5），结果等于 $close。
        """
        result = evaluate(parse("MAX($close, $open)"), panel)
        expected = panel["adj_close"]   # 因为 adj_close > open
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_names=False,
            check_dtype=False,
        )

    def test_max_elementwise_branch_with_float_constant(self, panel):
        """MAX($close, 5.0)：float 常量 → element-wise，不触发 rolling。"""
        result = evaluate(parse("MAX($close, 5.0)"), panel)
        aaa = _aaa(result)
        # AAA adj_close = 2..20，clip 下界到 5
        # iloc[0] = max(2, 5) = 5, iloc[1] = max(4, 5) = 5, iloc[2] = max(6, 5) = 6
        assert aaa.iloc[0] == 5.0
        assert aaa.iloc[1] == 5.0
        assert aaa.iloc[2] == 6.0
        assert aaa.iloc[9] == 20.0

    def test_min_rolling_branch(self, panel):
        """MIN($close, 3) rolling 分支。"""
        result = evaluate(parse("MIN($close, 3)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[2] == pytest.approx(2.0)   # min(2,4,6)
        assert aaa.iloc[9] == pytest.approx(16.0)  # min(16,18,20)

    def test_min_elementwise_with_integer_one(self, panel):
        """MIN($close, 1)：arg[1]=1 < 2 → element-wise（clip 上限到 1）。"""
        result = evaluate(parse("MIN($close, 1)"), panel)
        aaa = _aaa(result)
        # 所有 adj_close 都 ≥ 2，取 min 与 1 → 全部 1
        assert (aaa == 1.0).all()


# ---------------------------------------------------------------------------
# TestTickerIsolation — rolling / shift 不跨 ticker 串数据
# ---------------------------------------------------------------------------

class TestTickerIsolation:
    def test_ma_does_not_leak_across_tickers(self, panel):
        """AAA 第 3 行的 MA(3) 必须只用 AAA 的前 3 日，不掺 BBB。"""
        result = evaluate(parse("MA($close, 3)"), panel)

        # AAA 前 3 行 adj_close = 2,4,6 → mean = 4
        # 如果跨 ticker 串数据，BBB 的 40 会污染结果
        aaa_row3 = _aaa(result).iloc[2]
        assert aaa_row3 == pytest.approx(4.0)
        assert aaa_row3 != pytest.approx(
            (2 + 4 + 40) / 3, abs=1e-3
        ), "rolling 串了 BBB 的数据"

    def test_ref_shift_is_ticker_scoped(self, panel):
        """REF($close, 1) 对 BBB 首行 NaN，不把 AAA 的末值接过去。"""
        result = evaluate(parse("REF($close, 1)"), panel)
        bbb = _bbb(result)
        assert pd.isna(bbb.iloc[0])
        # BBB 第 2 行 = 40（BBB 首行 adj_close），不是 AAA 末行 20
        assert bbb.iloc[1] == 40.0


# ---------------------------------------------------------------------------
# TestEndToEnd — parse → evaluate 整链路
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_ma_diff(self, panel):
        """MA($close, 3) - MA($close, 5)：组合表达式跑通。"""
        result = evaluate(parse("MA($close, 3) - MA($close, 5)"), panel)
        # AAA adj_close = 2,4,6,8,10,12,14,16,18,20
        # 第 5 行（iloc=4）：MA3=(6+8+10)/3=8, MA5=(2+4+6+8+10)/5=6, diff=2
        aaa = _aaa(result)
        assert pd.isna(aaa.iloc[3])   # MA5 还没到位
        assert aaa.iloc[4] == pytest.approx(2.0)

    def test_nested_ma_ref(self, panel):
        """MA(REF($close, 1), 3)：嵌套函数调用。

        等价于：先对 close 做 shift(1)，再对结果做 MA(3)。
        AAA adj_close = 2,4,6,8,...；shift(1) = NaN,2,4,6,...
        MA(3) 需要窗口满 → iloc[3] = (2+4+6)/3 = 4
        """
        result = evaluate(parse("MA(REF($close, 1), 3)"), panel)
        aaa = _aaa(result)
        # 前 3 行 NaN（shift 吃 1 + 窗口前 2）
        assert pd.isna(aaa.iloc[0])
        assert pd.isna(aaa.iloc[1])
        assert pd.isna(aaa.iloc[2])
        assert aaa.iloc[3] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# TestEvaluatorErrors — 错误路径
# ---------------------------------------------------------------------------

class TestEvaluatorErrors:
    def test_non_multiindex_raises(self):
        """单层索引 panel → raise。"""
        df = pd.DataFrame({"adj_close": [1, 2, 3]})
        with pytest.raises(ExpressionError, match="MultiIndex"):
            Evaluator(df)

    def test_missing_ticker_level_raises(self):
        """MultiIndex 但无 ticker 一级 → raise。"""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["X"]],
            names=["date", "symbol"],   # 故意不叫 ticker
        )
        df = pd.DataFrame({"adj_close": [1, 2, 3]}, index=idx)
        with pytest.raises(ExpressionError, match="ticker"):
            Evaluator(df)

    def test_not_a_dataframe_raises(self):
        with pytest.raises(ExpressionError, match="DataFrame"):
            Evaluator([1, 2, 3])  # type: ignore[arg-type]

    def test_missing_column_raises_on_reference(self):
        """panel 无 adj_close 列时 evaluate $close → raise。"""
        idx = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=3), ["X"]],
            names=["date", "ticker"],
        )
        # 故意不放 adj_close
        df = pd.DataFrame({"open": [1, 2, 3]}, index=idx)
        with pytest.raises(ExpressionError, match="adj_close"):
            evaluate(parse("$close"), df)

    def test_three_level_index_raises(self):
        """(date, exchange, ticker) 三级索引必须被拒绝。

        否则 groupby(level='ticker') 只挑一级分组，rolling 会跨 exchange
        把同一 ticker 在不同 exchange 的行串成一段假连续序列，
        输出数值错误且无法被 unit test 捕获（bug 由 Codex P2 review 发现）。
        """
        idx = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01"), "NYSE",   "X"),
                (pd.Timestamp("2024-01-01"), "NASDAQ", "X"),
                (pd.Timestamp("2024-01-02"), "NYSE",   "X"),
                (pd.Timestamp("2024-01-02"), "NASDAQ", "X"),
            ],
            names=["date", "exchange", "ticker"],
        )
        df = pd.DataFrame({"adj_close": [1.0, 2.0, 3.0, 4.0]}, index=idx)
        with pytest.raises(ExpressionError, match="2 级"):
            Evaluator(df)


# ---------------------------------------------------------------------------
# TestEvaluatorEdgeCases — 回归测试（P2/P3 Codex review fixes）
# ---------------------------------------------------------------------------

class TestMaxMinScalarConstants:
    """MAX/MIN overload 在纯标量表达式上必须正常返回，不崩溃。

    Bug 由 Codex P3 review 发现：pure-scalar MAX(1, 2) 会错误地命中
    rolling 分支（因为 arg[1]=2 满足 int≥2 条件），然后把标量 1 传给
    _impl_max_rolling → .groupby(...) → AttributeError。
    修复：rolling 分支增加 isinstance(a, Series) 守卫，退化到 element-wise。
    """

    def test_max_two_int_constants(self, panel):
        """MAX(1, 2) 纯标量 → 返回 Python int 2。"""
        assert evaluate(parse("MAX(1, 2)"), panel) == 2
        # 类型保留为 Python int（不是 numpy 标量）
        assert type(evaluate(parse("MAX(1, 2)"), panel)) is int

    def test_min_two_int_constants(self, panel):
        """MIN(5, 3) 纯标量 → 3。"""
        assert evaluate(parse("MIN(5, 3)"), panel) == 3
        assert type(evaluate(parse("MIN(5, 3)"), panel)) is int

    def test_max_int_with_large_window_syntax(self, panel):
        """MAX(7, 10) 虽然 arg[1]=10 满足 rolling 语法，
        但 arg[0] 是标量，必须回落到 element-wise，返回 10。"""
        assert evaluate(parse("MAX(7, 10)"), panel) == 10

    def test_max_float_constants(self, panel):
        """MAX(1.5, 2.5)：float 常量本就走 elem-wise 分支。"""
        result = evaluate(parse("MAX(1.5, 2.5)"), panel)
        assert result == pytest.approx(2.5)
        assert isinstance(result, float)

    def test_max_series_with_rolling_window_still_works(self, panel):
        """回归测试：修复 P3 不能破坏正常的 rolling 分支。"""
        result = evaluate(parse("MAX($close, 3)"), panel)
        # 之前已经验证过的数值，必须依然正确
        assert _aaa(result).iloc[2] == pytest.approx(6.0)


# ===========================================================================
# m2 — 最小窗口校验（FunctionSpec.min_window）
# ===========================================================================
#
# SLOPE/RSQUARE/RESI/CORR/EMA 的 n=1 数学上退化（OLS 分母为 0，EMA α=1 不平滑），
# _validate_function_args 必须在 parse 阶段就 raise，不能让 n=1 流到 Evaluator。
# m1b 的 REF/MA/STD/SUM/DELTA/IDXMAX/IDXMIN 仍保持 min_window=1。
# ---------------------------------------------------------------------------

class TestMinWindowValidation:
    def test_slope_n1_rejected(self):
        """SLOPE 要求 min_window=2，n=1 → ParseError。"""
        with pytest.raises(ParseError, match="≥2"):
            parse("SLOPE($close, 1)")

    def test_ema_n1_rejected(self):
        """EMA 要求 min_window=2。"""
        with pytest.raises(ParseError, match="≥2"):
            parse("EMA($close, 1)")

    def test_corr_n1_rejected(self):
        """CORR 要求 min_window=2。"""
        with pytest.raises(ParseError, match="≥2"):
            parse("CORR($close, $open, 1)")

    def test_rsquare_n1_rejected(self):
        with pytest.raises(ParseError, match="≥2"):
            parse("RSQUARE($close, 1)")

    def test_resi_n1_rejected(self):
        with pytest.raises(ParseError, match="≥2"):
            parse("RESI($close, 1)")

    def test_ma_n1_still_allowed(self):
        """m1b 函数 min_window=1 保持不变，不应被本次改动破坏。"""
        # 不抛异常即通过
        parse("MA($close, 1)")
        parse("REF($close, 1)")

    def test_idxmax_n1_allowed(self):
        """IDXMAX 保持 min_window=1（单元素窗口 argmax=0 合法）。"""
        parse("IDXMAX($close, 1)")

    def test_min_window_field_exists_on_spec(self):
        """契约测试：FunctionSpec 必须暴露 min_window 字段。"""
        assert _REGISTRY["SLOPE"].min_window == 2
        assert _REGISTRY["MA"].min_window == 1
        assert _REGISTRY["CORR"].min_window == 2


# ===========================================================================
# m2 — 共享 rolling OLS kernel：SLOPE / RSQUARE / RESI
# ===========================================================================
#
# Panel 复用 m1b _make_panel() 的线性 adj_close：
#   AAA: adj_close = 2, 4, 6, ..., 20   （step = +2）
#   BBB: adj_close = 40, 38, 36, ..., 22 （step = -2）
# 完美线性 → 任意窗口 slope=±2.0，r²=1.0，resi=0.0。
# ---------------------------------------------------------------------------


def _make_panel_nonlinear() -> pd.DataFrame:
    """单 ticker Fibonacci 序列 + 第二个 ticker 保持线性，用于验证 r² < 1。"""
    rows = []
    fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    for i, date in enumerate(_DATES):
        # AAA: Fibonacci (非线性)
        rows.append({
            "date": date, "ticker": "AAA",
            "open": fib[i] - 0.5, "high": fib[i] + 0.5, "low": fib[i] - 1.0,
            "close": fib[i], "adj_close": float(fib[i]), "volume": 100 * (i + 1),
        })
        # BBB: 线性（对照组）
        close = 20 - i
        rows.append({
            "date": date, "ticker": "BBB",
            "open": close - 0.5, "high": close + 0.5, "low": close - 1.0,
            "close": close, "adj_close": float(close), "volume": 200 * (i + 1),
        })
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


@pytest.fixture
def panel_nonlinear() -> pd.DataFrame:
    return _make_panel_nonlinear()


class TestRollingOLS:
    def test_slope_linear_aaa_window_5(self, panel):
        """adj_close AAA = 2,4,...,20 (step=2) → 任意窗口 slope = 2.0。"""
        result = evaluate(parse("SLOPE($close, 5)"), panel)
        aaa = _aaa(result)
        # 前 n-1=4 行 NaN
        assert aaa.iloc[:4].isna().all()
        # 其余全为 2.0
        assert aaa.iloc[4:].to_numpy() == pytest.approx([2.0] * 6)

    def test_slope_linear_bbb_negative(self, panel):
        """BBB adj_close = 40,38,...,22 (step=-2) → slope = -2.0。"""
        result = evaluate(parse("SLOPE($close, 5)"), panel)
        bbb = _bbb(result)
        assert bbb.iloc[4:].to_numpy() == pytest.approx([-2.0] * 6)

    def test_slope_n_equals_2(self, panel):
        """最小窗口 n=2，两点 OLS = 相邻差。
        AAA step=2 → slope=2.0；BBB step=-2 → slope=-2.0。"""
        result = evaluate(parse("SLOPE($close, 2)"), panel)
        aaa = _aaa(result)
        bbb = _bbb(result)
        assert aaa.iloc[0] != aaa.iloc[0] or np.isnan(aaa.iloc[0])  # 第 0 行 NaN
        assert aaa.iloc[1:].to_numpy() == pytest.approx([2.0] * 9)
        assert bbb.iloc[1:].to_numpy() == pytest.approx([-2.0] * 9)

    def test_rsquare_linear_is_one(self, panel):
        """完美线性 → r² ≈ 1.0（两个 ticker 都一样）。"""
        result = evaluate(parse("RSQUARE($close, 5)"), panel)
        aaa = _aaa(result)
        bbb = _bbb(result)
        assert aaa.iloc[4:].to_numpy() == pytest.approx([1.0] * 6, abs=1e-12)
        assert bbb.iloc[4:].to_numpy() == pytest.approx([1.0] * 6, abs=1e-12)

    def test_rsquare_nonlinear_less_than_one(self, panel_nonlinear):
        """Fibonacci 输入 → r² < 1（非线性）；线性对照组 r² = 1。"""
        result = evaluate(parse("RSQUARE($close, 5)"), panel_nonlinear)
        aaa_r2 = _aaa(result).iloc[4:]   # Fibonacci
        bbb_r2 = _bbb(result).iloc[4:]   # 线性
        # AAA 非线性：每一个 r² 都 < 1（但仍 > 0.9，因为指数增长近似线性）
        assert (aaa_r2 < 1.0 - 1e-6).all()
        # BBB 线性对照：r² = 1
        assert bbb_r2.to_numpy() == pytest.approx([1.0] * 6, abs=1e-12)

    def test_resi_linear_is_zero(self, panel):
        """线性输入 → resi ≈ 0（两个 ticker 都一样）。"""
        result = evaluate(parse("RESI($close, 5)"), panel)
        aaa = _aaa(result)
        bbb = _bbb(result)
        assert aaa.iloc[4:].to_numpy() == pytest.approx([0.0] * 6, abs=1e-10)
        assert bbb.iloc[4:].to_numpy() == pytest.approx([0.0] * 6, abs=1e-10)

    def test_ols_nan_propagation_prefix(self, panel):
        """rolling 前 n-1 行 NaN 必须传播到 OLS 的所有输出（slope/rsquare/resi）。"""
        slope = evaluate(parse("SLOPE($close, 5)"), panel)
        rsq   = evaluate(parse("RSQUARE($close, 5)"), panel)
        resi  = evaluate(parse("RESI($close, 5)"), panel)
        assert _aaa(slope).iloc[:4].isna().all()
        assert _aaa(rsq).iloc[:4].isna().all()
        assert _aaa(resi).iloc[:4].isna().all()

    def test_resi_last_point_formula(self, panel_nonlinear):
        """RESI 必须返回最后一点的残差，不是整段残差序列。

        手算 AAA Fibonacci 窗口 [i=3..5] = [3, 5, 8]，n=3：
          Σy = 16, Σxy = Σ(i·y) − k_start·Σy where k_start = 3
               = (3·3 + 4·5 + 5·8) − 3·16
               = (9 + 20 + 40) − 48 = 21
          Σx = n(n-1)/2 = 3, Σxx = n(n-1)(2n-1)/6 = 5
          D  = n·Σxx − Σx² = 15 − 9 = 6
          slope     = (n·Σxy − Σx·Σy) / D = (3·21 − 3·16) / 6 = (63 − 48)/6 = 2.5
          intercept = (Σy − slope·Σx) / n = (16 − 7.5) / 3 = 8.5/3 ≈ 2.8333
          resi_t = y_t − (intercept + slope·(n−1))
                 = 8 − (2.8333 + 2.5·2) = 8 − 7.8333 = 0.1667
        """
        result = evaluate(parse("RESI($close, 3)"), panel_nonlinear)
        # AAA 第 i=5 行（窗口 [3..5]） → 对应 iloc[5]
        aaa_resi_5 = _aaa(result).iloc[5]
        assert aaa_resi_5 == pytest.approx(1.0 / 6.0, abs=1e-10)


# ===========================================================================
# m2 — EMA / IDXMAX / IDXMIN
# ===========================================================================

class TestEMA:
    def test_ema_hand_computed(self, panel):
        """n=3, alpha=2/4=0.5，IIR 递推 y_t = α·x_t + (1−α)·y_{t−1}。
        AAA adj_close = 2,4,6,... 前 2 行 NaN（min_periods=3），第 3 行起：
          y2 = 0.5·6 + 0.5·(0.5·4 + 0.5·2) = 3 + 0.5·3 = 4.5
          Wait: adjust=False 的 ewm 初始化用第一个观测值 y_0 = x_0 = 2，
                递推：y1 = 0.5·4 + 0.5·2 = 3
                      y2 = 0.5·6 + 0.5·3 = 4.5
                      y3 = 0.5·8 + 0.5·4.5 = 6.25
          min_periods=3 → 前 2 行输出 NaN，iloc[2]=4.5，iloc[3]=6.25
        """
        result = evaluate(parse("EMA($close, 3)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[0:2].isna().all()
        assert aaa.iloc[2] == pytest.approx(4.5)
        assert aaa.iloc[3] == pytest.approx(6.25)

    def test_ema_min_periods(self, panel):
        """min_periods=n → 前 n-1 行 NaN。"""
        result = evaluate(parse("EMA($close, 5)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[:4].isna().all()
        # iloc[4] 起有值
        assert not np.isnan(aaa.iloc[4])

    def test_ema_ticker_isolation(self, panel):
        """AAA/BBB EMA 不跨 ticker 串：两条 ticker 各自独立递推。"""
        result = evaluate(parse("EMA($close, 3)"), panel)
        aaa = _aaa(result)
        bbb = _bbb(result)
        # BBB adj_close = 40, 38, 36, ... 递减；前 2 行 NaN
        # y0=40, y1=0.5·38 + 0.5·40 = 39, y2=0.5·36 + 0.5·39 = 37.5
        assert bbb.iloc[2] == pytest.approx(37.5)
        # AAA 的 iloc[2] 应 = 4.5（与 BBB 独立）
        assert aaa.iloc[2] == pytest.approx(4.5)


class TestIdxMaxMin:
    def test_idxmax_rolling_3(self, panel):
        """AAA adj_close=2,4,...,20 递增，窗口 3 每行的 argmax 位置都是 2（最新 = 最大）。"""
        result = evaluate(parse("IDXMAX($close, 3)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[:2].isna().all()  # min_periods=3
        assert aaa.iloc[2:].to_numpy() == pytest.approx([2.0] * 8)

    def test_idxmin_rolling_3(self, panel):
        """AAA 递增序列 → 窗口 3 的 argmin 位置都是 0（最早 = 最小）。"""
        result = evaluate(parse("IDXMIN($close, 3)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[:2].isna().all()
        assert aaa.iloc[2:].to_numpy() == pytest.approx([0.0] * 8)

    def test_idxmax_bbb_decreasing(self, panel):
        """BBB 递减序列 → 窗口 3 argmax = 0（最早 = 最大）。"""
        result = evaluate(parse("IDXMAX($close, 3)"), panel)
        bbb = _bbb(result)
        assert bbb.iloc[2:].to_numpy() == pytest.approx([0.0] * 8)

    def test_idxmax_ties_first_position(self, panel):
        """平局（恒定序列）→ np.argmax 返回首个位置 0。
        造一个 AAA 与 BBB 都一致的常数 close，通过 SIGN($close - $close) + 1 构造常数 1。"""
        # 用表达式构造一个常数 Series：$close - $close + 1 → 全 1
        result = evaluate(parse("IDXMAX($close - $close + 1, 3)"), panel)
        aaa = _aaa(result)
        # 常数序列 [1,1,1] 的 argmax = 0
        assert aaa.iloc[2:].to_numpy() == pytest.approx([0.0] * 8)

    def test_idxmax_ticker_isolation(self, panel):
        """AAA 递增 / BBB 递减 在同一次调用中应各自独立 argmax。"""
        result = evaluate(parse("IDXMAX($close, 5)"), panel)
        # AAA 递增 → argmax = 4 (n-1)
        assert _aaa(result).iloc[4:].to_numpy() == pytest.approx([4.0] * 6)
        # BBB 递减 → argmax = 0
        assert _bbb(result).iloc[4:].to_numpy() == pytest.approx([0.0] * 6)

    def test_idxmax_n1_allowed(self, panel):
        """IDXMAX/IDXMIN 保持 min_window=1（单元素窗口 argmax=0 合法）。"""
        result = evaluate(parse("IDXMAX($close, 1)"), panel)
        # 每个窗口只有一个元素 → argmax = 0
        assert _aaa(result).to_numpy() == pytest.approx([0.0] * 10)


# ===========================================================================
# m2 — CORR（per-ticker rolling 相关系数）
# ===========================================================================

class TestCorrelation:
    def test_corr_identical_series(self, panel):
        """CORR($close, $close, 5) ≈ 1.0（恒等相关）。"""
        result = evaluate(parse("CORR($close, $close, 5)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[:4].isna().all()
        assert aaa.iloc[4:].to_numpy() == pytest.approx([1.0] * 6, abs=1e-10)

    def test_corr_anti_correlated(self, panel):
        """CORR($close, -$close, 5) = -1.0（完美反相关）。"""
        result = evaluate(parse("CORR($close, -$close, 5)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[4:].to_numpy() == pytest.approx([-1.0] * 6, abs=1e-10)

    def test_corr_constant_series_nan(self, panel):
        """恒定序列 → var=0 → 除零 → NaN，不应抛异常。"""
        # 构造一个恒定为 1 的 Series：$close - $close + 1
        result = evaluate(parse("CORR($close - $close + 1, $close, 5)"), panel)
        aaa = _aaa(result)
        # 窗口填满后所有位置应为 NaN（不是值也不是崩溃）
        assert aaa.iloc[4:].isna().all()

    def test_corr_ticker_isolation(self, panel):
        """CORR 在每 ticker 内独立计算，不跨 ticker 串。
        AAA close 与 volume 完全线性相关 → 1.0；BBB 两列也线性相关 → 1.0。"""
        result = evaluate(parse("CORR($close, $volume, 5)"), panel)
        aaa = _aaa(result).iloc[4:]
        bbb = _bbb(result).iloc[4:]
        # AAA close=1..10 递增，volume=100..1000 递增 → 线性相关
        assert aaa.to_numpy() == pytest.approx([1.0] * 6, abs=1e-10)
        # BBB close=20..11 递减，volume=200..2000 递增 → 反相关
        assert bbb.to_numpy() == pytest.approx([-1.0] * 6, abs=1e-10)

    def test_corr_output_index_matches_panel(self, panel):
        """CORR 输出 index 必须与 panel.index 完全对齐（droplevel + reindex 防御）。"""
        result = evaluate(parse("CORR($close, $open, 5)"), panel)
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.nlevels == 2
        # 必须能与 panel index 完全对齐（不多不少）
        assert set(result.index) == set(panel.index)


# ===========================================================================
# m2 — 横截面函数（RANK / QUANTILE）+ element-wise IF
# ===========================================================================
#
# Panel 设计确保两 ticker 每日可比：
#   AAA adj_close = 2, 4, ..., 20 （递增）
#   BBB adj_close = 40, 38, ..., 22（递减）
# Day 0: AAA=2  BBB=40 → AAA rank=0.5, BBB rank=1.0（BBB 大）
# Day 9: AAA=20 BBB=22 → AAA rank=0.5, BBB rank=1.0（BBB 仍大）
# BBB adj_close 始终 > AAA → BBB rank=1.0, AAA rank=0.5，每日恒定。
# ---------------------------------------------------------------------------


class TestCrossSectionRank:
    def test_rank_cross_section_two_tickers(self, panel):
        """BBB adj_close 每日都大于 AAA → BBB rank=1.0, AAA rank=0.5。"""
        result = evaluate(parse("RANK($close)"), panel)
        aaa = _aaa(result)
        bbb = _bbb(result)
        # pct=True + 2 个 ticker → 小的=0.5, 大的=1.0
        assert aaa.to_numpy() == pytest.approx([0.5] * 10)
        assert bbb.to_numpy() == pytest.approx([1.0] * 10)

    def test_rank_single_ticker_date(self):
        """单 ticker 退化：每日只有一个值 → rank=1.0（不 NaN 不崩）。"""
        df = pd.DataFrame(
            {"open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
             "close": [1.0, 2.0], "adj_close": [1.0, 2.0], "volume": [100, 200]},
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2024-01-01"), "AAA"),
                 (pd.Timestamp("2024-01-02"), "AAA")],
                names=["date", "ticker"],
            ),
        )
        result = evaluate(parse("RANK($close)"), df)
        assert result.to_numpy() == pytest.approx([1.0, 1.0])

    def test_rank_nan_input(self, panel):
        """NaN 输入保留 NaN（pandas rank 默认 na_option='keep'）。
        构造：LOG(0) = -inf（numpy 警告忽略），LOG(-1) = NaN。这里用 REF 制造前缀 NaN。"""
        result = evaluate(parse("RANK(REF($close, 3))"), panel)
        # 前 3 行 REF 是 NaN，因此 RANK 也应是 NaN
        aaa = _aaa(result)
        assert aaa.iloc[:3].isna().all()
        # 第 3 行起有值（两 ticker 都能参与排序）
        assert not aaa.iloc[3:].isna().any()


class TestCrossSectionQuantile:
    def test_quantile_median_broadcast(self, panel):
        """q=0.5 两 ticker → 每日返回两值中位数，AAA/BBB 同值。
        Day 0: median(2, 40) = 21.0  （adj_close 空间）
        Day 9: median(20, 22) = 21.0
        """
        result = evaluate(parse("QUANTILE($close, 0.5)"), panel)
        # 两 ticker 同一日期应得到相同值（广播）
        for date in _DATES:
            aaa_val = result.loc[(date, "AAA")]
            bbb_val = result.loc[(date, "BBB")]
            assert aaa_val == pytest.approx(bbb_val)
        # 第 0 天中位数
        assert result.loc[(_DATES[0], "AAA")] == pytest.approx(21.0)

    def test_quantile_multiindex_shape(self, panel):
        """QUANTILE 输出 index 与 panel.index 完全一致（shape preservation via transform）。"""
        result = evaluate(parse("QUANTILE($close, 0.5)"), panel)
        assert isinstance(result.index, pd.MultiIndex)
        assert set(result.index) == set(panel.index)
        assert len(result) == len(panel)

    def test_quantile_usable_in_if_predicate(self, panel):
        """QUANTILE 广播语义的意义：允许 IF($close > QUANTILE($close, 0.5), ...)。"""
        result = evaluate(
            parse("IF($close > QUANTILE($close, 0.5), 1, 0)"),
            panel,
        )
        # BBB 每日都高于中位数 → 1，AAA 每日都低于中位数 → 0
        assert _bbb(result).to_numpy() == pytest.approx([1] * 10)
        assert _aaa(result).to_numpy() == pytest.approx([0] * 10)


class TestIfFunction:
    def test_if_series_cond_series_branches(self, panel):
        """IF(Series, Series, Series) → 按 cond 逐点挑选。
        $close > $open 恒为 True（open=close-0.5）→ 全部取 $close。"""
        result = evaluate(parse("IF($close > $open, $close, $open)"), panel)
        pd.testing.assert_series_equal(
            result.sort_index(),
            panel["adj_close"].sort_index(),
            check_names=False,
        )

    def test_if_series_cond_scalar_branches(self, panel):
        """IF(Series, scalar, scalar) → 全 1（cond 恒 True）。"""
        result = evaluate(parse("IF($close > $open, 1, -1)"), panel)
        assert result.to_numpy() == pytest.approx([1] * 20)

    def test_if_scalar_cond_scalar_branches(self, panel):
        """IF(scalar, scalar, scalar) → 纯标量路径，返回原生数字。"""
        result = evaluate(parse("IF(1 > 0, 42, -1)"), panel)
        assert result == 42
        result2 = evaluate(parse("IF(0 > 1, 42, -1)"), panel)
        assert result2 == -1

    def test_if_scalar_cond_series_branches(self, panel):
        """IF(scalar, Series, Series) → 标量 cond 直接选分支（返回整条 Series）。"""
        result = evaluate(parse("IF(1 > 0, $close, $open)"), panel)
        # cond 为 True → 返回 $close = adj_close
        pd.testing.assert_series_equal(
            result.sort_index(),
            panel["adj_close"].sort_index(),
            check_names=False,
        )


class TestAdvancedIntegration:
    def test_nested_ma_slope(self, panel):
        """MA(SLOPE($close, 5), 3) — 验证 SLOPE 输出可嵌入另一个 rolling。
        线性 panel 下 SLOPE=2.0（AAA）/ -2.0（BBB），再 MA(.,3) 仍 = ±2.0。"""
        result = evaluate(parse("MA(SLOPE($close, 5), 3)"), panel)
        aaa = _aaa(result)
        bbb = _bbb(result)
        # SLOPE 前 4 行 NaN，再 MA 需要 3 行有效 → 前 4+2=6 行 NaN
        assert aaa.iloc[:6].isna().all()
        assert aaa.iloc[6:].to_numpy() == pytest.approx([2.0] * 4)
        assert bbb.iloc[6:].to_numpy() == pytest.approx([-2.0] * 4)

    def test_nested_ma_slope_lookback(self):
        """nested lookback：MA(SLOPE($close,5),3) = 5+3 = 8（嵌套相加）。"""
        assert parse("MA(SLOPE($close, 5), 3)").lookback() == 8

    def test_m2_impls_all_registered(self):
        """契约冒烟：m2 的 10 个函数必须全部有 impl 注册。"""
        m2_funcs = (
            "EMA", "SLOPE", "RSQUARE", "RESI", "CORR",
            "RANK", "QUANTILE", "IF", "IDXMAX", "IDXMIN",
        )
        for name in m2_funcs:
            assert _REGISTRY[name].impl is not None, (
                f"m2 函数 {name} 缺少 impl 注册"
            )


# ===========================================================================
# m3 — 时序滚动 TS_RANK / TS_QUANTILE
# ===========================================================================

class TestTsRank:
    def test_monotonic_increasing_rank_1(self, panel):
        """AAA close 递增 → 最后一行 rolling rank 应为 1.0 (最高百分位)。"""
        result = evaluate(parse("TS_RANK($close, 5)"), panel)
        aaa = _aaa(result)
        assert np.isnan(aaa.iloc[3])
        assert aaa.iloc[9] == pytest.approx(1.0)

    def test_monotonic_decreasing_rank(self, panel):
        """BBB close 递减 → 当前值总是窗口最小，rolling rank 应为最低百分位。"""
        result = evaluate(parse("TS_RANK($close, 5)"), panel)
        bbb = _bbb(result)
        for i in range(4, 10):
            assert bbb.iloc[i] == pytest.approx(1.0 / 5.0), f"row {i}"

    def test_min_periods_respected(self, panel):
        """窗口不满 → NaN (min_periods=window)。"""
        result = evaluate(parse("TS_RANK($close, 5)"), panel)
        aaa = _aaa(result)
        for i in range(4):
            assert np.isnan(aaa.iloc[i])

    def test_ties_average(self, panel):
        """Ties 用 average 策略（pandas 默认）。构造全等值窗口。"""
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=5, freq="D"), ["X"] * 5],
            names=["date", "ticker"],
        )
        p = pd.DataFrame({"adj_close": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=idx)
        result = evaluate(parse("TS_RANK($close, 3)"), p)
        assert result.iloc[4] == pytest.approx(2.0 / 3.0, abs=0.01)

    def test_ticker_isolation(self, panel):
        """TS_RANK 不跨 ticker 串数据。"""
        result = evaluate(parse("TS_RANK($close, 5)"), panel)
        assert result.index.equals(panel.sort_index().index)

    def test_lookback(self):
        ast = parse("TS_RANK($close, 10)")
        assert ast.lookback() == 10

    def test_min_window_validation(self):
        with pytest.raises(ParseError):
            parse("TS_RANK($close, 1)")


class TestTsQuantile:
    def test_q08_monotonic(self, panel):
        """AAA close 递增 → TS_QUANTILE($close, 5, 0.8) 应接近窗口第4大值。"""
        result = evaluate(parse("TS_QUANTILE($close, 5, 0.8)"), panel)
        aaa = _aaa(result)
        assert np.isnan(aaa.iloc[3])
        assert aaa.iloc[4] == pytest.approx(
            pd.Series([2.0, 4.0, 6.0, 8.0, 10.0]).quantile(0.8), abs=0.01
        )

    def test_q0_returns_min(self, panel):
        result = evaluate(parse("TS_QUANTILE($close, 5, 0)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[4] == pytest.approx(2.0)

    def test_q1_returns_max(self, panel):
        result = evaluate(parse("TS_QUANTILE($close, 5, 1)"), panel)
        aaa = _aaa(result)
        assert aaa.iloc[4] == pytest.approx(10.0)

    def test_q_validation_range(self):
        with pytest.raises(ParseError):
            parse("TS_QUANTILE($close, 5, 1.5)")
        with pytest.raises(ParseError):
            parse("TS_QUANTILE($close, 5, -0.1)")

    def test_min_window_validation(self):
        with pytest.raises(ParseError):
            parse("TS_QUANTILE($close, 1, 0.5)")

    def test_lookback(self):
        ast = parse("TS_QUANTILE($close, 20, 0.8)")
        assert ast.lookback() == 20

    def test_nested_lookback(self):
        ast = parse("TS_QUANTILE(REF($close, 3), 10, 0.5)")
        assert ast.lookback() == 13

    def test_m3_ts_impls_registered(self):
        """契约冒烟：m3 的 TS_RANK / TS_QUANTILE 必须有 impl 注册。"""
        for name in ("TS_RANK", "TS_QUANTILE"):
            assert _REGISTRY[name].impl is not None, (
                f"m3 函数 {name} 缺少 impl 注册"
            )


# ===========================================================================
# m3 — Alpha158 Loader / Registry / Lookback
# ===========================================================================

from stockbee.factor_data.alpha158 import Alpha158

_WINDOWS = [5, 10, 20, 30, 60]

_LOOKBACK_W = [
    "ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN",
    "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN", "IMXD",
    "CORR", "VMA", "VSTD",
]

_LOOKBACK_W_PLUS_1 = [
    "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
    "WVMA", "VSUMP", "VSUMN", "VSUMD",
]


class TestAlpha158Loader:
    def test_factor_count_158(self):
        assert len(Alpha158()) == 158

    def test_factor_names_unique(self):
        names = Alpha158().list_factor_names()
        assert len(names) == len(set(names))

    def test_kbar_9_factors_present(self):
        a = Alpha158()
        kbar = ["KMID", "KLEN", "KMID2", "KUP", "KUP2",
                "KLOW", "KLOW2", "KSFT", "KSFT2"]
        for name in kbar:
            assert name in a, f"KBAR factor {name} missing"

    def test_price_4_factors_present(self):
        a = Alpha158()
        for name in ["OPEN0", "HIGH0", "LOW0", "VWAP0"]:
            assert name in a, f"price factor {name} missing"

    def test_rolling_cartesian_expansion(self):
        """每个 rolling 算子对 5 个窗口展开。"""
        a = Alpha158()
        for prefix in _LOOKBACK_W + _LOOKBACK_W_PLUS_1:
            for w in _WINDOWS:
                name = f"{prefix}{w}"
                assert name in a, f"rolling factor {name} missing"

    def test_exact_order_frozen(self):
        """因子顺序是 contract，固定不变。"""
        a = Alpha158()
        names = a.list_factor_names()
        assert names[:9] == [
            "KMID", "KLEN", "KMID2", "KUP", "KUP2",
            "KLOW", "KLOW2", "KSFT", "KSFT2",
        ]
        assert names[9:13] == ["OPEN0", "HIGH0", "LOW0", "VWAP0"]
        assert names[13] == "ROC5"
        assert names[-1] == "VSUMD60"

    def test_unknown_name_raises(self):
        a = Alpha158()
        with pytest.raises(KeyError, match="未知因子"):
            a.get_expression("BOGUS")
        with pytest.raises(KeyError, match="未知因子"):
            a.get_expression_str("BOGUS")

    def test_all_158_parse_success(self):
        """全部 158 个表达式 parse 无异常。"""
        a = Alpha158()
        for name in a.list_factor_names():
            node = a.get_expression(name)
            assert isinstance(node, Node), f"{name} parse 返回类型错误"


class TestAlpha158Lookback:
    """全部 158 因子 lookback 参数化断言。"""

    def test_kbar_lookback_zero(self):
        a = Alpha158()
        for name in ["KMID", "KLEN", "KMID2", "KUP", "KUP2",
                      "KLOW", "KLOW2", "KSFT", "KSFT2"]:
            assert a.max_lookback(name) == 0, f"{name} expected lookback=0"

    def test_price_lookback_zero(self):
        a = Alpha158()
        for name in ["OPEN0", "HIGH0", "LOW0", "VWAP0"]:
            assert a.max_lookback(name) == 0, f"{name} expected lookback=0"

    @pytest.mark.parametrize("w", _WINDOWS)
    def test_simple_rolling_lookback_equals_window(self, w):
        """18 个简单滚动算子 lookback = window。"""
        a = Alpha158()
        for prefix in _LOOKBACK_W:
            name = f"{prefix}{w}"
            assert a.max_lookback(name) == w, (
                f"{name} expected lookback={w}, got {a.max_lookback(name)}"
            )

    @pytest.mark.parametrize("w", _WINDOWS)
    def test_ref_nested_lookback_equals_window_plus_1(self, w):
        """11 个 REF 嵌套算子 lookback = window + 1。"""
        a = Alpha158()
        for prefix in _LOOKBACK_W_PLUS_1:
            name = f"{prefix}{w}"
            assert a.max_lookback(name) == w + 1, (
                f"{name} expected lookback={w+1}, got {a.max_lookback(name)}"
            )


# ===========================================================================
# m3 — Alpha158 全量 Evaluation Smoke + 数值金标
# ===========================================================================

_EVAL_DATES = pd.date_range("2024-01-01", periods=80, freq="B")


def _make_eval_panel() -> pd.DataFrame:
    """2 tickers × 80 交易日合成 panel（覆盖 max lookback 61）。"""
    rows = []
    for ticker, base_close in (("AAA", 50.0), ("BBB", 100.0)):
        for i, date in enumerate(_EVAL_DATES):
            c = base_close + i * 0.5
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": c - 0.3,
                "high": c + 0.8,
                "low": c - 0.6,
                "close": c,
                "adj_close": c * 1.5,
                "volume": 1000.0 + i * 10.0,
                "vwap": c + 0.05,
            })
    return pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()


@pytest.fixture(scope="module")
def eval_panel() -> pd.DataFrame:
    return _make_eval_panel()


@pytest.fixture(scope="module")
def alpha() -> Alpha158:
    return Alpha158()


class TestAlpha158EvalSmoke:
    """全量 158 因子 evaluate smoke：无异常、返回 Series、index 对齐。"""

    def test_all_158_evaluate_no_exception(self, eval_panel, alpha):
        ev = Evaluator(eval_panel)
        for name in alpha.list_factor_names():
            ast = alpha.get_expression(name)
            result = ev.evaluate(ast)
            assert isinstance(result, pd.Series), (
                f"{name} evaluate 返回 {type(result).__name__}，期望 Series"
            )
            assert result.index.equals(eval_panel.sort_index().index), (
                f"{name} index 不对齐"
            )

    def test_kmid_numeric(self, eval_panel, alpha):
        """KMID = ($close-$open)/$open 数值核对。"""
        ev = Evaluator(eval_panel)
        result = ev.evaluate(alpha.get_expression("KMID"))
        p = eval_panel.sort_index()
        expected = (p["adj_close"] - p["open"]) / p["open"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_open0_numeric(self, eval_panel, alpha):
        """OPEN0 = $open/$close 数值核对。"""
        ev = Evaluator(eval_panel)
        result = ev.evaluate(alpha.get_expression("OPEN0"))
        p = eval_panel.sort_index()
        expected = p["open"] / p["adj_close"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_ma5_against_groundtruth(self, eval_panel, alpha):
        """MA5 = MEAN($close,5)/$close vs pandas groupby rolling。"""
        ev = Evaluator(eval_panel)
        result = ev.evaluate(alpha.get_expression("MA5"))
        p = eval_panel.sort_index()
        ma = (
            p["adj_close"]
            .groupby(level="ticker", sort=False, group_keys=False)
            .transform(lambda x: x.rolling(5, min_periods=5).mean())
        )
        expected = ma / p["adj_close"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_ts_rank5_monotonic(self, eval_panel, alpha):
        """TS_RANK5：AAA adj_close 单调递增 → rank = 1.0。"""
        ev = Evaluator(eval_panel)
        result = ev.evaluate(alpha.get_expression("RANK5"))
        aaa = result.xs("AAA", level="ticker").sort_index()
        for i in range(4, len(aaa)):
            assert aaa.iloc[i] == pytest.approx(1.0), f"row {i}"

    def test_cntp5_nan_comparison_behavior(self, eval_panel, alpha):
        """CNTP5 第一行 NaN 比较→False（qlib 兼容，非 NaN 传播）。"""
        ev = Evaluator(eval_panel)
        result = ev.evaluate(alpha.get_expression("CNTP5"))
        aaa = result.xs("AAA", level="ticker").sort_index()
        for i in range(4):
            assert np.isnan(aaa.iloc[i])
        assert not np.isnan(aaa.iloc[4])

    def test_vwap0_div_by_close(self, eval_panel, alpha):
        """VWAP0 = $vwap/$close，分母不为零时正常。"""
        ev = Evaluator(eval_panel)
        result = ev.evaluate(alpha.get_expression("VWAP0"))
        p = eval_panel.sort_index()
        expected = p["vwap"] / p["adj_close"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_wvma60_not_all_nan(self, eval_panel, alpha):
        """WVMA60 80 天 panel 最后几行应有值（非全 NaN）。"""
        ev = Evaluator(eval_panel)
        result = ev.evaluate(alpha.get_expression("WVMA60"))
        aaa = result.xs("AAA", level="ticker").sort_index()
        assert not aaa.iloc[-1:].isna().all()

    def test_157_evaluate_without_vwap(self, alpha):
        """无 vwap 列的 panel 上 157 因子（排除 VWAP0）全部正常。

        真实数据源 OHLCV_FIELDS 不含 vwap，VWAP0 是可选因子。
        此测试验证其余 157 个因子不依赖 vwap 列。
        """
        rows = []
        for ticker, base_close in (("AAA", 50.0), ("BBB", 100.0)):
            for i, date in enumerate(_EVAL_DATES):
                c = base_close + i * 0.5
                rows.append({
                    "date": date, "ticker": ticker,
                    "open": c - 0.3, "high": c + 0.8, "low": c - 0.6,
                    "close": c, "adj_close": c * 1.5,
                    "volume": 1000.0 + i * 10.0,
                })
        no_vwap_panel = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()
        ev = Evaluator(no_vwap_panel)
        non_vwap_factors = [n for n in alpha.list_factor_names() if n != "VWAP0"]
        assert len(non_vwap_factors) == 157
        for name in non_vwap_factors:
            ast = alpha.get_expression(name)
            result = ev.evaluate(ast)
            assert isinstance(result, pd.Series), f"{name} failed without vwap"

    def test_vwap0_raises_without_vwap(self, alpha):
        """无 vwap 列时 VWAP0 应 raise ExpressionError。"""
        idx = pd.MultiIndex.from_arrays(
            [pd.date_range("2024-01-01", periods=3, freq="D"), ["X"] * 3],
            names=["date", "ticker"],
        )
        p = pd.DataFrame({
            "open": [1.0, 2.0, 3.0], "high": [1.5, 2.5, 3.5],
            "low": [0.5, 1.5, 2.5], "adj_close": [1.0, 2.0, 3.0],
            "volume": [100.0, 200.0, 300.0],
        }, index=idx)
        ev = Evaluator(p)
        with pytest.raises(ExpressionError, match="vwap"):
            ev.evaluate(alpha.get_expression("VWAP0"))


# ---------------------------------------------------------------------------
# m4: ParquetFactorStore
# ---------------------------------------------------------------------------

from stockbee.factor_data.parquet_factor import ParquetFactorStore
from datetime import date as _date


def _make_factor_df(
    tickers: list[str],
    dates: list[str],
    columns: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """构造 MultiIndex (date, ticker) 因子 DataFrame 的测试辅助函数。"""
    idx_tuples = [
        (pd.Timestamp(d), t)
        for d in dates
        for t in tickers
    ]
    idx = pd.MultiIndex.from_tuples(idx_tuples, names=["date", "ticker"])
    n = len(idx_tuples)
    if columns is None:
        columns = {"factor_a": list(range(n))}
    data = {k: v for k, v in columns.items()}
    return pd.DataFrame(data, index=idx, dtype=float)


class TestParquetFactorStore:
    """m4: 预计算因子 Parquet 存储。"""

    def test_write_and_read_roundtrip(self, tmp_path):
        """write → read，shape 和值完全一致。"""
        store = ParquetFactorStore(tmp_path)
        df = _make_factor_df(["AAPL", "TSLA"], ["2024-01-01", "2024-01-02"])
        store.write_factors("fundamental", df)
        result = store.read_factors("fundamental")
        pd.testing.assert_frame_equal(result.sort_index(), df.sort_index())

    def test_read_filter_tickers(self, tmp_path):
        """write 2 tickers，read 只请求 1，返回只含该 ticker。"""
        store = ParquetFactorStore(tmp_path)
        df = _make_factor_df(["AAPL", "TSLA"], ["2024-01-01"])
        store.write_factors("fundamental", df)
        result = store.read_factors("fundamental", tickers=["AAPL"])
        assert list(result.index.get_level_values("ticker").unique()) == ["AAPL"]
        assert len(result) == 1

    def test_read_filter_date_range(self, tmp_path):
        """write 10 天，read [3,7]，返回 5 行（每天 1 ticker）。"""
        store = ParquetFactorStore(tmp_path)
        dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
        df = _make_factor_df(["AAPL"], dates)
        store.write_factors("fundamental", df)
        result = store.read_factors(
            "fundamental",
            start=_date(2024, 1, 3),
            end=_date(2024, 1, 7),
        )
        assert len(result) == 5
        dates_out = result.index.get_level_values("date")
        assert dates_out.min() == pd.Timestamp("2024-01-03")
        assert dates_out.max() == pd.Timestamp("2024-01-07")

    def test_merge_write_dedup_new_wins(self, tmp_path):
        """同 (date, ticker) 写两次，第二次值覆盖第一次。"""
        store = ParquetFactorStore(tmp_path)
        df1 = _make_factor_df(["AAPL"], ["2024-01-01"], {"factor_a": [1.0]})
        df2 = _make_factor_df(["AAPL"], ["2024-01-01"], {"factor_a": [99.0]})
        store.write_factors("fundamental", df1)
        store.write_factors("fundamental", df2)
        result = store.read_factors("fundamental")
        assert result.loc[(pd.Timestamp("2024-01-01"), "AAPL"), "factor_a"] == 99.0
        assert len(result) == 1

    def test_merge_write_new_columns_preserves_old(self, tmp_path):
        """第二次 write 加新列；旧列值也保留（不被丢弃）。"""
        store = ParquetFactorStore(tmp_path)
        df1 = _make_factor_df(["AAPL"], ["2024-01-01"], {"factor_a": [1.0]})
        df2 = _make_factor_df(["AAPL"], ["2024-01-01"], {"factor_b": [2.0]})
        store.write_factors("fundamental", df1)
        store.write_factors("fundamental", df2)
        result = store.read_factors("fundamental")
        # 两列都应存在
        assert "factor_a" in result.columns
        assert "factor_b" in result.columns
        # 旧值也保留
        assert result.loc[(pd.Timestamp("2024-01-01"), "AAPL"), "factor_a"] == 1.0
        assert result.loc[(pd.Timestamp("2024-01-01"), "AAPL"), "factor_b"] == 2.0

    def test_read_missing_group_returns_empty(self, tmp_path):
        """请求不存在的 group → 空 MultiIndex DataFrame，names 正确。"""
        store = ParquetFactorStore(tmp_path)
        result = store.read_factors("nonexistent")
        assert isinstance(result.index, pd.MultiIndex)
        assert list(result.index.names) == ["date", "ticker"]
        assert len(result) == 0

    def test_list_precomputed_factors_sorted(self, tmp_path):
        """write 多个 groups → list 返回 sorted stems，顺序确定。"""
        store = ParquetFactorStore(tmp_path)
        df = _make_factor_df(["AAPL"], ["2024-01-01"])
        store.write_factors("sentiment", df)
        store.write_factors("fundamental", df)
        store.write_factors("ml_score", df)
        result = store.list_precomputed_factors()
        assert result == sorted(result)
        assert set(result) == {"fundamental", "ml_score", "sentiment"}

    def test_read_none_tickers_returns_all(self, tmp_path):
        """tickers=None → 返回全部 tickers（不过滤）。"""
        store = ParquetFactorStore(tmp_path)
        df = _make_factor_df(["AAPL", "TSLA", "GOOG"], ["2024-01-01"])
        store.write_factors("fundamental", df)
        result = store.read_factors("fundamental", tickers=None)
        assert len(result) == 3

    def test_read_empty_tickers_list_returns_all(self, tmp_path):
        """tickers=[] → 与 tickers=None 行为相同，返回全部。"""
        store = ParquetFactorStore(tmp_path)
        df = _make_factor_df(["AAPL", "TSLA"], ["2024-01-01"])
        store.write_factors("fundamental", df)
        result = store.read_factors("fundamental", tickers=[])
        assert len(result) == 2

    def test_read_none_start_end_returns_all_dates(self, tmp_path):
        """start=None, end=None → 返回全量日期，不截断。"""
        store = ParquetFactorStore(tmp_path)
        dates = ["2024-01-01", "2024-06-15", "2024-12-31"]
        df = _make_factor_df(["AAPL"], dates)
        store.write_factors("fundamental", df)
        result = store.read_factors("fundamental", start=None, end=None)
        assert len(result) == 3

    def test_write_invalid_index_raises(self, tmp_path):
        """非 MultiIndex df → ValueError。"""
        store = ParquetFactorStore(tmp_path)
        df = pd.DataFrame({"factor_a": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
        with pytest.raises(ValueError, match="MultiIndex"):
            store.write_factors("fundamental", df)

    def test_write_wrong_index_names_raises(self, tmp_path):
        """MultiIndex 但 names 不是 ['date', 'ticker'] → ValueError。"""
        store = ParquetFactorStore(tmp_path)
        idx = pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-01"), "AAPL")],
            names=["timestamp", "symbol"],  # 错误的 names
        )
        df = pd.DataFrame({"factor_a": [1.0]}, index=idx)
        with pytest.raises(ValueError, match="names must be"):
            store.write_factors("fundamental", df)

    def test_invalid_group_name_raises(self, tmp_path):
        """包含路径穿越字符的 group 名 → ValueError。"""
        store = ParquetFactorStore(tmp_path)
        df = _make_factor_df(["AAPL"], ["2024-01-01"])
        with pytest.raises(ValueError, match="Invalid group name"):
            store.write_factors("../evil", df)
        with pytest.raises(ValueError, match="Invalid group name"):
            store.read_factors("../evil")
        with pytest.raises(ValueError, match="Invalid group name"):
            store.write_factors("", df)
        # backslash（Windows 路径穿越）
        with pytest.raises(ValueError, match="Invalid group name"):
            store.write_factors("..\\evil", df)

    # --- v3 新增：review 发现的 corner case ---

    def test_merge_write_nan_overwrites_existing(self, tmp_path):
        """新写入的 NaN 必须覆盖旧的非 NaN 值（新数据无条件胜出）。"""
        store = ParquetFactorStore(tmp_path)
        df1 = _make_factor_df(["AAPL"], ["2024-01-01"], {"factor_a": [1.0]})
        df2 = _make_factor_df(["AAPL"], ["2024-01-01"], {"factor_a": [float("nan")]})
        store.write_factors("fundamental", df1)
        store.write_factors("fundamental", df2)
        result = store.read_factors("fundamental")
        assert pd.isna(result.loc[(pd.Timestamp("2024-01-01"), "AAPL"), "factor_a"])

    def test_read_unknown_ticker_returns_empty(self, tmp_path):
        """请求存在的 group 但不存在的 ticker → 空 DataFrame。"""
        store = ParquetFactorStore(tmp_path)
        df = _make_factor_df(["AAPL"], ["2024-01-01"])
        store.write_factors("fundamental", df)
        result = store.read_factors("fundamental", tickers=["UNKNOWN"])
        assert len(result) == 0
        assert isinstance(result.index, pd.MultiIndex)

    def test_read_start_equals_end_single_day(self, tmp_path):
        """start == end → 只返回该日的数据。"""
        store = ParquetFactorStore(tmp_path)
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        df = _make_factor_df(["AAPL"], dates)
        store.write_factors("fundamental", df)
        result = store.read_factors(
            "fundamental",
            start=_date(2024, 1, 2),
            end=_date(2024, 1, 2),
        )
        assert len(result) == 1
        assert result.index.get_level_values("date")[0] == pd.Timestamp("2024-01-02")

    def test_write_empty_df_skipped(self, tmp_path):
        """写入空 DataFrame（0行）不创建文件。"""
        store = ParquetFactorStore(tmp_path)
        idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
        df = pd.DataFrame({"factor_a": pd.Series(dtype=float)}, index=idx)
        store.write_factors("fundamental", df)
        # 文件不应被创建
        assert not (tmp_path / "fundamental.parquet").exists()
        assert store.list_precomputed_factors() == []
