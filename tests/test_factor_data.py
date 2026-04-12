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
