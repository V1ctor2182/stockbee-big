"""10 个 Provider 的抽象接口定义。

每个 Provider 定义输入/输出类型签名，具体实现由各数据 Room 提供。
- CalendarProvider:    交易日历
- UniverseProvider:    股票池
- MarketDataProvider:  OHLCV 行情数据
- FactorProvider:      因子数据
- FundamentalProvider: 基本面数据
- NewsProvider:        新闻数据
- MacroProvider:       宏观经济数据
- SentimentProvider:   情绪评分
- BrokerProvider:      交易执行
- CacheProvider:       缓存管理（包装 CacheManager 为 Provider 接口）
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import date, datetime
from typing import Any

import pandas as pd

from .base import BaseProvider


# ---------------------------------------------------------------------------
# 1. CalendarProvider — 交易日历
# ---------------------------------------------------------------------------
class CalendarProvider(BaseProvider):
    """交易日历 Provider。

    回测: Qlib 日历 / 本地 CSV
    实盘: Alpaca API 日历
    """

    @abstractmethod
    def get_trading_days(
        self, start: date, end: date, exchange: str = "NYSE"
    ) -> list[date]:
        """返回指定范围内的交易日列表。"""
        ...

    @abstractmethod
    def is_trading_day(self, d: date, exchange: str = "NYSE") -> bool:
        """判断某天是否为交易日。"""
        ...

    @abstractmethod
    def next_trading_day(self, d: date, exchange: str = "NYSE") -> date:
        """返回下一个交易日。"""
        ...

    @abstractmethod
    def prev_trading_day(self, d: date, exchange: str = "NYSE") -> date:
        """返回上一个交易日。"""
        ...


# ---------------------------------------------------------------------------
# 2. UniverseProvider — 股票池
# ---------------------------------------------------------------------------
class UniverseProvider(BaseProvider):
    """股票池 Provider。

    管理四层漏斗: 广域(8000) → 广泛(4000) → 候选(500) → U100(100)
    """

    @abstractmethod
    def get_universe(
        self, level: str = "u100", as_of: date | None = None
    ) -> pd.DataFrame:
        """获取指定层级的股票池。

        Args:
            level: "broad_all" | "broad" | "candidate" | "u100"
            as_of: 时间切片（避免前看偏差）

        Returns:
            DataFrame with columns: ticker, sector, market_cap, avg_volume, short_able
        """
        ...

    @abstractmethod
    def refresh_universe(self, level: str = "broad_all") -> int:
        """刷新指定层级的股票池，返回更新的股票数量。"""
        ...

    @abstractmethod
    def upsert_members(
        self,
        level: str,
        members: pd.DataFrame,
        snapshot_date: date | None = None,
    ) -> int:
        """批量更新某层级的股票池成员，返回更新后的成员数。"""
        ...


# ---------------------------------------------------------------------------
# 3. MarketDataProvider — OHLCV 行情数据
# ---------------------------------------------------------------------------
class MarketDataProvider(BaseProvider):
    """OHLCV 行情数据 Provider。

    回测: Parquet 文件读取
    实盘: Alpaca API（fallback 到 Parquet 缓存）
    """

    @abstractmethod
    def get_daily_bars(
        self,
        tickers: list[str],
        start: date,
        end: date,
        fields: list[str] | None = None,
    ) -> pd.DataFrame:
        """获取日线 OHLCV 数据。

        Args:
            tickers: 股票代码列表
            start/end: 日期范围
            fields: 可选字段过滤, 默认 ["open","high","low","close","volume","adj_close"]

        Returns:
            MultiIndex DataFrame (date, ticker) x fields
        """
        ...

    @abstractmethod
    def get_latest_price(self, tickers: list[str]) -> dict[str, float]:
        """获取最新价格（实盘用）。"""
        ...


# ---------------------------------------------------------------------------
# 4. FactorProvider — 因子数据
# ---------------------------------------------------------------------------
class FactorProvider(BaseProvider):
    """因子数据 Provider。

    技术因子: 表达式引擎动态计算 (Alpha158)
    预计算因子: Parquet 读取 (基本面/情绪/ML预测)
    """

    @abstractmethod
    def get_factors(
        self,
        tickers: list[str],
        factor_names: list[str],
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """获取因子数据。

        Returns:
            MultiIndex DataFrame (date, ticker) x factor_names
        """
        ...

    @abstractmethod
    def list_factors(self) -> list[dict[str, str]]:
        """列出所有可用因子及其类型（expression / precomputed）。"""
        ...

    @abstractmethod
    def get_ic_report(
        self, factor_name: str, window: int = 252
    ) -> dict[str, float]:
        """返回因子的滚动 IC/ICIR 统计。"""
        ...


# ---------------------------------------------------------------------------
# 5. FundamentalProvider — 基本面数据
# ---------------------------------------------------------------------------
class FundamentalProvider(BaseProvider):
    """基本面数据 Provider。

    数据源: Polygon.io / Yahoo Finance fallback
    """

    @abstractmethod
    def get_fundamentals(
        self,
        tickers: list[str],
        metrics: list[str] | None = None,
        as_of: date | None = None,
    ) -> pd.DataFrame:
        """获取基本面数据。

        Args:
            metrics: 如 ["pe_ratio", "pb_ratio", "roe", "market_cap"]

        Returns:
            DataFrame (ticker) x metrics
        """
        ...

    @abstractmethod
    def get_financial_statements(
        self, ticker: str, statement_type: str = "income", periods: int = 4
    ) -> pd.DataFrame:
        """获取财务报表数据。

        Args:
            statement_type: "income" | "balance_sheet" | "cash_flow"
        """
        ...


# ---------------------------------------------------------------------------
# 6. NewsProvider — 新闻数据
# ---------------------------------------------------------------------------
class NewsProvider(BaseProvider):
    """新闻数据 Provider。

    回测: SQLite news_events 表
    实盘: NewsAPI + Perigon 实时抓取
    G1/G2/G3 三级漏斗处理后存储
    """

    @abstractmethod
    def get_news(
        self,
        tickers: list[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        min_importance: float = 0.0,
        g_level: int | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        """查询新闻事件。

        Returns:
            DataFrame: id, timestamp, source, tickers, headline, snippet,
                       sentiment_score, importance_score, reliability_score, g_level
        """
        ...

    @abstractmethod
    def ingest_news(self, source: str = "all") -> int:
        """拉取最新新闻并经过 G1/G2 处理，返回新增条数。"""
        ...


# ---------------------------------------------------------------------------
# 7. MacroProvider — 宏观经济数据
# ---------------------------------------------------------------------------
class MacroProvider(BaseProvider):
    """宏观经济数据 Provider。

    19 个 FRED 指标覆盖 8 大经济维度。
    回测: Parquet 时间序列
    实盘: FRED API
    """

    @abstractmethod
    def get_macro_indicators(
        self,
        indicators: list[str] | None = None,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.DataFrame:
        """获取宏观指标数据。

        Args:
            indicators: FRED 指标代码列表, None 表示全部 19 个

        Returns:
            DataFrame (date) x indicators
        """
        ...

    @abstractmethod
    def get_latest_z_scores(
        self, indicators: list[str] | None = None, window: int = 252
    ) -> dict[str, float]:
        """获取各指标最新 Z-score（相对历史窗口的标准化）。"""
        ...

    @abstractmethod
    def refresh(self) -> int:
        """从 FRED API 更新数据，返回更新的指标数量。"""
        ...


# ---------------------------------------------------------------------------
# 8. SentimentProvider — 情绪评分
# ---------------------------------------------------------------------------
class SentimentProvider(BaseProvider):
    """情绪评分 Provider。

    本地 FinBERT 模型推理或缓存查询。
    """

    @abstractmethod
    def score_texts(
        self, texts: list[str]
    ) -> list[dict[str, float]]:
        """批量计算文本情绪评分。

        Returns:
            [{"positive": 0.8, "negative": 0.1, "neutral": 0.1, "confidence": 0.8}, ...]
        """
        ...

    @abstractmethod
    def get_ticker_sentiment(
        self, ticker: str, lookback_days: int = 7
    ) -> dict[str, float]:
        """获取某只股票近期的综合情绪评分。"""
        ...


# ---------------------------------------------------------------------------
# 9. BrokerProvider — 交易执行
# ---------------------------------------------------------------------------
class BrokerProvider(BaseProvider):
    """交易执行 Provider。

    回测: 模拟执行（纸交易）
    实盘: Alpaca API
    """

    @abstractmethod
    def get_positions(self) -> pd.DataFrame:
        """获取当前持仓。

        Returns:
            DataFrame: ticker, qty, market_value, avg_cost, unrealized_pnl, side
        """
        ...

    @abstractmethod
    def get_account(self) -> dict[str, Any]:
        """获取账户信息（余额、购买力等）。"""
        ...

    @abstractmethod
    def submit_order(
        self,
        ticker: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        limit_price: float | None = None,
    ) -> dict[str, Any]:
        """提交订单。

        Args:
            side: "buy" | "sell"
            order_type: "market" | "limit" | "stop" | "stop_limit"

        Returns:
            订单确认信息 dict
        """
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """取消订单。"""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict[str, Any]:
        """查询订单状态。"""
        ...


# ---------------------------------------------------------------------------
# 10. CacheProviderInterface — 缓存管理（Provider 接口包装）
# ---------------------------------------------------------------------------
class CacheProviderInterface(BaseProvider):
    """缓存管理 Provider。

    将 CacheManager 暴露为标准 Provider 接口，
    供其他 Provider 通过 Registry 获取。
    """

    @abstractmethod
    def get_cached(
        self,
        key: str,
        fetcher: Any | None = None,
        ttl: float | None = None,
    ) -> Any | None:
        """从缓存获取数据，miss 时调用 fetcher。"""
        ...

    @abstractmethod
    def invalidate(self, key: str) -> None:
        """使缓存失效。"""
        ...

    @abstractmethod
    def report(self) -> dict[str, Any]:
        """返回缓存统计。"""
        ...
