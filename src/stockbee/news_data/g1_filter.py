"""G1 快速过滤管道 — 来源校验、时间校验、去重、实体识别。

G1 是新闻处理漏斗的第一级，目标淘汰 ~95% 的噪声。
通过 G1 的新闻写入 news_events 表，g_level=1。

过滤顺序（短路优化）：
1. 来源合法性（黑名单 domain）
2. 时间校验（拒绝超过 max_age_days 的旧新闻）
3. 内容校验（标题非空、非纯特殊字符）
4. 实体识别（从标题/摘要提取 ticker）
5. 去重由 SqliteNewsProvider 的 UNIQUE(headline, source) 保证
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 常见美股 ticker → 公司名映射（高频歧义词）
# 只收录容易和普通英语单词混淆的 ticker
_AMBIGUOUS_TICKERS: dict[str, list[str]] = {
    "AAPL": ["apple"],
    "AMZN": ["amazon"],
    "GOOG": ["google", "alphabet"],
    "GOOGL": ["google", "alphabet"],
    "META": ["meta platforms", "facebook"],
    "MSFT": ["microsoft"],
    "TSLA": ["tesla"],
    "NVDA": ["nvidia"],
    "AMD": ["advanced micro"],
    "NFLX": ["netflix"],
    "CRM": ["salesforce"],
    "INTC": ["intel"],
    "BA": ["boeing"],
    "DIS": ["disney", "walt disney"],
    "JPM": ["jpmorgan", "jp morgan"],
    "GS": ["goldman sachs", "goldman"],
    "V": ["visa"],
    "MA": ["mastercard"],
    "WMT": ["walmart"],
    "PG": ["procter & gamble", "procter and gamble"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "UNH": ["unitedhealth"],
    "HD": ["home depot"],
    "KO": ["coca-cola", "coca cola"],
    "PEP": ["pepsi", "pepsico"],
}

# 反向映射：公司名 → ticker
_COMPANY_TO_TICKER: dict[str, str] = {}
for _ticker, _names in _AMBIGUOUS_TICKERS.items():
    for _name in _names:
        _COMPANY_TO_TICKER[_name.lower()] = _ticker

# ticker 格式：1-5 个大写字母（可带 .A/.B 后缀）
_TICKER_PATTERN = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z]{1,2})?)\b")

# 短 ticker 语境判定关键词
_FINANCIAL_KEYWORDS = frozenset({
    "stock", "share", "buy", "sell", "price",
    "market", "trade", "invest", "earning", "$",
})

# 排除常见英语单词（会被误识别为 ticker）
_COMMON_WORDS = frozenset({
    "A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "IF",
    "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OK", "ON", "OR", "SO", "TO",
    "UP", "US", "WE", "AI", "CEO", "CFO", "CTO", "COO", "IPO", "SEC", "FBI",
    "FDA", "GDP", "CPI", "PPI", "ETF", "NYC", "USA", "UK", "EU", "UN", "WHO",
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HAD",
    "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "MAN", "NEW",
    "NOW", "OLD", "SEE", "WAY", "MAY", "SAY", "SHE", "TWO", "DAY", "DID",
    "GET", "HIM", "LET", "PUT", "TOP", "TOO", "USE", "YES", "BIG", "END",
    "FAR", "FEW", "GOT", "OWN", "RUN", "SET", "TRY", "WHY", "NET", "LOW",
    "HIGH", "JUST", "OVER", "SUCH", "TAKE", "YEAR", "THEM", "SOME", "THAN",
    "BEEN", "HAVE", "MANY", "VERY", "WHEN", "WHAT", "YOUR", "EACH", "MAKE",
    "LIKE", "LONG", "LOOK", "ONLY", "COME", "BACK", "GOOD", "GIVE", "MOST",
    "FIND", "HERE", "KNOW", "LAST", "MUCH", "WILL", "DOWN", "ALSO", "MADE",
    "WELL", "CALL", "SAID", "REAL", "BEST", "PLAN", "NEXT", "KEEP", "FREE",
    "POST", "THAT", "THIS", "WITH", "FROM", "THEY", "WERE", "THEIR", "WHICH",
    "COULD", "OTHER", "ABOUT", "AFTER", "FIRST", "STILL", "UNDER",
    "EVERY", "BEING", "THOSE", "SINCE", "WHERE", "WHILE", "WOULD", "THESE",
})

# 已知合法的短 ticker（确实是股票，不是英语单词）
_VALID_SHORT_TICKERS = frozenset({
    "A",    # Agilent
    "V",    # Visa
    "X",    # US Steel
    "F",    # Ford
    "T",    # AT&T
    "C",    # Citigroup
    "D",    # Dominion Energy
    "K",    # Kellanova
})


@dataclass
class G1Config:
    """G1 过滤器配置。"""
    # 来源黑名单（domain 列表）
    blacklist_domains: list[str] = field(default_factory=lambda: [
        "example.com", "test.com", "spam-news.com",
    ])
    # 最大新闻年龄（天）
    max_age_days: int = 7
    # 最短标题长度
    min_headline_length: int = 10
    # 是否要求至少匹配一个 ticker
    require_ticker: bool = False


@dataclass
class G1Result:
    """G1 过滤结果。"""
    passed: bool
    reason: str = ""  # 未通过时的原因
    tickers: list[str] = field(default_factory=list)  # 识别出的 ticker 列表


class G1Filter:
    """G1 快速过滤器。

    过滤顺序（短路，第一个失败就返回）：
    1. source_check — 黑名单 domain
    2. time_check — 旧新闻过滤
    3. content_check — 标题质量
    4. extract_tickers — 实体识别
    """

    def __init__(self, config: G1Config | None = None) -> None:
        self._config = config or G1Config()
        self._blacklist = set(d.lower() for d in self._config.blacklist_domains)

    def filter(
        self,
        headline: str,
        source: str,
        timestamp: str | datetime,
        snippet: str | None = None,
        source_url: str | None = None,
    ) -> G1Result:
        """对单条新闻运行 G1 全部过滤。"""

        # 1. 来源校验
        if not self._check_source(source, source_url):
            return G1Result(passed=False, reason=f"blacklisted source: {source}")

        # 2. 时间校验
        ts = self._parse_timestamp(timestamp)
        if ts is None:
            return G1Result(passed=False, reason="invalid or missing timestamp")
        if not self._check_time(ts):
            delta = datetime.now(timezone.utc) - ts
            if delta.total_seconds() < 0:
                return G1Result(passed=False, reason="timestamp in the future")
            return G1Result(passed=False, reason=f"too old: {delta.days} days")

        # 3. 内容校验
        if not self._check_content(headline):
            return G1Result(passed=False, reason="headline too short or invalid")

        # 4. 实体识别
        tickers = self.extract_tickers(headline, snippet)
        if self._config.require_ticker and not tickers:
            return G1Result(passed=False, reason="no ticker identified")

        return G1Result(passed=True, tickers=tickers)

    def filter_batch(
        self, events: list[dict]
    ) -> list[tuple[dict, G1Result]]:
        """批量过滤，返回 (event, result) 对列表。"""
        return [
            (event, self.filter(
                headline=event.get("headline", ""),
                source=event.get("source", ""),
                timestamp=event.get("timestamp", ""),
                snippet=event.get("snippet"),
                source_url=event.get("source_url"),
            ))
            for event in events
        ]

    # ------ 各项检查 ------

    def _check_source(self, source: str, source_url: str | None) -> bool:
        """检查来源是否在黑名单中。"""
        if source.lower() in self._blacklist:
            return False
        if source_url:
            try:
                domain = urlparse(source_url).netloc.lower()
                # 去掉 www. 前缀
                if domain.startswith("www."):
                    domain = domain[4:]
                if any(
                    domain == bl or domain.endswith("." + bl)
                    for bl in self._blacklist
                ):
                    return False
            except Exception:
                pass  # URL 解析失败不阻止，只检查 source 字段
        return True

    def _check_time(self, ts: datetime) -> bool:
        """检查新闻是否在允许的时间窗口内（不能太旧，也不能在未来）。"""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self._config.max_age_days)
        # 允许最多 1 小时的时钟偏差
        max_future = now + timedelta(hours=1)
        return cutoff <= ts <= max_future

    def _check_content(self, headline: str) -> bool:
        """检查标题质量。"""
        if not headline or not headline.strip():
            return False
        cleaned = headline.strip()
        if len(cleaned) < self._config.min_headline_length:
            return False
        # 纯特殊字符/数字
        if not re.search(r"[a-zA-Z\u4e00-\u9fff]", cleaned):
            return False
        return True

    def _parse_timestamp(self, ts: str | datetime) -> datetime | None:
        """解析时间戳为 UTC datetime。"""
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)
        if not ts or not isinstance(ts, str):
            return None
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except (ValueError, TypeError):
            return None

    # ------ 实体识别 ------

    def extract_tickers(
        self, headline: str, snippet: str | None = None
    ) -> list[str]:
        """从标题和摘要中提取股票 ticker。

        策略：
        1. 精确 ticker 匹配（$AAPL 或全大写 1-5 字母）
        2. 公司名 → ticker 模糊匹配
        3. 排除常见英语单词
        """
        text = headline or ""
        if snippet:
            text = f"{text} {snippet}"
        if not text.strip():
            return []

        found: set[str] = set()

        # 1. $TICKER 格式（最高置信度，大小写均可）
        for match in re.finditer(r"\$([A-Za-z]{1,5}(?:\.[A-Za-z]{1,2})?)\b", text):
            found.add(match.group(1).upper())

        # 2. 公司名匹配（词边界，避免 "pineapple" → AAPL）
        text_lower = text.lower()
        for company_name, ticker in _COMPANY_TO_TICKER.items():
            # 用 \b 词边界防止子串误匹配
            if re.search(r"\b" + re.escape(company_name) + r"\b", text_lower):
                found.add(ticker)

        # 3. ��写字母 ticker 匹配（从��文提取，保留大小��语境）
        for match in _TICKER_PATTERN.finditer(text):
            candidate = match.group(1)
            # 基础部分（去掉 .A/.B 后缀）用于常见词判定
            base = candidate.split(".")[0] if "." in candidate else candidate
            if len(base) == 1:
                # 单字母 ticker 只接受已知的
                if base in _VALID_SHORT_TICKERS:
                    # 还需要检查语境：前后是否有���融关键词
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    context = text[start:end].lower()
                    if any(kw in context for kw in _FINANCIAL_KEYWORDS):
                        found.add(candidate)
            elif base not in _COMMON_WORDS:
                # 2-5 字母：排除常见英语单词
                found.add(candidate)

        return sorted(found)
