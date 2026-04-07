"""news_data — 新闻数据存储与 G1/G2/G3 处理管道。

组件:
- SqliteNewsProvider: SQLite news_events 表 + NewsProvider 接口
- G1 Filter: 来源校验、去重、实体识别
- G2 Classifier: FinBERT 情绪分类 + 主题/重要度评分
- G3 Analyzer: Claude Haiku 深度分析
- NewsAPI/Perplexity Source: 新闻拉取
- NewsDataSyncer: 同步管道编排
"""

from .g1_filter import G1Config, G1Filter, G1Result
from .g2_classifier import G2Classifier, G2Config, G2Result
from .g3_analyzer import G3Analyzer, G3Config, G3Result
from .news_store import SqliteNewsProvider

__all__ = [
    "SqliteNewsProvider",
    "G1Filter",
    "G1Config",
    "G1Result",
    "G2Classifier",
    "G2Config",
    "G2Result",
    "G3Analyzer",
    "G3Config",
    "G3Result",
]
