# 股票数据存储 (01-stock-data)

## Intent

**Parquet 列存 OHLCV + SQLite 四层漏斗宇宙管理 + Alpaca API 同步管道**

5 个组件组成完整的股票数据基础设施：
- **ParquetMarketData** — 按 ticker 分文件 Parquet 存储，支持增量写入和缓存集成
- **SqliteUniverseProvider** — WAL 模式双表（members + snapshots），支持 as_of 时间切片
- **UniverseFunnel** — 四层漏斗 8000→4000→500→100，FunnelConfig 可配置
- **AlpacaMarketData** — alpaca-py SDK 实盘数据源
- **StockDataSyncer** — 编排周度同步：assets → OHLCV → liquidity → funnel

实现状态：已完成 ✅（2026-04-02），实测验证通过（2026-04-03）
测试覆盖：24 个单元测试

## Constraints

- **覆盖 ~12654 只可交易美股（broad_all）**，经漏斗筛选至 ~4000 → 500 → 100 — 来源: PRD §3.2
- **Parquet 按 ticker 分文件，6 列 OHLCV schema** — 来源: Tech Design §2.2
- **SQLite WAL 模式，universe_members + universe_snapshots 双表** — 来源: Tech Design §2.1
- **漏斗参数可配置（FunnelConfig dataclass）** — 来源: 实现决策
- **Alpaca 为唯一实盘数据源，alpaca-py 为可选依赖** — 来源: Tech Design §2.8

## Decisions

- **Parquet 列存 > Qlib .bin / HDF5** — 通用性强、8:1 压缩、跨语言兼容，长期可维护
- **四层漏斗 > 直接 U100** — 避免幸存者偏差，支持 as_of 时间切片回测
- **按 ticker 分文件 > 单一大 Parquet** — 增量更新只需重写单个文件，支持并行 I/O

## Contracts

- **ParquetMarketData** — get_daily_bars(tickers, start, end, fields) → MultiIndex DataFrame; write_ticker(ticker, df); get_latest_price; list_tickers; ticker_date_range
- **SqliteUniverseProvider** — get_universe(level, as_of) → DataFrame; upsert_members(level, members, snapshot_date) → int; get_member_count; get_all_level_counts
- **StockDataSyncer** — sync_assets; sync_ohlcv; sync_ohlcv_full; update_liquidity; run_funnel; run_weekly_sync

## Known Gaps

- **Issue #2**: market_cap / sector 缺失（Alpaca 不提供）→ 需 FundamentalProvider 补充
- **Issue #3**: 全量历史同步（4000只×5年）未实测 → 需验证耗时和限流
- **Issue #4**: adj_close 用 VWAP 近似 → 拆股/分红场景需从 Yahoo Finance 补正

---
_所有 spec 状态: active_
_spec.md 最后更新: 2026-04-03_
_specs 目录: 1 intent + 5 constraints + 3 decisions + 3 contracts + 1 context + 1 change = 14 个 spec 文件_
