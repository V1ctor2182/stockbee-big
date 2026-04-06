---
name: plan-milestones
description: >
  开发前的必经步骤：为 Feature Room 重新拆分 milestones。读 Tech Design + 已有 specs + 已完成模块的模式，
  生成适合实际开发的 milestone 列表。遇到不确定的拆分会询问开发者。
  触发词："拆 milestone"、"plan milestones news-data"、"重新拆分 macro-data 的步骤"、
  "准备开发 factor-storage"。在 dev skill 之前使用。
---

# Plan Milestones — 开发前的 Milestone 拆分

Feature Room 的 milestones 最初由 room-init 从 PRD 机械生成，粒度不适合实际开发。
这个 skill 在开发前重新拆分，确保每个 milestone = 一个合理的 commit 单位。

## 何时触发

- 开发一个新 Feature Room 之前（必经步骤）
- 用户说 "拆 milestone"、"plan milestones {room}"、"准备开发 {room}"
- dev skill 检测到 milestones 粒度不合理时建议先运行此 skill

## 前置条件

- Room 已由 room-init 创建（有 room.yaml, progress.yaml, spec.md）
- Tech Design 文档存在（BASIC/ 目录）
- 至少一个同类 Room 已完成（用作参考模式）

## 执行步骤

### Phase 0: 确认 Branch

在做任何事之前，**必须先确认开发者在正确的 branch 上**：

1. 运行 `git branch --show-current` 获取当前 branch
2. 运行 `git status` 检查是否有未提交的变更
3. 验证：
   - 当前 branch **不是 main** → 如果是 main，提示开发者先创建 feature branch
   - 当前 branch 名称和目标 Room **对得上** → 如果不对，提醒开发者确认
   - 工作区**干净**（无未提交变更） → 如果有未提交，提示先 commit 或 stash

```
示例输出：

⚠️ Branch 检查:
  当前 branch: main
  目标 Room: 02-news-data

  你在 main 上，建议先创建 feature branch:
    git checkout -b news-data

  要我帮你创建吗？(y/n)
```

4. **等待开发者确认或切换 branch 后再继续**

### Phase 1: 收集信息

1. 读取目标 Room 的当前状态：
   - `room.yaml`（依赖、优先级）
   - `progress.yaml`（现有 milestones）
   - `spec.md` + `specs/*.yaml`（intent, constraints）

2. 读取 Tech Design 中该 Room 相关的章节：
   - 从 intent spec 的 `provenance.source_ref` 定位 Tech Design 段落
   - 提取具体的技术要求、数据结构、接口定义

3. 读取已完成的**同类 Room** 的实际实现模式：
   - 文件结构（哪些文件、每个文件的职责）
   - milestone 实际粒度（每个 milestone 的代码量和范围）
   - 测试结构

   ```
   例如开发 02-news-data 前，参考已完成的 01-stock-data:
   - parquet_store.py (ParquetMarketData)     ← m1
   - universe_store.py (SqliteUniverseProvider) ← m2
   - funnel.py (UniverseFunnel)                ← m3
   - alpaca_market.py + sync.py                ← m4
   - test_stock_data.py                        ← m5
   ```

4. 读取 `providers/interfaces.py` 中该 Room 对应的 Provider 接口定义

### Phase 2: 生成 Milestone 方案

5. 基于收集的信息，生成 milestone 拆分方案。每个 milestone 应满足：

   **拆分原则**：
   | 原则 | 说明 |
   |------|------|
   | **一个 milestone = 一个可独立运行的产出** | 不是半成品，而是可以测试的完整单元 |
   | **50-200 行代码** | 太小浪费 commit，太大难 review |
   | **明确的输入/输出** | 下一个 milestone 知道从哪里接手 |
   | **测试作为独立 milestone 或合并到最后** | 视代码量决定 |
   | **顺序 = 依赖关系** | m2 可以 import m1 的产出 |

   **典型的 milestone 模式**（数据层 Room）：
   ```
   m1: Provider 实现（回测模式，本地存储）
   m2: 数据模型/Schema（SQLite 表或 Parquet 结构）
   m3: 核心业务逻辑（计算、转换、筛选）
   m4: API/外部数据源集成（实盘模式）
   m5: 同步管道（编排 m1-m4）
   m6: 测试
   ```

   但不是所有 Room 都适合这个模式 — **根据实际复杂度调整**。

6. 对每个 milestone 生成：
   ```yaml
   - id: "m{N}-{short-name}"
     name: "{中文描述}"
     status: pending
     estimated_files: ["src/stockbee/{module}/{file}.py"]  # 预估产出文件
     estimated_lines: 100-150                                # 预估代码量
     depends_on: []                                          # 依赖哪些前置 milestone
     description: |
       {2-3 句话描述这个 milestone 要做什么}
   ```

### Phase 3: 交互确认（核心）

7. 展示方案并**逐项和开发者确认**：

```
📋 Milestone Plan: 02-news-data (新闻数据)
────────────────────────────────────────────

参考模式: 01-stock-data (已完成, 5 milestones, 6 files)
Tech Design: §2.3 新闻数据处理管道

现有 milestones (room-init 生成):
  ❌ m1-news_events 表设计      → 粒度太细，只有表定义
  ❌ m2-G1 过滤入库              → 和 m1 高度耦合
  ❌ m3-分级字段存储              → 概念不清
  ❌ m4-测试                      → OK

建议重新拆分:

  m1: SqliteNewsProvider — news_events 表 + CRUD
      ├── SQLite WAL, news_events 表 schema
      ├── 实现 NewsProvider 接口 (get_news, ingest_news)
      ├── 支持按 ticker/时间/重要度/G 级别查询
      └── ~150 行, 1 file

  m2: G1/G2 新闻分级处理管道
      ├── G1: 时间戳校验、去重、数据源合法性
      ├── G2: LLM 分类 (情绪/主题/紧急度) — 接口预留，Phase 1 用规则
      └── ~120 行, 1 file

  m3: NewsAPI 数据同步 + LiveNewsProvider
      ├── NewsAPI SDK 集成 (实盘模式)
      ├── 增量拉取 + 去重写入 SQLite
      └── ~130 行, 1 file

  m4: NewsDataSyncer 同步管道
      ├── 编排: fetch → G1 filter → store → G2 classify
      └── ~80 行, 1 file

  m5: 测试
      ├── SQLite CRUD, G1/G2 分级, 去重, 时间查询
      └── ~200 行, 1 file

总计: ~680 行, 5 files, 5 milestones

─────────────────────────────────
有疑问的地方 (需要你决定):

  ❓ G2 分级: Phase 1 是用规则引擎模拟，还是直接调 LLM API？
     → 规则引擎更简单但不准确
     → LLM API 准确但需要 API key + 成本

  ❓ G3 深度分析: 是否包含在这个 Room？
     → Tech Design 说 G3 由 Claude Sonnet 执行，日限 10 篇
     → 建议 Phase 2，这个 Room 只做 G1+G2

请确认方案，或调整任何 milestone。
```

8. **等待开发者回答疑问和确认**。开发者可以：
   - 直接确认 "OK"
   - 调整某个 milestone（"m2 和 m3 合并"）
   - 回答疑问（"G2 先用规则，G3 不做"）
   - 增删 milestone（"加一个 m6 做 sentiment score 集成"）

9. 如果开发者修改了方案，展示修改后的版本再次确认。
   循环直到开发者说 OK。

### Phase 4: 写入

10. 确认后，更新 `progress.yaml`：
    - 替换全部 milestones（旧的机械生成的被新方案替换）
    - 所有新 milestone 状态为 `pending`
    - 保留已有的 commits 记录（如果有）

11. 更新 `spec.md`：
    - 在 "当前进度" 段落反映新的 milestone 列表

12. **不动** room.yaml, _tree.yaml, specs/*.yaml（这些是 dev/commit-sync 的职责）

### Phase 5: 输出

13. 展示最终确认：
```
✅ 02-news-data milestones 已更新 (4 → 5 milestones)

可以开始开发了:
  → "做 news-data m1" 或 "dev news-data/m1"
```

## 什么时候问开发者

| 场景 | 问什么 |
|------|-------|
| **Phase 1 的技术方案有多种选择** | "G2 用规则还是 LLM？" |
| **某个功能是否属于这个 Room** | "G3 深度分析放这个 Room 还是单独 Room？" |
| **milestone 粒度不确定** | "这两个合成一个 milestone，还是分开？" |
| **外部依赖不确定** | "这个 milestone 需要 API key 才能测试，要 mock 还是跳过？" |
| **Phase 归属不确定** | "RL 经验回放是 Phase 1 还是 Phase 2？" |
| **发现 Tech Design 和 PRD 不一致** | "Tech Design 说 19 个指标，但 FRED 只能映射 17 个，怎么处理？" |

**不问的情况**：
- 文件命名、目录结构 → 参考已完成 Room 的模式
- 代码风格 → 参考已完成 Room 的代码
- 测试是否独立 milestone → 根据代码量自动判断

## 和其他 skill 的关系

```
room-init → plan-milestones → dev (循环) → commit-sync
  创建 Room    拆分 milestone    逐个开发      逐个提交
  (一次)       (开发前一次)     (每个 m 一次)  (内置在 dev 中)
```
