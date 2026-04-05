# StockBEE Implementation Workflow

## 全局流程图

```
                    ┌──────────────┐
                    │  room-init   │  创建 Room 结构（初始 milestones 粗糙）
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  prompt-gen  │  Room 级: 生成完整上下文 prompt
                    │  (Room 级)   │  用于理解全貌、对齐设计
                    └──────┬───────┘
                           │
                    ┌──────▼────────────┐
                    │ plan-milestones   │  开发前: 重新拆分 milestones
                    │                   │  读 Tech Design + 已完成模式
                    │ 遇到问题会问开发者  │  确认后写入 progress.yaml
                    └──────┬────────────┘
                           │
              ┌────────────▼─────────────┐
              │                          │
              │  dev (循环，人驱动)        │
              │                          │
              │  ┌─────────────────────┐ │
              │  │ "做 room/m1"        │ │
              │  │                     │ │
              │  │ 1. context pull     │ │  ← 内置 milestone 级 prompt-gen
              │  │    (milestone 级)   │ │
              │  │                     │ │
              │  │ 2. 实现 + 测试      │ │
              │  │                     │ │
              │  │ 3. commit-sync      │ │  ← 内置 commit + spec 增量更新
              │  │    (milestone 级)   │ │
              │  │                     │ │
              │  │ 4. 暂停，等用户     │ │
              │  └────────┬────────────┘ │
              │           │              │
              │    用户: "下一个"          │
              │           │              │
              │  ┌────────▼────────────┐ │
              │  │ "做 room/m2"        │ │
              │  │  ...重复...          │ │
              │  └─────────────────────┘ │
              │                          │
              └────────────┬─────────────┘
                           │
                    ┌──────▼───────┐
                    │ Room 全部完成  │  每个 milestone 都 commit-sync
                    │              │  最后一个 milestone 的 commit-sync
                    │ 最后一个 m 的  │  额外触发:
                    │ commit-sync   │  - room.yaml lifecycle → active
                    │ 额外做:       │  - _tree.yaml status → 已完成
                    │ - spec 完善   │  - spec.md 完整重写
                    │ - status 传播 │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  room-status │  可选: 查看全局进度
                    └──────────────┘
```

## 日常开发示例

### 场景 1: 按 milestone 逐步开发

```
用户: 做 news-data m1
  AI: [pull context] → [实现 SQLite news_events 表] → [commit-sync]
  AI: ✅ m1 完成. 下一个: m2-G1/G2分级处理

用户: 下一个
  AI: [pull context, 含 m1 产出] → [实现 G1/G2 漏斗] → [commit-sync]
  AI: ✅ m2 完成. 下一个: m3-NewsAPI同步

用户: m3 先跳过，做 m4
  AI: [pull context] → [实现 m4] → [commit-sync]

用户: 回头做 m3
  AI: [pull context, 含 m1/m2/m4 产出] → [实现 m3] → [commit-sync]
  AI: ✅ news-data 全部完成 (5/5). Room status → 已完成.
```

### 场景 2: 跨 Room 切换

```
用户: 做 macro-data m1
  AI: ✅ m1 完成.

用户: 先去做 factor-storage m1
  AI: [切换 Room] → ✅ m1 完成.

用户: 回到 macro-data, 下一个
  AI: [读 progress.yaml, 找到 m2 pending] → ✅ m2 完成.
```

### 场景 3: 手动修改后 commit

```
用户: [手动在 IDE 改了代码]
用户: commit-sync
  AI: [检测 diff, 匹配 Room] → [commit + spec 更新]
```

## Skill 职责分工

| Skill | 触发 | 做什么 | 何时用 |
|-------|------|-------|-------|
| **dev** | "做 X m1"、"下一个" | context pull → 开发 → commit-sync（全自动） | **日常开发主入口** |
| **plan-milestones** | "拆 milestone X"、"准备开发 X" | 重新拆分 milestones，遇到问题问开发者 | **dev 之前必经** |
| prompt-gen | "promptgen X" | 只生成 prompt，不开发 | 想看 context 或给 Cursor 用 |
| commit-sync | "commit-sync" | 只 commit + spec 更新，不开发 | 手动改代码后提交 |
| room | "创建 room X" | Room CRUD | 项目初期规划 |
| room-init | "初始化 room X" | 生成 Room 模板文件 | 新 Room 骨架 |
| room-status | "项目进度" | 全局进度报告 | 想看全景 |

## Commit 粒度对照

| 方式 | commit 大小 | 适用场景 |
|------|-----------|---------|
| **dev (per milestone)** | 小（50-200行） | ✅ 推荐，日常开发 |
| 整个 Room 一次提交 | 大（500-1000行） | ❌ 避免 |
| 手动 git commit | 任意 | 紧急修复、小改动 |

## 每个 Milestone Commit 包含

```
代码文件（该 milestone 的实现 + 测试）
  +
meta/ 增量更新：
  - progress.yaml: 该 milestone → completed, 追加 commit hash
  - (最后一个 milestone 时) room.yaml, _tree.yaml, spec.md
```

**不包含**：整个 Room 的 spec 重写（那是最终完成时的事）。
