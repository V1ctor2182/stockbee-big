---
name: room-status
description: >
  Generate a project panorama report — timeline view, progress bars, milestone details, draft spec review, and issue detection. Scans all Rooms and aggregates into a human-readable dashboard (Layer 3 Projection). Use when the user says "项目状态", "看一下全景", "room status", "哪些room在开发", "有哪些spec没确认", "生成状态报告", "这周的报告", or any request about project progress, timeline view, or spec review status.
---

# Room Status — 项目全景 + 时间线 + Draft Review

扫描所有 Room，生成项目全景报告。把细粒度的 Room 状态聚合成人类可读的仪表盘（对应 VibeHub Layer 3 Projection）。

## 前置条件

- meta/ 目录已初始化
- 至少有一些 Room 和 specs
- timeline.yaml 和 progress.yaml 存在（由 timeline-init 生成）

## 用法

```
# 全景报告
> "看一下项目全景"

# 只看特定状态
> "哪些 room 在开发中？"

# 只看 draft specs
> "有哪些 spec 还没确认？"

# 输出报告到文件
> "生成这周的状态报告"
→ 输出到 meta/changelog/weekly-{YYYY}-W{NN}.md
```

## 执行步骤

### Step 1: 数据收集

1. 读取 `_tree.yaml` 获取所有 Room
2. 遍历每个 Room：
   - `room.yaml`: lifecycle 状态、spec 数量（按 type 分）、spec 状态（active/draft/stale）、prompt_test 状态
   - `progress.yaml`: completion 百分比、milestones 完成情况、最近 commits、最后更新时间
3. 读取 `timeline.yaml`，对比排期 vs 实际进度

### Step 2: 问题检测

检测以下问题并分级：

- 🔴 **Stale specs** — 代码改了 spec 没更新
- 🔴 **Behind schedule** — 实际进度落后于排期
- 🟡 **Draft specs** — 待 review 确认
- 🟡 **Missing prompts** — spec 就绪但没生成 prompt
- 🟡 **Empty rooms** — 创建了但没有任何 spec
- 🔵 **Blocked rooms** — depends_on 的 Room 还没 completed

### Step 3: 生成报告

报告包含以下段落：

**1. Overall Summary**
```
📊 Overall: {epic_count} Epic / {feature_count} Feature Rooms
✅ Completed: {N}    🔨 In-Dev: {N}
📝 Planning: {N}     ⚠️ Issues: {N}
📈 Project Completion: {X}%
```

**2. Timeline View（Gantt 式进度条）**
```
📅 TIMELINE (Phase 1: MVP1)
Target: 3/10 - 4/20  |  Today: {today}

foundation    ████████░░░░ 60%  ✅ on track
  3/10━━━━━━━━━━━━3/20
desktop-pet   ██░░░░░░░░░ 15%  ✅ on track
  3/18━━━━━━━━━━━━━━4/1
note-mode     ░░░░░░░░░░░  0%  ⏳ not started
  3/25━━━━━━━━━━━━━4/8
trigger-mode  ████░░░░░░░ 40%  ✅ on track
  4/1━━━━━━━━━━━━━━4/15
```

对比 timeline.yaml 中的 target_end 和当前日期 + completion，判断 on track / behind / ahead。

**3. In Development（带 Milestones）**
```
🔨 IN DEVELOPMENT

foundation (60%)
├─ ✅ DB 层设计
├─ ✅ API 框架搭建
├─ 🔨 权限管理
├─ ⬚ 错误处理
└─ ⬚ 测试
Specs: 5 ✅ | Commits: 8 | Last: 3/14

trigger-mode (40%)
├─ ✅ API 接口设计
├─ 🔨 CRUD API 实现
├─ ⬚ EventKit 时间触发
├─ ⬚ 语义匹配内容触发
└─ ⬚ CoreLocation 位置触发
Specs: 6 ✅ 1 ⚠️ | Commits: 3 | Last: 3/15
```

**4. Draft Specs 待确认**
```
📋 DRAFT SPECS 待确认 ({N} 条)

trigger-mode ({n} 条):
 [draft] intent-trigger-mode-001
   "三种 Trigger 自动触发 AI 处理"
   来源: prd_extraction | conf: 0.8
 [draft] constraint-trigger-mode-001
   "CPU≤3%, triggers≤50"
   来源: prd_extraction | conf: 0.9

→ 使用 room skill: "review trigger-mode 的 draft specs" 逐条确认
```

**5. Issues**
```
⚠️ ISSUES
├─ desktop-pet: 1 stale spec (pet-idle)
├─ knowledge-base: empty room
├─ trigger-mode: 3 draft specs pending
└─ note-mode: 未开始但排期 3/25 开始
```

### Step 4: 输出

- 默认直接在对话中展示报告
- 如果用户要求 "生成报告"，同时保存到 `meta/changelog/weekly-{YYYY}-W{NN}.md`

## 针对特定查询的简化输出

用户只问特定问题时，只展示相关段落：
- "哪些 room 在开发中？" → 只展示 In Development 段落
- "有哪些 spec 还没确认？" → 只展示 Draft Specs 段落
- "trigger-mode 进展如何？" → 只展示该 Room 的进度 + milestones
