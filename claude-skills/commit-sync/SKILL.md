---
name: commit-sync
description: >
  The core development skill — replaces git commit/push entirely. Analyzes code changes, maps them to Feature Rooms, checks spec consistency (anchor tracking + quality gate), auto-updates specs, detects draft-to-active upgrades, records change specs, updates progress/milestones, propagates status upstream, generates commit messages, and executes git add/commit/push. Use whenever the user says "帮我提交", "commit", "提交代码", "push", "检查代码改动", "dry-run", or wants to commit their work. This is the ONLY way to commit in the Nomi workflow — users should never git commit manually.
---

# Commit Sync — 检查 + Spec 更新 + 进度追踪 + 自动 Commit/Push

这是最核心的 skill。用户写完代码后，不需要自己 `git commit` / `git push`，直接 call 这个 skill。它完成从代码检查到 spec 更新到进度追踪到 commit+push 的完整流程。

所有 spec 更新发生在 commit 之前，确保 commit 出去的代码和 spec 始终一致。

## 前置条件

- 项目 meta/ 目录已初始化
- working directory 有未提交的变更

## 用法

```
# 基本用法（最常用）
> "帮我提交，我刚写完 trigger 的 CRUD API"

# 带描述（帮助更准确匹配 Room）
> "提交一下，改了 desktop-pet 的动画状态机，主要是加了 sleep 和 wake"

# dry-run（只检查不提交）
> "帮我检查一下现在的代码改动，但先不提交"

# 指定范围
> "只提交 src/triggers/ 下面的改动"
```

## 执行步骤

### Phase 1: 变更分析 + Room 分类

1. 运行 `git diff` + `git status` 扫描所有变更（staged + unstaged + untracked）
2. 如果用户指定了范围，只分析指定路径
3. 根据以下信息将文件匹配到对应 Room：
   - 文件路径 vs specs/*.yaml 的 `anchors.file` 字段
   - 用户的描述（如 "trigger 的 CRUD"）
   - 内容变更的语义分析
4. 无法匹配的文件标记为 "untracked file"

### Phase 2: Anchor Tracking

5. 对每个受影响的 Room，检查 specs/*.yaml 的 anchors：

| 变更类型 | 操作 | 自动/人工 |
|---------|------|----------|
| 文件重命名/移动 | 更新 `anchors.file` 路径 | 自动 |
| 函数/类签名变了 | 更新 `anchors.symbols` | 自动 |
| 逻辑修改（语义变更） | 标记 spec 为 `stale` | 自动标记，人工决定怎么改 |
| 新增代码无关联 spec | 建议创建新 spec | 自动建议 |

### Phase 3: Draft Spec 检测

6. 检查是否有 `state: draft` 的 spec 的功能被这次变更实现了：
   - 如果 draft spec 描述的功能对应的代码已写
   - 提示 "发现 draft spec X 的功能已实现，建议升为 active"

### Phase 4: 跨 Room 冲突检测

7. 检查变更是否涉及其他 Room 依赖的接口：
   - 读取 `type: contract` 的 spec
   - 如果 contract spec 的 anchor 被修改 → 查找 depends_on 这个 spec 的其他 Room
   - 输出 conflict warning

### Phase 5: 用户确认

8. 展示 Sync Report：

```
📋 Commit Sync Report
─────────────────────────
Files changed: 8
Rooms affected: 2
 - trigger-mode (5 files)
 - foundation (3 files)

Spec Updates:
 ✅ Auto-updated: 2 anchors
 ⚠️ Stale (needs review): 1
    → constraint-cpu-limit.yaml
      代码新增了 GPU 调用，可能影响 CPU 约束
 🆙 Draft → Active 建议: 1
    → intent-trigger-crud.yaml
      CRUD API 已实现，建议升为 active
 🆕 Untracked files: 1
    → src/triggers/scheduler.py

Conflicts:
 ⛔ Cross-room: 0

Progress:
 trigger-mode: 40% → 60%
   milestone "CRUD API" → completed ✅
 foundation: 60% (unchanged)
 project overall: 15% → 22%

Commit message:
 [trigger-mode] feat: add CRUD API for triggers

Actions:
 1. Auto-update 2 anchors ✅
 2. Mark 1 spec stale ⚠️
 3. Upgrade 1 draft → active? (y/n)
 4. Record change spec + update progress
 5. Commit + Push

确认提交？(y/n/编辑 commit message)
```

9. 等待用户确认。用户可以：
   - 直接确认 `y`
   - 编辑 commit message
   - 对 draft → active 建议逐条确认
   - 取消（先去修问题）

### Phase 6: 执行

10. 用户确认后，按顺序执行：

```
a. 更新 spec yaml 文件
   - auto-update anchors（路径/符号）
   - mark stale specs
   - upgrade confirmed drafts → active
   ↓
b. 创建 change spec
   - 在 Room 的 specs/ 创建 change-{date}-{short-desc}.yaml
   - 记录这次变更的摘要
   ↓
c. 更新 spec.md（同步 Human Projection）
   ↓
d. 更新 progress.yaml（当前 Room）
   - 追加 commit 记录
   - 更新相关 milestone 状态（pending → in_progress → completed）
   - 重算 completion 百分比
   ↓
e. 向上传播 Room 状态
   - 父 Epic Room 的 progress.yaml（重算 completion = avg(子 Room completion)）
   - 00-project-room/progress.yaml（重算整体 completion）
   - timeline.yaml（更新对应 Room 的 status + completion）
   ↓
f. git add（代码 + 所有更新过的 meta 文件一起 stage）
   ↓
g. git commit -m "[room-id] type: description"
   ↓
h. git push
```

### Phase 7: 提交后验证

11. commit 成功后，逐项验证（不通过则立即修复并 amend）：

```
a. Anchor 完整性
   - 对每个受影响 Room 下的 specs/*.yaml，确认 anchors 非空
   - anchors.file 指向的文件确实存在于本次 commit
   - anchors.symbols 在对应文件中存在

b. Progress 完整性
   - 对每个受影响 Room 的 progress.yaml，确认 commits 列表包含本次 commit hash
   - 新增/完成的 milestone 已记录

c. 向上传播完整性
   - 00-project-room/progress.yaml 的 completion 已更新
   - timeline.yaml 对应 Room 的 status + completion 已更新

d. Stage 完整性
   - git diff --name-only 确认无遗漏的 META 文件未 staged
```

如果验证发现遗漏，立即修复并 `git add + git commit --amend`。

12. 输出最终确认：
```
✅ Committed: abc1234
✅ Pushed to origin/main
📄 Specs updated: 3 files
📊 Progress: trigger-mode 40% → 60%
✅ Phase 7 验证通过
```

## Commit Message 格式

```
[room-id] type: description

# 多 Room 时
[trigger-mode][foundation] feat: add trigger CRUD with shared API conventions

# type 对照
feat     — 新功能
fix      — Bug 修复
refactor — 重构（无功能变更）
docs     — 文档/spec 更新
test     — 测试
chore    — 构建/工具链
```

## 状态向上传播规则

```
Feature Room progress 更新
  → 父 Epic Room progress.yaml 重算 completion = avg(子 Room completion)
  → 00-project-room/progress.yaml 重算 completion = avg(所有 Epic completion)
  → timeline.yaml 更新对应 Room 的 status + completion
```

| 子 Room 状态 | 父 Room 状态变化 |
|-------------|----------------|
| 第一个子 Room 开始 in_progress | 父 Room → in_progress |
| 所有子 Room completed | 父 Room → completed |
| 任一子 Room 有 stale spec | 父 Room issues += 1 |

## dry-run 模式

用户说 "检查但先不提交" 时：
- 运行 Phase 1-4 完整检查管线
- 展示 Sync Report
- 不执行 Phase 6（不做任何文件修改和 git 操作）
- 用户可以先修完问题再 call 正式提交

## Spec Type 参考

| Spec Type | 含义 | 示例 |
|-----------|------|------|
| **intent** | 做什么，功能目标 | "用户可以将商品加入购物车并结算" |
| **decision** | 已确认的技术/产品决策 | "使用 Stripe Checkout server-side 模式" |
| **constraint** | 不做什么，边界限制（must / must_not / should / should_not） | "Phase 1 不含优惠券计算" |
| **convention** | 团队约定/编码规范 | "所有 API 使用 JSON:API 规范" |
| **contract** | 接口契约、API Schema、模块间协议 | "RefundService 接口定义" |
| **context** | 背景信息、实验记录、数据假设 | "模型基线准确率 92%" |
| **change** | 一次具体变更的记录 | "本次 PR 把 session 改为 stateless" |

## progress.yaml commit 记录格式

```yaml
commits:
  - hash: "{git-hash}"
    date: "{today}"
    message: "{commit message}"
    files_changed: {N}
    specs_affected:
      - {spec-id} ({operation description})
    milestones_affected:
      - {milestone-id} (→ {new status})
```
