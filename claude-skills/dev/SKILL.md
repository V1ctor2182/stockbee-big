---
name: dev
description: >
  一句话触发 milestone 级开发。自动 pull context → 开发 → commit-sync。
  用户说 "做 news-data m1"、"开发 macro-data 的 Z-score 计算"、"下一个"、
  "dev 01-stock-data/m3" 时触发。这是日常开发的主入口 skill。
---

# Dev — Milestone 级开发（context pull → 实现 → commit-sync）

## 触发词

- "做 {room} m{N}"、"开发 {room} 的 {milestone 名}"
- "dev {room-id}/{milestone-id}"
- "下一个" / "继续" （在上一个 milestone 完成后）
- "实现 {功能描述}" （AI 推断 room + milestone）

## 前置条件

- meta/ 目录已初始化，Room 和 milestones 已定义
- Provider 架构已完成（07-provider-arch ✅）

## 执行步骤

### Phase 0: Repo 状态预检查

1. 运行 `git status`。如果有**非本 milestone 相关**的未提交改动（上次遗留 debug、别 room 修改等），
   问用户三选一：(a) 先 commit-sync 掉 / (b) git stash / (c) ignore（commit 时只 stage 本 milestone 文件）。
   改动就是本 milestone 续做则直接进 Phase 1。

### Phase 1: 定位 Milestone

2. 解析 Room + Milestone：
   - 显式："做 news-data m1" → room=02-news-data, milestone=m1
   - "下一个" → 读当前 Room progress.yaml 找下一个 `status: pending`
   - 语义推断："实现 Z-score 计算" → 匹配 milestone name

3. 读 Room progress.yaml，确认 milestone 为 `pending` / `in_progress`。
   `completed` → 提示用户 "已完成，重做还是跳下一个？"

4. 检查前置 milestone 已完成；未完成则警告。

### Phase 2: Context Pull（内置 prompt-gen 逻辑）

5. 按下表拉 milestone-scoped 上下文：

| 类别 | 内容 |
|------|------|
| 必拉 | Room intent spec、milestone 相关 constraint/contract、已完成 milestone 产出摘要、父 Room + 全局 constraints |
| 选拉 | Tech Design 相关段、兄弟 Room contract specs（跨 Room 依赖）、已有代码接口签名（本 milestone 实现 interface 时） |
| 不拉 | 未来 milestone 详细设计、不相关 Room specs |

6. 组装 milestone-scoped prompt（内部用，不展示）。

### Phase 2.5: 多方案规划 + 双重 Review（编码前必经）

7. **生成实施方案**：

   **快速路径**（主会话直接生成 1 个方案，不启 sub-agent）——同时满足：
   - 有可参考的同类已完成 milestone（模式明确）← 主导条件
   - 无 `complexity_flags`

   `estimated_lines` 不再是硬门槛——模式明确即使 280 行也走快速路径。

   **完整路径**（启 planning sub-agent，产出 Plan A + Plan B）——有 `complexity_flags`、无先例模式、或开发者显式要求。

   **Plan 模板（两路径共用）**：
   ```
   Plan {名称}:
     思路: {1-2 句}（完整路径必填，快速路径可省）
     步骤: numbered sub-tasks，每步含内容 / 预估行数 / 测试点
     Corner cases（必覆盖以下类别，缺项要说明为什么不适用）:
       - 空输入: tickers=[], factor_names=[], empty DataFrame
       - 类型边界: datetime.date vs Timestamp, scalar vs Series
       - 资源缺失: setter 未注入, 文件不存在, 字段不存在
       - 名字冲突 / 重复 / shadow
       - 错误传播: 下游 raise 时的处理方式
     优缺点: {至少各 1 条}
   ```

   有 `complexity_flags` 时，在方案里重点分析该类复杂度。
   有 `open_questions` 时，必须在方案里列出并标记需开发者决定。

8. **AI 自身 Review**（仅完整路径）：对 Plan A/B 做对比 review，给出推荐并说明：
   - 哪个更适合当前代码量/复杂度
   - corner cases 两套方案处理差异
   - 需锁定的不可逆设计决策
   - 是否合并/调整步骤

9. **展示方案 + Review + 等待确认**：

   ```
   📋 Milestone 实施方案: {room} {milestone}
   Plan A: {名称} [步骤 + corner cases + 优缺点]
   Plan B: {名称} [同上]  # 仅完整路径

   🔍 AI Review: 推荐 Plan {X}，理由: ...
   ⚠️ Open Questions:
     - Q1: {问题}
     - Q2: {问题}
   选择方案？(A/B/调整/回答问题)
   ```

   开发者可：选 A/B、要求调整、回答 OQ、要 sub-agent 深入分析、
   或对 OQ 说"你给建议" → AI 给 tradeoff 但**不替开发者拍板**。

   **Open Questions 全部锁定、方案确认后才进 Phase 3。未锁定不许"先写着看"**。

### Phase 3: 实现

10. **Smoke-test real interfaces**（写详细测试前必做）——用 2 秒脚本验证假设：

    ```bash
    python -c "
    from stockbee.factor_data import Alpha158
    a = Alpha158()
    print(a.list_factor_names()[:20])
    print('KMID' in a, a.max_lookback('MA5'))
    "
    ```
    确认：导入路径、公开 API 实际返回类型/形状/名字、依赖对象能构造。

11. 按选定方案逐步实现：严格按 sub-task 顺序、遵循已有代码模式、只做本 milestone 范围、写方案中列出的测试（含 corner case）。

12. 运行测试：新测试全过 + 已有测试不 break。

### Phase 3.5: Post-Implementation Code Review（非 trivial milestone 必经）

13. **跳过条件**（任一满足即可跳过）：
    - milestone < 50 行纯 glue 代码
    - 纯重命名 / 搬运，无新逻辑
    - 开发者显式说 "跳过 review，赶时间"

14. 否则并行启动两个 reviewer：
    - **Reviewer A: Codex CLI**（codex:rescue 或 codex review）——独立视角、查真实 API、找类型不匹配 / 签名错用 / 契约违反
    - **Reviewer B: Plan sub-agent**——纯代码角度、无 plan bias、找 edge case / 空输入 / 并发 / 资源生命周期

    每个 reviewer 的 prompt 必含：
    - 新增/修改文件的绝对路径
    - 依赖模块的接口签名
    - 关键决策背景
    - 要求按 severity（CRITICAL/HIGH/MEDIUM/LOW）分级 + 具体行号 + 修复建议

15. **汇总 findings → 过滤 → 修复**：
    - CRITICAL + HIGH：必修，补测试验证
    - MEDIUM：默认修，除非开发者明说 defer
    - LOW：展示给开发者逐条决定 take/defer

    ```
    🔍 Code Review 汇总
      Codex: 2 CRITICAL / 1 HIGH / 3 MEDIUM
      Subagent: 0 CRITICAL / 2 HIGH / 4 MEDIUM / 3 LOW
      必修: ✴️ [C1] date vs Timestamp 类型 mismatch (line 145) → normalize helper
            ✴️ [C2] 空 ohlcv 未短路 crash Evaluator (line 148) → empty guard
            ⚠️ [H1] IC 前向收益截尾 (line 264) → 多取 shift 天
      LOW（要 defer？）: · [L1] list_factors schema 不统一 → TypedDict / ...
    修复并重跑测试？(y/n/选择性)
    ```
   修复展示格式:
  每条: "[级别] [文件:行号] — 改了什么（一句话）— 为什么（一句话）"
  如需看完整改动: 用户说 "展开 C1" 才展示那一段 diff
  禁止: 用自然语言段落描述修改过程
16. 修复后重跑全量测试，必须全绿才进 Phase 4。

### Phase 4: Commit-Sync（内置 commit-sync 逻辑）

17. 生成 commit：格式 `[room-id] type: milestone 描述`；只 stage 本 milestone 相关文件（Phase 0 发现的无关改动不 stage）；commit + push。

18. 更新 Room 元数据（**只更新该 milestone 部分**）：
    - progress.yaml: milestone → `completed`、追加 commit hash
    - spec.md: 增量更新（不重写整个文件）
    - 如果是最后一个 milestone → 更新 room.yaml lifecycle、_tree.yaml status

19. **禁止**整个 Room 的 spec 大重写（那是全部完成后的事）。

### Phase 5: 暂停等待

20. 展示完成摘要：
    ```
    ✅ [03-macro-data] m1-Parquet时间序列 completed
       commit: abc1234  files: 2 added, 0 modified  tests: 8 new / 141 all pass
       review: 2 CRITICAL + 1 HIGH fixed, 3 LOW deferred
       progress: 03-macro-data 20% (1/5)
    下一个 milestone: m2-FRED指标采集
    说 "下一个" 继续，或给其他指令。
    ```

21. **等用户决定，不自动续做**。

## 特殊情况

### "下一个" 逻辑

- 同一 Room 内：找下一个 `pending` milestone
- 当前 Room 全部完成：提示 "Room X 已全部完成，要做哪个 Room？"
- 无上下文（新对话）：提示 "要做哪个 Room 的哪个 milestone？"

### 一个 milestone 太大

方案预估 >200 行时：拆成多个 sub-task（每个可独立 commit）。
milestone 状态**只在全部 sub-task 完成后**才标记 `completed`。
sub-task 粒度在 Phase 2.5 就和开发者确认好。

## 和其他 skill 的关系

| Skill | 何时用 | 关系 |
|-------|-------|------|
| prompt-gen | 想看完整 Room prompt 但不开发 | dev 内置 milestone 级 prompt-gen |
| commit-sync | 手动修改代码后想提交 | dev 内置 commit-sync |
| room-status | 看项目全局进度 | dev 完成后可调 room-status 查看 |
| room | 创建/修改 Room 结构 | dev 之前先用 room 创建好 Room 和 milestones |
