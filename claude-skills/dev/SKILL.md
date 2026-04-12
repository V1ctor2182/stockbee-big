---
name: dev
description: >
  一句话触发 milestone 级开发。自动 pull context → 开发 → commit-sync。
  用户说 "做 news-data m1"、"开发 macro-data 的 Z-score 计算"、"下一个"、
  "dev 01-stock-data/m3" 时触发。这是日常开发的主入口 skill。
---

# Dev — Milestone 级开发（context pull → 实现 → commit-sync）

一个 prompt 完成一个 milestone 的完整开发周期。用户不需要手动调 prompt-gen 和 commit-sync。

## 触发词

- "做 {room} m{N}"、"开发 {room} 的 {milestone 名}"
- "dev {room-id}/{milestone-id}"
- "下一个" / "继续" （在上一个 milestone 完成后）
- "实现 {功能描述}" （AI 推断 room + milestone）

## 前置条件

- meta/ 目录已初始化，Room 和 milestones 已定义
- Provider 架构已完成（07-provider-arch ✅）

## 执行步骤

### Phase 1: 定位 Milestone

1. 从用户输入解析 Room + Milestone：
   - 显式指定："做 news-data m1" → room=02-news-data, milestone=m1
   - "下一个" → 读当前 Room 的 progress.yaml，找下一个 `status: pending` 的 milestone
   - 语义推断："实现 Z-score 计算" → 匹配 milestone name

2. 读取 Room 的 progress.yaml，确认该 milestone 状态为 `pending` 或 `in_progress`
   - 如果 `completed` → 提示用户 "m1 已完成，要重做还是跳到下一个？"

3. 检查前置 milestone 是否已完成：
   - 如果 m3 依赖 m1/m2 的产出，但 m1/m2 未完成 → 警告用户

### Phase 2: Context Pull（内置 prompt-gen 逻辑）

4. 拉取**该 milestone 需要的**上下文（不是整个 Room 的全量上下文）：

   **必拉**：
   - Room 的 intent spec（理解整体目标）
   - 该 milestone 相关的 constraint / contract specs
   - 已完成 milestone 的产出摘要（作为前置条件）
   - 父 Room + 全局继承的 constraints

   **选拉**（根据 milestone 内容智能判断）：
   - Tech Design 中相关段落
   - 兄弟 Room 的 contract specs（如果有跨 Room 依赖）
   - 已有代码的接口签名（如果本 milestone 要实现某个 interface）

   **不拉**：
   - 未来 milestone 的详细设计
   - 不相关 Room 的 specs

5. 组装 milestone-scoped prompt（内部使用，不展示给用户）

### Phase 2.5: 多方案规划 + 双重 Review（编码前必经）

**不直接写代码。** 先生成方案，review 通过后再动手。

6. **生成实施方案**：

   **简单 milestone 走快速路径（主会话直接生成，不启动 sub-agent）**：
   当 milestone 同时满足以下条件时，主会话直接生成**单方案**，跳过多方案对比和 sub-agent：
   - 无 `complexity_flags`
   - `estimated_lines ≤ 120`
   - 有可参考的同类已完成 milestone（模式明确）

   快速路径产出：
   ```
   Plan: {名称}
     步骤: {numbered sub-tasks}
     每步: 内容 / 预估行数 / 测试点
     Corner cases: {简要列出}
     总计: ~{N} 行 code + ~{N} 行 test
   ```
   → 直接进入 step 8 展示给开发者确认。

   **复杂 milestone 走完整路径（启动 planning sub-agent）**：
   当 milestone 有 `complexity_flags`、`estimated_lines > 120`、或无先例模式时，
   启动 planning sub-agent（不写代码），产出两套方案：

   ```
   Plan A: {名称} (如 Bottom-Up)
     思路: {1-2 句}
     步骤: {numbered sub-tasks with dependencies}
     每步: 内容 / 预估行数 / 测试点
     Corner cases 识别:
       - {edge case 1}: 如何处理
       - {edge case 2}: 如何处理
     总计: ~{N} 行 code + ~{N} 行 test
     优点: ...
     缺点: ...

   Plan B: {名称} (如 Top-Down Vertical Slice)
     ...
   ```

   方案必须覆盖：
   - **正常路径**: 功能实现步骤
   - **Corner cases**: 空输入、异常数据、超时、并发、边界值
   - **Error handling**: 哪些错误需要 retry、fallback、还是直接失败
   - **测试策略**: 每个 sub-task 测什么、怎么 mock

   如果 milestone 有 `complexity_flags`（来自 plan-milestones），重点分析该类复杂度。
   如果有 `open_questions`，必须在方案中列出并标记需要开发者决定。

7. **AI 自身 Review**（仅完整路径，快速路径跳过此步）：

   对两套方案做对比 review，给出推荐并说明理由：
   - 哪个方案更适合当前代码量/复杂度
   - 哪些 corner cases 两套方案处理方式不同
   - 是否有需要锁定的设计决策（不可轻松逆转的）
   - 是否建议合并/调整步骤

8. **展示方案 + Review 结果，等待开发者确认**：

   ```
   📋 Milestone 实施方案: {room} {milestone}
   ────────────────────────────────

   Plan A: {名称}
     [步骤列表 + corner cases]

   Plan B: {名称}
     [步骤列表 + corner cases]

   🔍 AI Review:
     推荐 Plan {X}，理由: ...
     ⚠️ 需要你决定:
       - Q1: {设计问题}
       - Q2: {设计问题}

   选择方案？(A/B/调整后再看)
   ```

   开发者可以：
   - 选 A 或 B
   - 要求调整（"A 的步骤但用 B 的 error handling 策略"）
   - 回答设计问题
   - 要求 sub-agent 深入分析某个问题

   **循环直到开发者确认方案。确认前不写任何代码。**

### Phase 3: 实现（按确认的方案执行）

9. 按开发者选定的方案逐步实现：
   - 严格按方案的 sub-task 顺序执行
   - 遵循已有的代码模式（如 stock_data 的 Provider 模式）
   - 只做该 milestone 范围内的事
   - 写方案中列出的测试（包括 corner case 测试）

10. 运行测试验证：
   - 新测试必须全部通过
   - 已有测试不能 break

### Phase 4: Commit-Sync（内置 commit-sync 逻辑）

11. 分析本次变更，生成 commit：
    - commit message 格式：`[room-id] type: milestone 描述`
    - git add 只 stage 本 milestone 相关的文件
    - git commit + git push

12. 更新 Room 元数据（**只更新该 milestone 的部分**）：
    - progress.yaml: 该 milestone → `completed`，追加 commit hash
    - spec.md: 增量更新（不重写整个文件）
    - 如果是最后一个 milestone → 更新 room.yaml lifecycle、_tree.yaml status

13. **不做**整个 Room 的 spec 大重写（那是全部完成后的事）

### Phase 5: 暂停等待

14. 展示完成摘要：
```
✅ [03-macro-data] m1-Parquet时间序列 completed
   commit: abc1234
   files: 2 added, 0 modified
   tests: 8 new, 141 total (all pass)
   progress: 03-macro-data 20% (1/5 milestones)

下一个 milestone: m2-FRED指标采集
说 "下一个" 继续，或给其他指令。
```

15. **等待用户决定**，不自动继续下一个 milestone

## 特殊情况

### "下一个" 逻辑

- 在同一个 Room 内：找下一个 `pending` milestone
- 当前 Room 全部完成：提示 "Room X 已全部完成，要做哪个 Room？"
- 无上下文（新对话）：提示 "要做哪个 Room 的哪个 milestone？"

### 一个 milestone 太大

如果 Phase 2.5 的方案预估 >200 行代码，可以：
- 在方案中拆成多个 sub-task（每个 sub-task 可以独立 commit）
- milestone 状态只在全部 sub-task 完成后才标记 completed
- 方案的 sub-task 粒度应该在 Phase 2.5 就和开发者确认好

### milestone 之间有代码依赖

- m2 的代码 import 了 m1 的产出 → Phase 2 context pull 会自动包含 m1 的代码
- 不需要用户手动说明依赖关系

## 和其他 skill 的关系

| Skill | 何时用 | 和 dev 的关系 |
|-------|-------|--------------|
| prompt-gen | 想看完整 Room prompt 但不开发 | dev 内置了 milestone 级的 prompt-gen |
| commit-sync | 手动修改代码后想提交 | dev 内置了 commit-sync |
| room-status | 看项目全局进度 | dev 完成后可以调 room-status 查看 |
| room | 创建/修改 Room 结构 | dev 之前先用 room 创建好 Room 和 milestones |
