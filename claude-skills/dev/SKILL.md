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

### Phase 3: 实现

6. 根据 context 开发代码：
   - 遵循已有的代码模式（如 stock_data 的 Provider 模式）
   - 只做该 milestone 范围内的事
   - 写对应的测试

7. 运行测试验证：
   - 新测试必须全部通过
   - 已有测试不能 break

### Phase 4: Commit-Sync（内置 commit-sync 逻辑）

8. 分析本次变更，生成 commit：
   - commit message 格式：`[room-id] type: milestone 描述`
   - git add 只 stage 本 milestone 相关的文件
   - git commit + git push

9. 更新 Room 元数据（**只更新该 milestone 的部分**）：
   - progress.yaml: 该 milestone → `completed`，追加 commit hash
   - spec.md: 增量更新（不重写整个文件）
   - 如果是最后一个 milestone → 更新 room.yaml lifecycle、_tree.yaml status

10. **不做**整个 Room 的 spec 大重写（那是全部完成后的事）

### Phase 5: 暂停等待

11. 展示完成摘要：
```
✅ [03-macro-data] m1-Parquet时间序列 completed
   commit: abc1234
   files: 2 added, 0 modified
   tests: 8 new, 141 total (all pass)
   progress: 03-macro-data 20% (1/5 milestones)

下一个 milestone: m2-FRED指标采集
说 "下一个" 继续，或给其他指令。
```

12. **等待用户决定**，不自动继续下一个 milestone

## 特殊情况

### "下一个" 逻辑

- 在同一个 Room 内：找下一个 `pending` milestone
- 当前 Room 全部完成：提示 "Room X 已全部完成，要做哪个 Room？"
- 无上下文（新对话）：提示 "要做哪个 Room 的哪个 milestone？"

### 一个 milestone 太大

如果 AI 判断某个 milestone 的实现需要 >200 行代码，可以：
- 拆成多个 commit（同一 milestone 内）
- 但仍然在一次交互中完成
- milestone 状态只在全部完成后才标记 completed

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
