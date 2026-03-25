---
name: timeline-init
description: >
  Generate development timeline and milestones for the Nomi project after room-init. Creates timeline.yaml (project-level schedule with dependencies) and progress.yaml for each Room (milestones inferred from intent specs). Use when the user says "生成时间线", "排期", "timeline", "帮我排时间", or after room-init to set up progress tracking. Also use when new Rooms are added and timeline needs updating.
---

# Timeline Init — 生成开发排期 + Milestones

在 `room-init` 创建完 Room 树之后运行。读取现有的 Room 结构，自动生成开发时间线和每个 Room 的 milestones，为后续的进度追踪打基础。

## 前置条件

- `room-init` 已完成（`_tree.yaml` 和各 `room.yaml` 已存在）
- 各 Room 的 `specs/` 中至少有 intent specs
- 用户可选提供 deadline 约束（如 "4月底之前发 MVP1"）

## 执行步骤

### Phase 1: 依赖分析

1. 读取 `00-project-room/_tree.yaml` 获取所有 Room
2. 读取各 `room.yaml` 的 `depends_on` 字段
3. 构建依赖图（DAG）
4. 拓扑排序，确定开发顺序
5. 识别关键路径（最长依赖链）

### Phase 2: Milestone 推断

6. 对每个 Room，读取 `specs/` 中的 intent specs，推断需要完成的工作：
   - 每个 intent spec 对应一个 milestone
   - 有 contract spec 的加一个 "接口设计" milestone
   - 每个 Room 默认追加："测试" + "文档"

7. 估算每个 milestone 的工作量（粗略）：
   - 单个 intent → 2-3 天
   - 有 constraint 的 intent → +1 天
   - 有 contract 依赖的 → +1 天

### Phase 3: 排期生成

8. 根据依赖图 + 工作量估算生成时间线：
   - 无依赖的 Room 可以并行
   - 有依赖的 Room 串行
   - 如果有 deadline → 检查是否可行，不可行则提示

9. **展示排期规划，等待用户确认：**

```
📅 Timeline Plan

Phase 1: MVP1 (3/10 - 4/20)

3/10 ████████░░░░░░░░░░░░░ 3/20
     foundation (11d)
       milestones: DB层, API框架, 权限, 测试

3/18 ░░░░████████░░░░░░░░░ 4/1
     desktop-pet (14d)
       depends: foundation
       milestones: 渲染引擎, 动画状态机, 交互事件

3/25 ░░░░░░░░████████░░░░░ 4/8
     note-mode (14d)
       depends: foundation

4/1  ░░░░░░░░░░░░████████░ 4/15
     trigger-mode (15d)
       depends: foundation, ai-engine

⚠️ 关键路径: foundation → trigger-mode
   总计 36 天，deadline 41 天 ✅

确认？(y/n/调整)
```

10. 用户可能会：
    - 直接确认
    - 调整日期
    - 修改 milestones
    - 改变 Room 的开发顺序

### Phase 4: 生成文件

11. 用户确认后生成：

**`00-project-room/timeline.yaml`：**
```yaml
timeline:
  generated_at: "{today}"
  updated_at: "{today}"
  phases:
    - id: "phase-1"
      name: "{phase name}"
      target_start: "{date}"
      target_end: "{date}"
      rooms:
        - room: "{room-id}"
          order: 1
          target_start: "{date}"
          target_end: "{date}"
          status: planning
          completion: 0.0
          depends_on: ["{dep-room-id}"]
  overall:
    total_rooms: {N}
    completed: 0
    in_progress: 0
    planning: {N}
    completion: 0.0
```

**每个 Room 的 `progress.yaml`：**
```yaml
progress:
  completion: 0.0
  milestones:
    - id: "m1-{short-name}"
      name: "{milestone 名称}"
      status: pending
      completed_at: null
    - id: "m2-{short-name}"
      name: "{milestone 名称}"
      status: pending
      completed_at: null
  commits: []
```

**`00-project-room/progress.yaml`（项目整体）：**
```yaml
progress:
  completion: 0.0
  milestones: []      # 项目级不细分 milestone，从子 Room 聚合
  commits: []
```

12. 输出确认：
```
✅ Timeline Init 完成

生成了:
- timeline.yaml（{phase_count} 个阶段, {room_count} 个 Room）
- {room_count} 个 progress.yaml（共 {milestone_count} 个 milestones）

关键路径: {path}
预计完成: {date}

下一步: 开始开发后用 commit-sync 提交，进度会自动更新。
```

## 增量更新

如果已有 timeline.yaml，用户要求更新：
- 只处理新增的 Room
- 根据依赖关系插入到现有 timeline 中
- 不动已有 Room 的排期和 milestones
- 如果用户要求修改特定 Room 的 milestones，直接更新该 Room 的 progress.yaml

## 与其他 Skill 的关系

- **commit-sync** 每次提交后更新 progress.yaml（记录 commit + 更新 milestone 状态 + 向上传播完成度）
- **prompt-gen** pull 上下文时带上进度信息，让 AI 知道什么已经做了
- **room-status** 读取 timeline.yaml + progress.yaml 展示项目全景
