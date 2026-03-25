---
name: room
description: >
  Manage Nomi Feature Rooms through natural language — create, query, update, delete, move rooms and manage spec states. Use whenever the user wants to interact with Room structure: "给xx加一个room", "xx现在什么状态", "记录一个决策", "把xx标记为in-dev", "review draft specs", "加一条约束", or any CRUD operation on Feature Rooms and their specs. Also triggers on "room" mentions in the context of Nomi project management.
---

# Room — Feature Room 增删查改 + Spec 状态管理

通过自然语言管理 Feature Room 的一切操作。这是日常开发中最常用的 skill。

## 前置条件

- `meta/` 目录已初始化（至少有 `_tree.yaml`）
- 项目的 Room 结构已存在

## 操作类型

根据用户的自然语言输入，识别以下操作意图：

### 1. 创建 Room（Create）

用户说："给 quick-capture 加一个子 room，叫 trigger-mode"

执行：
1. 在父 Room 目录下创建新目录
2. 自动编号（查看同级目录确定下一个编号）
3. 推断 `type`（epic / feature）和 `owner`
4. 生成文件：
   - `room.yaml`（元数据）
   - `spec.md`（Human Projection 模板）
   - `progress.yaml`（空 milestones）
   - `specs/`（空目录）
5. 子 Room 自动继承父 Room 的 constraints 和 conventions
6. 更新 `_tree.yaml`
7. 展示创建结果

### 2. 查询 Room（Read）

用户说："trigger-mode 现在什么状态？" / "哪些 room 还没写 spec？"

执行：
1. 读取对应 room.yaml / 遍历所有 Room
2. 格式化展示：lifecycle、spec 数量（按 type 分）、stale 数、prompt_test 状态、进度
3. 如果是查询 spec 内容，读取 specs/ 目录

### 3. 修改 Room（Update）

用户说："把 trigger-mode 标记为 in-dev" / "给 trigger-mode 加一条约束"

分两类操作：

**修改 Room 元数据：**
- 更新 room.yaml 的对应字段（lifecycle, owner, contributors 等）
- 同步更新 _tree.yaml

**添加/修改 Spec：**
- 在 specs/ 中创建或更新 yaml 文件
- 同步更新 spec.md（Human Projection）
- spec yaml 格式：

```yaml
spec_id: "{type}-{room-id}-{NNN}"
type: {intent|decision|constraint|contract|convention|change|context}
state: {draft|active}
intent:
  summary: "{一句话}"
  detail: |
    {详细描述}
constraints: []
indexing:
  type: {same as type above}
  priority: {P0|P1|P2}
  layer: {epic|feature}
  domain: "{domain}"
  tags: [{tags}]
provenance:
  source_type: manual_input
  confidence: 1.0
  source_ref: "用户手动记录"
relations: []
anchors: []
```

用户手动添加的 spec 可以直接设为 `active`（不需要经过 draft）。

### 4. Spec 状态管理（Review）

用户说："review 一下 desktop-pet 的 draft specs" / "把 EventKit 决策升为 active"

**单条操作：**
- 找到对应 spec yaml
- 更新 `state` 字段（draft → active）
- 同步更新 spec.md

**批量 Review：**
1. 读取指定 Room 的所有 `state: draft` 的 spec
2. 逐条展示，包含 summary、detail、provenance（来源和置信度）
3. 对每条让用户选择：
   - ✅ 升为 active
   - ✏️ 编辑后升为 active
   - ⏸️ 保持 draft
   - ❌ 删除
4. 批量执行用户的选择
5. 同步更新 spec.md

### 5. 删除/归档（Delete/Archive）

用户说："calendar-sync 先不做了，归档掉"

- 更新 room.yaml 的 `lifecycle: archived`
- 更新 _tree.yaml 标记
- 不删除目录（归档 ≠ 删除）

### 6. 移动 Room（Move）

用户说："把 map-widget 从 knowledge-base 移到 desktop-pet 下面"

1. 移动目录
2. 更新 _tree.yaml
3. 更新 room.yaml 的 parent
4. 重新继承新父 Room 的 specs
5. 检查是否有 broken 的 depends_on 引用

## 执行流程

```
用户自然语言输入
  ↓
1. 意图识别：create / read / update / delete / move / review
  ↓
2. 实体提取：room name、parent、owner、lifecycle、spec 内容等
  ↓
3. 缺失信息推断或追问：
   - 能推断的直接推断（如 "AI 引擎" → owner=backend）
   - 无法推断的追问（如 "放在哪个 Epic 下？"）
  ↓
4. 执行操作 + 更新相关文件
  ↓
5. 输出操作摘要 + 受影响的文件列表
```

## spec.md 同步规则

每次操作 specs/ 中的 yaml 后，都需要同步更新 spec.md：
- Intent 变更 → 更新 "Intent" 段落
- Decision 变更 → 更新 "Decisions" 表格
- Constraint 变更 → 更新 "Constraints" 段落
- Contract 变更 → 更新 "Contracts" 段落
- 其他类型 → 添加/更新对应段落

spec.md 是人类可读视图，specs/*.yaml 是机器可读的源数据。两者保持同步。
