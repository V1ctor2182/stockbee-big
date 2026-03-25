---
name: room-init
description: >
  Initialize the Nomi project's Feature Room tree from a PRD document. Creates the complete meta/ directory structure with Room directories, initial intent and constraint specs, _tree.yaml, and project.yaml. Use this skill when starting a new Nomi project, when the user says "初始化项目", "init rooms", "从PRD创建Room", or when setting up the VibeHub Feature Room structure for the first time. Also use when a PRD is updated and new modules need to be added to the existing Room tree.
---

# Room Init — 从 PRD 冷启动 Feature Room 树

这个 skill 在项目启动时运行一次。输入一份 PRD 文档，自动分析功能模块，创建完整的 Feature Room 树结构，并为每个 Room 生成默认的 `intent` 和 `constraint` spec。

## 前置条件

- 用户提供了 PRD 文件路径（如 `nomi-prd.md`）
- 可选：MVP 范围文件（如 `nomi-prd-mvp1.md`）
- 项目的 `meta/` 目录尚未初始化（或只需要增量添加新 Room）

## 执行步骤

### Phase 1: PRD 分析

1. 读取用户指定的 PRD 文件
2. 提取功能模块，识别：
   - **Epic 级模块**（大的功能域，如"桌宠系统"、"Quick Capture"）
   - **Feature 级模块**（具体功能点，如"Note 模式"、"Trigger 模式"）
   - **父子关系**（哪些 Feature 属于哪个 Epic）
3. 对每个模块提取：
   - 功能目标（将成为 `intent` spec）
   - 约束条件（将成为 `constraint` spec）
   - 技术栈偏好（将成为 `room.yaml` 的 `owner` 字段）

### Phase 2: Room 树规划（先确认，不直接创建）

4. 生成 Room 树规划并展示给用户：
   - 自动编号：`00-project-room`, `01-xxx`, `02-xxx`...
   - 推断 `type`（epic / feature）
   - 推断 `owner`（frontend / backend / fullstack）
   - 如果有 MVP 文件，设置 `lifecycle`（planning / backlog）

展示格式：
```
我从 PRD 中识别出以下 Room 结构：

00-project-room/
├── 01-foundation/          [Epic] backend
├── 02-desktop-pet/         [Epic] frontend
├── 03-quick-capture/       [Epic] frontend
│   ├── 01-note-mode/       [Feature] frontend
│   └── 02-trigger-mode/    [Feature] backend
├── 04-knowledge-base/      [Epic] backend
└── 05-ai-engine/           [Epic] backend

每个 Room 的 intent 摘要：
- 01-foundation: 基础架构层，DB + API + 权限 + 错误处理
- 02-desktop-pet: 桌宠常驻桌面，作为 Nomi 入口
  ...

请确认或调整。
```

5. **等待用户确认**。用户可能会：
   - 直接确认
   - 调整 Room 结构（增删、改名、改层级）
   - 修改 owner 或 lifecycle

### Phase 3: 批量创建

6. 用户确认后，创建目录结构。对每个 Room 执行：

**创建目录和文件：**
```
{room-name}/
├── room.yaml         # Room 元数据
├── spec.md           # Human Projection（聚合视图）
├── progress.yaml     # 进度追踪（初始为空 milestones）
└── specs/            # Spec Objects 目录
    ├── intent-{name}.yaml
    └── constraint-{name}.yaml（如果 PRD 中有约束）
```

**room.yaml 格式：**
```yaml
room:
  id: "{room-id}"
  name: "{room-name-中文}"
  parent: "{parent-id}"
  lifecycle: planning        # 或 backlog（根据 MVP 范围）
  created_at: "{today}"
  updated_at: "{today}"
  owner: {owner}
  contributors: []
  prompt_test:
    passable: false
    last_tested: null
    token_count: null
  depends_on: []             # 用户后续可以手动添加
```

**intent spec yaml 格式：**
```yaml
spec_id: "intent-{room-id}-001"
type: intent
state: draft
intent:
  summary: "{从 PRD 提取的一句话功能目标}"
  detail: |
    从 PRD 提取：
    {PRD 中关于这个功能的详细描述}
constraints: []
indexing:
  type: intent
  priority: high
  layer: {epic 或 feature}
  domain: "{domain}"
  tags: [{relevant tags}]
provenance:
  source_type: prd_extraction
  confidence: 0.8
  source_ref: "{prd-filename}#{section}"
relations: []
anchors: []
```

**constraint spec yaml 格式类似，`type: constraint`，`confidence` 通常 0.85-0.9。**

7. 创建项目级文件：
   - `00-project-room/_tree.yaml`：完整 Room 树索引
   - `meta/project.yaml`：项目配置（技术栈从 PRD 提取）
   - `00-project-room/specs/`：项目级 convention 和 constraint

### Phase 4: 输出报告

8. 展示初始化报告：
```
🚀 Room Init 完成

PRD: nomi-prd.md
创建了 {N} 个 Room（{epic_count} Epic + {feature_count} Feature）
生成了 {spec_count} 个 Spec（{intent_count} intent + {constraint_count} constraint）
所有 spec 状态: draft（需要 review 后升为 active）

⚠️ 需要人工补充：
- {列出 PRD 中信息不足的 Room}

下一步：
1. 运行 timeline-init 生成开发排期
2. 用 room skill 补充缺失的 spec
3. review draft specs 并升为 active
```

## 重要规则

- 所有从 PRD 提取的 spec 默认 `state: draft`，`provenance.source_type: prd_extraction`
- 子 Room 自动继承父 Room 的 constraints 和 conventions（在 spec.md 的 "Inherited Specs" 段落中体现）
- Room 编号 = 大致的开发顺序，不是优先级
- `_tree.yaml` 是 Room 树的唯一索引，其他 skill 依赖它来遍历所有 Room
- 如果是增量初始化（PRD 更新了新模块），只创建新增的 Room，不动已有结构

## Spec Object 的 7 种类型

供参考，room-init 主要创建 intent 和 constraint 两种：

| type | 是什么 | 例子 |
|------|-------|------|
| `intent` | 为什么做这件事 | "桌宠常驻桌面，作为 Nomi 的入口" |
| `decision` | 为什么选 A 不选 B | "时间触发用 EventKit 而非自建调度器" |
| `constraint` | 不能做什么 / 边界 | "后台 CPU ≤3%" |
| `contract` | 组件之间的接口约定 | "Trigger CRUD API" |
| `convention` | 团队怎么做事 | "RESTful API 规范" |
| `change` | 一次具体变更的记录 | "本次 PR 把 session 改为 stateless" |
| `context` | 背景信息 | "目标用户画像" |
