---
name: prompt-gen
description: >
  Pull Feature Room context and enhance user prompts for AI coding assistants (Cursor/Claude). Automatically gathers specs, decisions, constraints, contracts, conventions, and progress from the relevant Room(s), then wraps the user's own prompt with full project context. Use when the user says "帮我写prompt", "增强prompt", "pull context", "帮我实现xxx", "给Cursor写prompt", "promptgen", or when the user wants a context-rich prompt for AI-assisted development. Also use for prompt testing ("测一下spec够不够"). Supports both Room-level and Milestone-level prompts.
---

# Prompt Gen — Context Pull + 用户 Prompt 增强

用户写自己的 prompt（想让 AI 做什么），这个 skill 自动 pull 相关 Room 的上下文，把用户的 prompt 增强为一个带完整背景的 prompt。核心逻辑是 **context pulling**，不是替用户写 prompt。

支持两种粒度：
- **Room 级**：`promptgen stock-data` — 拉整个 Room 的上下文（用于规划）
- **Milestone 级**：`promptgen stock-data/m1` — 只拉该 milestone 需要的上下文（用于开发）

## 前置条件

- meta/ 目录已初始化，至少有 Room 结构和一些 specs
- 用户提供了自己的 prompt + 指定（或可推断）Room

## 执行步骤

### Phase 0: 确定粒度

- 用户指定了 milestone（如 "promptgen macro-data/m3"）→ **Milestone 模式**
- 用户只指定 Room（如 "promptgen macro-data"）→ **Room 模式**（现有行为）

**Milestone 模式额外行为**：
- 只拉该 milestone 相关的 specs（而非全部）
- 已完成 milestone 的产出作为 "前置条件" 段落
- 未来 milestone 不包含在 prompt 中
- token 预算更紧凑（目标 2K-5K）

### Phase 1: 确定 Room

优先级：**用户显式指定 > AI 推断**

- 用户说了 room name（如 "我在 trigger-mode"）→ 直接用，不问
- 用户没说 → 从 `_tree.yaml` + 各 room spec.md 语义匹配 → **先向用户确认**再 pull
- 涉及多个 Room（如 "trigger-mode 和 knowledge-base 的集成"）→ 用户指定的全部 pull；AI 推断的先确认

### Phase 2: Pull Context

对每个涉及的 Room，收集以下内容：

**a. specs/*.yaml 中的 Spec Objects（按 state 过滤）：**

| Spec 状态 | 是否 Pull | 说明 |
|-----------|----------|------|
| active | ✅ 一定 pull | 核心上下文 |
| draft | ⚠️ 标注后 pull | 标注 "[draft]"，提醒可能未确认 |
| stale | ⚠️ 标注后 pull | 标注 "[stale ⚠️]"，提醒可能过时 |
| deprecated | ❌ 不 pull | 已废弃 |
| superseded | ❌ 不 pull | 被替代，pull 替代者 |

Pull 的 spec 类型包括全部 7 种：
- `intent` — 功能目标
- `decision` — 关键决策（含 ADR 推理）
- `constraint` — 边界和红线
- `contract` — 接口约定
- `convention` — 规范（可能从父 Room 继承）
- `context` — 背景信息
- `change` — 最近的变更记录（只取最新几条）

**b. progress.yaml — 开发进度：**
- 已完成的 milestones（告诉 AI 什么已经做了）
- 当前 in_progress 的 milestone（AI 知道正在做什么）
- 最近的 commits（AI 知道最近改了什么）
- completion 百分比

**c. meta/project.yaml（技术栈、全局配置）**

### Phase 3: 智能筛选

根据用户 prompt 的意图，重点 pull 不同类型的上下文：
- 用户说 "后端 API" → 重点 pull constraints、contract specs、API 规范
- 用户说 "UI 动画" → 重点 pull 设计约束、交互 specs
- 用户说 "集成" → 重点 pull 两个 Room 的 contract specs

### Phase 4: 组装最终 Prompt

```markdown
# Enhanced Prompt — {Room} {任务简述}

<!-- context_from: {room-ids} -->
<!-- token_count: XXXX | prompt_test: pass/fail -->

## 项目背景
[从 project.yaml: 技术栈、产品定位]

## 上下文

### 来自 {room-id}/specs/
[intent] {summary}
[decision] {summary}（{选择理由}）
[constraint] {summary}
[contract] {接口定义}

### 来自 00-project-room/specs/（继承）
[convention] {规范}
[constraint] {全局约束}

## 当前进度

{room} 完成度: {X}%
- ✅ {completed milestone} (completed {date})
- 🔨 {in_progress milestone} (in_progress)
- ⬚ {pending milestone}

最近 commit:
- [{date}] {commit message}

## 你的任务

{用户原始 prompt，原封不动}
```

### Phase 5: Prompt 测试

计算最终 prompt 的 token 数：
- < 3000 tokens → ⚠️ "上下文偏少，spec 可能不够充分"
- 3000-8000 tokens → ✅ "粒度合适"
- > 8000 tokens → ⚠️ "上下文过多，建议缩小范围或拆 Room"

### Phase 6: 输出

1. 展示增强后的 prompt（用户可直接复制到 Cursor/Claude）
2. 展示 prompt 测试结果
3. 可选：保存到 Room 的 `prompt.md`

## 纯 Prompt 测试模式

用户说 "测一下 trigger-mode 的 spec 够不够生成 prompt" 时：
- 只做 Prompt 测试：拉 context，计算 token 数
- 不需要用户提供 prompt
- 报告粒度是否合适
- 这是验证 Room 拆分是否合理的代理指标

## Milestone 模式输出模板

```markdown
# Milestone Prompt — {Room} / {milestone-id}: {milestone-name}

<!-- context_from: {room-id}/{milestone-id} -->
<!-- token_count: XXXX | prompt_test: pass/fail -->

## 目标

{milestone 的具体目标，从 progress.yaml 和 intent spec 提取}

## 前置条件（已完成的 milestones）

{列出该 Room 已完成的 milestones 及其产出摘要}
- ✅ m1-xxx: {一句话描述产出，含关键文件路径}
- ✅ m2-xxx: ...

## 上下文

{只包含该 milestone 相关的 specs，精简版}

## 约束

{只包含和该 milestone 直接相关的 constraints}

## 已有代码参考

{如果本 milestone 要实现某个 interface，列出接口签名}
{如果要 import 已完成 milestone 的代码，列出 import 路径}

## 你的任务

{用户原始 prompt 或 milestone 描述}
```

## 重要规则

- 增强 prompt 中，用户原始 prompt 放在最后 "你的任务" 段落，**原封不动**
- draft 和 stale 的 spec 要明确标注状态，让使用 prompt 的 AI 知道这些信息可能不准确
- 从父 Room 继承的 specs 要标注来源（"来自 00-project-room"）
- token 估算可以用字符数 / 4 作为粗略近似
- **Milestone 模式下 token 目标 2K-5K**，Room 模式下 3K-8K
