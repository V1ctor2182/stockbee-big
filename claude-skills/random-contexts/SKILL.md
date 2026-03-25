---
name: random-contexts
description: >
  Parse unstructured content (chat logs, meeting notes, random thoughts) into structured Spec Objects and route them to the correct Feature Rooms. Use when the user pastes chat records, discussion logs, meeting notes, or random ideas and wants to extract decisions, constraints, and context from them. Triggers on "帮我整理", "这段聊天", "会议笔记", "粘贴", "extract specs", "从聊天记录提取", or when the user pastes a block of unstructured text that contains design discussions or technical decisions.
---

# Random Contexts — 聊天记录/笔记 → Spec Objects → Room 归位

用户把和别人的聊天记录、会议笔记、随手想法等非结构化内容粘贴进来，自动解析出 spec 信息，创建 Spec Objects，路由到正确的 Room。

这验证了 VibeHub 最核心的假设：日常讨论中隐含大量的设计决策和约束，但通常会丢失。

## 前置条件

- meta/ 目录已初始化（至少有 `_tree.yaml`）
- 用户粘贴了一段非结构化文本

## 执行步骤

### Phase 1: 内容解析

1. 分析用户粘贴的文本
2. 识别讨论单元（一个独立的观点/决策/约束/信息）
3. 拆分多人对话中的不同话题
4. 过滤噪音（寒暄、无关内容）

5. 对每个讨论单元，判断 Spec 类型：

| 信号 | Spec Type | 例子 |
|------|-----------|------|
| "用 X 不用 Y，因为..." | decision | "时间触发用 EventKit 而非自建" |
| "必须/不能/上限/下限..." | constraint | "CPU≤3%、triggers≤50" |
| "接口是这样的..." | contract | "REST API: POST /api/triggers" |
| "我们约定都用 xxx..." | convention | "统一用 RESTful" |
| "用户反馈说.../试了一下..." | context | "Spotlight 风格比悬浮窗好" |
| "X 改成了 Y" | change | "session 改为 stateless" |
| "这个功能要做的是..." | intent | "支持三种触发类型" |

### Phase 2: Room 路由

6. 确定每条 spec 属于哪个 Room：
   - 用户已指定 Room（"这段是关于 trigger-mode 的"）→ 全部归入该 Room
   - 用户未指定 → AI 语义匹配（读取 `_tree.yaml` + 各 room 的 `spec.md`）
   - 涉及多个 Room → 分别归类
   - 项目级内容（如 "前后端统一用 REST"）→ `00-project-room`
   - 无法归类的 → 标记出来让用户指定

### Phase 3: 用户确认

7. **展示解析结果，等待确认：**

```
📋 解析结果（共 {N} 条）

→ trigger-mode ({n} 条)
  [decision] EventKit 做时间触发（conf: 0.85）
  [decision] CoreLocation 做位置触发（conf: 0.85）
  [constraint] geofence ≤ 20 个（conf: 0.9）

→ desktop-pet ({n} 条)
  [context] Lottie 动画方案在评估中（conf: 0.7）

→ 00-project-room ({n} 条)
  [decision] 前后端用 REST（conf: 0.85）

⚠️ {n} 条无法归类，需要你指定 Room:
  "数据加密用 AES-256" → 哪个 Room？

确认创建？(y/n/编辑)
```

8. 用户可以：
   - 直接确认
   - 编辑某条的类型、内容或 Room 归属
   - 删除不需要的条目
   - 指定无法归类的条目的 Room

### Phase 4: 创建 Spec Objects

9. 确认后，对每条创建 yaml 文件：

```yaml
spec_id: "{type}-{room-id}-{NNN}"
type: {decision|constraint|context|...}
state: draft
intent:
  summary: "{一句话总结}"
  detail: |
    来源：{聊天/会议/笔记}

    {具体内容和推理}

    原始讨论：
    > {引用原文}
constraints: []
indexing:
  type: {same as type}
  priority: {inferred}
  layer: {epic|feature}
  domain: "{domain}"
  tags: [{tags}]
provenance:
  source_type: chat_extraction
  confidence: {0.5-0.9}
  source_ref: "{来源描述}"
relations: []
anchors: []
```

10. 写入对应 Room 的 specs/ 目录
11. 同步更新各 Room 的 spec.md

12. 输出报告：
```
✅ Created: {N} specs across {M} rooms
📝 All specs are draft (needs review)
平均 confidence: {X}

下一步: 用 room skill "review {room} 的 draft specs" 逐条确认
```

## Confidence 评分规则

| 信号特征 | confidence | 说明 |
|---------|-----------|------|
| 明确的 "我们决定用 X" | 0.9 | 明确决策，高置信 |
| "用 X 吧" + 给出理由 | 0.85 | 有共识倾向 + 理由 |
| "我觉得应该用 X" | 0.7 | 个人观点，未达成共识 |
| "X 也行，Y 也行" | 0.5 | 讨论中，未决 |
| 隐含的约束（从讨论推断） | 0.6 | AI 推断，需要人工确认 |
| 明确的数字约束 "≤ 20 个" | 0.9 | 明确数字，高置信 |

## 重要规则

- 所有创建的 spec 默认 `state: draft`，`provenance.source_type: chat_extraction`
- 保留原始讨论的引用（在 detail 字段中引用原文）
- confidence 分数要诚实，不确定的就标低
- 宁可漏提取也不要误提取（precision > recall）——用户可以手动补充
- 同一段对话可能包含多个不同 Room 的信息，需要正确路由
