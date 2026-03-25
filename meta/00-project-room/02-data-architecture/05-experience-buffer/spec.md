# 经验回放缓冲 (05-experience-buffer)

## Intent

**Ring Buffer 50K，内存常驻，DQN/SAC 训练数据源**

FIFO 覆盖策略，50K transition ≈ 35MB。可选 Priority Experience Replay (Phase 2)。

来源：Tech Design §2.6
研究状态：研究完成 (research-experience-buffer-20260317.md)

## Constraints

- **容量 50K transition** — 来源: Tech Design §2.6
- **内存占用 ≈ 35MB** — 来源: Tech Design §2.6

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
