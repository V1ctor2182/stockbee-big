# 强化学习算法 (03-rl-algorithms)

## Intent

**Phase 1: DQN+PPO; Phase 2: SAC+Ensemble**

选股环境（DQN，100→8 离散决策）+ 权重环境（PPO，连续 softmax）。Walk-forward 验证防过拟合。Phase 2 升级 SAC + 集合策略应对制度切换。

来源：Tech Design §3.4
研究状态：待研究 (research-rl-algorithms.md 有初步方案)

## Constraints

- **多 seed 验证** — 来源: PRD §3.3
- **训练硬件 RTX 3070 (8GB VRAM)** — 来源: Tech Design §7.1

## Decisions

_暂无决策记录_

## Contracts

_暂无接口约定_

---
_所有 spec 状态: draft（需要 review 后升为 active）_
_spec.md 由 room-init 自动生成，specs/*.yaml 为源数据_
