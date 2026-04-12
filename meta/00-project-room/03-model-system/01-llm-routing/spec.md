# LLM 多模型路由 (01-llm-routing)

## Intent

**按任务选模型，litellm 统一接口，降级链，成本追踪**

3 个核心组件：
- **LLMRouter** — route(task_type, prompt) → LLMResponse。litellm 统一调用，降级链（主模型→备选→raise），JSON 解析 + schema 验证，set_completion_fn() 支持 mock 测试
- **TaskConfig** — 5 种任务类型（G1_FILTER, G2_CLASSIFY, G3_ANALYSIS, SEC_PARSE, MACRO_ANALYSIS），每种指定模型、备选、token 限制、输出格式
- **CostTracker** — SQLite 历史记录，按任务类型累计成本，月度预算检查（$15），prompt hash 响应缓存

实现状态：已完成 ✅（2026-04-08）
测试覆盖：28 个单元测试（mock，不调真实 API）

## Constraints

- **月度 LLM 成本 < $15** — G1($1) + G2($2) + G3($5) + SEC($3) + 宏观($4)
- **litellm 为可选依赖** — 回测/测试不需要安装
- **Router 是 task-agnostic** — 不含 G1/G2/G3 业务逻辑，只提供路由接口

## Decisions

- **litellm > 直接 OpenAI+Anthropic SDK** — 一行代码切换模型，统一接口
- **Router 不管 G1/G2/G3 业务逻辑** — prompt 模板和处理逻辑在调用方（news-data 等）

## Contracts

- **LLMRouter** — route(task_type, prompt, system_prompt?, output_schema?) → LLMResponse; get_config(task_type); list_task_types(); set_completion_fn(fn)
- **CostTracker** — record_call; get_cached; is_over_budget; monthly_spent; monthly_report; cache_size

---
_所有 spec 状态: active_
_spec.md 最后更新: 2026-04-08_
