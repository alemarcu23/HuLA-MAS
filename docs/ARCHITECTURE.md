# HuLA-MAS — Developer Guide

This document includes information on HuLA-MAS, how to add new agents, how to add new modules and how to run the simulations.

---

## 1. Running & configuring

```bash
python main.py                       # team of LLM agents, no humans
python main.py --config my.yaml      # custom config
python main_multi_human.py           # human(s) + RescueBot, browser lobby
```

`main.py` does only four things: load `config.yaml`, size the LLM thread pool
([`init_agent_pool`](../matrx/agents/llm/modules/agent_infra/async_model_prompting.py)),
build the world ([`create_builder`](../matrx/grid_world_creation/WorldBuilder.py)), and run it.
Config in: [`config.yaml`](../config.yaml):

| Section | Key | What it does |
| --- | --- | --- |
| `map` | — | World to load (`official`, or a `maps/<name>.yaml` path). |
| `llm` | `model`, `backend`, `api_base` | Model id; backend is `ollama_sdk` or `transformers`; Ollama endpoint. |
| `llm` | `sampling` | `max_tokens`/`temperature`/`top_p`/… applied to every LLM call. |
| `agents` | `num_agents` | Number of LLM rescue agents. |
| `agents` | `capability_presets` + `presets` | Per-agent capability levels (vision/strength/medical/speed). |
| `agents` | `roles` + `role_goals` | Per-agent fixed role and the mission text injected as its goal. |
| `agents` | `capability_knowledge` | `informed` (knows limits) vs `discovery` (learns by failing). |
| `agents` | `reasoning_strategies` | Per-agent reasoning: `io \| cot \| reflexion \| self_refine`. |
| `agents` | `planning_strategies` | Per-agent planning: `io \| deps \| critic \| cot`. |
| `agents` | `joint_action_ask_trigger` | `auto_bridge` (action auto-sends ask_help) vs `llm_explicit`. |
| `world` | `seed` | Int for reproducibility, `null` for map default. |
| `server` | `enable_gui`, `vis_port`, `api_port` | Browser visualizer + MATRX REST API. |
| `task` | `game_rules` | Task description injected into every LLM stage. `{drop_zone}` is filled at runtime. |

Make sure that you fill enough values for roles, capabilities, reasoning and planning as the number of agents you add.
---

## 2. The game loop

The simulation is **tick-based** (MATRX engine, `tick_duration=0.1` -> ~10 ticks/s). The
loop is found in [`GridWorld.__step()`](../matrx/grid_world.py) (called from `run()`):

```
each tick:
  1. build complete world state, check simulation goal (all target victims rescued?)
  2. run loggers (ActionLogger writes per-tick CSV)
  3. for each agent NOT mid-action:
         filter_observations(state)  ->  decide_on_action(state)  ->  (action, params)
         mark agent busy for the action's duration (in ticks)
     for each agent finishing its action this tick: enqueue it in the action buffer
  4. route messages between agents, execute buffered actions, push state to the API/GUI
```

**`decide_on_action` must return immediately.** Actions have a *duration* and
an agent is "busy" (skipped) until its action's ticks is finished. The LLM is slow, so the agent
**never blocks**: it submits the model call to a thread pool and returns `Idle` while the
future is pending, polling it on later ticks.

---

## 3. The LLM agent

```
LlmAgent(JointAction, LlmBrain)
└── LlmBrain(ArtificialBrain, Perception, Execution, Communication, AsyncLLMCalls)
```

- [`LlmBrain`](../matrx/agents/llm/LlmBrain.py) — infrastructure only (any LLM agent should extend this to 
  implement Perception, Execution and Communication)
- [`LlmAgent`](../matrx/agents/llm/llm_agent.py) — the actual planning and reasoning calls.

### The cognitive pipeline

`decide_on_actions` runs once per tick and does:

1. `update_knowledge` — perception: rebuild `WORLD_STATE` / `_world_state_belief`, read messages (roles, plans, capabilities, help requests), update area-exploration map.
2. `_run_infra` — carry retry, navigation steps.
3. Poll the pending LLM future. If still running -> `Idle`. If done -> handle the result.
4. Otherwise advance the pipeline state machine:

```
IDLE -> PLANNING -> REASONING -> EXECUTE (or COMM_DISPATCH) -> IDLE
```

### Agent modules (`matrx/agents/llm/modules/`)

| Module | Responsibility |
| --- | --- |
| [`planning_module.py`](../matrx/agents/llm/modules/planning_module.py) | Planner prompt + `PLANNING_STRATEGY_REGISTRY` (`io/deps/critic/cot`). |
| [`reasoning_module.py`](../matrx/agents/llm/modules/reasoning_module.py) | Reasoning prompt + `REASONING_STRATEGY_REGISTRY` (`io/cot/reflexion/self_refine`); two-pass strategies hook `on_llm_result`. |
| [`profile_module.py`](../matrx/agents/llm/modules/profile_module.py) | Capabilities + role + goal; preset -> capability resolution. |
| [`memory_module.py`](../matrx/agents/llm/modules/memory_module.py) | `BaseMemory` (rolling log, compresses), `EpisodicMemory` (per-cycle episodes), `SharedMemory`. |
---

## 4. How to extend

### Add a new metric
1. Add a field + `record_*()` method to `AgentMetricsTracker` ([agent_metrics.py](../matrx/metrics/agent_metrics.py)) and include it in `to_dict()`.
2. Call `self.metrics.record_*()` at the relevant point in the pipeline (`agent_sar.py` / `LlmBrain`).

### Add a new reasoning / planning strategy
1. Subclass `ReasoningBase` (override `_strategy_addendum`, and `on_llm_result` for two-pass
   strategies) or `PlanningBase` (set `prefix`/`output_schema`).
2. Register it in `REASONING_STRATEGY_REGISTRY` / `PLANNING_STRATEGY_REGISTRY`.
3. Reference the new name in `config.yaml` (`reasoning_strategies` / `planning_strategies`).

### Add a new action (tool)
1. Define an `@tool` function in [tool_registry.py](../matrx/agents/llm/tool_registry.py) and add it to `ALL_ACTION_TOOLS`.
2. Add validation rules in [`ActionValidator`](../matrx/helpers/logic_module.py).
3. Map it to a MATRX action in the execution path; if it's a 2-agent action, route it through `JointAction`.
4. If it needs a custom engine action, add it to [`CustomActions.py`](../matrx/actions/CustomActions.py).

### Add a new agent type
1. Subclass [`LlmBrain`](../matrx/agents/llm/LlmBrain.py) (for an LLM agent) or
   [`ArtificialBrain`](../matrx/agents/ArtificialBrain.py) (rule-based — see
   [`RuleBasedAgent.py`](../matrx/agents/RuleBasedAgent.py)). Implement `decide_on_actions`
   returning `(action_name, params)` **without blocking**.
2. Register it in [`WorldBuilder.add_agents`](../matrx/grid_world_creation/WorldBuilder.py)
   (`builder.add_agent(...)`), wiring config into its constructor.
