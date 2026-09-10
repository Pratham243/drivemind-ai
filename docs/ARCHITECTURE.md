# DriveMind AI — Architecture

This document is a focused technical deep-dive into the request pipeline. For the
project overview, dataset details, evaluation results, and setup instructions, see the
main [README](../README.md).

## Pipeline

```
User
  │
  ▼
API / UI                    FastAPI (src/api/) or Streamlit (app/streamlit_app.py) —
  │                         the UI talks to the API over plain HTTP, never to agent
  │                         internals directly.
  ▼
NLU                         src/nlu/router.py — NLURouter.process()
  │
  ├─▶ Intent Classifier     src/nlu/intent_classifier.py → src/models/model_manager.py
  │                         Selects the trained model (TF-IDF+LogReg baseline, or
  │                         DistilBERT transformer if NLU_MODEL=transformer). If
  │                         confidence < threshold (0.5), falls back to the LLM
  │                         layer (src/nlu/llm_provider.py) for a structured
  │                         intent/entity guess — MockLLMProvider by default,
  │                         OpenAIProvider if OPENAI_API_KEY is set.
  │
  └─▶ Entity Extraction     src/nlu/entity_extractor.py — deterministic
                            regex/gazetteer extraction, scoped to the entity keys
                            valid for the resolved intent (configs/intents.yaml).
  │
  ▼
Agent Controller             src/agents/agent.py — AgentController.handle()
  │                          1. Splits the message into sub-requests on
  │                             conjunctions (src/agents/planner.py) for
  │                             multi-step requests ("battery level AND find
  │                             a charger").
  │                          2. Runs NLU (above) on each sub-request.
  ▼
Tool Selection                src/agents/planner.py: tool_for_intent(intent)
  │                            Looks up configs/intents.yaml's intent → tool
  │                            mapping. Some intents (general_question) have no
  │                            tool — those go straight to response generation.
  ▼
Safety Validation              src/safety/validator.py — SafetyValidator.validate()
  │                            Checks the proposed (tool_name, entities) against
  │                            src/safety/policies.py: unknown tool → reject,
  │                            missing required parameter → reject, wrong type →
  │                            reject, out-of-range value → reject (temperature
  │                            16–30°C, volume 0–100, ...), context-dependent
  │                            policy → reject (e.g. window blocked above
  │                            200 km/h). A rejection here means the request
  │                            never reaches the tool at all.
  ▼
Tool Execution                 src/agents/tool_registry.py: execute_tool()
  │                            Only reached if safety validation passed. Looks
  │                            up the tool function, forwards only the entity
  │                            keys the function's signature accepts, and calls
  │                            it — with the shared VehicleSimulator instance
  │                            for stateful tools.
  ▼
Vehicle / Service State         src/tools/vehicle.py — VehicleSimulator
  │                            The single in-memory source of truth for battery,
  │                            temperature, speed, range, windows, tire pressure,
  │                            seats, ambient light, climate mode, navigation, and
  │                            media state. Stateless tools (charging stations,
  │                            restaurants, weather, traffic) query a
  │                            deterministic simulated backend instead
  │                            (src/tools/charging.py, restaurant.py, weather.py,
  │                            navigation.py:get_traffic) — see README "Vehicle
  │                            simulator" for why these are simulated, not real
  │                            API integrations.
  ▼
Response                        src/agents/agent.py: AgentController._response_for_step()
                                A template per tool renders the ToolResult's data
                                into a sentence; get_tire_pressure has two
                                templates because its ToolResult shape depends on
                                whether one tire or all were requested. Tool-less
                                intents (general_question) and unsafe/failed
                                calls get their response from the LLM layer or a
                                fixed "Sorry, I couldn't do that: <reason>."
                                message respectively.
  │
  ▼
User
```

Every step above produces a `StepTrace` (`src/agents/state.py`), and a request's full
list of `StepTrace`s is returned as the API response's `steps` field — this is the same
object rendered in the Streamlit "Technical trace" panel, so what you see there is
literally the pipeline's internal state, not a separate summary.

## The LLM does not directly control vehicle state

This is the single most important architectural constraint in the project, and it's
enforced structurally, not by convention:

- The `LLMProvider` interface (`src/nlu/llm_provider.py`) exposes exactly two methods:
  `generate()` (free-form text) and `structured_output()` (intent + entities as a
  Python dict). **Neither method has any reference to `VehicleSimulator` or any tool.**
  It is architecturally impossible for the LLM layer to call a tool directly — it can
  only produce data that flows back through the same NLU → planner → safety → tool
  path every other request takes.
- Even when the LLM *is* used (low-confidence fallback, or generating the reply for a
  tool-less intent), its output for a would-be tool call still passes through
  `SafetyValidator.validate()` exactly like a request that came from the trained ML
  classifier — see `src/agents/agent.py: AgentController._process_step()`, which calls
  `self.safety.validate(...)` unconditionally for any resolved tool, regardless of
  which layer (ML model or LLM) produced the entities.
- The same rule applies to non-agent paths: `POST /vehicle/climate` in
  `src/api/routes.py` calls `SafetyValidator.validate("set_temperature", ...)` before
  touching vehicle state, even though that endpoint doesn't go through the agent or
  LLM at all. There is no code path — LLM-driven or direct REST call — that reaches
  `VehicleSimulator`'s mutating methods without passing through the safety layer first.

This is verified, not just asserted: `tests/test_safety.py` has 13 dedicated rejection
tests, and `tests/test_agent.py::test_scenario_safety_rejection_extreme_temperature` /
`tests/test_api.py::test_vehicle_climate_endpoint_rejects_unsafe_temperature` confirm the
end-to-end behavior for both the agent and direct-REST paths.

## Key module map

| Concern | Module |
|---|---|
| Text cleaning (two separate paths — see module docstring for why) | `src/preprocessing/cleaner.py` |
| Dataset schema + validation | `src/annotation/schema.py`, `src/annotation/validator.py` |
| Data quality analysis | `src/annotation/quality.py` |
| TF-IDF + Logistic Regression model | `src/models/baseline.py` |
| DistilBERT fine-tuning | `src/models/transformer.py` |
| Active-model selection/fallback | `src/models/model_manager.py` |
| Intent → tool → entity taxonomy (config, not code) | `configs/intents.yaml` |
| Safety limits (config, not code) | `configs/config.yaml: safety` → `src/safety/policies.py` |
| Tool implementations | `src/tools/*.py` (one module per domain) |
| Vehicle state | `src/tools/vehicle.py` |
| Agent orchestration | `src/agents/agent.py`, `planner.py`, `tool_registry.py`, `state.py` |
| ML metrics + agent evaluation harness | `src/evaluation/*.py` |
| REST API | `src/api/main.py`, `routes.py`, `schemas.py` |
| Dashboard | `app/streamlit_app.py` |

## Design decisions not obvious from the diagram

See the README's ["Architecture decisions & trade-offs"](../README.md#architecture-decisions--trade-offs)
table for the reasoning behind: rule-based (not LLM-driven) multi-step planning,
deterministic (not ML-based) entity extraction, why the transformer isn't the default
model despite outperforming the baseline, and why the safety layer is a separate module
rather than validation embedded in each tool.
