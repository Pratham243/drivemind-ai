# 🚗 DriveMind AI — Agentic NLP & LLM-Based Intelligent In-Car Assistant

A Master's-level portfolio project demonstrating an end-to-end automotive conversational AI
system: NLP preprocessing → traditional ML + transformer NLU → LLM fallback → an agent
controller that selects and safely executes tools against a simulated vehicle backend,
exposed through a FastAPI service and a Streamlit dashboard.

> Built to be **explainable in a 20–30 minute technical interview** — every component is
> small, testable, and has a documented reason for existing the way it does.

---

## Table of contents

- [Overview](#overview)
- [Why automotive AI?](#why-automotive-ai)
- [Architecture](#architecture)
- [System workflow](#system-workflow)
- [Features](#features)
- [Dataset](#dataset)
- [Annotation & data quality](#annotation--data-quality)
- [ML baseline](#ml-baseline)
- [Transformer model](#transformer-model)
- [LLM layer](#llm-layer)
- [Agent architecture](#agent-architecture)
- [Tools](#tools)
- [Vehicle simulator](#vehicle-simulator)
- [Safety layer](#safety-layer)
- [FastAPI](#fastapi)
- [Streamlit dashboard](#streamlit-dashboard)
- [Docker](#docker)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Evaluation & results](#evaluation--results)
- [Error analysis](#error-analysis)
- [Installation & local setup](#installation--local-setup)
- [Example requests](#example-requests)
- [Example agent traces](#example-agent-traces)
- [Project structure](#project-structure)
- [Future improvements](#future-improvements)
- [Limitations](#limitations)
- [Responsible AI / safety considerations](#responsible-ai--safety-considerations)
- [Resume bullet points](#resume-bullet-points)
- [Interview talking points](#interview-talking-points)
- [Architecture decisions & trade-offs](#architecture-decisions--trade-offs)

---

## Overview

DriveMind AI simulates the software stack behind a modern in-car voice assistant
(comparable in spirit to Mercedes MBUX, BMW Intelligent Personal Assistant, or Amazon Alexa
Auto). A user types (or would speak) a natural-language request — *"Find a charging station
near Berlin with at least 150 kW"* — and the system:

1. Classifies the **intent** (`find_charging_station`) using a trained ML model.
2. Extracts structured **entities** (`location=Berlin`, `minimum_power_kw=150`).
3. Routes the request through an **agent controller** that selects the matching **tool**
   (`find_charging_stations`).
4. Validates the tool call against a **safety layer** (range/type/schema checks) before it is
   allowed to touch vehicle state.
5. Executes the tool against a **simulated vehicle backend**.
6. Generates a natural-language **response**.

Every step of that pipeline is a real, independently-tested component — not a single prompt
to an LLM. The LLM is used only where it adds value: as a fallback for ambiguous utterances
the trained ML models are unsure about, and as an offline-safe MockLLMProvider by default.

## Why automotive AI?

In-car assistants are a uniquely good domain for demonstrating applied AI/ML engineering
because they force you to solve problems that a generic chatbot demo does not:

- **Safety-critical action execution** — an LLM must never be allowed to directly open a
  window at 200 km/h or set an unsafe cabin temperature. This requires a real
  validate-before-execute architecture, not just prompt engineering.
- **Structured, multi-modal outputs** — "set the temperature to 22°C" needs a *typed*
  parameter (`temperature: int`), not free text.
- **Multi-step task decomposition** — "What's my battery level and find the nearest
  charger" requires planning and executing two tool calls, not one.
- **A closed, well-defined tool surface** — ideal for demonstrating agentic tool-calling
  patterns without the scope explosion of a general-purpose assistant.

It is also a domain directly relevant to Germany's automotive + AI job market (OEMs,
Tier-1 suppliers, and their software subsidiaries all build exactly this kind of system).

## Architecture

```
                     USER
                       |
                       v
             Text / Voice Input
                       |
                       v
             NLP Preprocessing
                       |
                       v
             +-------------------+
             |   NLU / ML Layer  |
             |                   |
             | TF-IDF + LogReg   |
             | DistilBERT        |
             | LLM fallback      |
             +---------+---------+
                       |
                       v
             Agent Controller (planner)
                       |
                       v
               Tool Selection
                       |
                       v
             Safety Validation  <---- rejects unsafe/malformed calls
                       |
        +--------------+--------------+-------------------+
        |              |              |                   |
        v              v              v                   v
    Climate        Navigation     Charging      Media / Comms / Comfort / ...
     Tool             Tool          Tool                  Tools
        |              |              |                   |
        +--------------+--------------+-------------------+
                       |
                       v
              Simulated Vehicle (in-memory state)
                       |
                       v
                 Tool Result
                       |
                       v
             Response Generator
                       |
                       v
                     USER
```

Data pipeline (offline, run once to produce trained artifacts):

```
Dataset generation → Annotation schema validation → Data-quality checks
    → Train/Val/Test split → Model training (baseline + transformer)
    → Evaluation → Model artifacts (models/) → Loaded by the Inference API
```

## System workflow

1. **Input**: a `POST /chat` request (or a message typed into the Streamlit chat box).
2. **NLU routing** (`src/nlu/router.py`): the configured classifier (baseline or
   transformer) predicts an intent + confidence. If confidence is below threshold, the
   `LLMProvider` is asked for a structured fallback prediction.
3. **Entity extraction** (`src/nlu/entity_extractor.py`): deterministic, regex/gazetteer
   based, scoped to the entities valid for the resolved intent.
4. **Planning** (`src/agents/planner.py`): the utterance is split into sub-requests on
   coordinating conjunctions ("and", ",", ";") to support multi-step requests; each
   sub-request is routed through NLU independently.
5. **Tool selection**: `configs/intents.yaml` maps each intent to (at most) one tool.
6. **Safety validation** (`src/safety/validator.py`): every tool call is checked for
   missing/invalid/out-of-range parameters *before* execution.
7. **Tool execution** (`src/agents/tool_registry.py` + `src/tools/*`): mutates or reads the
   simulated vehicle state.
8. **Response generation**: a template per tool, or the `LLMProvider` for tool-less
   (`general_question`) intents.
9. Everything above is captured in an `AgentState`/`StepTrace` — the full trace is returned
   by the API and rendered in the Streamlit "Technical trace" panel.

## Features

- 27-intent automotive NLU taxonomy with a 2,715-example synthetic dataset.
- Reproducible dataset generation (seeded) with realistic imbalance, typos, and
  paraphrasing — not uniform template noise.
- Pydantic-based annotation schema + validator (catches invalid intents, malformed
  entities, out-of-range values, duplicates).
- Automated data-quality report with real computed statistics + charts.
- TF-IDF + Logistic Regression baseline (trains in seconds).
- DistilBERT transformer classifier (fine-tunable on a CPU laptop).
- Deterministic rule-based entity extraction (fast, fully unit-testable, zero API cost).
- LLM abstraction with automatic offline fallback (`MockLLMProvider`) — no API key required
  to run the full system.
- Agent controller with real multi-step task decomposition and execution.
- 20 automotive tools across climate, navigation, charging, media, comms, comfort, and
  information lookup.
- In-memory vehicle simulator that is *actually mutated* by tool calls.
- Safety validation layer that rejects out-of-range, malformed, or unknown tool calls
  *before* they reach vehicle state — with real rejection tests.
- FastAPI backend with full OpenAPI docs.
- Streamlit dashboard showing vehicle state, conversation, and the full agent trace.
- 79 automated tests (pytest) covering happy paths *and* negative/safety cases.
- Docker + docker-compose for the API and dashboard.
- GitHub Actions CI: lint → test → train → build Docker image.
- Real evaluation reports (no fabricated numbers) under `reports/evaluation/`.

## Dataset

Generated by `scripts/generate_dataset.py` — **2,715 examples across 27 intents**
(seeded, reproducible with `configs/config.yaml: dataset.seed`).

| Field | Description |
|---|---|
| `id` | unique example id |
| `text` | the utterance |
| `intent` | ground-truth intent label |
| `entities` | ground-truth structured entities |
| `split` | `train` / `val` / `test` (stratified per intent, 80/10/10) |
| `source` | always `synthetic_template` for this dataset |
| `difficulty` | `easy` / `medium` / `hard`, assigned per template |

Variation is generated via:
- Multiple templates per intent (short and long phrasings).
- Random polite (*"Could you please..."*) / informal (*"Hey, ..."*) / plain phrasing wrappers.
- Light spelling-noise injection (character drop/double/swap) on a subset of examples.
- A deliberately **imbalanced** per-intent target count (e.g. `battery_status`: 160 examples,
  `general_question`: 60) — real automotive usage is not uniform across intents, and the
  data-quality pipeline is built to detect and report this imbalance, not hide it.

Regenerate with:
```bash
python scripts/generate_dataset.py
```

## Annotation & data quality

- **Schema** (`src/annotation/schema.py`): a Pydantic `AnnotatedExample` model is the single
  source of truth for what a valid example looks like.
- **Validator** (`src/annotation/validator.py`) checks, on the *entire* dataset:
  - missing/empty text or intent
  - unknown intents
  - entities not valid for their intent, wrong entity type, or out-of-range values
    (e.g. `temperature=999`)
  - exact-text duplicates
- **Data quality** (`src/annotation/quality.py`) computes, from the real dataset:
  total samples, missing labels, duplicates, class distribution + imbalance ratio, text
  length stats, entity coverage, and a heuristic "suspicious sample" flag.

Run:
```bash
python scripts/validate_dataset.py
```
This produces `reports/evaluation/data_quality_report.json` and two charts under
`reports/figures/` (`class_distribution.png`, `split_distribution.png`).

**Real finding from this run**: the validator flagged **260 duplicate texts** (out of 2,715)
— an honest artifact of template-based generation combined with a small amount of
intentional near-duplicate noise, not a hidden bug. This is exactly the kind of issue a data
quality pipeline is supposed to catch, and is discussed further in
[Error analysis](#error-analysis).

## ML baseline

`src/models/baseline.py` — TF-IDF (1–2 grams, 5000 features) → Logistic Regression,
trained via `scripts/train_baseline.py`.

**Actual results on the held-out test set (275 examples)**:

| Metric | Value |
|---|---|
| Accuracy | **0.9927** |
| Macro F1 | **0.9922** |
| Weighted F1 | ~0.9927 |

(Full metrics including per-class confusion matrix: `reports/evaluation/baseline_metrics.json`.)

This is the **default** model (`NLU_MODEL=baseline`) — it trains in a few seconds, needs no
GPU, and is what the API uses out of the box.

## Transformer model

`src/models/transformer.py` — fine-tunes `distilbert-base-uncased` (chosen specifically
because it is small enough — 66M params — to fine-tune on a CPU laptop in a few minutes;
larger encoders were deliberately avoided for that reason).

```bash
pip install -r requirements-ml.txt
python scripts/train_transformer.py
```

**Actual results** (CPU-only training, 3 epochs, batch size 16, lr 5e-5 — see
`configs/config.yaml: transformer_model`):

| Epoch | Train loss | Val accuracy |
|---|---|---|
| 1 | 1.7904 | 0.944 |
| 2 | 0.2518 | 0.9851 |
| 3 | 0.1034 | 0.9925 |

**Test set**: accuracy **0.9891**, macro F1 **0.9881**. Training took **427.6 s (~7.1 min)**
on CPU for all 3 epochs over 2,172 training examples — comfortably within "runnable on a
laptop." Full numbers: `reports/evaluation/transformer_metrics.json`.

### Baseline vs. transformer — the real comparison

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + Logistic Regression | **0.9927** | **0.9922** |
| DistilBERT (fine-tuned) | 0.9891 | 0.9881 |

The baseline slightly **outperforms** the transformer here — a genuine, non-cherry-picked
result (see `reports/figures/model_comparison.png`). This is expected, not a bug: with a
template-generated dataset this clean and lexically distinctive per intent, TF-IDF features
already separate the classes almost perfectly, and 2,172 training examples is a small
fine-tuning set for a 66M-parameter model. The transformer is *not* the default model in
this project for that reason (`NLU_MODEL=baseline`). Its value here is demonstrating a
correct, working fine-tuning pipeline; on a larger, more lexically diverse (non-templated)
real-world dataset, the transformer's contextual embeddings would be expected to close the
gap and likely overtake the linear baseline — particularly on typo/paraphrase robustness
(see [Error analysis](#error-analysis)).

## LLM layer

`src/nlu/llm_provider.py` defines an `LLMProvider` interface with two implementations:

- **`MockLLMProvider`** (default, always available): keyword-heuristic structured
  extraction + templated response generation. Zero cost, zero external dependency, fully
  deterministic — used automatically whenever `OPENAI_API_KEY` is not set.
- **`OpenAIProvider`**: real LLM via the OpenAI Chat Completions API, prompted to return
  strict JSON `{"intent": ..., "entities": {...}}`. Only instantiated if `openai` is
  installed *and* an API key is configured.

`get_llm_provider()` picks automatically — **the system is fully demonstrable without any
paid API**, per the project's core design constraint.

The LLM is invoked in two places:
1. As a **fallback** inside `NLURouter` when the ML classifier's confidence is below
   threshold (ambiguous utterances).
2. As the **response generator** for tool-less intents (`general_question`).

## Agent architecture

`src/agents/agent.py` (`AgentController`) orchestrates the full pipeline described in
[System workflow](#system-workflow). Key design points:

- **State** (`src/agents/state.py`): every request produces an `AgentState` containing a
  list of `StepTrace` (one per sub-request), the composed final response, and any errors.
  This is the object serialized as the "agent trace" in both the API and the UI.
- **Planner** (`src/agents/planner.py`): multi-step decomposition is rule-based (split on
  conjunctions), not LLM-based — deliberately, so multi-step requests work identically with
  or without an LLM configured, and remain deterministic/fast.
- **Tool registry** (`src/agents/tool_registry.py`): maps tool names to callables and uses
  `inspect.signature` to forward only the entity keys each tool actually accepts, so adding
  a new tool never requires touching the agent's dispatch logic.
- Tool execution errors are caught and surfaced as a failed step rather than crashing the
  request (`AgentController._process_step`).

## Tools

20 tools across 8 modules in `src/tools/`:

| Module | Tools |
|---|---|
| `vehicle.py` | vehicle state + `get_vehicle_status`, `get_battery_level`, `get_range`, `get_tire_pressure` |
| `climate.py` | `set_temperature`, `increase_temperature`, `decrease_temperature`, `set_climate_mode` |
| `navigation.py` | `start_navigation`, `cancel_navigation`, `search_navigation`, `get_traffic`, `find_parking` |
| `charging.py` | `find_charging_stations` |
| `media.py` | `play_music`, `pause_music`, `next_song`, `change_volume` |
| `comfort.py` | `open_window`, `close_window`, `adjust_seat`, `set_ambient_lighting` |
| `communication.py` | `make_phone_call`, `send_message` |
| `restaurant.py` / `weather.py` | `search_restaurants`, `get_weather` |

Every tool returns a `ToolResult(success, data, error)` — a single, uniform contract the
agent (and tests) can reason about regardless of which tool ran. Search-style tools
(charging/restaurants/weather/traffic) use a deterministic hash-based simulated backend
(no external API dependency, fully reproducible in tests).

## Vehicle simulator

`src/tools/vehicle.py` (`VehicleSimulator`) holds a single in-memory `VehicleState`:
battery %, temperature, speed, range, per-window open/closed state, tire pressures, seat
adjustments, ambient light color, navigation state, and media state. A process-wide
singleton (`get_vehicle()`) is shared by the API/agent so `GET /vehicle/status` reflects
whatever the agent actually did — e.g. after *"Set temperature to 23"*, the very next status
read shows `temperature_c: 23`. Tests use fresh, non-shared instances so they never
interfere with each other.

## Safety layer

`src/safety/validator.py` + `src/safety/policies.py`. Every tool call — whether it comes
from the agent or a direct REST call to `/vehicle/climate` — passes through
`SafetyValidator.validate(tool_name, arguments)` before touching vehicle state:

- **Unknown tool** → rejected.
- **Missing required parameter** → rejected (e.g. `start_navigation` with no destination).
- **Wrong type** → rejected (e.g. `temperature="warm"`).
- **Out-of-range value** → rejected (`temperature` must be 16–30°C, `volume` 0–100).
- **Unknown enum value** → rejected (e.g. opening a `"sunroof"`, which isn't a modeled
  window).
- **Context-dependent policy** → rejected (windows cannot open above 200 km/h).

Rejections are logged (`logging.warning`, never silently dropped) and returned as a
**user-safe message** — the raw exception/internal reason is never leaked verbatim to the
LLM or the end user beyond the "why" the safety layer already computed. See
`tests/test_safety.py` for 13 dedicated negative tests.

## FastAPI

`src/api/main.py` + `src/api/routes.py`. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/chat` | full agent pipeline, returns structured trace |
| POST | `/intent` | NLU only (intent + entities, no tool execution) |
| POST | `/agent` | alias of `/chat` (kept distinct per spec for clarity) |
| GET | `/vehicle/status` | full simulated vehicle state |
| GET | `/vehicle/battery` | battery level only |
| POST | `/vehicle/climate` | direct climate control (still safety-validated) |
| GET | `/charging-stations` | charging station search |
| GET | `/health` | service health + active NLU model + LLM provider |

Interactive docs at `/docs` (Swagger UI) once the server is running.

## Streamlit dashboard

`app/streamlit_app.py` — talks to the FastAPI backend over plain HTTP (no direct import of
agent internals), so it exercises the exact same API contract any external client would.
Shows: vehicle status (battery/range/temperature/speed/windows/media), a chat interface,
and a full technical trace panel (intent, entities, tool, safety status, raw tool result,
multi-step breakdown, and model/LLM-provider info).

## Docker

```bash
docker compose up --build
```
Builds and runs two services:
- `api` (Dockerfile) — FastAPI backend on port 8000, `NLU_MODEL=baseline` by default (no
  heavy ML dependencies in this image, keeps build fast).
- `dashboard` (`docker/Dockerfile.streamlit`) — Streamlit UI on port 8501, pointed at the
  `api` service.

> **Status: NOT VERIFIED in this environment.** Docker Engine is not installed on the
> machine this project was built on, so `docker compose up --build` has not been run here.
> The Dockerfiles/compose file are written and reviewed but unbuilt — please verify on a
> machine with Docker installed before relying on them.

## Testing

```bash
pytest tests/ -v
```
**79 tests, all passing**, across:
- `test_preprocessing.py` — cleaning/tokenization
- `test_nlu.py` — intent classification, entity extraction, LLM mock, NLU router
- `test_tools.py` — all 20 tools, including negative cases (invalid window, negative power, etc.)
- `test_safety.py` — 13 dedicated safety-rejection tests (temperature, volume, unknown tool,
  unknown window, high-speed window block, missing params, wrong types)
- `test_agent.py` — all 10 canonical scenarios from the spec, plus multi-step and
  garbage-input robustness
- `test_api.py` — FastAPI endpoints, including safety-rejection status codes
- `test_dataset_quality.py` — annotation validator + data-quality analyzer edge cases

Tests that depend on a trained model (`test_nlu.py`, `test_agent.py`, `test_api.py`)
`pytest.skip()` cleanly if `models/baseline/model.joblib` doesn't exist yet, rather than
failing — run `python scripts/train_baseline.py` first.

## CI/CD

`.github/workflows/ci.yml` — on every push/PR to `main`:
1. Install dependencies.
2. Lint (`ruff check`) and format-check (`black --check`).
3. Generate the dataset and validate it.
4. Train the baseline model.
5. Run the full pytest suite.
6. (second job) Regenerate data/model artifacts and build the API Docker image.

> **Status: config written, not run.** This CI has not executed on GitHub Actions from this
> environment (no push has been made to a remote yet). Locally, the exact same lint/format/
> test commands the workflow runs were executed and passed (see [Testing](#testing)).

## Evaluation & results

All numbers below are read directly from `reports/evaluation/*.json`, produced by
`scripts/train_baseline.py`, `scripts/train_transformer.py`, and `scripts/evaluate.py` —
**nothing here is invented**.

### Model comparison (`model_comparison.json`)

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.9927 | 0.9922 | 0.9927 |
| DistilBERT (transformer) | 0.9891 | 0.9881 | 0.9890 |

### Agent evaluation (`agent_evaluation.json`)

Run against 16 hand-written scenarios covering all 10 canonical spec scenarios plus extra
safety/edge cases (`scripts/evaluate.py: build_agent_eval_cases`):

| Metric | Value |
|---|---|
| Intent accuracy | 1.0 |
| Tool selection accuracy | 1.0 |
| Entity extraction accuracy | 1.0 |
| Task success rate | 1.0 |
| Safety rejection accuracy | 1.0 |
| Fallback rate (LLM used) | 0.0 |
| Error rate | 0.0 |
| Avg latency | ~8 ms/request |

These scenarios are intentionally the same style of utterance as the training data — a
perfect score here demonstrates the pipeline wiring is correct, **not** that the system
generalizes to arbitrary unseen phrasing (see [Limitations](#limitations)).

## Error analysis

`reports/evaluation/error_analysis.json` — real misclassifications from the baseline model
on the 275-example test set (2 out of 275):

1. `"What's my TEA?"` (a typo of "ETA" injected by the dataset's noise generator) →
   predicted `range_query`, true label `navigation`. A legitimately confusable pair once the
   "ETA" signal is corrupted into "TEA" — a genuine tokenization/vocabulary limitation of a
   bag-of-words model, not a bug.
2. `"Opne the driver window"` (typo of "Open") → predicted `close_window`, true label
   `open_window`. The typo destroys the strongest lexical signal ("Open"), and the model
   falls back on weaker cues.

Both are real limitations of (a) a linear bag-of-words classifier and (b) an intentionally
noisy synthetic dataset — exactly the failure mode a larger/more diverse training set or a
transformer with subword tokenization (which is more typo-robust than whole-word TF-IDF
features) would be expected to reduce. This is discussed further in
[Future improvements](#future-improvements).

## Installation & local setup

**Requirements**: Python 3.10+, git. Docker optional (not verified in this environment —
see [Docker](#docker)).

```bash
git clone <this-repo>
cd DriveMindAi

python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt          # core (always required)
# pip install -r requirements-ml.txt     # optional: transformer support
# pip install -r requirements-llm.txt    # optional: real OpenAI LLM support

cp .env.example .env                     # edit if you want to set OPENAI_API_KEY etc.

python scripts/generate_dataset.py
python scripts/validate_dataset.py
python scripts/train_baseline.py
python scripts/evaluate.py

pytest tests/ -v
```

### Running the API
```bash
uvicorn src.api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### Running the dashboard
```bash
streamlit run app/streamlit_app.py
# (make sure the API above is running first)
```

### Running with Docker (unverified — see Docker section)
```bash
docker compose up --build
```

### Environment variables (`.env`)
See `.env.example`. Everything has a safe default; the system runs fully offline with none
of these set.

## Example requests

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Find a charging station near Berlin with at least 150 kW"}'
```
```json
{
  "message": "Find a charging station near Berlin with at least 150 kW",
  "intent": "find_charging_station",
  "entities": {"location": "Berlin", "minimum_power_kw": 150},
  "tool": "find_charging_stations",
  "tool_arguments": {"location": "Berlin", "minimum_power_kw": 150},
  "safety": {"status": "passed", "reason": null},
  "tool_result": {"success": true, "data": {"location": "Berlin", "count": 4, "stations": [...]}, "error": null},
  "response": "Found 4 charging station(s) near Berlin.",
  "steps": [...],
  "errors": []
}
```

```bash
curl -X POST http://localhost:8000/vehicle/climate \
  -H "Content-Type: application/json" -d '{"temperature": 100}'
# -> HTTP 400 {"detail": "temperature 100 outside safe range [16, 30]°C"}
```

## Example agent traces

**Multi-step**: *"What's my battery level and find the nearest charging station."*
```
Step 1: intent=battery_status        tool=get_battery_level       safety=passed
Step 2: intent=find_charging_station tool=find_charging_stations  safety=passed
Response: "Your battery is at 68%. Found 5 charging station(s) near your current location."
```

**Safety rejection**: *"Set temperature to 100 degrees."*
```
intent=set_temperature  entities={temperature: 100}
safety: REJECTED — "temperature 100 outside safe range [16, 30]°C"
Response: "Sorry, I couldn't do that: temperature 100 outside safe range [16, 30]°C."
```

## Project structure

```
DriveMindAi/
├── data/{raw,annotated,processed}/     # generated, not committed
├── notebooks/                          # 01 data analysis, 02 baseline training, 03 evaluation
├── src/
│   ├── preprocessing/                  # traditional-ML vs transformer text cleaning
│   ├── annotation/                     # schema, validator, data-quality
│   ├── nlu/                            # intent classifier, entity extractor, LLM provider, router
│   ├── models/                         # baseline (TF-IDF+LogReg), transformer (DistilBERT), model manager
│   ├── agents/                         # state, planner, tool registry, agent controller
│   ├── tools/                          # vehicle sim + 8 tool modules
│   ├── safety/                         # policies + validator
│   ├── evaluation/                     # shared metrics, model + agent evaluation
│   ├── api/                            # FastAPI app, routes, schemas
│   └── config/                         # settings (.env) + YAML config loading
├── app/streamlit_app.py                # dashboard
├── tests/                              # 79 pytest tests
├── configs/{config.yaml,intents.yaml}  # hyperparameters + intent taxonomy
├── scripts/                            # generate/validate/train/evaluate CLIs
├── models/                             # trained artifacts (gitignored, reproducible)
├── reports/{evaluation,figures}/       # real metrics + charts (gitignored, reproducible)
├── docker/, Dockerfile, docker-compose.yml
└── .github/workflows/ci.yml
```

## Future improvements

- Replace the deterministic gazetteer entity extractor with a trained NER model
  (spaCy custom pipeline or transformer token classification) for open-vocabulary entities
  like arbitrary destinations.
- Collect/incorporate real (non-templated) user utterances to stress-test generalization —
  the current 99%+ accuracy is partly a ceiling effect of a template-generated dataset.
- Add conversation memory / multi-turn context (currently every `/chat` call is stateless).
- Add a real speech-to-text integration behind the same NLU interface (PHASE 18 designed
  the seam for this; not implemented in this pass).
- Add LangGraph-based explicit graph orchestration if/when the agent's control flow grows
  beyond what the current linear planner handles cleanly (deliberately not added now — see
  [Architecture decisions](#architecture-decisions--trade-offs)).
- Persist conversation/evaluation history to SQLite instead of purely in-memory.

## Limitations

- **NOT COMPLETED**: Docker build/run has not been verified — Docker Engine is not
  installed in this development environment.
- **NOT COMPLETED**: GitHub Actions CI has not executed on a real GitHub remote from this
  session — the workflow file is written and its steps verified manually/locally instead.
- **NOT COMPLETED**: the three notebooks under `notebooks/` could not be executed
  end-to-end via `jupyter nbconvert` in this sandboxed development environment — every
  attempt (including a minimal one-cell smoke test) fails with a ZeroMQ
  "Bad file descriptor" kernel-startup error, which is an environment/sandbox socket
  restriction, not a notebook code issue. Every cell's code is the same code already
  verified working via the equivalent scripts (`scripts/generate_dataset.py`,
  `scripts/validate_dataset.py`, `scripts/train_baseline.py`, `scripts/evaluate.py`), so
  the logic is exercised and correct, but the `.ipynb` files themselves carry no saved
  cell outputs. Please re-run them with Jupyter on a normal (non-sandboxed) machine to
  confirm end-to-end, or run `jupyter lab` / `jupyter notebook` interactively.
- The dataset is entirely synthetic/template-generated; near-perfect classifier accuracy
  reflects that, not necessarily generalization to real-world spoken/typed input.
- Entity extraction is deterministic/rule-based; it will not generalize to destinations,
  contacts, etc. outside its gazetteer lists without extension.
- No real speech-to-text, telephony, or navigation/maps API integration — all "external"
  tools (charging stations, restaurants, weather, traffic) use a deterministic simulated
  backend, by design (see PHASE 13/34 constraints — no external API dependency required to
  run the project).
- No persistent conversation memory across `/chat` calls.

## Responsible AI / safety considerations

- The LLM (real or mock) is **never** given direct write access to vehicle state. Every
  mutation goes through `SafetyValidator` first, including requests that bypass the
  agent entirely (e.g. calling `/vehicle/climate` directly).
- Safety rejections are logged, not silently swallowed, and returned to the user as a
  clear, non-technical message.
- The system is designed to **fail closed**: an unrecognized tool, missing parameter, or
  out-of-range value is rejected by default rather than executed with best-effort guessing.
- No user data is sent to a third-party LLM unless the user explicitly configures
  `OPENAI_API_KEY` — the default operating mode is fully offline.

## Resume bullet points

- Designed and built an end-to-end agentic NLP system for automotive voice assistants,
  combining a TF-IDF/Logistic Regression baseline, a fine-tuned DistilBERT transformer, and
  an LLM fallback layer behind a single NLU interface.
- Implemented a safety-validation layer that mediates every LLM/agent tool call against a
  simulated vehicle backend, enforcing type, range, and policy constraints before any state
  mutation — verified with 13 dedicated negative tests.
- Built a rule-based multi-step task planner enabling compound natural-language requests
  (e.g. "check battery and find a charger") to be decomposed and executed as multiple tool
  calls in one turn.
- Generated and validated a 2,700+ example synthetic NLU dataset with a custom annotation
  schema, automated data-quality pipeline, and reproducible seeded generation.
- Shipped a FastAPI backend + Streamlit dashboard exposing a full agent execution trace
  (intent, entities, tool, safety status, latency) for debuggability.
- Achieved 99.3% test-set intent classification accuracy on the baseline model and 100%
  task success on a 16-scenario agent evaluation suite, with all metrics computed from real
  training/evaluation runs and documented error analysis (no fabricated numbers).

## Interview talking points

- **Why not just prompt an LLM for everything?** Cost, latency, determinism, and safety —
  an LLM-only design can't guarantee it will never accept `set_temperature(1000)`; a typed,
  validated tool-calling architecture can.
- **Why TF-IDF *and* a transformer?** To have an honest, real comparison rather than
  asserting "transformers are better" — in this specific clean/templated dataset regime,
  the linear baseline already saturates, which itself is a useful, real finding to discuss.
- **Why is entity extraction rule-based, not ML-based?** Closed vocabulary automotive
  domain + need for 100% predictable behavior in a safety-adjacent system; trade-off is
  explicitly documented as a scalability limitation for open-vocabulary entities.
- **How would this change for production?** Real STT, real maps/POI APIs behind the tool
  interfaces (the `ToolResult` contract wouldn't need to change), persistent conversation
  state, and a formal red-teaming pass on the safety layer.

## Architecture decisions & trade-offs

| Decision | Alternative considered | Why this choice |
|---|---|---|
| Rule-based multi-step planner | LLM-driven planning (ReAct/LangGraph) | Deterministic, free, works identically with/without an LLM configured; sufficient for the bounded automotive tool surface |
| Deterministic entity extraction | spaCy/transformer NER | Predictable, unit-testable, zero training cost for a closed-vocabulary domain |
| MockLLMProvider default | Require an API key | Core project constraint: must be fully demonstrable offline/free |
| DistilBERT over BERT-base/RoBERTa | Larger encoders | Must fine-tune in minutes on a CPU laptop |
| Safety layer as a separate module, not embedded in tools | Validate inside each tool function | Centralizes policy, makes it auditable in one file, and guarantees no tool can be reached without passing through it |
| In-memory vehicle singleton | SQLite-backed state | Simpler for a single-driver demo; the `ToolResult`/tool-function contract doesn't change if this is swapped out later |
