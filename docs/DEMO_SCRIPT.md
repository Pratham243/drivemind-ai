# DriveMind AI — Demo Script

A ~10-minute walkthrough for a live demo or interview. Every example below has been
run against the actual system; outputs shown are real, not illustrative.

## Setup (before the demo starts)

```bash
python scripts/generate_dataset.py
python scripts/train_baseline.py
uvicorn src.api.main:app --reload --port 8000          # terminal 1
streamlit run app/streamlit_app.py                     # terminal 2
```
Open the Streamlit URL it prints (typically `http://localhost:8501`).

## 1. Show the vehicle state panel (30s)

Point out the right-hand panel: battery 68%, range 320 km, temperature 20°C, speed
0 km/h, expandable windows/tire-pressure/media state, and the "Model Info" box showing
`NLU model: baseline` and `LLM provider: mock` — call out that this is running with
**zero external API dependency** by default.

## 2. A simple, single-tool request

Type: **"Find a charging station near Berlin with at least 150 kW"**

Expected response: *"Found 4 charging station(s) near Berlin."*

Open the "Technical trace" panel below and walk through it live:
- Intent: `find_charging_station`
- Tool: `find_charging_stations`
- Safety: `PASSED`
- Entities: `{"location": "Berlin", "minimum_power_kw": 150}`
- Tool result: the actual station list with power/distance/availability

This demonstrates NLU → entity extraction → tool selection → safety → execution in one
visible trace.

## 3. A request that actually mutates state

Type: **"Turn on the AC"**

Expected response: *"Climate mode set to ac_on."*

Then open the vehicle status panel (or click Refresh) and point out `Climate: ac_on`
now reflects the change — this is not a canned response; the underlying
`VehicleSimulator` object was actually mutated, and any subsequent request (or a direct
`GET /vehicle/status` call) sees the new state.

## 4. A safety rejection (the most important demo moment)

Type: **"Set temperature to 100 degrees"**

Expected response: *"Sorry, I couldn't do that: temperature 100 outside safe range
[16, 30]°C."*

This is the moment to explain the architecture's core guarantee: the LLM/agent never
writes to vehicle state directly — every tool call passes through a safety validator
first. Point out the Safety trace shows `REJECTED`, and that the vehicle's temperature
in the status panel is unchanged. See `docs/ARCHITECTURE.md`'s "The LLM does not
directly control vehicle state" section for the deeper explanation if asked.

## 5. Multi-step request

Type: **"What's my battery level and find the nearest charging station"**

Expected response: *"Your battery is at 68%. Found 5 charging station(s) near your
current location."*

Expand the trace and show two steps executed from one message — point out this is a
small rule-based conjunction-splitter (`src/agents/planner.py`), not an LLM call, so it
works identically whether or not an OpenAI key is configured.

## 6. Show the offline LLM fallback

Type something the training data doesn't resemble, e.g.: **"purple elephant dance"**

Expected: intent resolves to `general_question` with a response prefixed
`[MockLLMProvider]`, and the trace's `model` field shows something like
`baseline+llm_fallback(mock)`.

Explain: when the ML classifier's confidence drops below threshold, the system falls
back to the LLM layer for a best-effort structured guess. With no `OPENAI_API_KEY`
configured it's `MockLLMProvider` (deterministic keyword heuristics); setting the key
swaps in `OpenAIProvider` transparently — same interface, no code changes needed
elsewhere.

## 7. (Optional, if time and a GPU-less wait are acceptable) Model comparison

If asked "why two models": open `reports/figures/model_comparison.png` and
`reports/evaluation/model_comparison.json`, and mention the real, measured finding —
the fine-tuned DistilBERT transformer (99.27% test accuracy) actually outperforms the
TF-IDF baseline (98.55%), specifically by recovering 2 of the baseline's 4
typo-corrupted misclassifications, while the baseline remains the default for its
~60x-faster training time. Full discussion in the README's
["Transformer model"](../README.md#transformer-model) section.

## 8. Wrap-up talking point

Close with: this project's strongest technical claim isn't the accuracy numbers — it's
that every layer (NLU, entity extraction, agent, safety, tools) is independently unit
tested (118 tests) and the safety layer specifically has dedicated negative tests
proving it rejects malformed/unsafe/unknown requests, which is the property that
actually matters for a system that's allowed to act on a vehicle.
