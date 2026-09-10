# DriveMind AI — Interview Guide

Expanded Q&A depth beyond the README's condensed
["Interview talking points"](../README.md#interview-talking-points). Every number here
is sourced from `reports/evaluation/*.json` — regenerate with `python scripts/evaluate.py`
to confirm before an interview if the codebase has changed since.

## "Walk me through what happens when a user sends a message"

Give the pipeline from `docs/ARCHITECTURE.md`, but the version worth saying out loud:
"Text goes through NLU — either a trained scikit-learn TF-IDF/LogReg model or a
fine-tuned DistilBERT, your choice via an env var — which gives me an intent and a
confidence score. I run deterministic regex/gazetteer entity extraction scoped to
whatever entities are valid for that intent. The agent looks up which tool that intent
maps to, and *before* the tool runs, a safety validator checks the tool name and every
argument against explicit policies — range checks, type checks, required-field checks.
Only if that passes does the tool actually touch the simulated vehicle's state. The
whole trace — intent, entities, tool, safety verdict, tool result — comes back as
structured JSON, which is also what the debug panel in the UI renders."

## "Why is entity extraction rule-based instead of a trained NER model?"

Two reasons, and be honest that trade-off cuts both ways:
1. The domain is closed-vocabulary — a bounded set of cities, music genres, window
   positions, contacts. A regex/gazetteer approach is 100% predictable, which matters
   in a system that's allowed to affect vehicle state, and it costs nothing to run or
   train.
2. It doesn't generalize to open vocabulary (an arbitrary destination name, an
   unlisted contact) — that's a real, documented limitation, not glossed over. The
   natural fix is a trained NER model (spaCy or transformer token classification), and
   the project deliberately didn't build one because there was no dataset/training
   signal to justify it yet — see README "Future improvements."

If pushed on "why not just have the LLM extract entities always instead of a
classifier + rules": cost, latency, and determinism. The rule-based path costs
nothing and is instant; the LLM is used *only* as a fallback when the trained
classifier's confidence is low (`NLURouter`, threshold 0.5).

## "You trained two models — which one is better, and why keep both?"

Be precise with the actual numbers (test set, 275 examples):

| Model | Accuracy | Macro F1 | Training time |
|---|---|---|---|
| TF-IDF + Logistic Regression | 98.55% | 0.9851 | seconds |
| DistilBERT (fine-tuned) | 99.27% | 0.9918 | ~10.5 min (CPU) |

The transformer wins, and the *why* is the interesting part, not just the number: of
the baseline's 4 test-set errors, all 4 are typo-corrupted inputs (e.g. "Make it
ocoler"). Re-running the transformer on those exact 4 examples, it recovers 2 — because
DistilBERT's subword tokenizer still extracts partial signal from a misspelled word
("ocoler" still shares subword pieces with "cooler"), where whole-word TF-IDF just
treats it as an unknown token and loses the signal entirely. That's a genuinely
explainable mechanism, not a coincidence.

Why the baseline still ships as default: a 0.72-point accuracy gap doesn't justify
~10 minutes of CPU training and a >1GB PyTorch/transformers dependency for every
deployment of a small, well-defined intent set. `NLU_MODEL=transformer` is one env var
if someone wants the extra robustness.

If asked "isn't this dataset too clean for a fair comparison" — agree openly. It's
template-generated; 98–99% accuracy on either model partly reflects that. The
interesting finding (typo robustness) is real precisely *because* it shows up even in
an otherwise easy, saturated benchmark.

## "How do you know the safety layer actually works, not just that you wrote code for it?"

`tests/test_safety.py` has 13 tests that assert specific rejections: temperature above
30°C or below 16°C, volume outside 0–100, an unknown tool name, an unknown window
identifier, a window-open request above 200 km/h, missing required parameters, wrong
argument types. `tests/test_api.py::test_vehicle_climate_endpoint_rejects_unsafe_temperature`
proves the same policy applies to a direct REST call that never touches the agent or
LLM at all — the safety layer isn't something the agent remembers to call, it's the
only path into `VehicleSimulator`'s mutating methods.

Also worth mentioning if it comes up naturally: an engineering audit pass on this
project found and fixed a case where a tool (`set_climate_mode`) was wired into the
tool registry incorrectly and silently failed every time it was actually invoked,
despite looking correct on inspection — the fix included adding a whole-system
invariant test (every registered tool's expected-vehicle-argument flag must match its
function signature) specifically so that class of bug can't reappear silently. That's
a good story about the difference between "code exists" and "code is verified to work."

## "What would you change for a production deployment?"

Concrete, not hand-wavy:
- Replace the simulated backends (charging stations, restaurants, weather, traffic)
  with real APIs behind the same `ToolResult` contract — no agent/safety code changes
  needed, only the tool function bodies.
- Real speech-to-text behind `src/voice/stt.py`'s existing `SpeechToTextProvider`
  interface (currently a documented no-op stub, by design).
- Persistent conversation state (currently every `/chat` call is stateless) and
  persistent vehicle state (currently in-memory, reset on restart).
- A formal red-team pass on the safety policies specifically — the current policies
  are the ones an obvious threat model suggests, but haven't been adversarially tested
  by someone trying to find a bypass.

## "What's the single most defensible claim you can make about this project?"

Not the accuracy numbers — those are a property of a clean, synthetic dataset. The
defensible claim is architectural: an LLM-driven agent that can only ever reach vehicle
state through a mandatory, independently-tested validation layer, with 118 automated
tests covering the ML models, entity extraction, every tool, the safety policies, the
agent's multi-step planning, and the API — not a demo that happens to work when you
type the expected phrases.
