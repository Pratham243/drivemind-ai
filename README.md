# 🚗 DriveMind AI — Agentic In-Car Voice Assistant

> **92% intent-classification accuracy** on a custom-annotated 500+ example automotive dataset.
> Multi-domain agentic architecture with safety-validation for real-vehicle behavior.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent%20Orchestration-FF6F00?style=flat)](https://github.com/langchain-ai/langgraph)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![CI](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=flat&logo=github-actions&logoColor=white)](https://github.com/features/actions)

---

## What This Is

DriveMind is an agentic voice assistant for in-car use — the kind of AI that understands multi-step, ambiguous requests across vehicle domains ("Turn on the AC, navigate home, and play my driving playlist") and executes them safely without hallucinating commands.

**Why this is hard:** In-car voice AI can't afford hallucinations. A misunderstood "turn off" command on the highway has consequences. This system has a safety-validation layer that constrains agent-issued commands before execution.

---

## Results

| Metric | Value |
|---|---|
| Intent classification accuracy | **92%** |
| Dataset size | **500+ annotated examples** |
| Domains covered | Climate · Navigation · Media · Charging |
| Model comparison | TF-IDF+LR vs. BERT vs. RoBERTa (full benchmark) |

---

## Architecture

```
Voice Input
    │
    ▼
Speech-to-Text
    │
    ▼
┌──────────────────────────────────┐
│   Intent Classification          │
│   (RoBERTa fine-tuned)          │
│   + NER (spaCy — entities)      │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│   LangGraph Agentic Router       │
│   (multi-step reasoning graph)   │
└──┬──────────┬──────┬──────┬──────┘
   │          │      │      │
   ▼          ▼      ▼      ▼
Climate   Navigation  Media  Charging
 Agent      Agent    Agent   Agent
   │          │      │      │
   └──────────┴──────┴──────┘
               │
               ▼
   ┌───────────────────────┐
   │  Safety Validator     │
   │  (constraint checker) │
   └───────────┬───────────┘
               │
               ▼
         Execute Command
```

---

## Dataset & Annotation

The 500+ example dataset was created and annotated personally from scratch — not scraped from an existing dataset:

- **Schema design**: 4 domains × ~15 intents each, with entity slots per intent
- **Annotation process**: intent labels + entity spans (slot filling)
- **QA pipeline**: class-imbalance detection, duplicate-label detection, missing-entity checks
- **Split**: 80/10/10 train/val/test with stratified sampling

This matters because real automotive NLU datasets don't exist publicly — the model has to be built from domain knowledge up.

---

## Model Benchmarking

| Model | Accuracy | F1 (macro) |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 74% | 0.71 |
| BERT (bert-base-uncased, fine-tuned) | 89% | 0.87 |
| **RoBERTa (roberta-base, fine-tuned)** | **92%** | **0.91** |

RoBERTa selected based on F1 and confusion-matrix analysis (fewer critical-domain misclassifications vs. BERT).

---

## Stack

| Component | Technology |
|---|---|
| Agent Orchestration | LangGraph |
| NLP / Intent Classification | HuggingFace Transformers (RoBERTa), spaCy |
| Deep Learning | PyTorch |
| Data Annotation & QA | Custom pipeline (Label Studio, Python QA scripts) |
| Backend API | FastAPI |
| Testing | pytest |
| Containerization | Docker |
| CI/CD | GitHub Actions |

---

## How to Run

```bash
git clone https://github.com/Pratham243/drivemind-ai
cd drivemind-ai
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Key Engineering Decisions

- **RoBERTa over GPT for intent classification**: Lower latency, fully deterministic, no API dependency for real-time in-car use
- **LangGraph for multi-domain routing**: Explicit state graph prevents the "one long prompt" anti-pattern; each domain agent is isolated and testable
- **Safety validator as a separate layer**: Keeps constraint logic outside agent code, easier to audit and update independently

---

*M.Sc. AI project · by [Prathamesh Shimpi](https://linkedin.com/in/prathamesh-shimpi0501)*
