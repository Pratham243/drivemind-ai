"""DistilBERT-based intent classifier.

This module is only importable when the optional ML dependencies
(``requirements-ml.txt``: torch, transformers, datasets) are installed.
The rest of the application never imports this module at top level —
it is loaded lazily by :mod:`src.models.model_manager` so that the core
system (baseline NLU, agent, tools, API, UI) works without a ~1GB
PyTorch/transformers install.

Model choice: ``distilbert-base-uncased`` — 66M parameters, ~260MB,
chosen specifically because it fine-tunes in a few minutes on a CPU-only
laptop, unlike larger encoders (BERT-base, RoBERTa) which are impractical
to train without a GPU for a student project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without ML extras
    TRANSFORMERS_AVAILABLE = False


class _IntentDataset(Dataset if TRANSFORMERS_AVAILABLE else object):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class TransformerIntentClassifier:
    """Fine-tuned DistilBERT sequence classifier for intent detection."""

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        max_length: int = 32,
        device: str | None = None,
    ) -> None:
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "torch/transformers not installed. Run: " "pip install -r requirements-ml.txt"
            )
        self.base_model = base_model
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.label2id: dict[str, int] = {}
        self.id2label: dict[int, str] = {}

    def _init_model(self, labels: list[str]) -> None:
        self.label2id = {label: i for i, label in enumerate(sorted(labels))}
        self.id2label = {i: label for label, i in self.label2id.items()}
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model, num_labels=len(self.label2id)
        ).to(self.device)

    def _encode(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def fit(
        self,
        train_texts: list[str],
        train_labels: list[str],
        val_texts: list[str] | None = None,
        val_labels: list[str] | None = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        all_labels: list[str] | None = None,
        verbose: bool = True,
    ) -> list[dict[str, float]]:
        """Fine-tune DistilBERT on the given training data. Returns a list
        of per-epoch training-loss / validation-accuracy dicts (real
        numbers, not simulated).
        """
        label_universe = all_labels or sorted(set(train_labels))
        self._init_model(label_universe)

        train_enc = self._encode(train_texts)
        train_ids = [self.label2id[label] for label in train_labels]
        dataset = _IntentDataset(train_enc, train_ids)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        total_steps = len(loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        history = []
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / max(1, len(loader))

            val_acc = None
            if val_texts is not None and val_labels is not None:
                preds = self.predict(val_texts)
                val_acc = float(np.mean([p == y for p, y in zip(preds, val_labels)]))

            record = {"epoch": epoch + 1, "train_loss": round(avg_loss, 4)}
            if val_acc is not None:
                record["val_accuracy"] = round(val_acc, 4)
            history.append(record)
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}: {record}")

        return history

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[str]:
        self.model.eval()
        enc = self._encode(texts)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        return [self.id2label[i] for i in pred_ids]

    @torch.no_grad()
    def predict_one(self, text: str) -> tuple[str, float]:
        self.model.eval()
        enc = self._encode([text])
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        best_id = int(torch.argmax(probs).item())
        return self.id2label[best_id], round(float(probs[best_id].item()), 4)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        import json

        with open(path / "label_map.json", "w") as f:
            json.dump(self.id2label, f)

    @classmethod
    def load(cls, path: str | Path, max_length: int = 32) -> "TransformerIntentClassifier":
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("torch/transformers not installed.")
        import json

        path = Path(path)
        instance = cls.__new__(cls)
        instance.base_model = str(path)
        instance.max_length = max_length
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"
        instance.tokenizer = AutoTokenizer.from_pretrained(path)
        instance.model = AutoModelForSequenceClassification.from_pretrained(path).to(
            instance.device
        )
        with open(path / "label_map.json") as f:
            id2label = json.load(f)
        instance.id2label = {int(k): v for k, v in id2label.items()}
        instance.label2id = {v: k for k, v in instance.id2label.items()}
        return instance
