"""TF-IDF + Logistic Regression intent classification baseline.

This is the fast, always-available NLU path: no GPU, no large model
download, trains in seconds on a laptop. It is the default model used by
the API (``NLU_MODEL=baseline``) unless a transformer/LLM is configured.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocessing.cleaner import clean_for_traditional_ml


class BaselineIntentClassifier:
    """TF-IDF + Logistic Regression pipeline for intent classification."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 2),
        C: float = 5.0,
        max_iter: int = 1000,
    ) -> None:
        self.pipeline: Pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        preprocessor=clean_for_traditional_ml,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(C=C, max_iter=max_iter),
                ),
            ]
        )
        self._fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> "BaselineIntentClassifier":
        self.pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict(self, texts: list[str]) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Model has not been fit or loaded yet.")
        return list(self.pipeline.predict(texts))

    def predict_proba(self, text: str) -> dict[str, float]:
        if not self._fitted:
            raise RuntimeError("Model has not been fit or loaded yet.")
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        return {cls: round(float(p), 4) for cls, p in zip(classes, proba)}

    def predict_one(self, text: str) -> tuple[str, float]:
        """Return (predicted_intent, confidence) for a single utterance."""
        probs = self.predict_proba(text)
        best_intent = max(probs, key=probs.get)
        return best_intent, probs[best_intent]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "BaselineIntentClassifier":
        instance = cls()
        instance.pipeline = joblib.load(path)
        instance._fitted = True
        return instance

    @property
    def classes(self) -> list[str]:
        return list(self.pipeline.classes_) if self._fitted else []
