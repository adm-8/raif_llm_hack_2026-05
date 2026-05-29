from __future__ import annotations

import typing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.models import (
    CATEGORIES,
    Conversation,
)

CONFIDENCE_THRESHOLD = 0.70


def _to_features(conv: Conversation) -> str:
    return f"{conv.client_messages_as_string}\n{conv.as_string}"


def _build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced")),
    ])


class FastClassifier:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._label_encoder = LabelEncoder()

    def fit(self, conversations: list[Conversation]) -> FastClassifier:
        texts = [_to_features(c) for c in conversations]
        labels = [c.category for c in conversations]
        self._label_encoder.fit(CATEGORIES)
        self._pipeline = _build_pipeline()
        self._pipeline.fit(texts, self._label_encoder.transform(labels))
        return self

    def predict(self, conv: Conversation) -> tuple[str, float]:
        if self._pipeline is None:
            raise RuntimeError("Call fit() first.")
        proba: typing.Any = self._pipeline.predict_proba([_to_features(conv)])[0]
        best_idx = int(proba.argmax())
        return str(self._label_encoder.inverse_transform([best_idx])[0]), float(proba[best_idx])

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None
