import typing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from app.models import Conversation

CONFIDENCE_THRESHOLD = 0.70

CATEGORIES = [
    "clear",
    "information_extraction",
    "transaction_coercion",
    "policy_manipulation",
    "identity_deception",
    "adversarial_attack",
    "scope_violation",
]


def _conversation_to_features(conv: Conversation) -> str:
    """Combine client + full conversation text as a single string for TF-IDF."""
    client_text = conv.client_messages_as_string
    full_text = conv.as_string
    return f"{client_text}\n{full_text}"


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=30_000,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            C=1.0,
        )),
    ])


class FastClassifier:
    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._label_encoder = LabelEncoder()

    def fit(self, conversations: list[Conversation]):
        texts = [_conversation_to_features(c) for c in conversations]
        labels = [c.category for c in conversations]

        self._label_encoder.fit(CATEGORIES)
        encoded = self._label_encoder.transform(labels)

        self._pipeline = build_pipeline()
        self._pipeline.fit(texts, encoded)
        return self

    def predict(self, conv: Conversation) -> tuple[str, float]:
        """Return (predicted_category, confidence)."""
        if self._pipeline is None:
            msg = "Classifier is not fitted. Call fit() first."
            raise RuntimeError(msg)

        text = [_conversation_to_features(conv)]
        proba: typing.Any = self._pipeline.predict_proba(text)[0]
        best_idx = int(proba.argmax())
        confidence = float(proba[best_idx])
        category = str(self._label_encoder.inverse_transform([best_idx])[0])
        return category, confidence

    def needs_llm_fallback(self, conv: Conversation) -> bool:
        """True when the model is uncertain and LLM should be consulted."""
        _, confidence = self.predict(conv)
        return confidence < CONFIDENCE_THRESHOLD
