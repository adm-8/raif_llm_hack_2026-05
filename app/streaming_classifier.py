from __future__ import annotations

import dataclasses
import json
import pathlib
import typing

from sklearn.preprocessing import LabelEncoder

from app.classifier import _build_pipeline, _to_features
from app.models import (
    CATEGORIES,
    CLEAR_CATEGORY,
    Conversation,
    Message,
    RedFlag,
)

if typing.TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

# Phrase lexicon used to augment training (one short labeled example per phrase).
PHRASES_PATH = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "phrases.json"
# Sample weight for phrase examples relative to real dialogues (1.0). Kept low
# so the ~120 short phrases enrich vocabulary without skewing class balance.
PHRASE_WEIGHT = 0.05

# Rolling window: how many trailing turns feed each message-level classification.
WINDOW_SIZE = 4
# Decay applied to the running per-label risk before folding in the new score.
DECAY = 0.9
# Aggregated risk above this -> escalate; above MEDIUM -> monitor; else safe.
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.40

Decision = typing.Literal["safe", "monitor", "escalate"]
EscalateFn = typing.Callable[["StreamingResult"], None]


@dataclasses.dataclass
class StreamingResult:
    """Outcome of streaming a full conversation."""

    final_category: str
    confidence: float
    risk_scores: dict[str, float]
    decision: Decision
    triggering_message: int | None


def load_phrase_conversations(path: pathlib.Path = PHRASES_PATH) -> list[Conversation]:
    """Turn artifacts/phrases.json into single-message labeled Conversations.

    Each trigger phrase becomes one short training example for its category,
    enriching the TF-IDF vocabulary — especially useful for the thin minority
    classes (4 dialogues each). Keys starting with "_" (e.g. ``_comment``) are
    skipped. Returns an empty list if the file is missing.
    """
    if not path.exists():
        return []
    data: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
    return [
        Conversation(
            session_id="",
            messages=[Message(role="user", content=phrase)],
            expected_red_flags=[RedFlag(category=category)],
        )
        for category, phrases in data.items()
        if not category.startswith("_")
        for phrase in phrases
    ]


def _window_text(messages: list[Message]) -> str:
    """Feature text for a window: client lines first, then the full window."""
    client = "\n".join(m.content for m in messages if m.role == "user")
    full = "\n".join(f"{m.role}: {m.content}" for m in messages)
    return f"{client}\n{full}"


def _windows(conv: Conversation) -> list[str]:
    """Overlapping windows of the last WINDOW_SIZE turns, one per message."""
    windows: list[str] = []
    for end in range(1, len(conv.messages) + 1):
        start = max(0, end - WINDOW_SIZE)
        windows.append(_window_text(conv.messages[start:end]))
    return windows


def _decision_for(risk: float) -> Decision:
    if risk >= HIGH_THRESHOLD:
        return "escalate"
    if risk >= MEDIUM_THRESHOLD:
        return "monitor"
    return "safe"


class StreamingClassifier:
    def __init__(
        self,
        escalate_fn: EscalateFn | None = None,
        *,
        augment_with_phrases: bool = True,
        phrase_weight: float = PHRASE_WEIGHT,
    ) -> None:
        self._pipeline: Pipeline | None = None
        self._label_encoder = LabelEncoder()
        # Escalation seam: where a heavier model (LLMClient / BertClassifier)
        # would plug in for uncertain or rising risk. Not invoked yet.
        self._escalate_fn = escalate_fn
        # Fold artifacts/phrases.json into the training set (off for a clean
        # train.json-only comparison). Phrase examples are down-weighted so they
        # enrich the vocabulary without skewing the class balance.
        self._augment_with_phrases = augment_with_phrases
        self._phrase_weight = phrase_weight

    def fit(self, conversations: list[Conversation]) -> StreamingClassifier:
        # Trained on whole-conversation features — same recipe as FastClassifier,
        # which gives the strongest final-label signal on this dataset. The same
        # pipeline also scores per-message windows for the streaming trace.
        real = list(conversations)
        phrases = load_phrase_conversations() if self._augment_with_phrases else []
        training = real + phrases
        texts = [_to_features(c) for c in training]
        labels = [c.category for c in training]
        self._label_encoder.fit(CATEGORIES)
        self._pipeline = _build_pipeline()
        fit_params: dict[str, typing.Any] = {}
        if phrases:
            fit_params["clf__sample_weight"] = (
                [1.0] * len(real) + [self._phrase_weight] * len(phrases)
            )
        self._pipeline.fit(texts, self._label_encoder.transform(labels), **fit_params)
        return self

    def stream_predict(self, conv: Conversation) -> StreamingResult:
        """Stream the dialogue: per-message risk trace + whole-context verdict."""
        if self._pipeline is None:
            raise RuntimeError("Call fit() first.")

        red_flag_labels = [c for c in CATEGORIES if c != CLEAR_CATEGORY]
        index_of = {label: int(self._label_encoder.transform([label])[0]) for label in red_flag_labels}
        risk: dict[str, float] = dict.fromkeys(red_flag_labels, 0.0)
        peak_score: dict[str, float] = dict.fromkeys(red_flag_labels, 0.0)
        triggered: dict[str, int] = {}

        # --- Streaming pass: per-message windows feed the per-label risk trace.
        for idx, window in enumerate(_windows(conv)):
            proba: typing.Any = self._pipeline.predict_proba([window])[0]
            for label in red_flag_labels:
                score = float(proba[index_of[label]])
                # Decayed running max: a strong single-message signal persists
                # but stale signals fade as the dialogue moves on.
                risk[label] = max(DECAY * risk[label], score)
                # Triggering message = the message whose window scored highest.
                if score > peak_score[label]:
                    peak_score[label] = score
                    triggered[label] = idx

        top_streaming_risk = max(risk.values()) if risk else 0.0

        # --- Verdict: committed category comes from the whole-conversation signal.
        full_proba: typing.Any = self._pipeline.predict_proba([_to_features(conv)])[0]
        best_idx = int(full_proba.argmax())
        final_category = str(self._label_encoder.inverse_transform([best_idx])[0])
        confidence = float(full_proba[best_idx])

        # Decision blends the live streaming risk with the committed verdict so a
        # confident red-flag verdict never reads "safe": a clear verdict can still
        # be "monitor" if the streaming risk is rising, and a red-flag verdict is
        # graded by its own confidence.
        verdict_risk = confidence if final_category != CLEAR_CATEGORY else 0.0
        decision = _decision_for(max(top_streaming_risk, verdict_risk))

        triggering_message = triggered.get(final_category) if final_category != CLEAR_CATEGORY else None

        return StreamingResult(
            final_category=final_category,
            confidence=confidence,
            risk_scores=risk,
            decision=decision,
            triggering_message=triggering_message,
        )

    def predict(self, conv: Conversation) -> tuple[str, float]:
        """Drop-in interface matching FastClassifier."""
        result = self.stream_predict(conv)
        return result.final_category, result.confidence

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None
