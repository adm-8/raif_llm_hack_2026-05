"""RescueCascade — combine a precise clean model with an augmented recall model.

Two base learners with complementary, empirically-verified failure modes:

* **clean** (combined-trained full-text char model): excellent `clear` precision
  (25/26 on the real LOO panel) but misses violations diluted in long dialogues
  (`scope_violation` 0/4).
* **augmented** (same recipe, trained on length-augmented data): catches the
  buried violations (`scope` 4/4, every minority class 3-4/4) but over-flags
  `clear` (8/26) — it learned "violation content anywhere ⇒ violation".

Neither ships alone. The cascade keeps the clean model's verdict by default and
only lets the augmented model **rescue** a dialogue the clean model dismissed as
`clear` — and only when the augmented model is confident above ``rescue_threshold``.
So `clear` precision is preserved (we override `clear` only on strong evidence)
while buried violations get recovered.

``rescue_threshold`` is the single tunable knob; it trades `clear` recall against
violation recall. Drop-in: ``fit(list[Conversation]) -> self`` /
``predict(conv) -> (cat, conf)``.
"""

from __future__ import annotations

import typing

from app.augment import AugmentedClassifier
from app.classifier import _build_pipeline as _build_char_pipeline
from app.models import CLEAR_CATEGORY, Conversation
from app.window_classifier import WindowClassifier

Factory = typing.Callable[[], typing.Any]

# Above this augmented-model confidence, a clean-`clear` verdict is overridden.
# 0.60 is the robust operating point on the real-50 LOO panel: it preserves the
# majority `clear` class (25/26) and synthetic CV (~0.97) while recovering buried
# minority violations (LOO ~0.82). Lowering toward 0.50 trades `clear` recall for
# more minority recall (LOO ~0.84, clear 20/26) — pick per deployment priors.
DEFAULT_RESCUE_THRESHOLD = 0.60


def _default_clean() -> WindowClassifier:
    # Full-text char features + window pooling; small margin keeps clear high.
    return WindowClassifier(clear_margin=0.10, pipeline_factory=_build_char_pipeline, full_text=True)


def _default_augmented() -> AugmentedClassifier:
    # No clear-margin: we WANT high violation recall here; the cascade gates it.
    return AugmentedClassifier(
        lambda: WindowClassifier(clear_margin=0.0, pipeline_factory=_build_char_pipeline, full_text=True))


class RescueCascade:
    def __init__(
        self,
        clean_factory: Factory = _default_clean,
        augmented_factory: Factory = _default_augmented,
        *,
        rescue_threshold: float = DEFAULT_RESCUE_THRESHOLD,
    ) -> None:
        self._clean = clean_factory()
        self._augmented = augmented_factory()
        self._rescue_threshold = rescue_threshold

    def fit(self, conversations: list[Conversation]) -> RescueCascade:
        self._clean.fit(conversations)
        self._augmented.fit(conversations)
        return self

    def predict(self, conv: Conversation) -> tuple[str, float]:
        clean_cat, clean_conf = self._clean.predict(conv)
        if clean_cat != CLEAR_CATEGORY:
            # Trust the precise model's positive verdicts.
            return clean_cat, clean_conf
        # Clean says `clear` — let the augmented detector rescue strong violations.
        aug_cat, aug_conf = self._augmented.predict(conv)
        if aug_cat != CLEAR_CATEGORY and aug_conf >= self._rescue_threshold:
            return aug_cat, aug_conf
        return CLEAR_CATEGORY, clean_conf

    @property
    def is_fitted(self) -> bool:
        return getattr(self._clean, "is_fitted", False)
