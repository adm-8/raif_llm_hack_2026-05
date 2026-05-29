"""Length/position data augmentation — purely offline, no LLM.

Root problem (verified): synthetic dialogues are short (~6 turns, violation in
turn 1); real ones are long (~15, violation buried late among legit banking
chatter). A model trained on synthetic learns "violation = short, up-front" and
defaults long real dialogues to `clear`.

We manufacture the missing distribution by **recombination** of existing labeled
turns:

1. *Late-violation-in-context*: splice a violation's turns into a random position
   inside a `clear` dialogue; keep the violation label. Teaches "a violation
   anywhere in a long thread is still a violation, regardless of position".
2. *Long-clear*: concatenate two `clear` dialogues; keep `clear`. Crucial
   counterweight — without it the model would just learn "long ⇒ violation".

Augmentation is applied **only to training data** (via :class:`AugmentedClassifier`,
which augments inside ``fit``), so cross-validation and the held-out real set
stay leak-free.
"""

from __future__ import annotations

import random
import typing

from app.models import CLEAR_CATEGORY, Conversation

# Augmented variants generated per source violation dialogue.
DEFAULT_VARIANTS = 2


def augment_conversations(
    conversations: list[Conversation],
    *,
    variants_per_violation: int = DEFAULT_VARIANTS,
    seed: int = 42,
) -> list[Conversation]:
    """Return NEW synthetic-from-recombination conversations (not the originals)."""
    rng = random.Random(seed)  # noqa: S311 — recombination sampling, not cryptographic
    clear = [c for c in conversations if c.category == CLEAR_CATEGORY]
    violations = [c for c in conversations if c.category != CLEAR_CATEGORY]
    if not clear or not violations:
        return []

    augmented: list[Conversation] = []

    # 1. Late-violation-in-context.
    for v in violations:
        for k in range(variants_per_violation):
            ctx = rng.choice(clear)
            cut = rng.randint(0, len(ctx.messages))
            messages = ctx.messages[:cut] + v.messages + ctx.messages[cut:]
            augmented.append(Conversation(
                session_id=f"aug_late_{v.session_id}_{k}",
                messages=messages,
                expected_red_flags=v.expected_red_flags,
            ))

    # 2. Long-clear counterweight (match the count so length stays label-neutral).
    n_long_clear = len(violations) * variants_per_violation
    has_pair = len(clear) >= 2  # noqa: PLR2004 — need two distinct clears to concatenate
    for k in range(n_long_clear):
        a, b = rng.sample(clear, 2) if has_pair else (clear[0], clear[0])
        augmented.append(Conversation(
            session_id=f"aug_clear_{k}",
            messages=a.messages + b.messages,
            expected_red_flags=[],
        ))

    return augmented


class AugmentedClassifier:
    """Wraps a base classifier: augments its training set inside ``fit``.

    Keeps the augmentation leak-free by construction — only the conversations
    passed to ``fit`` (the training fold) are recombined.
    """

    def __init__(
        self,
        base_factory: typing.Callable[[], typing.Any],
        *,
        variants_per_violation: int = DEFAULT_VARIANTS,
        seed: int = 42,
    ) -> None:
        self._base = base_factory()
        self._variants = variants_per_violation
        self._seed = seed

    def fit(self, conversations: list[Conversation]) -> AugmentedClassifier:
        extra = augment_conversations(
            conversations, variants_per_violation=self._variants, seed=self._seed)
        self._base.fit(conversations + extra)
        return self

    def predict(self, conv: Conversation) -> tuple[str, float]:
        return self._base.predict(conv)  # type: ignore[no-any-return]

    @property
    def is_fitted(self) -> bool:
        return getattr(self._base, "is_fitted", False)
