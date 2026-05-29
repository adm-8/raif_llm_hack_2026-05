"""WindowClassifier — recovers the diluted late/short-violation signal.

Whole-conversation features average a brief, late violation into a mostly-legit
thread; this classifier max-pools per-window scores so it survives.

Synthetic dialogues are short (~6 turns, flag in turn 1); real ones are long
(~15, flag buried late). A single TF-IDF vector over the whole conversation
averages a brief late violation into a mostly-legit thread, so the verdict
defaults to `clear`. We instead score overlapping windows of the dialogue and
**max-pool** each class's probability across windows, then blend with the
whole-conversation probability::

    p = blend * max_over_windows(p_window) + (1 - blend) * p_whole

so a strong signal in any single window survives. Clear-margin abstention (as in
:class:`~app.tuned_classifier.TunedClassifier`) still corrects the balanced
synthetic prior's tendency to over-flag `clear`.

The TF-IDF + LogReg pipeline is trained on whole-conversation client text (same
recipe as TunedClassifier) and reused to score windows — no separate weak-label
training, which keeps it simple and leak-free.

Drop-in: ``fit(list[Conversation]) -> self`` / ``predict(conv) -> (cat, conf)``.
"""

from __future__ import annotations

import typing

import numpy as np

from app.models import CLEAR_CATEGORY, Conversation, Message
from app.tuned_classifier import DEFAULT_CLEAR_MARGIN, _build_pipeline

# Sliding window of trailing messages fed to each per-window score.
WINDOW_SIZE = 4
# Weight of the window-max signal vs the whole-conversation signal.
DEFAULT_BLEND = 0.5


def _user_text(messages: list[Message]) -> str:
    return "\n".join(m.content for m in messages if m.role == "user")


def _full_text(messages: list[Message]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in messages)


def _windows(conv: Conversation, *, full_text: bool = False) -> list[str]:
    """Text for each trailing window; drops empty (bot-only) windows.

    ``full_text`` includes every role (matches FastClassifier features); otherwise
    only the client's lines (matches TunedClassifier features).
    """
    extract = _full_text if full_text else _user_text
    texts: list[str] = []
    for end in range(1, len(conv.messages) + 1):
        start = max(0, end - WINDOW_SIZE)
        text = extract(conv.messages[start:end])
        if text.strip():
            texts.append(text)
    return texts


PipelineFactory = typing.Callable[[], typing.Any]


class WindowClassifier:
    def __init__(
        self,
        clear_margin: float = DEFAULT_CLEAR_MARGIN,
        blend: float = DEFAULT_BLEND,
        pipeline_factory: PipelineFactory = _build_pipeline,
        *,
        full_text: bool = False,
    ) -> None:
        self._pipeline: typing.Any = None
        self._classes: list[str] = []
        self._clear_margin = clear_margin
        self._blend = blend
        self._pipeline_factory = pipeline_factory
        self._full_text = full_text

    def _whole_text(self, conv: Conversation) -> str:
        # full_text mirrors FastClassifier (client lines + full role-tagged text);
        # otherwise client-only, matching TunedClassifier.
        if self._full_text:
            return f"{conv.client_messages_as_string}\n{conv.as_string}"
        return conv.client_messages_as_string

    def fit(self, conversations: list[Conversation]) -> WindowClassifier:
        self._pipeline = self._pipeline_factory()
        self._pipeline.fit([self._whole_text(c) for c in conversations],
                            [c.category for c in conversations])
        self._classes = list(self._pipeline.named_steps["clf"].classes_)
        return self

    def _pooled_proba(self, conv: Conversation) -> np.ndarray:
        whole = self._pipeline.predict_proba([self._whole_text(conv)])[0]
        windows = _windows(conv, full_text=self._full_text)
        if not windows:
            return np.asarray(whole)
        window_max = self._pipeline.predict_proba(windows).max(axis=0)
        return np.asarray(self._blend * window_max + (1 - self._blend) * whole)

    def predict(self, conv: Conversation) -> tuple[str, float]:
        if self._pipeline is None:
            raise RuntimeError("Call fit() first.")
        proba = self._pooled_proba(conv)
        order = np.argsort(proba)[::-1]
        top = self._classes[int(order[0])]
        # Abstain to `clear` when the top red-flag fails to beat `clear` by the
        # margin. Skipped if `clear` was absent from training (defensive).
        if self._clear_margin > 0 and top != CLEAR_CATEGORY and CLEAR_CATEGORY in self._classes:
            clear_i = self._classes.index(CLEAR_CATEGORY)
            if proba[order[0]] - proba[clear_i] < self._clear_margin:
                return CLEAR_CATEGORY, float(proba[clear_i])
        return top, float(proba[int(order[0])])

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None
