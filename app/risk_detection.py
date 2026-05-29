"""Real risk detection wired into ``check_dialogue`` without touching ``check.py``.

``check.py`` calls ``app.models.process_risk_detection(llm_client, raw_text)``.
That function delegates here, so all behaviour changes live in this file.

Pipeline
--------
1. Parse the formatted ``raw_text`` ("role: content" per line) back into a
   :class:`~app.models.Conversation`.
2. Run the :class:`~app.streaming_classifier.StreamingClassifier` (trained once
   at first use from ``artifacts/train.json`` plus the phrase lexicon in
   ``artifacts/phrases.json``). It walks the dialogue message-by-message and
   commits the whole-conversation verdict.
3. Return ``{"category": ...}`` for a red flag, or ``None`` for "clear".
"""

from __future__ import annotations

import json
import pathlib
import typing

from app.models import (
    CLEAR_CATEGORY,
    Conversation,
    LLMClient,
    Message,
)
from app.streaming_classifier import StreamingClassifier

_KNOWN_ROLES = {"user", "support", "chatbot", "assistant"}
_TRAIN_PATH = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "train.json"


def _parse_raw_text(raw_text: str) -> Conversation:
    messages: list[Message] = []
    for line in raw_text.splitlines():
        prefix, sep, rest = line.partition(": ")
        if sep and prefix in _KNOWN_ROLES:
            messages.append(Message(role=prefix, content=rest))
        elif messages:
            messages[-1].content += f"\n{line}"
        else:
            messages.append(Message(role="user", content=line))
    return Conversation(session_id="", messages=messages)


_classifier: StreamingClassifier | None = None


def _get_classifier() -> StreamingClassifier:
    """Train the classifier once (train.json + phrase lexicon) and cache it."""
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        raw = json.loads(_TRAIN_PATH.read_text(encoding="utf-8"))
        conversations = [Conversation.from_dict(item) for item in raw]
        _classifier = StreamingClassifier().fit(conversations)
    return _classifier


def run_detection(
    llm_client: LLMClient,  # noqa: ARG001
    raw_text: str,
) -> dict[str, typing.Any] | None:
    """Entry point used by app.models.process_risk_detection.

    Returns ``{"category": ...}`` for a detected red flag, or ``None`` for clear.
    """
    conv = _parse_raw_text(raw_text)
    category, _confidence = _get_classifier().predict(conv)

    if category == CLEAR_CATEGORY:
        return None
    return {"category": category}
