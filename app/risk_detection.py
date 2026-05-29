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
import logging
import pathlib
import typing

from app.ensemble_classifier import RescueCascade
from app.models import (
    CLEAR_CATEGORY,
    Conversation,
    LLMClient,
    Message,
)

_KNOWN_ROLES = {"user", "support", "chatbot", "assistant"}
_SYNTHETIC_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "train" / "synthetic_train.json"
_TRAIN_PATH = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "train.json"

detection_logger = logging.getLogger("uvicorn.error")


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


_classifier: RescueCascade | None = None


def _get_classifier() -> RescueCascade:
    """Train the production RescueCascade once (synthetic + real) and cache it."""
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        conversations: list[Conversation] = []
        for path in (_SYNTHETIC_PATH, _TRAIN_PATH):
            if path.exists():
                conversations += [Conversation.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
        _classifier = RescueCascade().fit(conversations)
    return _classifier


def run_detection(
    llm_client: LLMClient,  # noqa: ARG001
    raw_text: str,
) -> dict[str, typing.Any] | None:
    """Entry point used by app.models.process_risk_detection.

    Returns ``{"category": ...}`` for a detected red flag, or ``None`` for clear.
    """
    try:
        conv = _parse_raw_text(raw_text)
        category, _confidence = _get_classifier().predict(conv)
    except Exception:
        detection_logger.exception("Детекция упала — возвращаю clear")
        return None

    if category == CLEAR_CATEGORY:
        return None
    return {"category": category}
