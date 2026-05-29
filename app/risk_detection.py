"""Real risk detection wired into ``check_dialogue`` without touching ``check.py``.

``check.py`` calls ``app.models.process_risk_detection(llm_client, raw_text)``.
That function delegates here, so all behaviour changes live in this file.

Pipeline
--------
1. Parse the formatted ``raw_text`` ("role: content" per line) back into a
   :class:`~app.models.Conversation`.
2. Run the rule-based :class:`~app.regex_classifier.RegexClassifier` — no
   training step required, predictions are instant.
3. Return ``{"category": ...}`` for a red flag, or ``None`` for "clear".
"""

from __future__ import annotations

import typing

from app.models import (
    CLEAR_CATEGORY,
    Conversation,
    LLMClient,
    Message,
)
from app.regex_classifier import RegexClassifier

_KNOWN_ROLES = {"user", "support", "chatbot", "assistant"}

_classifier = RegexClassifier()


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


def run_detection(
    llm_client: LLMClient,  # noqa: ARG001
    raw_text: str,
) -> dict[str, typing.Any] | None:
    """Entry point used by app.models.process_risk_detection.

    Returns ``{"category": ...}`` for a detected red flag, or ``None`` for clear.
    """
    conv = _parse_raw_text(raw_text)
    category, _confidence = _classifier.predict(conv)

    if category == CLEAR_CATEGORY:
        return None
    return {"category": category}
