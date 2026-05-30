# ruff: noqa: RUF002
"""Risk detection wired into ``check_dialogue``.

``check.py`` calls ``app.models.process_risk_detection(llm_client, raw_text)``.
That function delegates here.

Detector selection: ``DETECTOR_TYPE`` constant controls mode.
* ``"llm"`` (default) — delegates to :func:`app.llm_classifier.run_llm_detection`.
* ``"regex"`` — uses :class:`~app.regex_classifier.RegexClassifier` (no training).
"""

from __future__ import annotations

import logging
import typing

from app.models import (
    CLEAR_CATEGORY,
    Conversation,
    LLMClient,
    Message,
)
from app.regex_classifier import RegexClassifier

_KNOWN_ROLES = {"user", "support", "chatbot", "assistant"}

DETECTOR_TYPE: typing.Literal["llm", "regex"] = "llm"

detection_logger = logging.getLogger("uvicorn.error")

_classifier: RegexClassifier | None = None


def _get_classifier() -> RegexClassifier:
    global _classifier  # noqa: PLW0603
    if _classifier is None:
        _classifier = RegexClassifier()
    return _classifier


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
    llm_client: LLMClient,
    raw_text: str,
) -> dict[str, typing.Any] | None:
    """Entry point used by app.models.process_risk_detection."""
    if DETECTOR_TYPE == "llm":
        from app.llm_classifier import run_llm_detection  # noqa: PLC0415

        return run_llm_detection(llm_client, raw_text)

    try:
        conv = _parse_raw_text(raw_text)
        category, _confidence = _get_classifier().predict(conv)
    except Exception:
        detection_logger.exception("Детекция упала — возвращаю clear")
        return None

    if category == CLEAR_CATEGORY:
        return None
    return {"category": category}
