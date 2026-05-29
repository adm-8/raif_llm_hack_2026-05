# ruff: noqa: RUF002
"""Real risk detection wired into ``check_dialogue`` without touching ``check.py``.

``check.py`` calls ``app.models.process_risk_detection(llm_client, raw_text)``.
That function delegates here, so all behaviour changes live in this file.

Detector selection
------------------
Модуль поддерживает два режима, переключаемых константой ``DETECTOR_TYPE``:

* ``"llm"`` (по умолчанию) — делегирует в :func:`app.llm_classifier.run_llm_detection`,
  который оборачивает диалог в промпт и вызывает OpenRouter.
* ``"streaming"`` — использует офлайн :class:`~app.streaming_classifier.StreamingClassifier`,
  обучаемый один раз из ``artifacts/train.json`` плюс лексикон ``artifacts/phrases.json``.

Чтобы переключиться, поменяйте значение ``DETECTOR_TYPE`` в этом файле.
В обоих режимах возвращается ``{"category": ...}`` для red flag или ``None`` для ``clear``.
"""

from __future__ import annotations

import json
import logging
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

DETECTOR_TYPE: typing.Literal["llm", "streaming"] = "llm"

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
    llm_client: LLMClient,
    raw_text: str,
) -> dict[str, typing.Any] | None:
    """Entry point used by app.models.process_risk_detection.

    Returns ``{"category": ...}`` for a detected red flag, or ``None`` for clear.
    Маршрут выбора детектора управляется константой :data:`DETECTOR_TYPE`.
    """
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
