# ruff: noqa: RUF001, RUF002
"""Файл для тестирования с eval сервисом, желательно не трогать."""

import time
import typing

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.models import CLEAR_CATEGORY, Conversation, Message

check_router = APIRouter(tags=["Dialogue Check"])


@typing.final
class DialogueMessage(BaseModel):
    role: str = Field(description="Роль отправителя сообщения (user, support, assistant)")
    content: str = Field(description="Содержимое сообщения")


@typing.final
class DialogueCheckRequest(BaseModel):
    session_id: str = Field(description="Идентификатор пользовательской сессии")
    messages: list[DialogueMessage] = Field(description="Список сообщений в диалоге")


@typing.final
class RedFlagItem(BaseModel):
    category: str = Field(description="Категория обнаруженного риска")


@typing.final
class DialogueCheckResponse(BaseModel):
    session_id: str = Field(description="Идентификатор сессии")
    predicted_red_flags: list[RedFlagItem] = Field(
        description="Список предсказанных нарушений (сравнивается eval-сервисом с expected_red_flags)",
    )
    processing_time_ms: int = Field(description="Время обработки сессии в миллисекундах")


@check_router.post("/check")
def check_dialogue(
    http_request: Request,
    request_body: DialogueCheckRequest,
) -> DialogueCheckResponse:
    start_time = time.perf_counter()

    clf = http_request.app.state.streaming_classifier
    conv = Conversation(
        session_id=request_body.session_id,
        messages=[Message(role=m.role, content=m.content) for m in request_body.messages],
    )

    # Offline RescueCascade is the sole decision-maker (no LLM fallback): the old
    # low-confidence branch re-ran a weaker train.json-only model and could
    # override the stronger combined-trained cascade verdict.
    category, _confidence = clf.predict(conv)

    predicted_red_flags = [] if category == CLEAR_CATEGORY else [RedFlagItem(category=category)]

    processing_time_ms = int((time.perf_counter() - start_time) * 1000)

    return DialogueCheckResponse(
        session_id=request_body.session_id,
        predicted_red_flags=predicted_red_flags,
        processing_time_ms=processing_time_ms,
    )
