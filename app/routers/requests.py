# ruff: noqa: RUF001
"""Эндпоинт для просмотра запросов, сохранённых middleware'ом /check."""

import typing

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

if typing.TYPE_CHECKING:
    from app.request_store import CheckRequestStore

requests_router = APIRouter(tags=["Saved Requests"])


@typing.final
class StoredRequestItem(BaseModel):
    received_at: str = Field(description="Время получения запроса (UTC, ISO 8601)")
    client: str | None = Field(default=None, description="Адрес клиента в формате host:port")
    body: typing.Any = Field(default=None, description="Тело исходного запроса к /check")


@typing.final
class SavedRequestsResponse(BaseModel):
    total: int = Field(description="Всего сохранённых запросов")
    count: int = Field(description="Сколько записей возвращено в этом ответе")
    requests: list[StoredRequestItem] = Field(description="Сохранённые запросы")


@requests_router.get("/requests")
def list_saved_requests(
    http_request: Request,
    limit: typing.Annotated[int | None, Query(ge=1, description="Вернуть только последние N записей")] = None,
    session_id: typing.Annotated[str | None, Query(description="Фильтр по session_id")] = None,
) -> SavedRequestsResponse:
    store: CheckRequestStore = http_request.app.state.request_store
    records = store.list_records(limit=limit, session_id=session_id)
    return SavedRequestsResponse(
        total=store.count(),
        count=len(records),
        requests=[StoredRequestItem(**record) for record in records],
    )
