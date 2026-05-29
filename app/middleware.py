# ruff: noqa: RUF002
"""ASGI-middleware для перехвата и сохранения запросов к /check.

Реализовано как чистый ASGI-middleware (а не BaseHTTPMiddleware), чтобы
читать тело запроса без риска «съесть» его до того, как до него доберётся
обработчик /check. Тело копится по чанкам и проксируется дальше без изменений.
"""

import typing

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.request_store import CheckRequestStore


@typing.final
class CheckRequestCaptureMiddleware:
    """Сохраняет тело каждого POST-запроса к указанному пути."""

    def __init__(self, app: ASGIApp, store: CheckRequestStore, *, path: str = "/check") -> None:
        self._app = app
        self._store = store
        self._path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not (scope["type"] == "http" and scope.get("method") == "POST" and scope.get("path") == self._path):
            await self._app(scope, receive, send)
            return

        body_chunks: list[bytes] = []
        captured = False
        client = self._format_client(scope.get("client"))

        async def capturing_receive() -> Message:
            nonlocal captured
            message = await receive()
            if message["type"] == "http.request":
                body_chunks.append(message.get("body", b""))
                if not message.get("more_body", False) and not captured:
                    captured = True
                    self._store.record(b"".join(body_chunks), client)
            return message

        await self._app(scope, capturing_receive, send)

    @staticmethod
    def _format_client(client: tuple[str, int] | None) -> str | None:
        if not client:
            return None
        host, port = client
        return f"{host}:{port}"
