# ruff: noqa: RUF001, RUF002
"""Хранилище входящих запросов к эндпоинту /check.

Каждый запрос сохраняется в памяти и дописывается в JSONL-файл, поэтому
сохранённые запросы переживают перезапуск процесса. Все операции потокобезопасны.
"""

import json
import logging
import threading
import typing
from datetime import UTC, datetime
from pathlib import Path

store_logger = logging.getLogger("uvicorn.error")


class StoredCheckRequest(typing.TypedDict):
    """Одна сохранённая запись о запросе к /check."""

    received_at: str
    client: str | None
    body: typing.Any


@typing.final
class CheckRequestStore:
    """Потокобезопасное хранилище запросов с персистентностью в JSONL."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._lock = threading.Lock()
        self._records: list[StoredCheckRequest] = []
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._log_path.exists():
            return
        try:
            with self._log_path.open(encoding="utf-8") as log_file:
                for raw_line in log_file:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        self._records.append(json.loads(line))
                    except json.JSONDecodeError:
                        store_logger.warning("Пропущена битая строка в %s", self._log_path)
        except OSError:
            store_logger.warning("Не удалось прочитать %s", self._log_path)

    def record(self, raw_body: bytes, client: str | None) -> None:
        """Сохраняет запрос. Никогда не бросает исключений (чтобы не ломать /check)."""
        try:
            parsed = self._parse_body(raw_body)
            entry: StoredCheckRequest = {
                "received_at": datetime.now(UTC).isoformat(),
                "client": client,
                "body": parsed,
            }
            with self._lock:
                self._records.append(entry)
                self._append_to_disk(entry)
        except Exception:
            store_logger.exception("Не удалось сохранить запрос к /check")

    @staticmethod
    def _parse_body(raw_body: bytes) -> typing.Any:  # noqa: ANN401 — тело запроса произвольной формы
        if not raw_body:
            return None
        try:
            return json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw_body.decode("utf-8", errors="replace")

    def _append_to_disk(self, entry: StoredCheckRequest) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def list_records(
        self,
        *,
        limit: int | None = None,
        session_id: str | None = None,
    ) -> list[StoredCheckRequest]:
        """Возвращает сохранённые запросы с опциональной фильтрацией."""
        with self._lock:
            records = list(self._records)

        if session_id is not None:
            records = [
                record
                for record in records
                if isinstance(record["body"], dict) and record["body"].get("session_id") == session_id
            ]
        if limit is not None:
            records = records[-limit:]
        return records

    def count(self) -> int:
        """Общее число сохранённых запросов."""
        with self._lock:
            return len(self._records)
