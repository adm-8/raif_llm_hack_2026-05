"""LLM-клиент и детектор red flags."""

from __future__ import annotations

import dataclasses
import os
import typing

import httpx
from pydantic import BaseModel, Field

OPENROUTER_MODEL = "anthropic/claude-sonnet-4.6"
CLEAR_CATEGORY = "clear"

CATEGORIES = [
    "clear",
    "information_extraction",
    "transaction_coercion",
    "policy_manipulation",
    "identity_deception",
    "adversarial_attack",
    "scope_violation",
]


class Message(BaseModel):
    role: str = Field(description="Роль отправителя сообщения (user, support, chatbot)")
    content: str = Field(description="Содержимое сообщения")

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Message:
        return cls(role=data["role"], content=data["content"])


class RedFlag(BaseModel):
    category: str = Field(description="Категория обнаруженного риска")

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> RedFlag:
        return cls(category=data["category"])


@dataclasses.dataclass
class Conversation:
    session_id: str
    messages: list[Message]
    expected_red_flags: list[RedFlag] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, typing.Any]) -> Conversation:
        return cls(
            session_id=data["session_id"],
            messages=[Message.from_dict(m) for m in data["messages"]],
            expected_red_flags=[RedFlag.from_dict(f) for f in data.get("expected_red_flags", [])],
        )

    @property
    def bot_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "chatbot"]

    @property
    def client_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]

    @property
    def as_string(self) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in self.messages)

    @property
    def bot_messages_as_string(self) -> str:
        return "\n".join(m.content for m in self.bot_messages)

    @property
    def client_messages_as_string(self) -> str:
        return "\n".join(m.content for m in self.client_messages)

    @property
    def category(self) -> str:
        if not self.expected_red_flags:
            return CLEAR_CATEGORY
        return self.expected_red_flags[0].category


@typing.final
class LLMClient:
    """chat-completions via OpenRouter."""

    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")

    def request_completion(self, prompt_text: str, *, json_mode: bool = True) -> str | None:
        if not self.api_key:
            return None

        request_payload: dict[str, typing.Any] = {
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
        }
        if json_mode:
            request_payload["response_format"] = {"type": "json_object"}

        try:
            response = httpx.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=request_payload,
            )
            return str(response.json()["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001
            return None


def load_llm() -> LLMClient:
    """Создаёт LLM-клиент при старте приложения."""
    return LLMClient()
