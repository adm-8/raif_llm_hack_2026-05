"""Base class for embedding-based classifiers (E5, MiniLM, …).

Subclasses override ``MODEL_NAME`` and ``_prefix`` to adapt to any
HuggingFace encoder that supports mean-pool + L2-normalise.

Requires the ``hf`` dependency group: ``uv sync --group hf``.
"""

from __future__ import annotations

import typing

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer

from app.models import CATEGORIES, Conversation

MAX_LENGTH = 512
BATCH_SIZE = 32


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _to_features(conv: Conversation, *, user_messages_only: bool) -> str:
    if user_messages_only:
        return conv.client_messages_as_string
    return f"{conv.client_messages_as_string}\n{conv.as_string}"


class EmbeddingClassifier:
    """Mean-pool + L2-normalise + LogReg head over any HF encoder.

    Parameters
    ----------
    user_messages_only:
        ``True``  — embed only user turns (all categories still predicted).
        ``False`` — embed client text + full role-tagged dialogue (default).
    model_name:
        HuggingFace model id. Falls back to the subclass ``MODEL_NAME``.
    prefix:
        String prepended to every input before tokenisation.
        E5 models need ``"query: "``; MiniLM needs ``""``.
    device:
        ``"cuda"`` / ``"cpu"`` / ``"mps"``. Auto-detected when ``None``.
    """

    MODEL_NAME: str = ""
    _DEFAULT_PREFIX: str = ""

    def __init__(
        self,
        *,
        user_messages_only: bool = False,
        model_name: str | None = None,
        prefix: str | None = None,
        device: str | None = None,
    ) -> None:
        self._user_messages_only = user_messages_only
        self._model_name = model_name or self.MODEL_NAME
        self._prefix = prefix if prefix is not None else self._DEFAULT_PREFIX
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer: typing.Any = None
        self._model: typing.Any = None
        self._head: LogisticRegression | None = None
        self._label_encoder = LabelEncoder()

    def _ensure_encoder(self) -> None:
        if self._model is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = (
                AutoModel.from_pretrained(self._model_name).to(self._device).eval()
            )

    @torch.no_grad()
    def _embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_encoder()
        prefixed = [self._prefix + t for t in texts]
        vectors: list[np.ndarray] = []
        for start in range(0, len(prefixed), BATCH_SIZE):
            batch = prefixed[start : start + BATCH_SIZE]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self._device)
            output = self._model(**encoded)
            pooled = _mean_pool(output.last_hidden_state, encoded["attention_mask"])
            normed = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(normed.cpu().float().numpy())
        return np.vstack(vectors)

    def fit(self, conversations: list[Conversation]) -> EmbeddingClassifier:
        texts = [_to_features(c, user_messages_only=self._user_messages_only) for c in conversations]
        labels = [c.category for c in conversations]
        self._label_encoder.fit(CATEGORIES)
        embeddings = self._embed(texts)
        self._head = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
        self._head.fit(embeddings, self._label_encoder.transform(labels))
        return self

    def predict(self, conv: Conversation) -> tuple[str, float]:
        if self._head is None:
            raise RuntimeError("Call fit() first.")
        text = _to_features(conv, user_messages_only=self._user_messages_only)
        embedding = self._embed([text])
        proba: typing.Any = self._head.predict_proba(embedding)[0]
        best_idx = int(proba.argmax())
        return (
            str(self._label_encoder.inverse_transform([best_idx])[0]),
            float(proba[best_idx]),
        )

    @property
    def is_fitted(self) -> bool:
        return self._head is not None
