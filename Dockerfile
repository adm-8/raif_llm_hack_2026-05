# syntax=docker/dockerfile:1.7

# --- Стадия 1: сборка зависимостей в .venv через uv ---
FROM ghcr.io/astral-sh/uv:0.7-python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1

WORKDIR /app

# Ставим основные зависимости + группу v2 (sentence-transformers, torch, xgboost).
# Слой кешируется, пока не меняются uv.lock / pyproject.toml.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --group v2


# --- Стадия 2: минимальный runtime-образ ---
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    # HuggingFace кеш моделей — хранится внутри образа, загружается при сборке.
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# Непривилегированный пользователь для запуска приложения.
RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app

WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app app ./app
# Артефакты классификатора (train.json + phrases.json + model pkl-ы) — без них /check падает с 500.
COPY --chown=app:app artifacts ./artifacts
# Обучающие данные и реальная разметка.
COPY --chown=app:app data/train ./data/train
COPY --chown=app:app data/requests.json ./data/requests.json

# Создаём директорию HF-кеша и предзагружаем encoder-модели в образ,
# чтобы первый predict не требовал интернета и не вызывал таймаут.
RUN mkdir -p /app/hf_cache \
    && python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small'); \
SentenceTransformer('sentence-transformers/LaBSE'); \
print('HF models cached.')" \
    && chown -R app:app /app/hf_cache

# Предобучаем RescueCascade (fallback-модель, ~8 c) и сериализуем в artifacts/model.pkl.
RUN python -m app.model_loader \
    && chown -R app:app /app/artifacts

# Если artifacts/e5_labse_model.pkl уже скопирован выше — пропускаем обучение.
# Иначе обучаем V2-модель прямо при сборке (медленно, ~15 мин, но однократно).
RUN if [ ! -f /app/artifacts/e5_labse_model.pkl ]; then \
        echo "e5_labse_model.pkl not found — training V2 model at build time..."; \
        python -m app.v2.model_loader; \
        chown -R app:app /app/artifacts; \
    else \
        echo "e5_labse_model.pkl found — skipping V2 build-time training."; \
    fi

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:application", "--host", "0.0.0.0", "--port", "8000"]
