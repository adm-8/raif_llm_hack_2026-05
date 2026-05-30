import collections.abc
import contextlib
import logging
import os
from pathlib import Path

from fastapi import FastAPI

from app.middleware import CheckRequestCaptureMiddleware
from app.model_loader import load_model
from app.models import load_llm
from app.request_store import CheckRequestStore
from app.routers import check_router, health_router, requests_router

app_logger = logging.getLogger("uvicorn.error")

# Suppress noisy INFO-level HTTP logs from huggingface_hub / httpx.
for _noisy_logger in ("httpx", "huggingface_hub", "huggingface_hub.file_download"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

CHECK_REQUESTS_LOG = Path(os.getenv("CHECK_REQUESTS_LOG", "artifacts/check_requests.jsonl"))
request_store = CheckRequestStore(CHECK_REQUESTS_LOG)


@contextlib.asynccontextmanager
async def run_lifespan(fastapi_app: FastAPI) -> collections.abc.AsyncIterator[None]:
    fastapi_app.state.llm_client = load_llm()

    # Try V3 (E5-only, fast) → V2 (E5+LaBSE) → RescueCascade fallback.
    try:
        from app.v3.model_loader import load_model as load_v3_model  # noqa: PLC0415

        fastapi_app.state.streaming_classifier = load_v3_model()
        app_logger.info("Using V3 classifier (E5-only, no LaBSE)")
    except Exception:  # noqa: BLE001
        app_logger.warning("V3 classifier unavailable — trying V2")
        try:
            from app.v2.model_loader import load_model as load_v2_model  # noqa: PLC0415

            fastapi_app.state.streaming_classifier = load_v2_model()
            app_logger.info("Using V2 classifier (E5+LaBSE)")
        except Exception:  # noqa: BLE001
            app_logger.exception("V2 classifier unavailable — falling back to RescueCascade")
            fastapi_app.state.streaming_classifier = load_model()

    fastapi_app.state.request_store = request_store

    server_port = os.getenv("DEV_PORT", "8787")
    app_logger.info("Server: http://localhost:%s", server_port)
    app_logger.info("Docs:   http://localhost:%s/docs", server_port)

    yield


application = FastAPI(
    title="Red Flag Detector API",
    version="1.0.0",
    description="API для детекции red flags в диалогах.",
    lifespan=run_lifespan,
)

application.add_middleware(CheckRequestCaptureMiddleware, store=request_store)

application.include_router(health_router)
application.include_router(check_router)
application.include_router(requests_router)
