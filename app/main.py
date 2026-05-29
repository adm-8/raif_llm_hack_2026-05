import collections.abc
import contextlib
import logging
import os
from pathlib import Path

from fastapi import FastAPI

from app.middleware import CheckRequestCaptureMiddleware
from app.models import load_llm
from app.request_store import CheckRequestStore
from app.routers import check_router, health_router, requests_router

app_logger = logging.getLogger("uvicorn.error")

CHECK_REQUESTS_LOG = Path(os.getenv("CHECK_REQUESTS_LOG", "artifacts/check_requests.jsonl"))
request_store = CheckRequestStore(CHECK_REQUESTS_LOG)


@contextlib.asynccontextmanager
async def run_lifespan(fastapi_app: FastAPI) -> collections.abc.AsyncIterator[None]:
    fastapi_app.state.llm_client = load_llm()
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
