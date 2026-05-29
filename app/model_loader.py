"""Pre-fit / load the production RescueCascade.

The cascade trains two TF-IDF models on augmented data (~8.5 s, ~330 MB peak),
which is too heavy to do inside the container's startup hook — a slow/OOM boot
can leave the server unhealthy while the evaluator still gets notified, scoring
0. So we fit **once at Docker build time** and pickle the result; the container
then loads it in well under a second.

Usage:
* build time: ``python -m app.model_loader``  → writes the pickle to MODEL_PATH.
* runtime:    ``load_model()``                 → unpickle, else fit as fallback.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
import pickle

from app.ensemble_classifier import RescueCascade
from app.models import Conversation

logger = logging.getLogger("uvicorn.error")

_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = pathlib.Path(os.getenv("MODEL_PATH", str(_ROOT / "artifacts" / "model.pkl")))
_TRAIN_PATHS = (
    _ROOT / "data" / "train" / "synthetic_train.json",
    _ROOT / "artifacts" / "train.json",
)


def _load_conversations() -> list[Conversation]:
    convs: list[Conversation] = []
    for path in _TRAIN_PATHS:
        if path.exists():
            convs += [Conversation.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    return convs


def build_model() -> RescueCascade:
    """Fit a fresh RescueCascade on all available data (synthetic + real)."""
    convs = _load_conversations()
    clf = RescueCascade()
    if convs:
        clf.fit(convs)
        logger.info("RescueCascade fitted on %d conversations", len(convs))
    return clf


def save_model(path: pathlib.Path = MODEL_PATH) -> None:
    """Build and pickle the model (called at Docker build time)."""
    clf = build_model()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(clf, handle)
    logger.info("RescueCascade pickled to %s", path)


def load_model(path: pathlib.Path = MODEL_PATH) -> RescueCascade:
    """Load the pre-fitted pickle; fall back to fitting fresh if unavailable.

    The fallback guarantees the server still serves a working model even if the
    pickle is missing or was written by an incompatible build.
    """
    if path.exists():
        try:
            with path.open("rb") as handle:
                clf = pickle.load(handle)  # noqa: S301 — our own build artifact
        except Exception:
            logger.exception("Failed to load pickled model at %s — refitting", path)
        else:
            if isinstance(clf, RescueCascade) and clf.is_fitted:
                logger.info("Loaded pre-fitted RescueCascade from %s", path)
                return clf
            logger.warning("Pickle at %s is not a fitted RescueCascade — refitting", path)
    else:
        logger.warning("No pickled model at %s — fitting at startup (slow path)", path)
    return build_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    save_model()
