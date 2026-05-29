"""Pre-fit / load the MultilingualE5Classifier (V2: E5 + LaBSE + TF-IDF MLP).

Usage
-----
* build time / local training:
    ``python -m app.v2.model_loader``
    → trains on all available data, writes artifacts/e5_labse_model.pkl

* runtime (inside FastAPI lifespan):
    ``from app.v2.model_loader import load_model``
    → loads pre-saved pkl; falls back to fitting if the file is missing.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

logger = logging.getLogger("uvicorn.error")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = pathlib.Path(
    os.getenv("E5_MODEL_PATH", str(_ROOT / "artifacts" / "e5_labse_model.pkl"))
)

# Sources used for training (all real + synthetic data combined).
_REAL_DATA_PATH = _ROOT / "data" / "requests.json"
_SYNTHETIC_DATA_PATH = _ROOT / "data" / "train" / "synthetic_train.json"
_ARTIFACT_TRAIN_PATH = _ROOT / "artifacts" / "train.json"


def _load_conversations():  # noqa: ANN201
    from app.models import CLEAR_CATEGORY, Conversation  # noqa: PLC0415

    convs = []

    # Real labelled data (150 dialogs)
    if _REAL_DATA_PATH.exists():
        data = json.loads(_REAL_DATA_PATH.read_text(encoding="utf-8"))
        for req in data["requests"]:
            label = req["category"]
            conv = Conversation.from_dict(
                {
                    "session_id": req["body"]["session_id"],
                    "messages": req["body"]["messages"],
                    "expected_red_flags": (
                        [] if label == CLEAR_CATEGORY else [{"category": label}]
                    ),
                }
            )
            convs.append(conv)
        logger.info("Loaded %d real conversations from %s", len(convs), _REAL_DATA_PATH)

    # Synthetic training data
    for path in (_SYNTHETIC_DATA_PATH, _ARTIFACT_TRAIN_PATH):
        if path.exists():
            items = json.loads(path.read_text(encoding="utf-8"))
            batch = [Conversation.from_dict(d) for d in items]
            convs.extend(batch)
            logger.info("Loaded %d conversations from %s", len(batch), path)

    return convs


def build_model():  # noqa: ANN201
    """Fit a fresh MultilingualE5Classifier on all available data."""
    from app.v2.e5_xgboost import MultilingualE5Classifier  # noqa: PLC0415

    convs = _load_conversations()
    clf = MultilingualE5Classifier()
    clf.fit(convs)
    logger.info("MultilingualE5Classifier fitted on %d conversations", len(convs))
    return clf


def save_model(path: pathlib.Path = MODEL_PATH) -> None:
    """Build, fit, and save the model (called at build time or offline)."""
    clf = build_model()
    clf.save(path)


def load_model(path: pathlib.Path = MODEL_PATH):  # noqa: ANN201
    """Load pre-saved pkl; fall back to fitting fresh if unavailable."""
    from app.v2.e5_xgboost import MultilingualE5Classifier  # noqa: PLC0415

    if path.exists():
        try:
            clf = MultilingualE5Classifier.load(path)
            if clf.is_fitted:
                logger.info("Loaded pre-fitted MultilingualE5Classifier from %s", path)
                return clf
            logger.warning("Pkl at %s is not fitted — refitting", path)
        except Exception:
            logger.exception("Failed to load V2 model from %s — refitting", path)
    else:
        logger.warning("No V2 pkl at %s — fitting at startup (slow path)", path)

    return build_model()


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")
    save_model()
