"""Load / train the V3 classifier (E5-only, no LaBSE — faster inference).

Usage
-----
* build time / local training:
    ``python -m app.v3.model_loader``
    → trains on all available data, writes artifacts/e5_v3_model.pt

* runtime (inside FastAPI lifespan):
    ``from app.v3.model_loader import load_model``
    → loads pre-saved .pt; falls back to fitting if the file is missing.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib

logger = logging.getLogger("uvicorn.error")

_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_PATH = pathlib.Path(
    os.getenv("E5_V3_MODEL_PATH", str(_ROOT / "artifacts" / "e5_v3_model.pt"))
)

_REAL_DATA_PATH = _ROOT / "data" / "requests.json"
_SYNTHETIC_DATA_PATH = _ROOT / "data" / "train" / "synthetic_train.json"
_ARTIFACT_TRAIN_PATH = _ROOT / "artifacts" / "train.json"


def _load_conversations():  # noqa: ANN201
    from app.models import CLEAR_CATEGORY, Conversation  # noqa: PLC0415

    convs = []

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

    for path in (_SYNTHETIC_DATA_PATH, _ARTIFACT_TRAIN_PATH):
        if path.exists():
            items = json.loads(path.read_text(encoding="utf-8"))
            batch = [Conversation.from_dict(d) for d in items]
            convs.extend(batch)
            logger.info("Loaded %d conversations from %s", len(batch), path)

    return convs


def build_model():  # noqa: ANN201
    from app.v3.best_version import MultilingualE5Classifier  # noqa: PLC0415

    convs = _load_conversations()
    clf = MultilingualE5Classifier()
    clf.fit(convs)
    logger.info("V3 MultilingualE5Classifier fitted on %d conversations", len(convs))
    return clf


def save_model(path: pathlib.Path = MODEL_PATH) -> None:
    clf = build_model()
    clf.save(path)


def load_model(path: pathlib.Path = MODEL_PATH):  # noqa: ANN201
    from app.v3.best_version import MultilingualE5Classifier  # noqa: PLC0415

    if path.exists():
        try:
            clf = MultilingualE5Classifier.load(path)
            if clf.is_fitted:
                logger.info("Loaded pre-fitted V3 classifier from %s", path)
                return clf
            logger.warning("V3 model at %s is not fitted — refitting", path)
        except Exception:
            logger.exception("Failed to load V3 model from %s — refitting", path)
    else:
        logger.warning("No V3 model at %s — fitting at startup (slow path)", path)

    return build_model()


if __name__ == "__main__":
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")
    save_model()
