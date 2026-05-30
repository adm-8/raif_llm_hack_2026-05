"""Returns the production classifier. No training or pickling needed."""

from __future__ import annotations

from app.regex_classifier import RegexClassifier


def load_model() -> RegexClassifier:
    return RegexClassifier()
