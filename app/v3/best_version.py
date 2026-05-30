"""MultilingualE5Classifier — intfloat/multilingual-e5-small + PyTorch MLP.

Embedding strategy  ← best validated configuration (F1 macro 0.782)
------------------
Each message is embedded separately:

    h1 = enc("query: " + user_turn_1)
    h2 = enc("query: " + bot_turn_1)
    h3 = enc("query: " + user_turn_2)
    ...

All turn vectors are aggregated via three pooling heads concatenated together:

1. **Mean pooling** — unweighted average of all turn vectors (384-dim).
2. **Attention pooling** — softmax(v · h_i)-weighted average where v is fitted
   as the normalised mean of all training turn vectors (384-dim).
3. **Last-K pooling** — mean of the last ``last_k`` turns (384-dim). Default K=6.

Total E5 contribution: 384 × 3 = 1152-dim.

TF-IDF features
---------------
word(1,2) + char_wb(3,5) TF-IDF on the full role-prefixed dialogue, reduced to
``tfidf_components`` dimensions via TruncatedSVD (LSA) → default 100-dim.

Classification head
-------------------
Final feature vector: [mean | attn | lastK | TF-IDF] = 1252-dim ->

    Linear(1252, 256) -> ReLU -> Dropout(0.2) -> Linear(256, num_classes)

Trained with CrossEntropyLoss + Adam.
Optional early stopping on a stratified validation split (disabled by default).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import re
import sys
import time
from collections import Counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder

from app.models import CLEAR_CATEGORY, Conversation

# =============================================================================
# Configuration
# =============================================================================

@dataclasses.dataclass
class E5ClassifierConfig:
    clear_margin: float = 0.20
    last_k: int = 6
    tfidf_components: int = 100
    epochs: int = 50
    patience: int = 0          # 0 = disabled; set >0 to enable early stopping
    val_fraction: float = 0.15
    batch_size: int = 32
    learning_rate: float = 1e-3
    torch_device: str | None = None
    encoder_device: str | None = None
    # Message-level marker filtering. One of:
    #   "none"          — use every turn (baseline, 1252-dim)
    #   "weighted_head" — keep every turn, add a marker-weighted pooling head (+384)
    #   "hard_filter"   — drop non-marker turns before embedding, add 2 scalar features
    message_filter: str = "none"
    marker_top_n: int = 40     # markers kept per category (by log-odds z-score)
    marker_prior: float = 0.01  # Dirichlet smoothing per word (flat prior)


DEFAULT_CONFIG = E5ClassifierConfig()

E5_MODEL = "intfloat/multilingual-e5-small"
E5_DIM = 384
E5_PREFIX = "query: "

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # letters only (latin + cyrillic), no digits


def _tokenize(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if len(t) >= 2]


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Main Classifier
# =============================================================================

class MultilingualE5Classifier:
    """Per-message E5 embeddings (mean + attention + last-K pooling) + TF-IDF + MLP."""

    def __init__(self, config: E5ClassifierConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        self._torch_device = self._get_torch_device()
        self._encoder: Any = None
        self._attn_v: np.ndarray | None = None
        self._tfidf_union: FeatureUnion | None = None
        self._svd: TruncatedSVD | None = None
        self._model: Classifier | None = None
        self._le: LabelEncoder | None = None
        self._marker_weights: dict[str, float] = {}

    def _get_torch_device(self) -> torch.device:
        if self.config.torch_device:
            return torch.device(self.config.torch_device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # -------------------------------------------------------------------------
    # Encoder
    # -------------------------------------------------------------------------
    def _ensure_encoder(self) -> None:
        if self._encoder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install with: uv sync --group v2"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.config.encoder_device:
            kwargs["device"] = self.config.encoder_device
        self._encoder = SentenceTransformer(E5_MODEL, **kwargs)

    def _embed_turns(self, messages: list) -> np.ndarray:
        """Embed a list of messages separately -> (n_turns, 384)."""
        self._ensure_encoder()
        texts = [E5_PREFIX + m.content for m in messages]
        return self._encoder.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True,
        )

    def _embed_messages(self, conv: Conversation) -> np.ndarray:
        """Embed every turn of a conversation -> (n_turns, 384)."""
        return self._embed_turns(conv.messages)

    # -------------------------------------------------------------------------
    # Marker lexicon (weighted log-odds, Monroe et al. "Fightin' Words")
    # -------------------------------------------------------------------------
    def _fit_markers(self, conversations: list[Conversation]) -> None:
        """Build a per-word marker weight = max log-odds z-score across red-flag
        categories vs. the rest. Fitted on the training split only (no leakage)."""
        self._marker_weights = {}
        if self.config.message_filter == "none":
            return

        per_cat: dict[str, Counter] = {}
        total: Counter = Counter()
        for c in conversations:
            toks: list[str] = []
            for m in c.messages:
                toks.extend(_tokenize(m.content))
            per_cat.setdefault(c.category, Counter()).update(toks)
            total.update(toks)

        vocab = list(total)
        a = self.config.marker_prior
        a0 = a * len(vocab)
        grand = sum(total.values())

        for cat, t_counts in per_cat.items():
            if cat == CLEAR_CATEGORY:
                continue                      # markers describe red-flags, not "clear"
            n_t = sum(t_counts.values())
            n_r = grand - n_t
            scored: list[tuple[str, float]] = []
            for w in vocab:
                y_t = t_counts.get(w, 0)
                y_r = total[w] - y_t
                # log-odds-ratio with informative Dirichlet prior + z-score
                delta = (
                    np.log((y_t + a) / (n_t + a0 - y_t - a))
                    - np.log((y_r + a) / (n_r + a0 - y_r - a))
                )
                var = 1.0 / (y_t + a) + 1.0 / (y_r + a)
                z = delta / np.sqrt(var)
                if z > 0:
                    scored.append((w, float(z)))
            scored.sort(key=lambda kv: kv[1], reverse=True)
            for w, z in scored[: self.config.marker_top_n]:
                if z > self._marker_weights.get(w, 0.0):
                    self._marker_weights[w] = z

    def _marker_score(self, msg) -> float:
        """Sum of marker z-weights for the terms in a single message."""
        if not self._marker_weights:
            return 0.0
        return sum(self._marker_weights.get(t, 0.0) for t in _tokenize(msg.content))

    def _select_messages(self, conv: Conversation) -> tuple[list, int, int]:
        """Hard filter: keep marker-bearing turns + the last user turn.
        Returns (kept_messages, n_removed, n_total). Always keeps >= 1 turn."""
        msgs = list(conv.messages)
        n_total = len(msgs)
        if n_total == 0:
            return msgs, 0, 0
        last_user = max((i for i, m in enumerate(msgs) if m.role == "user"), default=-1)
        kept = [m for i, m in enumerate(msgs) if self._marker_score(m) > 0 or i == last_user]
        if not kept:
            kept = [msgs[-1]]                 # never emit an empty conversation
        return kept, n_total - len(kept), n_total

    # -------------------------------------------------------------------------
    # Pooling heads
    # -------------------------------------------------------------------------
    @staticmethod
    def _mean_pool(vecs: np.ndarray) -> np.ndarray:
        return vecs.mean(axis=0)

    @staticmethod
    def _attn_pool(vecs: np.ndarray, attn_v: np.ndarray | None) -> np.ndarray:
        if attn_v is None or len(vecs) <= 1:
            return vecs.mean(axis=0)
        scores = vecs @ attn_v
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        return (vecs * weights[:, None]).sum(axis=0)

    def _lastk_pool(self, vecs: np.ndarray) -> np.ndarray:
        return vecs[-self.config.last_k:].mean(axis=0)

    @staticmethod
    def _marker_pool(vecs: np.ndarray, scores: list[float]) -> np.ndarray:
        """Marker-weighted mean; falls back to plain mean if no turn scores."""
        s = np.asarray(scores, dtype=np.float64)
        if len(vecs) == 0 or s.sum() <= 0:
            return vecs.mean(axis=0)
        w = s / s.sum()
        return (vecs * w[:, None]).sum(axis=0)

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    def _fit_attn_v(self, conversations: list[Conversation]) -> None:
        all_vecs = np.vstack([self._embed_messages(c) for c in conversations])
        v = all_vecs.mean(axis=0)
        norm = np.linalg.norm(v)
        self._attn_v = v / norm if norm > 0 else v

    def _conv_e5_row(self, conv: Conversation) -> np.ndarray:
        """Per-conversation E5 feature row, honouring config.message_filter."""
        mode = self.config.message_filter
        if mode == "hard_filter":
            msgs, n_removed, n_total = self._select_messages(conv)
        else:
            msgs, n_removed, n_total = list(conv.messages), 0, len(conv.messages)

        vecs = self._embed_turns(msgs)
        heads = [
            self._mean_pool(vecs),
            self._attn_pool(vecs, self._attn_v),
            self._lastk_pool(vecs),
        ]
        if mode == "weighted_head":
            heads.append(self._marker_pool(vecs, [self._marker_score(m) for m in msgs]))
        elif mode == "hard_filter":
            removed_frac = n_removed / n_total if n_total else 0.0
            heads.append(np.array([removed_frac, float(len(msgs))], dtype=np.float32))
        return np.concatenate(heads)

    def _e5_features(self, conversations: list[Conversation]) -> np.ndarray:
        """Per-message pooling -> (N, D) where D depends on config.message_filter."""
        return np.vstack([self._conv_e5_row(c) for c in conversations])

    @staticmethod
    def _conv_text(conv: Conversation) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in conv.messages)

    def _fit_tfidf(self, conversations: list[Conversation]) -> np.ndarray:
        texts = [self._conv_text(c) for c in conversations]
        self._tfidf_union = FeatureUnion([
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30_000, sublinear_tf=True)),
        ])
        sparse = self._tfidf_union.fit_transform(texts)
        n_components = min(self.config.tfidf_components, sparse.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        return self._svd.fit_transform(sparse)

    def _transform_tfidf(self, conversations: list[Conversation]) -> np.ndarray:
        texts = [self._conv_text(c) for c in conversations]
        return self._svd.transform(self._tfidf_union.transform(texts))

    def _build_features(self, conversations: list[Conversation], *, fit: bool) -> np.ndarray:
        """[mean | attn | lastK | (marker head / filter stats) | TF-IDF] -> (N, D).

        D = 1252 ("none"), 1636 ("weighted_head"), or 1254 ("hard_filter").
        TF-IDF always sees the full dialogue text regardless of message_filter.
        """
        e5 = self._e5_features(conversations)
        tfidf = self._fit_tfidf(conversations) if fit else self._transform_tfidf(conversations)
        return np.hstack([e5, tfidf])

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def fit(self, conversations: list[Conversation]) -> "MultilingualE5Classifier":
        """Fit the classifier. Set config.patience > 0 to enable early stopping."""
        self._ensure_encoder()
        self._fit_markers(conversations)
        self._fit_attn_v(conversations)

        categories = sorted({c.category for c in conversations})
        self._le = LabelEncoder().fit(categories)
        y_np = self._le.transform([c.category for c in conversations])
        x_np = self._build_features(conversations, fit=True)

        feat_dim, num_classes = x_np.shape[1], len(categories)

        use_early_stop = self.config.patience > 0 and self.config.val_fraction > 0
        if use_early_stop:
            tr_idx, val_idx = train_test_split(
                np.arange(len(x_np)), test_size=self.config.val_fraction,
                stratify=y_np, random_state=42,
            )
            x_tr, y_tr = x_np[tr_idx], y_np[tr_idx]
            x_val = torch.tensor(x_np[val_idx], dtype=torch.float32).to(self._torch_device)
            y_val = torch.tensor(y_np[val_idx], dtype=torch.long).to(self._torch_device)
        else:
            x_tr, y_tr = x_np, y_np

        x_t = torch.tensor(x_tr, dtype=torch.float32).to(self._torch_device)
        y_t = torch.tensor(y_tr, dtype=torch.long).to(self._torch_device)

        self._model = Classifier(feat_dim, num_classes).to(self._torch_device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_val_loss, best_state, no_improve = float("inf"), None, 0

        for epoch in range(1, self.config.epochs + 1):
            self._model.train()
            perm = torch.randperm(len(x_t), device=self._torch_device)
            for start in range(0, len(x_t), self.config.batch_size):
                idx = perm[start: start + self.config.batch_size]
                optimizer.zero_grad()
                criterion(self._model(x_t[idx]), y_t[idx]).backward()
                optimizer.step()

            if use_early_stop:
                self._model.eval()
                with torch.no_grad():
                    val_loss = criterion(self._model(x_val), y_val).item()
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss, no_improve = val_loss, 0
                    best_state = copy.deepcopy(self._model.state_dict())
                else:
                    no_improve += 1
                    if no_improve >= self.config.patience:
                        print(f"    early stop at epoch {epoch}  val_loss={best_val_loss:.4f}")
                        break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    def predict(self, conv: Conversation) -> tuple[str, float]:
        if self._model is None or self._le is None:
            raise RuntimeError("Call fit() first.")
        x_t = torch.tensor(
            self._build_features([conv], fit=False), dtype=torch.float32,
        ).to(self._torch_device)
        with torch.no_grad():
            proba = torch.softmax(self._model(x_t), dim=-1).cpu().numpy()[0]

        best_idx = int(proba.argmax())
        best_cat = str(self._le.inverse_transform([best_idx])[0])
        best_conf = float(proba[best_idx])

        if self.config.clear_margin > 0 and best_cat != CLEAR_CATEGORY:
            try:
                clear_idx = int(self._le.transform([CLEAR_CATEGORY])[0])
                clear_conf = float(proba[clear_idx])
                if best_conf - clear_conf < self.config.clear_margin:
                    return CLEAR_CATEGORY, clear_conf
            except ValueError:
                pass
        return best_cat, best_conf

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, path: str | pathlib.Path) -> None:
        """Persist the fitted classifier to a single file.

        Stores the MLP weights, fitted attention vector, TF-IDF union, SVD,
        label encoder and config. The E5 encoder is *not* saved — it is
        re-loaded lazily from the hub on first use after ``load()``.
        """
        if self._model is None or self._le is None:
            raise RuntimeError("Call fit() before save().")
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": dataclasses.asdict(self.config),
                "attn_v": self._attn_v,
                "tfidf_union": self._tfidf_union,
                "svd": self._svd,
                "label_encoder": self._le,
                "marker_weights": self._marker_weights,
                "model_state": self._model.state_dict(),
                "feat_dim": self._model.net[0].in_features,
                "num_classes": self._model.net[-1].out_features,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: str | pathlib.Path, *, config: E5ClassifierConfig | None = None,
    ) -> "MultilingualE5Classifier":
        """Load a classifier saved with :meth:`save`.

        Pass ``config`` to override runtime knobs (e.g. ``torch_device``);
        otherwise the config persisted at save time is restored.
        """
        path = pathlib.Path(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        clf = cls(config or E5ClassifierConfig(**blob["config"]))
        clf._attn_v = blob["attn_v"]
        clf._tfidf_union = blob["tfidf_union"]
        clf._svd = blob["svd"]
        clf._le = blob["label_encoder"]
        clf._marker_weights = blob.get("marker_weights", {})
        model = Classifier(blob["feat_dim"], blob["num_classes"])
        model.load_state_dict(blob["model_state"])
        clf._model = model.to(clf._torch_device).eval()
        return clf


# =============================================================================
# Evaluation (when run as script)
# =============================================================================

def _cross_validate(
    all_convs: list[Conversation],
    all_labels: list[str],
    syn_convs: list[Conversation],
    config: E5ClassifierConfig,
    *,
    n_splits: int = 5,
) -> list[str]:
    """Stratified K-fold out-of-fold predictions for a given config."""
    preds: list[str] = [""] * len(all_convs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(all_convs, all_labels):
        train_set = [all_convs[i] for i in train_idx] + syn_convs
        clf = MultilingualE5Classifier(config)
        clf.fit(train_set)
        for i in test_idx:
            preds[i], _ = clf.predict(all_convs[i])
    return preds


if __name__ == "__main__":
    ROOT = pathlib.Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from sklearn.metrics import confusion_matrix, f1_score  # noqa: PLC0415

    data = json.loads((ROOT / "data" / "requests.json").read_text(encoding="utf-8"))
    all_convs: list[Conversation] = []
    all_labels: list[str] = []
    for req in data["requests"]:
        label = req["category"]
        conv = Conversation.from_dict({
            "session_id": req["body"]["session_id"],
            "messages": req["body"]["messages"],
            "expected_red_flags": [] if label == CLEAR_CATEGORY else [{"category": label}],
        })
        all_convs.append(conv)
        all_labels.append(label)

    syn_path = ROOT / "data" / "train" / "synthetic_train.json"
    syn_convs: list[Conversation] = []
    if syn_path.exists():
        syn_convs = [Conversation.from_dict(item) for item in json.loads(syn_path.read_text(encoding="utf-8"))]
        print(f"Loaded {len(syn_convs)} synthetic samples.")

    n_splits = 5
    modes = ["none", "weighted_head", "hard_filter"]
    print(
        f"A/B: {n_splits}-fold CV on {len(all_convs)} real + {len(syn_convs)} synthetic "
        f"across message_filter ∈ {modes}\n"
    )

    results: dict[str, dict] = {}
    for mode in modes:
        cfg = E5ClassifierConfig(message_filter=mode)
        t0 = time.perf_counter()
        preds = _cross_validate(all_convs, all_labels, syn_convs, cfg, n_splits=n_splits)
        dt = time.perf_counter() - t0
        acc = sum(p == g for p, g in zip(preds, all_labels)) / len(all_labels)
        f1m = f1_score(all_labels, preds, average="macro")
        f1w = f1_score(all_labels, preds, average="weighted")
        results[mode] = {"preds": preds, "acc": acc, "f1_macro": f1m, "f1_weighted": f1w, "time": dt}
        print(f"  {mode:<14} acc={acc:.3f}  F1_macro={f1m:.4f}  F1_weighted={f1w:.4f}  ({dt:.0f}s)")

    print("\n=== A/B summary (sorted by F1 macro) ===")
    print(f"{'message_filter':<16}{'accuracy':>10}{'F1_macro':>12}{'F1_weighted':>14}{'time(s)':>10}")
    print("-" * 62)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["f1_macro"], reverse=True)
    baseline_f1 = results["none"]["f1_macro"]
    for mode, r in ranked:
        delta = r["f1_macro"] - baseline_f1
        tag = "  (baseline)" if mode == "none" else f"  ({delta:+.4f} vs none)"
        print(f"{mode:<16}{r['acc']:>10.3f}{r['f1_macro']:>12.4f}{r['f1_weighted']:>14.4f}{r['time']:>10.0f}{tag}")

    best_mode = ranked[0][0]
    preds = results[best_mode]["preds"]
    print(f"\n=== Detailed report for best mode: {best_mode!r} ===\n")

    cats = sorted(set(all_labels) | set(preds))
    cm = confusion_matrix(all_labels, preds, labels=cats)
    col_w = max(len(c) for c in cats) + 2
    header = f"{'':>{col_w}}" + "".join(f"{c:>{col_w}}" for c in cats)
    print("Confusion matrix (rows=true, cols=predicted):")
    print(header)
    print("-" * len(header))
    for row_cat, row in zip(cats, cm):
        print(f"{row_cat:>{col_w}}" + "".join(f"{v:>{col_w}}" for v in row))

    print("\nPer-class precision / recall / F1:")
    print(f"{'category':>{col_w}}  {'prec':>6}  {'rec':>6}  {'f1':>6}  {'support':>8}")
    print("-" * (col_w + 36))
    for i, cat in enumerate(cats):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        print(f"{cat:>{col_w}}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}  {int(cm[i, :].sum()):>8}")
