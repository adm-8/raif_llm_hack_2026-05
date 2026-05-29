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
Final feature vector: [E5 mean|attn|lastK | LaBSE user|bot|interaction | TF-IDF]
= 1152 + 5377 + 100 = 6629-dim ->

    Linear(6629, 256) -> ReLU -> Dropout(0.2) -> Linear(256, num_classes)

Trained with CrossEntropyLoss + Adam.
Optional early stopping on a stratified validation split (disabled by default).
"""

from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import pickle
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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


DEFAULT_CONFIG = E5ClassifierConfig()

E5_MODEL = "intfloat/multilingual-e5-small"
E5_DIM = 384
E5_PREFIX = "query: "
LABSE_MODEL = "sentence-transformers/LaBSE"
LABSE_DIM = 768


class Classifier(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
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
        self._labse_encoder: Any = None   # second encoder — LaBSE
        self._attn_v: np.ndarray | None = None
        self._tfidf_union: FeatureUnion | None = None
        self._svd: TruncatedSVD | None = None
        self._model: Classifier | None = None
        self._le: LabelEncoder | None = None

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

    def _embed_messages(self, conv: Conversation) -> np.ndarray:
        """Embed each turn separately -> (n_turns, 384)."""
        self._ensure_encoder()
        texts = [E5_PREFIX + m.content for m in conv.messages]
        return self._encoder.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True,
        )

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

    def _pool(self, vecs: np.ndarray) -> np.ndarray:
        """[mean | attn | lastK] -> (1152,)."""
        return np.concatenate([
            self._mean_pool(vecs),
            self._attn_pool(vecs, self._attn_v),
            self._lastk_pool(vecs),
        ])

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    def _fit_attn_v(self, conversations: list[Conversation]) -> None:
        all_vecs = np.vstack([self._embed_messages(c) for c in conversations])
        v = all_vecs.mean(axis=0)
        norm = np.linalg.norm(v)
        self._attn_v = v / norm if norm > 0 else v

    def _e5_features(self, conversations: list[Conversation]) -> np.ndarray:
        """Per-message pooling -> (N, 1152)."""
        return np.vstack([self._pool(self._embed_messages(c)) for c in conversations])

    # -------------------------------------------------------------------------
    # LaBSE encoder + pooling
    # -------------------------------------------------------------------------
    def _ensure_labse(self) -> None:
        if self._labse_encoder is not None:
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
        self._labse_encoder = SentenceTransformer(LABSE_MODEL, **kwargs)

    def _embed_role_labse(self, conv: Conversation, user: bool) -> np.ndarray:
        """Embed user or bot turns with LaBSE -> (n, 768). Zeros if no turns."""
        self._ensure_labse()
        msgs = [m for m in conv.messages if (m.role == "user") == user]
        if not msgs:
            return np.zeros((1, LABSE_DIM), dtype=np.float32)
        return self._labse_encoder.encode(
            [m.content for m in msgs],
            batch_size=32, show_progress_bar=False, normalize_embeddings=True,
        )

    @staticmethod
    def _pool_labse(vecs: np.ndarray) -> np.ndarray:
        """mean + last + max pooling -> (768*3,) = (2304,)."""
        if len(vecs) == 0:
            return np.zeros(LABSE_DIM * 3, dtype=np.float32)
        return np.concatenate([
            vecs.mean(axis=0),
            vecs[-1],
            vecs.max(axis=0),
        ])

    def _labse_features(self, conversations: list[Conversation]) -> np.ndarray:
        """LaBSE user/bot pools + user-bot interaction -> (N, 2304+2304+769) = (N, 5377)."""
        rows = []
        for conv in conversations:
            u_vecs = self._embed_role_labse(conv, user=True)
            b_vecs = self._embed_role_labse(conv, user=False)

            u_pooled = self._pool_labse(u_vecs)
            b_pooled = self._pool_labse(b_vecs)

            # Interaction: last-user-turn minus mean-bot + cosine similarity
            u_last = u_vecs[-1]
            b_mean = b_vecs.mean(axis=0)
            diff = u_last - b_mean
            cos_sim = np.array([
                cosine_similarity(u_last.reshape(1, -1), b_mean.reshape(1, -1))[0, 0]
            ])
            interaction = np.concatenate([diff, cos_sim])  # 769-dim

            rows.append(np.concatenate([u_pooled, b_pooled, interaction]))
        return np.vstack(rows)

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
        """[E5 mean|attn|lastK | LaBSE user|bot|interaction | TF-IDF] -> (N, 6629)."""
        e5 = self._e5_features(conversations)
        labse = self._labse_features(conversations)
        tfidf = self._fit_tfidf(conversations) if fit else self._transform_tfidf(conversations)
        return np.hstack([e5, labse, tfidf])

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def fit(self, conversations: list[Conversation]) -> "MultilingualE5Classifier":
        """Fit the classifier. Set config.patience > 0 to enable early stopping."""
        self._ensure_encoder()
        self._ensure_labse()
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

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------
    def save(self, path: str | pathlib.Path) -> None:
        """Save the fitted classifier to *path* (a .pkl file).

        The pre-trained sentence-transformer weights (E5 and LaBSE) are NOT
        stored — they are re-downloaded / re-loaded from cache on the next
        ``load()``.  Everything else (sklearn pipelines, MLP weights,
        label encoder, attention vector, config) is serialised.
        """
        if self._model is None or self._le is None:
            raise RuntimeError("Call fit() before save().")

        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save PyTorch model architecture params + state dict separately so
        # they survive across torch versions better than pickling nn.Module.
        model_payload = {
            "dim": self._model.net[0].in_features,
            "num_classes": self._model.net[-1].out_features,
            "state_dict": {k: v.cpu() for k, v in self._model.state_dict().items()},
        }

        payload = {
            "config": self.config,
            "attn_v": self._attn_v,
            "tfidf_union": self._tfidf_union,
            "svd": self._svd,
            "le": self._le,
            "model": model_payload,
        }

        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved classifier to {path}")

    @classmethod
    def load(cls, path: str | pathlib.Path, config: E5ClassifierConfig | None = None) -> "MultilingualE5Classifier":
        """Load a previously saved classifier from *path*.

        The sentence-transformer encoders (E5, LaBSE) are NOT stored in the
        file — they will be loaded from HuggingFace cache on first use.
        """
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(config or payload["config"])
        obj._attn_v = payload["attn_v"]
        obj._tfidf_union = payload["tfidf_union"]
        obj._svd = payload["svd"]
        obj._le = payload["le"]

        mp = payload["model"]
        obj._model = Classifier(mp["dim"], mp["num_classes"]).to(obj._torch_device)
        obj._model.load_state_dict({k: v.to(obj._torch_device) for k, v in mp["state_dict"].items()})
        obj._model.eval()

        print(f"Loaded classifier from {path}")
        return obj

    @property
    def is_fitted(self) -> bool:
        return self._model is not None


# =============================================================================
# Evaluation (when run as script)
# =============================================================================

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
    print(f"Running {n_splits}-fold CV on {len(all_convs)} real + {len(syn_convs)} synthetic ...\n")

    preds: list[str] = [""] * len(all_convs)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    t0 = time.perf_counter()

    for fold, (train_idx, test_idx) in enumerate(skf.split(all_convs, all_labels), 1):
        train_set = [all_convs[i] for i in train_idx] + syn_convs
        test_set = [all_convs[i] for i in test_idx]
        clf = MultilingualE5Classifier()
        clf.fit(train_set)
        for i, conv in zip(test_idx, test_set):
            preds[i], _ = clf.predict(conv)
        fold_acc = sum(preds[i] == all_labels[i] for i in test_idx) / len(test_idx)
        print(f"  Fold {fold}/{n_splits}  acc={fold_acc:.3f}  ({time.perf_counter()-t0:.0f}s)")

    total_time = time.perf_counter() - t0
    cats = sorted(set(all_labels) | set(preds))
    correct = sum(p == g for p, g in zip(preds, all_labels))
    accuracy = correct / len(all_labels)
    f1_macro = f1_score(all_labels, preds, average="macro")
    f1_weighted = f1_score(all_labels, preds, average="weighted")

    print(f"\nAccuracy:    {correct}/{len(all_labels)} = {accuracy:.3f}  ({total_time:.0f}s total)")
    print(f"F1 macro:    {f1_macro:.4f}")
    print(f"F1 weighted: {f1_weighted:.4f}\n")

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
