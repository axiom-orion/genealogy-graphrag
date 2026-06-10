"""Dense embedding pipeline. SentenceTransformer (MiniLM) with on-disk caching so
repeated eval runs are fast and deterministic."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .config import settings

if TYPE_CHECKING:
    from .attest import Attestation


class Embedder:
    def __init__(self, model_name: str | None = None, cache_dir: Path | None = None):
        self.model_name = model_name or settings.embed_model
        self.cache_dir = cache_dir or settings.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None  # lazy: don't pay load cost until first encode

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def attest(self) -> Attestation:
        """Fingerprint the loaded embedder weights (Paramesphere S0). Loads the model."""
        from .attest import attest, named_tensors_from_state_dict
        model = self._load()
        return attest(self.model_name, named_tensors_from_state_dict(model.state_dict()))

    def _key(self, texts: list[str]) -> Path:
        h = hashlib.sha256(
            (self.model_name + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:24]
        return self.cache_dir / f"emb-{h}.npy"

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        """Return L2-normalised float32 embeddings, shape (n, embed_dim)."""
        if use_cache:
            ck = self._key(texts)
            if ck.exists():
                return np.load(ck)
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True,
                            show_progress_bar=False, batch_size=64)
        vecs = np.asarray(vecs, dtype=np.float32)
        if use_cache:
            np.save(self._key(texts), vecs)
        return vecs
