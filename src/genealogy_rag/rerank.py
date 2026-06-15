"""Optional cross-encoder reranker. Scores (query, passage) jointly -- the single
highest-precision stage. Degrades to a no-op with a warning if the model is absent."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from .attest import Attestation


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None, revision: str | None = None):
        self.model_name = model_name or settings.rerank_model
        self.revision = revision if revision is not None else settings.rerank_revision
        self._model = None
        self.available = True

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                # Load at the pinned HF revision when configured (S0, §10), else latest.
                kw = {"revision": self.revision} if self.revision else {}
                self._model = CrossEncoder(self.model_name, **kw)
            except Exception as e:  # pragma: no cover
                warnings.warn(f"reranker unavailable ({e}); skipping rerank stage", stacklevel=2)
                self.available = False
        return self._model

    def attest(self) -> Attestation | None:
        """Fingerprint the loaded reranker weights (Paramesphere S0). None if unavailable.

        Lands the loaded-state fingerprint with the pinned HF revision and an at-rest
        ``artifact_sha256`` (from local weight files when present; ``null`` otherwise) — the
        §10 S0 record. Degrades to ``None`` if the reranker can't load, never crashing the
        pipeline. Same-model tamper/swap on a fixed weight set — not cross-model identity,
        not quantization-robust.
        """
        from .attest import (
            artifact_paths_from_dir,
            attest,
            named_tensors_from_state_dict,
            resolve_model_dir,
        )
        model = self._load()
        if not self.available or model is None:
            return None
        inner = getattr(model, "model", model)   # CrossEncoder wraps the HF model in `.model`
        return attest(
            self.model_name,
            named_tensors_from_state_dict(inner.state_dict()),
            revision=self.revision,
            artifact_paths=artifact_paths_from_dir(resolve_model_dir(inner)),
        )

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        model = self._load()
        if not self.available or model is None:
            return [0.0] * len(passages)
        scores = model.predict([(query, p) for p in passages],
                              show_progress_bar=False)
        return [float(s) for s in scores]
