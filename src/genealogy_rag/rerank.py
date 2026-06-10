"""Optional cross-encoder reranker. Scores (query, passage) jointly -- the single
highest-precision stage. Degrades to a no-op with a warning if the model is absent."""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from .config import settings

if TYPE_CHECKING:
    from .attest import Attestation


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model
        self._model = None
        self.available = True

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name)
            except Exception as e:  # pragma: no cover
                warnings.warn(f"reranker unavailable ({e}); skipping rerank stage", stacklevel=2)
                self.available = False
        return self._model

    def attest(self) -> Attestation | None:
        """Fingerprint the loaded reranker weights (Paramesphere S0). None if unavailable."""
        from .attest import attest, named_tensors_from_state_dict
        model = self._load()
        if not self.available or model is None:
            return None
        inner = getattr(model, "model", model)   # CrossEncoder wraps the HF model in `.model`
        return attest(self.model_name, named_tensors_from_state_dict(inner.state_dict()))

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        model = self._load()
        if not self.available or model is None:
            return [0.0] * len(passages)
        scores = model.predict([(query, p) for p in passages],
                              show_progress_bar=False)
        return [float(s) for s in scores]
