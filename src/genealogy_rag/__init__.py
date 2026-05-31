"""genealogy_rag: a hybrid (lexical + dense + graph) retrieval pipeline for
provenance-grounded genealogical question answering."""
from .config import Settings, settings
from .corpus import Document, load_documents, load_questions
from .pipeline import GenealogyRAG, RetrievalConfig

__all__ = [
    "Settings", "settings", "Document", "load_documents", "load_questions",
    "GenealogyRAG", "RetrievalConfig",
]
__version__ = "0.1.0"
