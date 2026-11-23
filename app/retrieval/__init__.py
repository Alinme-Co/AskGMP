"""
app.retrieval — finding the passages that bear on a question.

    from app.retrieval.retriever import Retriever
    from app.retrieval.schemas import RetrievedEvidence

Independent of chat: this package knows how to rank passages and nothing about
prompts, citations or source tables. It depends on app.infrastructure for the
clients and the vector builders, and on nothing else in the project.
"""

from app.retrieval.schemas import RetrievedChunk, RetrievedEvidence

__all__ = ["RetrievedChunk", "RetrievedEvidence"]
