"""
app.knowledge_base — the persistent regulatory corpus.

    KnowledgeBase/_registry/documents_registry.csv   the source of truth
      → registry.py     what to index, and every metadata field
        → indexer.py    extract (app.documents) → chunk → embed → Qdrant
          → index_state.py   fingerprints, so a rerun touches only what moved

Depends on app.documents for extraction and app.infrastructure for Qdrant and
OpenAI. Nothing here depends on app.retrieval, and app.retrieval does not
depend on this: the two meet at the collection contract in
app/core/constants.py and app/infrastructure/qdrant.py, not at each other.
"""
