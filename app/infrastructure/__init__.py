"""
app.infrastructure — the external services AskGMP talks to.

Qdrant and OpenAI, and the vector building that has to agree between them.
Both the knowledge-base indexer and the retriever depend on this package;
neither depends on the other.
"""
