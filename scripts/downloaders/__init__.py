"""
scripts.downloaders — one module per issuing authority.

    python -m scripts.downloaders.eudralex
    python -m scripts.downloaders.ich
    ...

Each writes into KnowledgeBase/<authority>/ and is safe to re-run: a document
already on disk is not fetched again. After a download, refresh the registry
with `python -m scripts.build_document_registry` so the new files become
indexable.
"""
