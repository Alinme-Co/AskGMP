"""
service.py — the knowledge base as an operation, not a script.

    from app.knowledge_base.service import index_knowledge_base, plan

    index_knowledge_base(mode="delta", limit=5)

Two callers want this: scripts/index_knowledge_base.py, and anything later that
wants to trigger indexing without shelling out. Both get the real pipeline —
this module holds no business logic of its own, it only names the entry points
so a caller does not have to reach into `indexer`'s argv handling.
"""

from __future__ import annotations

from pathlib import Path

from app import config
from app.knowledge_base import indexer
from app.knowledge_base.index_state import StateStore
from app.knowledge_base.registry import discover_documents


def index_knowledge_base(
    mode: str = "auto",
    limit: int | None = None,
    workers: int | None = None,
    dry_run: bool = False,
    reextract: bool = False,
    ocr: bool = False,
    registry_path: Path | None = None,
) -> int:
    """
    Run the full pipeline: registry → extraction → chunking → embeddings →
    Qdrant → delta state. Returns the process exit code.
    """
    argv: list[str] = ["--mode", mode]
    if dry_run:
        argv.append("--dry-run")
    if reextract:
        argv.append("--reextract")
    if ocr:
        argv.append("--ocr")
    if limit is not None:
        argv += ["--limit", str(limit)]
    if workers is not None:
        argv += ["--workers", str(workers)]
    if registry_path is not None:
        argv += ["--registry", str(registry_path)]

    return indexer.main(argv)


def plan(limit: int | None = None, registry_path: Path | None = None):
    """What the registry currently selects for ingestion, without indexing it."""
    return discover_documents(
        knowledge_base_dir = config.KNOWLEDGE_BASE_DIR,
        registry_path      = registry_path or config.REGISTRY_PATH,
        limit              = limit,
    )


def state() -> StateStore:
    """The delta state for the regulatory corpus — what is currently indexed."""
    return StateStore(indexer.SOURCE)
