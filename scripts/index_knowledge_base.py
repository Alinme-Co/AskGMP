"""
index_knowledge_base.py — bring the regulatory index up to date.

    python -m scripts.index_knowledge_base                # auto: init, then delta
    python -m scripts.index_knowledge_base --list         # the plan only
    python -m scripts.index_knowledge_base --dry-run      # no OpenAI or Qdrant calls
    python -m scripts.index_knowledge_base --limit 5      # small trial
    python -m scripts.index_knowledge_base --mode init    # force a rebuild

Safe to run on a schedule: the indexer decides for itself whether anything
changed, and on an unchanged corpus it makes no OpenAI calls at all.

A wrapper and nothing more. The pipeline lives in app/knowledge_base/indexer.py,
so there is one implementation to fix when it is wrong — not a copy here that
drifts.
"""

from __future__ import annotations

import sys

from app.knowledge_base.indexer import main

if __name__ == "__main__":
    sys.exit(main())
