"""
config.py — every tunable for AskGMP, in one place.

Retrieval numbers, model names, paths and limits are read here and imported by
name, so nothing is repeated at a call site. Everything is overridable through
the environment.

This module reads configuration and does nothing else. It creates no clients,
touches no network, and does not modify sys.path — importing it is free, which
is what lets `--dry-run`, `--help` and the unit tests run without OpenAI,
Qdrant or an ONNX runtime installed. Client construction lives in
app/infrastructure/; the Qdrant field names live in app/core/constants.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
APP_DIR  = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

KNOWLEDGE_BASE_DIR = BASE_DIR / "KnowledgeBase"
REGISTRY_PATH      = KNOWLEDGE_BASE_DIR / "_registry" / "documents_registry.csv"

DATA_DIR   = BASE_DIR / "data"
STATE_DIR  = DATA_DIR / ".index_state"
CACHE_DIR  = DATA_DIR / ".cache"
EXTRACT_DIR = CACHE_DIR / "kb_extracted"
UPLOAD_DIR = DATA_DIR / "uploads"


# ── .env ──────────────────────────────────────────────────────────────────────
# Deliberately hand-rolled: one small function is cheaper than a dependency, and
# the indexer has to run in environments where only pandas is present.
def _load_dotenv() -> None:
    for directory in (BASE_DIR, *BASE_DIR.parents[:3]):
        candidate = directory / ".env"
        if not candidate.exists():
            continue
        for line in candidate.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        return


_load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


# ── Qdrant ────────────────────────────────────────────────────────────────────
# COLLECTION_NAME is the name this project's .env already uses;
# QDRANT_COLLECTION is accepted as an alias so either spelling works.
COLLECTION_NAME = (
    os.getenv("COLLECTION_NAME")
    or os.getenv("QDRANT_COLLECTION")
    or "askgmp_knowledge_base"
)

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None
QDRANT_TIMEOUT = _int("QDRANT_TIMEOUT", 120)


# ── OpenAI ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Without an OpenAI key, embeddings go through OpenRouter: same model, same
# 3072 dimensions, so vectors from either route share one space.
EMBED_VIA_OPENROUTER = not OPENAI_API_KEY and bool(os.getenv("OPENROUTER_API_KEY"))
EMBED_MODEL    = os.getenv(
    "EMBED_MODEL",
    "openai/text-embedding-3-large" if EMBED_VIA_OPENROUTER else "text-embedding-3-large",
)
DENSE_DIM      = _int("EMBED_DIM", 3072)

TEMPERATURE = _float("ASKGMP_TEMPERATURE", 0.0)


# ── Chat LLM (OpenRouter) ─────────────────────────────────────────────────────
# Answers go through OpenRouter's OpenAI-compatible endpoint; embeddings stay on
# OpenAI above, because the stored vectors were built with that model.
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
LLM_BASE_URL       = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
CHAT_MODEL         = os.getenv("ASKGMP_CHAT_MODEL", "anthropic/claude-sonnet-5.5")


# ── CAG: the cached core library ──────────────────────────────────────────────
# These documents are sent whole, in a prompt-cached prefix, on every question;
# RAG covers everything else. The set is the regulations most questions land
# on, kept around 90k tokens so the prefix is cheap to read from cache.
CAG_ENABLED = _bool("CAG_ENABLED", True)
CAG_CORE_DOCUMENTS = [
    doc.strip()
    for doc in os.getenv(
        "CAG_CORE_DOCUMENTS",
        ",".join(
            [f"EUDRALEX_EU_GMP_CHAPTER_{n}" for n in range(1, 10)]
            + ["EUDRALEX_01_GLOSSARY", "FDA_21_CFR_PART_210", "FDA_21_CFR_PART_211",
               "ICH_ICH_Q9_R1", "ICH_ICH_Q10",
               # Annex 11: small, and hybrid search loses it to the larger
               # data-integrity guidances on audit-trail questions.
               "EUDRALEX_09_COMPUTERISED_SYSTEMS"]
        ),
    ).split(",")
    if doc.strip()
]
# Cache lifetime: "5m" (write 1.25x input price) or "1h" (write 2x). Use 1h only
# when questions arrive more than 5 minutes apart but within the hour; else 5m.
CAG_CACHE_TTL = os.getenv("CAG_CACHE_TTL", "5m")
CAG_CACHE_CONTROL = (
    {"type": "ephemeral", "ttl": "1h"} if CAG_CACHE_TTL == "1h" else {"type": "ephemeral"}
)
# Sticky routing key: keeps requests on the provider that holds the warm cache.
CAG_SESSION_ID = os.getenv("CAG_SESSION_ID", "askgmp-cag-core")


# ── Chunking ──────────────────────────────────────────────────────────────────
# 1000 tokens is the AskGMP default. The indexer this grew out of used 8000,
# which put the two sentences that actually matched somewhere inside a wall of
# text — the old value is still reachable through the environment for anyone
# who wants to compare.
MAX_CHUNK_TOKENS = _int("REG_MAX_CHUNK_TOKENS", 1000)

# Below this, a chunk is a fragment. Used only to let an over-budget split's
# small tail rejoin its predecessor, never to merge unrelated sections.
MIN_CHUNK_TOKENS = _int("REG_MIN_CHUNK_TOKENS", 40)


# ── Knowledge-base indexing ───────────────────────────────────────────────────
EXTRACT_WORKERS   = _int("KB_EXTRACT_WORKERS", 4)
EMBED_BATCH_SIZE  = _int("KB_EMBED_BATCH_SIZE", 16)
UPLOAD_BATCH_SIZE = _int("KB_UPLOAD_BATCH_SIZE", 32)

# A document that extracts to less than this is almost always a download that
# went wrong rather than a genuinely short guideline — typically a JavaScript
# landing page saved instead of the document it links to.
THIN_DOCUMENT_TOKENS = _int("KB_THIN_DOCUMENT_TOKENS", 200)


# ── Retrieval ─────────────────────────────────────────────────────────────────
# Recall first, precision second: fetch a wide candidate set from each half of
# the hybrid search, fuse, then narrow. The narrowing is what keeps the prompt
# small — see MAX_CONTEXT_CHUNKS.
CANDIDATE_LIMIT = _int("ASKGMP_CANDIDATE_LIMIT", 20)   # after fusion
PREFETCH_LIMIT  = _int("ASKGMP_PREFETCH_LIMIT", 40)    # per vector, before fusion

# How many chunks actually reach the model.
MAX_CONTEXT_CHUNKS = _int("ASKGMP_MAX_CONTEXT_CHUNKS", 8)

# Diversity guard. Without it a single long guideline can occupy every context
# slot, and the answer then cites one document for a question that several
# authorities address — which is exactly the comparison a GMP user wants.
MAX_CHUNKS_PER_DOCUMENT = _int("ASKGMP_MAX_CHUNKS_PER_DOCUMENT", 3)

# Characters of chunk text per source block in the prompt. Chunks are ~1000
# tokens, so this is generous enough to keep whole passages intact.
MAX_CHUNK_CHARS = _int("ASKGMP_MAX_CHUNK_CHARS", 4000)

# Below this fused score a result is treated as noise rather than evidence.
# RRF scores are small and rank-based, so this is deliberately permissive; the
# real "is there an answer here" judgement is the model's, reported through
# has_sufficient_evidence.
MIN_SCORE = _float("ASKGMP_MIN_SCORE", 0.0)

# A question that retrieves nothing above this gets the no-answer response
# without an LLM call at all.
MIN_RESULTS_FOR_ANSWER = _int("ASKGMP_MIN_RESULTS_FOR_ANSWER", 1)

# Regulatory state. Superseded and withdrawn guidance is still in the index —
# the registry records status precisely so it can be filtered rather than
# deleted — but answering from it by default would be a real hazard.
EFFECTIVE_ONLY   = _bool("ASKGMP_EFFECTIVE_ONLY", True)
EFFECTIVE_STATUS = os.getenv("ASKGMP_EFFECTIVE_STATUS", "effective")


# ── Uploaded documents ────────────────────────────────────────────────────────
# How many passages from the user's own uploads reach the prompt. Separate from
# MAX_CONTEXT_CHUNKS, and additive to it: when someone asks "is this a
# deviation?" over their SOP, the answer needs both what their procedure says
# and what the regulation requires, so the two budgets must not compete.
MAX_DOCUMENT_CHUNKS = _int("ASKGMP_MAX_DOCUMENT_CHUNKS", 6)

# Chunks sent to the summariser when no question was asked. A cap rather than
# the whole document: a 300-page scan would otherwise blow the context window,
# and the opening of a controlled document is where its purpose and scope live.
MAX_SUMMARY_CHUNKS = _int("ASKGMP_MAX_SUMMARY_CHUNKS", 30)

# OCR uploaded documents by default. Costly, but a scanned SOP is unreadable
# without it, and an unreadable SOP silently answers nothing.
UPLOAD_OCR = _bool("ASKGMP_UPLOAD_OCR", True)

MAX_UPLOAD_BYTES = _int("ASKGMP_MAX_UPLOAD_BYTES", 100 * 1024 * 1024)


# ── Generation ────────────────────────────────────────────────────────────────
NO_ANSWER_TEXT = (
    "I could not find enough information in the available AskGMP documents "
    "to answer this confidently."
)


# ── API / frontend ────────────────────────────────────────────────────────────
API_HOST = os.getenv("ASKGMP_API_HOST", "127.0.0.1")
API_PORT = _int("ASKGMP_API_PORT", 8000)
API_URL  = os.getenv("ASKGMP_API_URL", f"http://{API_HOST}:{API_PORT}")
