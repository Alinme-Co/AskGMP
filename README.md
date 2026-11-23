# AskGMP · CAG-RAG

**Ask a GMP question, get an answer you can audit.** Grounded Q&A over ~90
pharmaceutical GMP regulations — EudraLex, ICH, FDA, EMA, MHRA, PIC/S and WHO —
plus your own uploaded documents.

This fork adds **CAG-RAG**: the 15 regulations most questions land on sit
*whole* in a prompt-cached block the model sees on every question, and hybrid
search covers the rest of the corpus.

> AskGMP never returns a substantive answer without showing which document(s)
> and which page(s) support it, wherever page metadata exists.

Every citation is checked against what the model was actually given. A
reference it invented is stripped before the answer reaches you.

[![CAG-RAG launch video](videos/v4-voice2/brag.jpg)](videos/v4-voice2/brag.mp4)

---

## Why CAG + RAG

Plain RAG sends the model the top few chunks a search found. That works until
the right passage loses the ranking — a small regulation outranked by a long
guidance that mentions the same words, or a retention rule split across chunks.

CAG (cache-augmented generation) removes the search step for the documents that
matter most: they are always in the prompt, in full. Prompt caching makes that
affordable — after the first call, those ~133k tokens are read from cache at a
fraction of the normal price.

| | Plain RAG | CAG-RAG |
|---|---|---|
| Core regulations | only the chunks search finds | always in the prompt, whole |
| Long tail (~75 docs) | hybrid search | hybrid search |
| Prompt per question | ~9k tokens | ~140k tokens, ~133k cached |
| Cost per question | ~$0.03 | ~$0.04 warm · ~$0.36 cold |

**The core library** (`CAG_CORE_DOCUMENTS` in `app/config.py`):

- EU GMP Part I, Chapters 1–9, and the EudraLex glossary
- 21 CFR Parts 210 and 211
- ICH Q9(R1) and ICH Q10
- EU GMP Annex 11 (computerised systems) — small, and hybrid search kept losing
  it to the larger data-integrity guidances on audit-trail questions

---

## Benchmark

`scripts/eval_cag.py` asks 12 fixed questions twice — with the cached core and
without it — and scores what can be checked without an LLM judge:

- **hit** — the document that should answer the question was cited
- **facts** — the key facts expected in the answer are all present
- **rejected** — citations the validator had to strip (invented by the model)

Half the questions are answered by core documents, half by the long tail, plus
one that needs both — so the run also shows whether the big core distracts the
model from retrieved passages.

Final configuration (run 3, `anthropic/claude-sonnet-5.5` via OpenRouter):

| Mode | Cited right doc | All facts | Invented citations | Avg latency | Cost (12 Qs) |
|---|---|---|---|---|---|
| **CAG-RAG** | **12 / 12** | **12 / 12** | 0 | 13.1 s | $0.81 |
| Plain RAG | 10 / 12 | 11 / 12 | 0 | 12.0 s | $0.39 |

Where plain RAG missed:

- *"How long must batch documentation be retained under EU GMP?"* — RAG found
  the facts but never cited EU GMP Chapter 4. CAG had the chapter in front of it.
- *"What does EU GMP Annex 11 require for audit trails?"* — RAG answered from
  data-integrity guidances and missed Annex 11 entirely. In run 1 both modes
  missed it; Annex 11 joined the core from run 2 on, and CAG has cited it since.

Across all three runs CAG-RAG never scored below plain RAG on document hits
(11, 12, 12 vs 10, 10, 10). Neither mode produced a single invented citation.
Outside Annex 11, both modes cited the right long-tail document every time,
and CAG's only other slip in 36 answers was one missing expected phrase on the
FDA OOS question in run 2: the core does not crowd out retrieval.

**What it costs.** CAG costs about a cent more per warm question, and the first
question after the cache expires costs ~$0.36 to write it. The cost total above
includes that one cold call. CAG pays for accuracy, not savings.

```bash
venv/Scripts/python.exe -m scripts.eval_cag --out data/eval/run4.json   # ~$1.20
```

---

## Architecture

```
KnowledgeBase/                          data/uploads/
     │                                       │
     ▼                                       ▼
        ┌──────────────────────────────────────┐
        │   app/documents — shared extraction  │
        │   pdf docx xlsx pptx html json csv   │
        │   OCR · pages · slides · cells       │
        │   one record model · one chunker     │
        └──────────────────────────────────────┘
     │                                       │
     ▼  app/knowledge_base                   ▼  app/uploads
  registry → indexer → delta state      storage → in-memory index
     │                                       │
     ▼                                       │
   Qdrant  (dense + sparse)                  │
     │          │                            │
     │          ▼  app/cag                   │
     │      CoreLibrary: 15 core docs,       │
     │      loaded once, fixed order         │
     │          │                            │
     ▼  app/retrieval                        │
  hybrid search → RRF → diversify            │
  (core hits dropped: already in prompt)     │
     │          │                            │
     └──────────┴───┬────────────────────────┘
                    ▼  app/chat
   ┌────────────────────────────────────────┐
   │ system: instructions                   │
   │ system: core library  ◄ cache_control  │  static, byte-identical
   ├────────────────────────────────────────┤
   │ user:   retrieved passages + question  │  dynamic
   └────────────────────────────────────────┘
                    │  OpenRouter → Claude Sonnet
                    ▼
        citation validation + source rows
                    │
                    ▼  app/api  →  app/frontend
```

### How CAG stays cheap and correct

Prompt caching only hits if every byte before the breakpoint is identical, and
citations only work if a document gets the same key every time. So:

- **Deterministic core.** `app/cag/core.py` loads the core chunks from Qdrant
  once per process and sorts them by configured document order, then chunk
  number. Same bytes, every question.
- **One breakpoint.** `cache_control` sits only on the core system block.
  Retrieved passages and the question come after it, in the user message.
- **Stable keys.** Core chunks go first, so `assign_citation_keys` always gives
  core documents the same short keys.
- **No duplicates.** Retrieval ranks the whole corpus; core hits are dropped
  afterwards, so questions the core answers send few extra chunks.
- **Sticky routing.** A fixed OpenRouter `session_id` keeps requests on the
  provider that holds the warm cache.
- **Clean source table.** Core documents appear only if the answer cites them,
  narrowed to the cited pages.
- **Escape hatch.** An `authority` filter skips the core and falls back to
  plain RAG; `CAG_ENABLED=false` turns it off entirely.

Check it is working: `GET /api/health` → `cag_last_usage.cached_tokens` should
be ~133k on a warm call. Zero on repeat calls means the prefix changed.

### Packages

| Package | Responsibility |
|---|---|
| `app/core` | Field names, payload discriminators, exception types. No logic. |
| `app/config.py` | Every tunable, read from the environment. Creates no clients. |
| `app/documents` | File formats: extraction, OCR, the record model, chunking. |
| `app/knowledge_base` | The persistent regulatory corpus: registry, indexer, delta state. |
| `app/uploads` | One user's file: storage, reading, in-memory RAG, summarising. |
| `app/infrastructure` | Qdrant and OpenAI clients, dense and sparse vector building. |
| `app/retrieval` | Hybrid search, ranking, filters, and the evidence model. |
| `app/cag` | The cached core library: loads the core documents in a fixed order. |
| `app/chat` | Prompts, the structured call, citation validation, source rows. |
| `app/api` | FastAPI endpoints. |
| `app/frontend` | Streamlit chat and uploader. |
| `scripts` | Downloaders and operational entry points. |

### Dependency direction

```
              app.documents          app.infrastructure
                 ▲      ▲                ▲         ▲
                 │      │                │         │
    app.knowledge_base  └── app.uploads ─┘         │
                                  │                │
                                  └──► app.retrieval
                                            ▲
                                            │
                                        app.chat
                                            ▲
                                            │
                                         app.api
                                            ▲
                                            │
                                       app.frontend
```

`knowledge_base` and `retrieval` never import each other. The indexer writes
points and the retriever reads them, but what they agree on — the collection
name, the two named vectors, the payload field names, the deterministic point-ID
scheme and the sparse-vector algorithm — lives in `app/core/constants.py` and
`app/infrastructure/`. A separately fitted sparse vocabulary on the query side
would put term IDs in a different space and quietly break the keyword half of
every search; sharing the builder is what prevents that.

---

## Shared document processing

Knowledge-base ingestion and user uploads run through **one** extraction stack.
Adding a format means adding one module under `app/documents/extractors/` and
one line in its registry — both workflows gain it at once.

The two callers want different things from the same file, so the dispatcher
takes a **profile** rather than pretending they are identical:

- **`PROFILE_STRUCTURAL`** — what the corpus is indexed with. `pymupdf4llm`
  renders a PDF as markdown, preserving the heading structure a regulation is
  cited by; a slide becomes one record. Fast, and no OCR unless a document
  yields no text at all.
- **`PROFILE_RICH`** — what an upload gets. Scanned pages are detected and
  OCR-ed, pictures inside a `.docx` or `.pptx` are read, tables and rows become
  their own records with cell-level provenance.

HTML, Markdown, JSON, CSV and plain text ignore the profile entirely and share
a single implementation — which is most of what the unification removed. PDF
and PowerPoint keep both strategies in one module rather than in two packages.

Everything produced carries the same record shape (`app/documents/models.py`)
and goes through the same chunker, so an uploaded SOP and a EudraLex annex flow
into citation validation with no translation step.

---

## Commands

Run everything from the project root, inside the venv.

```bash
# Qdrant
docker start cag-rag-qdrant        # or: docker run -d -p 6333:6333 qdrant/qdrant

# Index the knowledge base (auto: full load first time, delta afterwards)
python -m scripts.index_knowledge_base

python -m scripts.index_knowledge_base --list      # the plan, no work
python -m scripts.index_knowledge_base --dry-run   # no OpenAI or Qdrant calls
python -m scripts.index_knowledge_base --mode init # force a rebuild

# Start the API
uvicorn app.api.main:app --reload --port 8000

# Start the frontend (expects the API on ASKGMP_API_URL, default :8000)
streamlit run app/frontend/streamlit_app.py

# Tests (no network)
python -m pytest tests -q

# CAG vs plain RAG benchmark (~$1.20)
python -m scripts.eval_cag --out data/eval/runN.json
```

A fresh Qdrant needs `--mode init`: the committed `data/.index_state` says
everything is already indexed, so delta mode would skip it all.

Refreshing the corpus:

```bash
python -m scripts.downloaders.eudralex        # and ich, ema, who_production, ...
python -m scripts.build_document_registry     # then review the registry CSV
python -m scripts.index_knowledge_base        # delta: only what changed
```

---

## Configuration

`.env` at the project root:

```
OPENROUTER_API_KEY=...
```

That is all CAG-RAG needs. Answers go to `anthropic/claude-sonnet-5.5` through
OpenRouter. With no `OPENAI_API_KEY`, embeddings also go through OpenRouter
(`openai/text-embedding-3-large`, same 3072 dims). Qdrant defaults to
`http://localhost:6333`; set `NO_PROXY=127.0.0.1,localhost` if you run behind
an HTTP proxy.

| Variable | Default | |
|---|---|---|
| `CAG_ENABLED` | true | false = plain RAG |
| `CAG_CORE_DOCUMENTS` | 15 docs above | comma-separated `document_id`s |
| `CAG_CACHE_TTL` | 5m | `1h` if questions come 5–60 min apart |
| `ASKGMP_CHAT_MODEL` | anthropic/claude-sonnet-5.5 | |

Changing the core set or the system prompt invalidates the cache (one ~$0.36
write). After re-indexing, restart the API: the core loads once per process.

Everything else is optional and lives in `app/config.py` with its default and
the reason for it — retrieval limits, chunk budget, the effective-only filter,
OCR on uploads, model names. Each is overridable through the environment.

---

## Data

| Path | What it is |
|---|---|
| `KnowledgeBase/` | The regulatory source documents, by authority. |
| `KnowledgeBase/_registry/documents_registry.csv` | The indexer's source of truth: what to index, and every metadata field. Curated columns survive a rebuild. |
| `data/.cache/kb_extracted/` | Extraction results, keyed by file SHA-256. |
| `data/.index_state/` | Fingerprints and checkpoints, so a rerun touches only what changed. |
| `data/uploads/` | User uploads, named by content hash. |

Nothing under `KnowledgeBase/` or `data/` is inside the application package, and
the registry is never rewritten by the indexer — only read.

---

## Further reading

[`app/README.md`](app/README.md) covers the chat layer in detail: how citation
keys are derived, the three independent mechanisms that keep an answer
traceable, and how the source table aggregates chunks into document rows.

---

## Regulatory state

Superseded and withdrawn guidance stays in the index; the registry records
`status` precisely so it can be *filtered* rather than deleted. Retrieval
applies `status=effective` by default, inside both halves of the hybrid query
rather than after fusion — filtering afterwards would let superseded chunks
consume candidate slots and quietly shrink the result set.

An uploaded document is deliberately **not** written into Qdrant. It has no
authority, no status and no registry row, and a site's draft procedure must
never look like published guidance. Uploads are indexed in memory and their
regulatory payload fields are left empty rather than invented.
