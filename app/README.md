# AskGMP — chat layer

How a question becomes a cited answer. For the overall architecture and the
commands, see the [root README](../README.md).

```
question
   │
   ▼  app/retrieval/retriever.py
hybrid Qdrant search (dense + sparse, RRF), status=effective only
   │
   ▼  diversify: cap chunks per document, keep the best 8
   │
   ▼  app/chat/prompts.py      → [SOURCE n] blocks with IDs, pages, sections
   │
   ▼  app/chat/service.py      → structured LLM call (Pydantic)
   │
   ▼  app/chat/citations.py    → validate every citation against retrieval
   │
   ▼  answer with [DOC_ID, p. N]  +  aggregated source rows
```

## Run it

```bash
# 1. Qdrant must be up (the knowledge base lives there)
docker start talentlens-qdrant

# 2. Backend
venv/Scripts/python.exe -m uvicorn app.api.main:app --reload --port 8000

# 3. Frontend, in a second terminal
venv/Scripts/python.exe -m streamlit run app/frontend/streamlit_app.py
```

Then open http://localhost:8501. The API alone:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"Which documents discuss data integrity?"}'
```

## The one promise

AskGMP never shows a substantive claim it cannot trace to a document and, where
the document has pages, a page. Three mechanisms enforce that, and they are
independent on purpose:

1. **The model never supplies metadata.** It returns prose, a confidence flag,
   and citations consisting of a key and a page range — nothing else. Titles,
   authorities, reference codes and URLs are copied from the Qdrant payload
   after generation. There is no way to tell, from output alone, a URL the
   model read from one it composed, so it is never asked for one.

2. **Every citation is checked against what was retrieved.** An unknown key is
   stripped from the prose and reported in `rejected_citations`. A page the
   document did not contribute is corrected to the nearest range it did — the
   document is right, only the locator is off. A page cited for a document with
   no page metadata is removed. A section reference is kept only if its
   identifying token appears in that document's retrieved text.

3. **No context, no call.** A question that retrieves nothing returns the
   no-answer response without reaching the model, because there would be
   nothing for it to be grounded in.

If validation empties the answer, or the model reports insufficient evidence,
the response becomes the no-answer text plus the closest documents found. The
source table renders either way.

## Citation keys

The registry's `document_id` values run to a hundred characters
(`EUDRALEX_01_GUIDELINES_ON_GOOD_MANUFACTURING_PRACTICE_SPECIFIC_TO_ADVANCED_THERAPY_MEDICINAL_PRODUCTS`),
which is unusable mid-sentence. Each document therefore gets a short
`citation_key` derived from stored metadata — the `reference_code` where the
registry has one (`ICH_Q9_R1`, `EU_GMP_CHAPTER_8`, `PE_009_17_ANNEXES`),
otherwise a trimmed `document_id`.

It is derived, never invented, and always reconciled to a real `document_id`
before anything is returned. Where two documents share a reference code — ICH
Q7 the guideline and ICH Q7 the Q&A — the content role disambiguates
(`ICH_Q7` / `ICH_Q7_QA`). The API returns both `citation_key` and
`document_id`; the table shows the key, because that is what appears in the
answer and what the reader needs to match against.

## Source table

One row per document, not per chunk. Five chunks from one guideline aggregate
into one row with merged page ranges. Contiguous pages join (`11-13`);
non-contiguous ones stay separate (`4, 40-41`), because a document that answers
on page 4 and again on page 40 is naming two places to look and `4-40` would be
a false statement about where the evidence is.

`Official Document` links to `document_url`, falling back to `source_page_url`.
Most rows are empty today — the registry has URLs for only 7 of 90 documents —
so the UI says so rather than showing a dead column.

Rows the answer actually cited sort first and are flagged; the rest are still
shown, because "here is what I looked at" helps someone trying to locate a
document.

## Configuration

Everything tunable is in `app/config.py`, overridable by environment variable.

| Variable | Default | |
|---|---|---|
| `ASKGMP_CANDIDATE_LIMIT` | 20 | candidates after RRF fusion |
| `ASKGMP_PREFETCH_LIMIT` | 40 | per vector, before fusion |
| `ASKGMP_MAX_CONTEXT_CHUNKS` | 8 | chunks sent to the model |
| `ASKGMP_MAX_CHUNKS_PER_DOCUMENT` | 3 | diversity cap |
| `ASKGMP_EFFECTIVE_ONLY` | true | filter out superseded/withdrawn/draft |
| `ASKGMP_CHAT_MODEL` | gpt-4o | |
| `ASKGMP_API_URL` | http://127.0.0.1:8000 | where the frontend looks |

## Limits of V1

No conversation memory — each question is retrieved and answered independently,
so follow-ups like "and what about the FDA?" do not inherit context. No
reranking model beyond RRF plus the diversity cap. No auth, no streaming.
