# AskGMP: Pharmaceutical GMP Questions with Source Citations

AskGMP is an [Alinme](https://www.alinme.com) AI engineering tutorial that combines **Cache-Augmented Generation (CAG)** and hybrid **Retrieval-Augmented Generation (RAG)** to answer pharmaceutical Good Manufacturing Practice (GMP) questions with document references and page-level citations when available.

![AskGMP banner with a magnifying glass over pharmaceutical regulatory documents](docs/askgmp-banner.jpg)

[Quick start](#quick-start) · [API and OpenAPI](#api-and-openapi) · [How CAG and RAG work](#how-cag-and-rag-work) · [Benchmark](#cag-rag-vs-plain-rag-benchmark) · [License](#license)

## What can I ask AskGMP?

- How long must batch documentation be retained under EU GMP?
- What does EU GMP Annex 11 say about audit trails?
- What sources cover out-of-specification (OOS) results in pharmaceutical production?

The [document registry](KnowledgeBase/_registry/documents_registry.csv) includes sources from the European Commission, FDA, ICH, EMA, MHRA, PIC/S, and WHO. You can also upload a site document such as a standard operating procedure (SOP) through the Streamlit app. Uploads are searchable alongside the corpus and are labeled as uploads in the response.

## Quick start

Use Python 3.12, Docker for local Qdrant, and an [OpenRouter API key](https://openrouter.ai/keys) for live answers and embeddings. The tests run without an API key or Qdrant.

```bash
git clone https://github.com/Alinme-Co/AskGMP.git
cd AskGMP
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` in Bash on macOS or Linux, or `& .\.venv\Scripts\Activate.ps1` in PowerShell. Then install dependencies and run the tests.

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q
```

### Prepare the regulatory corpus

The public checkout includes the registry metadata. Download the source files into the paths listed in its `local_path` column under `KnowledgeBase/`. The following commands prepare the authorities used by the default 15-document core. Use the remaining [downloaders](scripts/downloaders/) to add wider coverage.

```bash
python -m scripts.downloaders.eudralex
python -m scripts.downloaders.ich
python -m scripts.downloaders.fda_regulations
python -m scripts.convert_fda_to_json
python -m scripts.index_knowledge_base --list
```

Review the missing-file report before indexing. The benchmark used a wider corpus than this core subset. To refresh the registry after updating the corpus, run `python -m scripts.build_document_registry` and review the resulting CSV.

### Start Qdrant, the API, and the web app

Add `OPENROUTER_API_KEY=your-key` to a local `.env` file at the project root. Start a fresh local Qdrant and index the available documents.

```bash
docker run -d --name askgmp-qdrant -p 127.0.0.1:6333:6333 qdrant/qdrant
python -m scripts.index_knowledge_base --mode init
```

For later updates, run `python -m scripts.index_knowledge_base` to index changes, then restart the API to reload the cached core. Start each of the following commands in its own terminal with the environment activated.

```bash
python -m uvicorn app.api.main:app --port 8000
python -m streamlit run app/frontend/streamlit_app.py
```

Open [the local Streamlit app](http://localhost:8501) to ask a question or upload a document. Use documents approved for external processing because the model provider receives the question and selected document context.

## API and OpenAPI

The FastAPI server exposes [interactive API docs](http://127.0.0.1:8000/docs) and a machine-readable [OpenAPI schema](http://127.0.0.1:8000/openapi.json).

| Endpoint | Contract |
| --- | --- |
| `POST /api/chat` | Requires `question`. Optional fields are `authority`, `effective_only`, and `document_hashes` for files already stored by the app. |
| `POST /api/documents/summary` | Summarizes an existing upload by `file_hash`. Accepts an optional `filename`. |
| `GET /api/health` | Reports Qdrant connectivity, collection size, and cache metrics when available. |
| `GET /api/config` | Reports the configured model, retrieval settings, and CAG options. |

Example request body for `POST /api/chat`:

```json
{
  "question": "What does EU GMP Annex 11 say about audit trails?"
}
```

The response includes `answer`, `has_sufficient_evidence`, `sources`, and `rejected_citations`. To identify sources used in the answer, check `sources[].cited`. Each row can carry a `citation_key`, `origin`, page ranges, and an `official_url`. The complete request and response models live in [`app/chat/schemas.py`](app/chat/schemas.py).

## How CAG and RAG work

AskGMP combines cache-augmented generation (CAG) with retrieval-augmented generation (RAG). CAG sends 15 core documents in a stable, prompt-cached prefix. RAG retrieves other relevant passages through Qdrant's dense and sparse search, ranks the results, and adds them alongside any matching upload passages.

The default core contains EU GMP Part I Chapters 1–9, the EudraLex glossary and Annex 11, FDA 21 CFR Parts 210 and 211, and ICH Q9(R1) and Q10. The [core loader](app/cag/core.py) loads indexed content in a fixed order. CAG skips the core if a request filters by authority. Set `CAG_ENABLED=false` for plain RAG on every request.

### How citations and document status are handled

AskGMP checks citation keys against the documents provided to the model and strips unrecognized references. It checks inline page numbers against available source pages. Review the cited passage in its official context before relying on an answer for a quality or compliance decision.

Regulatory search filters for effective documents by default. The registry retains superseded and withdrawn entries for historical research. Uploaded documents use an in-memory retrieval index and stay separate from published regulatory sources.

## CAG-RAG vs. plain RAG benchmark

The repository's recorded comparison used 12 fixed GMP questions and Claude Sonnet 5.5 through OpenRouter. The [evaluation script](scripts/eval_cag.py) checks whether each answer cites the expected document and contains all expected facts. The final configuration, reported as run 3, produced these results.

| Mode | Cited expected document | All expected facts | Rejected citations | Mean latency | Recorded cost for 12 questions |
| --- | ---: | ---: | ---: | ---: | ---: |
| CAG-RAG | 12/12 | 12/12 | 0 | 13.1 s | $0.81 |
| Plain RAG | 10/12 | 11/12 | 0 | 12.0 s | $0.39 |

In this run, plain RAG missed citations to EU GMP Chapter 4 on batch record retention and Annex 11 on audit trails. CAG-RAG cited both. The result reflects this question set, model, and corpus. The CAG-RAG cost includes a cold cache write. Across all three reported runs, CAG-RAG cited the expected document in 11, 12, and 12 answers. Plain RAG did so in 10 answers each time.

To compare both modes on your own indexed corpus, run `python -m scripts.eval_cag --out data/eval/run.json`. This makes billable requests.

## Configuration and repository guide

| Setting | Default or behavior |
| --- | --- |
| `OPENROUTER_API_KEY` | Required for live answers. Embeddings also use OpenRouter unless `OPENAI_API_KEY` is set. |
| `QDRANT_URL` | `http://localhost:6333`. For local services behind an HTTP proxy, set `NO_PROXY=127.0.0.1,localhost`. |
| `ASKGMP_CHAT_MODEL` | `anthropic/claude-sonnet-5.5`. |
| `CAG_ENABLED` | `true`. Set to `false` for plain RAG. |
| `CAG_CORE_DOCUMENTS` | Comma-separated registry document IDs. |
| `CAG_CACHE_TTL` | `5m` by default. `1h` is also supported. |

Use `GET /api/health` and inspect `cag_last_usage.cached_tokens` after repeat questions to check provider-reported cache reads. See [`app/config.py`](app/config.py) for the full settings.

| Path | Purpose |
| --- | --- |
| [`app/api/`](app/api/) and [`app/frontend/`](app/frontend/) | FastAPI routes and the Streamlit app. |
| [`app/chat/`](app/chat/) and [`app/README.md`](app/README.md) | Prompts, citation validation, API schemas, and chat internals. |
| [`app/cag/`](app/cag/) and [`app/retrieval/`](app/retrieval/) | Core prompt context and hybrid Qdrant search. |
| [`app/documents/`](app/documents/) and [`app/uploads/`](app/uploads/) | Shared extraction, OCR, upload search, and summaries. |
| [`app/knowledge_base/`](app/knowledge_base/) and [`scripts/`](scripts/) | Registry loading, indexing, data downloaders, and evaluation. |
| [`tests/`](tests/) | Offline tests for citations, CAG, document processing, and indexing. |

## License

Alinme publishes this proprietary source code for viewing and educational reference. See the [license and usage terms](LICENSE.md) for permissions.
