"""
indexer.py — Index the AskGMP regulatory knowledge base into Qdrant, driven by
KnowledgeBase/_registry/documents_registry.csv.

    python -m scripts.index_knowledge_base --list       # the plan only
    python -m scripts.index_knowledge_base --dry-run    # no writes
    python -m scripts.index_knowledge_base --limit 5    # small trial
    python -m scripts.index_knowledge_base              # auto: init, then delta
    python -m scripts.index_knowledge_base --mode init  # force a rebuild

What changed from the regulations indexer this grew out of
----------------------------------------------------------
That script read a folder of EudraLex and ICH PDFs and worked out what each one
was from its path: the family from the folder, the document type from a regex
over the filename, the title from the first non-boilerplate line of page one.
Three things broke that approach for AskGMP.

  * Seven authorities, not two. "Is this FDA guidance or an EMA Q&A?" stopped
    being answerable from a folder name.

  * Five file formats. rglob("*.pdf") silently skipped the eCFR HTML, the EMA
    Q&A JSON and MHRA's OOS deck — three of the most citable things in the
    corpus.

  * Regulatory state. A guideline becomes `superseded` on a date that appears
    nowhere in its own bytes. No amount of filename parsing finds that, and an
    index that keeps answering from a withdrawn guideline is worse than one
    that has never heard of it.

So the registry became authoritative. Nothing here infers authority,
jurisdiction, reference code, topic, status, version or URL — it reads them.
The indexer's remaining judgement is confined to what the registry genuinely
cannot know: how to get text out of a given file format, and where to put the
chunk boundaries.

Where the chunks land
---------------------
doc_type = "regulation", source = "askgmp", in the shared collection. Not
doc_type="sop": these are external regulatory instruments, and a retriever that
cannot tell a European Commission annex from a site's own procedure cannot
answer "what does GMP require?" any differently from "what do we do?".

Pipeline
--------
    load registry (ingest AND is_preferred_source)
      → resolve files, report missing / unregistered
        → key each unit by artifact_id
          → extract per format, page/slide/section aware, cached by file hash
            → chunk to a 1000-token budget, atomic units kept whole
              → dense embeddings + sparse vectors
                → upsert to Qdrant, checkpointing after every document

Full vs delta
-------------
Chosen automatically: full when nothing is tracked, delta afterwards. A delta
compares each document's fingerprint — file hash *and* registry metadata —
against the previous run, and re-embeds only what moved. Both modes checkpoint
per document, so an interrupted run resumes instead of paying to embed the
corpus twice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from app import config
from app.core.constants import (
    DOC_TYPE_FIELD,
    DOC_TYPE_REGULATION,
    SOURCE_ASKGMP,
    SOURCE_FIELD,
)
from app.documents import cache as extraction_cache
from app.documents import chunking
from app.documents.chunking import MAX_CHUNK_TOKENS, chunk_records, token_count
from app.documents.extractors import PROFILE_STRUCTURAL
from app.documents.text_utils import BOILERPLATE, clean_line
from app.infrastructure.embeddings import embed_texts
from app.infrastructure.openai_client import get_openai_client
from app.infrastructure.qdrant import (
    count_points,
    delete_by_filter,
    delete_point_ids,
    ensure_collection,
    get_client,
    make_point,
    point_id,
    upsert_in_batches,
)
from app.knowledge_base import registry as registry_module
from app.knowledge_base.index_state import (
    KNOWLEDGE_BASE_DIR,
    REGISTRY_PATH,
    DeltaPlan,
    StateStore,
    banner,
    hash_file,
    hash_text,
    plan_delta,
)
from app.knowledge_base.registry import (
    FINGERPRINT_FIELDS,
    METADATA_FIELDS,
    discover_documents,
)

# ── Configuration ─────────────────────────────────────────────────────────────
SOURCE      = SOURCE_ASKGMP
DOC_TYPE    = DOC_TYPE_REGULATION
EXTRACT_DIR = config.EXTRACT_DIR

COLLECTION_NAME = config.COLLECTION_NAME
DENSE_DIM       = config.DENSE_DIM
EMBED_MODEL     = config.EMBED_MODEL

EXTRACT_WORKERS   = config.EXTRACT_WORKERS
EMBED_BATCH_SIZE  = config.EMBED_BATCH_SIZE
UPLOAD_BATCH_SIZE = config.UPLOAD_BATCH_SIZE

# A document that extracts to less than this is almost always a download that
# went wrong rather than a genuinely short guideline — typically a JavaScript
# landing page saved instead of the document it links to. It is still indexed
# (an accurate abstract is worth citing), but it is reported, because the fix
# is to re-download it and no one will look unless they are told.
THIN_DOCUMENT_TOKENS = config.THIN_DOCUMENT_TOKENS

# Bump when extraction, chunking or payload construction changes in a way that
# makes already-stored chunks stale. It is mixed into every document's
# fingerprint, so bumping it makes the next delta run rebuild everything without
# anyone having to delete state files by hand.
CONFIG_VERSION = "askgmp-1"


# ── Fingerprints ──────────────────────────────────────────────────────────────
def document_fingerprint(entry: dict, file_hash: str) -> str:
    """
    Changes when the file changes *or* when metadata that reaches the payload
    changes.

    The metadata half is the point. A guideline that gets superseded, or gains
    the official URL it was missing, is byte-identical on disk — and the chunks
    already in Qdrant would go on claiming it is effective, and go on failing to
    cite it, until someone deleted the state file. Folding the registry fields
    into the fingerprint means the next delta run notices.
    """
    parts: list[Any] = [
        file_hash,
        entry.get("document_key"),
        CONFIG_VERSION,
        MAX_CHUNK_TOKENS,
    ]
    parts.extend(entry.get(field) for field in FINGERPRINT_FIELDS)
    return hash_text(*parts)


# ── Payloads ──────────────────────────────────────────────────────────────────
def build_payload(entry: dict, chunk: dict, chunk_no: int, chunk_id: str) -> dict:
    """
    One chunk's Qdrant payload: registry metadata, chunk locators, citation URLs.

    The URLs matter more than they look. They come from the registry (or, for a
    Q&A/CFR record, from the record itself), which means the model can cite the
    real page instead of assembling a plausible-looking ema.europa.eu URL that
    does not resolve. A fabricated citation on a GMP answer is worse than no
    citation at all.
    """
    meta = chunk.get("meta") or {}

    payload: dict[str, Any] = {
        "text": chunk["text"],

        # Chunk locators.
        "chunk_id" : chunk_id,
        "chunk_no" : chunk_no,
        "section"  : chunk.get("section"),
        "page"     : chunk.get("page_start"),
        "page_end" : chunk.get("page_end"),

        # Provenance on disk.
        "document_key": entry.get("document_key"),
        "source_file" : entry.get("source_file"),
        "source_path" : entry.get("local_path"),

        DOC_TYPE_FIELD: DOC_TYPE,
        SOURCE_FIELD  : SOURCE,
    }

    # Registry metadata, verbatim. `status`, `effective_date` and
    # `supersedes_document_id` are copied exactly as recorded — never
    # normalised to "effective" — so a retriever can filter on regulatory state.
    for field in METADATA_FIELDS:
        payload[field] = _clean(entry.get(field))

    # A per-record URL (an eCFR section, an EMA Q&A anchor) is more precise than
    # the document-level one, so it is preferred where extraction found one.
    record_url = meta.get("record_url")
    if record_url and not payload.get("document_url"):
        payload["document_url"] = record_url
    if record_url:
        payload["record_url"] = record_url

    # The single URL a citation should use, resolved once here rather than by
    # the model at answer time.
    payload["citation_url"] = (
        record_url
        or payload.get("document_url")
        or payload.get("source_page_url")
    )

    for key in ("question", "qa_section", "qa_topic"):
        if meta.get(key):
            payload[key] = meta[key]

    return payload


def _clean(value: Any) -> Any:
    """NaN and empty strings out of the payload; keep the key, drop the noise."""
    return registry_module.clean_value(value)


def build_points(entry: dict, chunks: list[dict], dense_vectors: list[list[float]]):
    points, ids = [], []

    for number, (chunk, dense) in enumerate(zip(chunks, dense_vectors)):
        chunk_id = f"{entry['document_key']}#chunk-{number:04d}"
        pid = point_id(DOC_TYPE, chunk_id)
        payload = build_payload(entry, chunk, number, chunk_id)
        points.append(make_point(pid, dense, chunk["text"], payload))
        ids.append(pid)

    return points, ids


# ── Titles ────────────────────────────────────────────────────────────────────
def resolve_title(entry: dict, records: list[dict]) -> str:
    """
    The registry title wins. Always, when it is there.

    Reading a title out of a PDF's first page is guesswork that the old indexer
    had to do because nothing else knew. Here it is a fallback for a blank
    registry cell, nothing more — a curated "ICH Q9(R1) Quality Risk Management"
    must never be overwritten by whatever line the cover page happened to start
    with.
    """
    title = registry_module.clean_value(entry.get("document_title"))
    if title:
        return str(title)

    for record in records[:3]:
        for line in (record.get("text") or "").splitlines():
            line = clean_line(line)
            if not (8 <= len(line) <= 140):
                continue
            if BOILERPLATE.search(line):
                continue
            if not any(character.isalpha() for character in line):
                continue
            return line

    return Path(entry["local_path"]).stem.replace("_", " ")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index the AskGMP regulatory knowledge base into Qdrant."
    )
    parser.add_argument("--mode", choices=["auto", "init", "delta"], default="auto",
                        help="auto (default): full load when nothing is tracked, "
                             "delta afterwards.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load, extract and chunk, but make no OpenAI or "
                             "Qdrant calls.")
    parser.add_argument("--list", action="store_true",
                        help="Print the registry plan and exit.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Index at most N documents (trial runs).")
    parser.add_argument("--workers", type=int, default=EXTRACT_WORKERS,
                        help=f"Parallel extraction workers (default {EXTRACT_WORKERS}).")
    parser.add_argument("--reextract", action="store_true",
                        help="Ignore the extraction cache and re-read every file.")
    parser.add_argument("--ocr", action="store_true",
                        help="OCR PDFs that yield no text at all (needs Tesseract).")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH,
                        help="Path to documents_registry.csv.")
    return parser.parse_args(argv)


def print_registry_plan(discovery, limit: int | None) -> None:
    print(f"Registry rows        : {discovery.total_rows}")
    print(f"Selected for ingest  : {len(discovery.documents)}"
          + (f"  (--limit {limit})" if limit else ""))
    print(f"ingest = False       : {len(discovery.not_ingested)}")
    print(f"Missing files        : {len(discovery.missing)}")
    print(f"Alternate formats    : {len(discovery.skipped_representations)}")
    print(f"Unregistered files   : {len(discovery.unregistered)}")
    print()

    header = f"{'document_id':<44} {'authority':<21} {'type':<16} {'status':<12} {'fmt':<5} local_path"
    print(header)
    print("-" * len(header))
    for entry in discovery.documents:
        print(
            f"{str(entry['document_key'])[:44]:<44} "
            f"{str(entry.get('authority') or '-')[:21]:<21} "
            f"{str(entry.get('document_type') or '-')[:16]:<16} "
            f"{str(entry.get('status') or '-')[:12]:<12} "
            f"{str(entry.get('file_type') or entry['suffix'].lstrip('.'))[:5]:<5} "
            f"{entry['local_path']}"
        )
    print()

    _print_problems(discovery)


def _print_problems(discovery) -> None:
    if discovery.missing:
        print(f"! {len(discovery.missing)} registry row(s) point at a file that is not "
              f"on disk — skipped:")
        for row in discovery.missing[:20]:
            print(f"    {row.get('document_id')}: {row.get('local_path')} "
                  f"({row.get('problem')})")
        if len(discovery.missing) > 20:
            print(f"    ... and {len(discovery.missing) - 20} more")
        print()

    if discovery.skipped_representations:
        print(f"  {len(discovery.skipped_representations)} alternate representation(s) "
              f"skipped in favour of a more structured format:")
        for row in discovery.skipped_representations:
            print(f"    {row['local_path']}  ->  {row['superseded_by_path']}")
        print()

    collisions = [entry for entry in discovery.documents if entry.get("id_collision")]
    if collisions:
        print(f"  {len(collisions)} document(s) share a document_id with a different "
              f"document and were kept separately:")
        for entry in collisions:
            print(f"    {entry['document_key']}")
        print()

    if discovery.unregistered:
        print(f"! {len(discovery.unregistered)} unregistered knowledge-base file(s) found "
              f"— NOT indexed:")
        for path in discovery.unregistered[:20]:
            print(f"    {path}")
        if len(discovery.unregistered) > 20:
            print(f"    ... and {len(discovery.unregistered) - 20} more")
        print("  Run python -m scripts.build_document_registry to review them.")
        print()

    if discovery.not_ingested:
        print(f"  {len(discovery.not_ingested)} row(s) have ingest = False and were "
              f"never considered.")
        print()

    # A chunk with no URL can be quoted but not linked. Q&A and CFR records
    # carry their own per-record URLs, so those documents are fine even when
    # the registry cell is empty; the rest depend entirely on the registry.
    unlinked = [
        entry for entry in discovery.documents
        if not entry.get("document_url") and not entry.get("source_page_url")
        and entry["suffix"] not in {".json"}
    ]
    if unlinked:
        print(f"! {len(unlinked)} of {len(discovery.documents)} document(s) have neither "
              f"document_url nor source_page_url in the registry.")
        print("  They will be indexed and quotable, but an answer citing them cannot "
              "link to the official source.")
        print("  Filling those two columns is the single highest-value edit to the "
              "registry.")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Python block-buffers stdout when it is a pipe rather than a terminal, so a
    # run piped to a log or a CI job shows nothing at all until it finishes —
    # for a job that takes tens of minutes, the progress output is the point.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass

    state = StateStore(SOURCE)
    mode = args.mode
    if mode == "auto":
        mode = "delta" if state.is_initialised else "init"

    banner("AskGMP Knowledge Base Indexer", [
        ("Knowledge base", KNOWLEDGE_BASE_DIR),
        ("Registry", args.registry),
        ("Target", f"{COLLECTION_NAME} ({DOC_TYPE_FIELD}={DOC_TYPE}, "
                   f"{SOURCE_FIELD}={SOURCE})"),
        ("Model", f"{EMBED_MODEL} ({DENSE_DIM}d)"),
        ("Chunk budget", f"{MAX_CHUNK_TOKENS} tokens"
                         + ("" if chunking.tokenizer_available()
                            else "  (tiktoken missing — estimating)")),
        ("Mode", mode.upper() + (" (dry run)" if args.dry_run else "")),
        ("Last run", state.read_last_update_time() or "never"),
    ])

    # 1 ── registry-driven discovery
    discovery = discover_documents(
        knowledge_base_dir = KNOWLEDGE_BASE_DIR,
        registry_path      = args.registry,
        limit              = args.limit,
    )

    if args.list:
        print_registry_plan(discovery, args.limit)
        return 0

    _print_problems(discovery)

    documents = discovery.documents
    if not documents:
        print("No documents selected for ingestion.")
        return 0

    # 2 ── delta plan
    current: dict[str, str] = {}
    for entry in documents:
        entry["file_hash"] = hash_file(Path(entry["path"]))
        current[entry["document_key"]] = document_fingerprint(entry, entry["file_hash"])

    previous = {} if mode == "init" else state.fingerprints

    resumed: dict[str, str] = {}
    if mode == "init" and not args.dry_run:
        # Not on a dry run: begin_init writes the crash-recovery file, and a
        # run that promises to change nothing must not leave one behind for the
        # next real run to interpret as an interrupted load.
        resumed = state.begin_init(total=len(documents))
        previous = resumed          # already-uploaded documents survive a crash

    plan: DeltaPlan = plan_delta(current, previous)

    # --limit only narrows what this run *looks at*. Without this guard the
    # documents outside the limit look like they were deleted from the registry,
    # and a trial run would quietly delete the rest of the index.
    if args.limit and plan.removed:
        print(f"  --limit is set: leaving {len(plan.removed)} untouched document(s) "
              f"in place rather than treating them as removed.")
        plan.removed = {}

    print(f"Delta plan: {plan.summary()}\n")

    if not plan.has_work:
        print("Nothing to do — the index is already up to date.")
        state.write_marker(mode, {"documents": len(documents), "chunks_indexed": 0})
        state.save()
        _print_summary(discovery, state, extracted=0, skipped=0, chunks_created=0,
                       chunks_written=0, chunks_removed=0,
                       unchanged=len(plan.unchanged), total=None)
        return 0

    by_key = {entry["document_key"]: entry for entry in documents}
    todo = [by_key[key] for key in plan.to_index]

    # 3 ── extract
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reextract:
        for entry in todo:
            extraction_cache.cache_path(EXTRACT_DIR, entry["file_hash"]).unlink(missing_ok=True)

    print(f"Extracting {len(todo)} document(s) with {args.workers} worker(s)...")
    jobs = [
        {
            "document_key": entry["document_key"],
            "path"        : entry["path"],
            "suffix"      : entry["suffix"],
            "file_hash"   : entry["file_hash"],
            "cache_dir"   : str(EXTRACT_DIR),
            "use_ocr"     : args.ocr or bool(entry.get("needs_ocr")),
            "force"       : args.reextract,
            # The structural profile: markdown-aware PDF extraction and one
            # record per slide, which is what the indexed chunks were built
            # from. The rich profile is for uploads — see app/documents.
            "profile"     : PROFILE_STRUCTURAL,
        }
        for entry in todo
    ]
    extraction = extraction_cache.extract_all(jobs, workers=max(1, args.workers))

    failures = [
        result for result in extraction.values()
        if result["status"].startswith(("failed", "empty"))
    ]
    if failures:
        print(f"\n  ! {len(failures)} document(s) yielded no usable text and are skipped:")
        for result in failures[:15]:
            print(f"      {result['document_key']}: {result['status']}")
    print()

    # 4 ── chunk
    print("Chunking...")
    prepared: list[tuple[dict, list[dict]]] = []
    skipped_documents = 0

    for entry in todo:
        result = extraction.get(entry["document_key"], {})
        if result.get("status", "").startswith(("failed", "empty")):
            skipped_documents += 1
            continue

        records = extraction_cache.load_records(EXTRACT_DIR, entry["file_hash"])
        entry["document_title"] = resolve_title(entry, records)

        chunks = chunk_records(records)
        if not chunks:
            print(f"    {entry['source_file']}: no extractable text — skipped")
            skipped_documents += 1
            continue

        prepared.append((entry, chunks))

    chunks_created = sum(len(chunks) for _, chunks in prepared)
    print(f"    {len(prepared)} document(s) -> {chunks_created} chunk(s)\n")

    thin = [
        (entry, sum(token_count(chunk["text"]) for chunk in chunks))
        for entry, chunks in prepared
    ]
    thin = [(entry, size) for entry, size in thin if size < THIN_DOCUMENT_TOKENS]
    if thin:
        print(f"  ! {len(thin)} document(s) extracted to very little text — likely a "
              f"landing page rather than the document itself. They are still indexed, "
              f"but are worth re-downloading:")
        for entry, size in thin[:15]:
            print(f"      {entry['local_path']}  ({size} tokens)")
        print()

    if args.dry_run:
        _print_dry_run(prepared, plan, state)
        _print_summary(discovery, state, extracted=len(prepared),
                       skipped=skipped_documents, chunks_created=chunks_created,
                       chunks_written=0, chunks_removed=0,
                       unchanged=len(plan.unchanged), total=None)
        return 0

    # 5 ── connect
    openai_client = get_openai_client()
    qdrant = get_client()
    ensure_collection(qdrant)

    # A clean full load starts by clearing the previous knowledge base. A
    # resumed one must not — its earlier chunks are exactly what we are keeping.
    if mode == "init" and not resumed:
        delete_by_filter(qdrant, {DOC_TYPE_FIELD: DOC_TYPE, SOURCE_FIELD: SOURCE})
        # The corpus is gone from Qdrant, so the tracked keys describing it must
        # go too — otherwise documents removed from the registry since the last
        # run would linger in state forever.
        state.reset()
    print()

    # 6 ── retire points that no longer belong
    orphans: list[str] = []
    for key in plan.removed:
        orphans.extend(state.forget(key))
    stale_owners = set(plan.changed)
    if orphans:
        deleted = delete_point_ids(qdrant, orphans)
        print(f"Deleted {deleted} point(s) from {len(plan.removed)} removed document(s).")

    # 7 ── embed + upload, one document at a time so progress is durable
    chunks_written = 0
    for number, (entry, chunks) in enumerate(prepared, 1):
        print(f"[{number}/{len(prepared)}] {entry['source_file']} "
              f"({len(chunks)} chunk(s))")

        dense = embed_texts(
            openai_client,
            EMBED_MODEL,
            [chunk["text"] for chunk in chunks],
            batch_size = EMBED_BATCH_SIZE,
            label      = "chunks",
        )

        points, ids = build_points(entry, chunks, dense)

        # A re-indexed document that got shorter leaves points behind, because
        # upsert only overwrites the IDs it writes. Drop the surplus.
        if entry["document_key"] in stale_owners:
            surplus = [
                pid for pid in state.point_ids_for(entry["document_key"])
                if pid not in set(ids)
            ]
            if surplus:
                delete_point_ids(qdrant, surplus)
                orphans.extend(surplus)
                print(f"    retired {len(surplus)} stale chunk(s)")

        upsert_in_batches(
            qdrant, COLLECTION_NAME, points,
            batch_size = UPLOAD_BATCH_SIZE,
        )

        digest = current[entry["document_key"]]
        state.record(
            entry["document_key"], digest, ids,
            document_id   = entry.get("document_id"),
            authority     = entry.get("authority"),
            status        = entry.get("status"),
            source_path   = entry.get("local_path"),
            chunks        = len(ids),
        )
        state.save()
        if mode == "init":
            state.commit_init(entry["document_key"], digest)

        chunks_written += len(ids)

    # 8 ── finalise
    state.write_marker(mode, {
        "documents"     : len(state.fingerprints),
        "chunks_indexed": chunks_written,
        "chunks_removed": len(orphans),
    })
    state.save()
    if mode == "init":
        state.end_init()

    total = count_points(qdrant, {DOC_TYPE_FIELD: DOC_TYPE, SOURCE_FIELD: SOURCE})
    _print_summary(discovery, state, extracted=len(prepared), skipped=skipped_documents,
                   chunks_created=chunks_created, chunks_written=chunks_written,
                   chunks_removed=len(orphans), unchanged=len(plan.unchanged),
                   total=total)
    return 0


def _print_dry_run(prepared, plan, state) -> None:
    print("Dry run — stopping before any OpenAI or Qdrant call.\n")

    for entry, chunks in prepared[:3]:
        first = chunks[0]
        chunk_id = f"{entry['document_key']}#chunk-0000"
        payload = build_payload(entry, first, 0, chunk_id)

        print(f"  {entry['document_key']}  —  {entry['document_title']}")
        print(f"    {entry['local_path']}  |  {len(chunks)} chunk(s)  |  "
              f"{token_count(first['text'])} tokens in chunk 0")
        print(f"    section      : {payload['section']}")
        print(f"    authority    : {payload['authority']} / {payload['jurisdiction']}"
              f"  |  status: {payload['status']}"
              f"  |  weight: {payload['regulatory_weight']}")
        print(f"    citation_url : {payload['citation_url']}")
        for key in ("question", "qa_section", "qa_topic"):
            if key in payload:
                print(f"    {key:<13}: {str(payload[key])[:90]}")
        print(f"    text         : {first['text'][:200].strip()!r}")
        print()

    removals = sum(len(state.point_ids_for(key)) for key in plan.removed)
    if removals:
        print(f"  Would delete {removals} orphaned point(s) from "
              f"{len(plan.removed)} removed document(s).\n")


def _print_summary(discovery, state, *, extracted, skipped, chunks_created,
                   chunks_written, chunks_removed, unchanged, total) -> None:
    rows = [
        ("Registry documents", discovery.total_rows),
        ("Selected for ingestion", len(discovery.documents)),
        ("Missing files", len(discovery.missing)),
        ("Unregistered files", len(discovery.unregistered)),
        ("Alternate formats", len(discovery.skipped_representations)),
        ("Documents extracted", extracted),
        ("Documents skipped", skipped),
        ("Documents unchanged", unchanged),
        ("Chunks created", chunks_created),
        ("Chunks written", chunks_written),
        ("Chunks removed", chunks_removed),
        ("Documents tracked", len(state.fingerprints)),
        ("Collection name", COLLECTION_NAME),
    ]
    if total is not None:
        rows.append(("Points in collection", total))
    rows.append(("State", state.tracked_path))

    print()
    banner("Done", rows)


if __name__ == "__main__":
    sys.exit(main())
