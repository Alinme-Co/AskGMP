"""
cache.py — extract many documents in parallel, and never read one twice.

Extraction is the expensive half of indexing: a PDF page costs about five
seconds, so re-reading the corpus on every run would dominate the whole
pipeline. Results are cached under data/.cache/kb_extracted/ and a resumed or
repeated run re-reads nothing that has not changed.

Workers run in separate processes, so every heavyweight import (pymupdf, bs4,
pptx) happens inside the extractor that needs it and the records travel via the
cache file rather than back across the process boundary.

This is the batch driver only. What a file extracts *to* is
app/documents/extractors/'s job, and the uploads path uses that directly
without touching the cache — an upload is read once, in the request that
uploaded it.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from app.documents.extractors import PROFILE_STRUCTURAL, extract_records
from app.documents.text_utils import strip_front_matter


def cache_path(cache_dir: Path, file_hash: str) -> Path:
    """
    Cache entries are named by the SHA-256 of the file they came from.

    Keyed by content, not by registry key: what a file extracts to depends on
    its bytes, not on what the registry currently calls it. Renaming a
    document_id, or moving to artifact_id as the unit key, therefore costs
    nothing — where keying by the registry id would silently throw away the
    whole cache and re-read every PDF.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(file_hash))[:80]
    return cache_dir / f"{safe}.json"


def extract_document(job: dict) -> dict:
    """
    Extract one document, honouring the cache. Runs in a worker process, so it
    returns a small summary and leaves the records in the cache file.
    """
    cache_dir = Path(job["cache_dir"])
    cache = cache_path(cache_dir, job["file_hash"])

    if not job.get("force"):
        try:
            if cache.exists():
                cached = json.loads(cache.read_text(encoding="utf-8"))
                if cached.get("file_hash") == job["file_hash"]:
                    return {
                        "document_key": job["document_key"],
                        "records"     : len(cached.get("records", [])),
                        "status"      : "cached",
                    }
        except (json.JSONDecodeError, OSError):
            pass                                  # unreadable cache: re-extract

    try:
        records, status = extract_records(
            job["path"],
            suffix  = job.get("suffix"),
            use_ocr = job.get("use_ocr", False),
            profile = job.get("profile", PROFILE_STRUCTURAL),
        )
    except Exception as exc:                      # noqa: BLE001 - one bad file
        return {                                  # must not abort the corpus
            "document_key": job["document_key"],
            "records"     : 0,
            "status"      : f"failed: {type(exc).__name__}: {exc}",
        }

    if job.get("strip_front_matter", True):
        records = strip_front_matter(records)

    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps({
            "document_key": job["document_key"],
            "file_hash"   : job["file_hash"],
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "records"     : records,
        }, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "document_key": job["document_key"],
        "records"     : len(records),
        "status"      : status if records else "empty: no extractable text",
    }


def load_records(cache_dir: Path, file_hash: str) -> list[dict]:
    cache = cache_path(cache_dir, file_hash)
    if not cache.exists():
        return []
    try:
        return json.loads(cache.read_text(encoding="utf-8")).get("records", [])
    except (json.JSONDecodeError, OSError):
        return []


def extract_all(jobs: list[dict], workers: int = 4) -> dict[str, dict]:
    """Extract many documents, in parallel unless workers <= 1."""
    results: dict[str, dict] = {}
    total = len(jobs)

    if workers <= 1:
        for index, job in enumerate(jobs, 1):
            result = extract_document(job)
            results[result["document_key"]] = result
            _report(index, total, job, result)
        return results

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_document, job): job for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:              # noqa: BLE001
                result = {
                    "document_key": job["document_key"],
                    "records"     : 0,
                    "status"      : f"failed: {type(exc).__name__}: {exc}",
                }
            results[result["document_key"]] = result
            _report(index, total, job, result)

    return results


def _report(index: int, total: int, job: dict, result: dict) -> None:
    name = Path(job["path"]).name
    print(f"    [{index}/{total}] {name} — {result['records']} record(s) "
          f"({result['status']})")
