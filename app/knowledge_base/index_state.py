"""
index_state.py — Delta-load bookkeeping for the knowledge-base indexer.

The problem: re-embedding everything on each run is slow and costs money, so a
run must be able to answer "what actually changed since last time?" and touch
only that.

The answer here is a *fingerprint state file* per source. Each indexable unit
(a deviation record, an SOP chunk, a regulation document) is reduced to a stable
key plus a content hash. Comparing this run's fingerprints with the previous
run's yields four sets:

    added      key is new                        → embed and upsert
    changed    key exists, hash differs          → re-embed and upsert
    unchanged  key exists, hash identical        → skip entirely (the saving)
    removed    key was there, is gone now        → delete its points

Because point IDs are deterministic (app.infrastructure.qdrant.point_id), an
upsert of a changed unit overwrites its previous point in place — no duplicates.

State lives in data/.index_state/ as plain JSON so it can be inspected, and
deleted to force a full rebuild.

Nothing in here talks to OpenAI or Qdrant; it is pure bookkeeping, which keeps
it easy to test and safe to call from a --dry-run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from app import config

# Re-exported from app/config.py, which owns every path in the project. Kept as
# module-level names because the indexer reads them constantly and
# `config.KNOWLEDGE_BASE_DIR` at each site adds nothing.
BASE_DIR           = config.BASE_DIR
KNOWLEDGE_BASE_DIR = config.KNOWLEDGE_BASE_DIR
REGISTRY_PATH      = config.REGISTRY_PATH

STATE_DIR = config.STATE_DIR
CACHE_DIR = config.CACHE_DIR

# Delta detection is hash-based, so this timestamp is informational only — a
# human-readable "when did this last run". The buffer mirrors the convention of
# recording a slightly earlier time than "now", so that a source system with
# clock skew or late-arriving writes is not missed by a timestamp-based reader.
LAST_UPDATE_BUFFER_HOURS = 24


# ── Hashing ───────────────────────────────────────────────────────────────────
def hash_text(*parts: Any) -> str:
    """Content fingerprint for a unit built from in-memory values."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="replace"))
        digest.update(b"\x00")  # separator, so ("ab","c") != ("a","bc")
    return digest.hexdigest()


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Content fingerprint for a file on disk.

    Deliberately content-based rather than mtime-based: copying documents onto
    the machine, restoring a backup or re-downloading an unchanged PDF all
    rewrite mtime without changing a single byte, and each of those would
    otherwise trigger a pointless full re-embed of the corpus.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


# ── Delta plan ────────────────────────────────────────────────────────────────
@dataclass
class DeltaPlan:
    """What a run has to do, derived from two fingerprint maps."""
    added:     dict[str, str] = field(default_factory=dict)
    changed:   dict[str, str] = field(default_factory=dict)
    unchanged: dict[str, str] = field(default_factory=dict)
    removed:   dict[str, str] = field(default_factory=dict)

    @property
    def to_index(self) -> list[str]:
        """Keys needing embedding work, in a stable order."""
        return sorted(self.added) + sorted(self.changed)

    @property
    def has_work(self) -> bool:
        return bool(self.added or self.changed or self.removed)

    def summary(self) -> str:
        return (
            f"new={len(self.added)}  changed={len(self.changed)}  "
            f"unchanged={len(self.unchanged)}  removed={len(self.removed)}"
        )


def plan_delta(current: dict[str, str], previous: dict[str, str]) -> DeltaPlan:
    plan = DeltaPlan()

    for key, digest in current.items():
        if key not in previous:
            plan.added[key] = digest
        elif previous[key] != digest:
            plan.changed[key] = digest
        else:
            plan.unchanged[key] = digest

    for key, digest in previous.items():
        if key not in current:
            plan.removed[key] = digest

    return plan


# ── State store ───────────────────────────────────────────────────────────────
class StateStore:
    """
    Fingerprint + progress state for one source ("regulations", "sop",
    "deviations").

    Files written per source, in data/.index_state/:

        <source>_vectordb_update_last_update_info.txt
            Human-readable marker: last successful run, and the mode it ran in.

        <source>_tracked_document_names.json
            The real state: {key: content_hash} for everything currently
            indexed, plus the point IDs each key owns so that a shrinking
            document (10 chunks → 6) has its 4 orphans deleted rather than
            left behind as stale hits.

        <source>_init_in_progress.json
            Written at the start of a FULL load and deleted at the end. Its
            presence on startup means the previous full load crashed part-way;
            the keys already committed inside it are treated as done, so the
            rerun resumes instead of paying for the whole corpus again.
    """

    def __init__(self, source: str):
        self.source = source
        STATE_DIR.mkdir(parents=True, exist_ok=True)

        self.marker_path  = STATE_DIR / f"{source}_vectordb_update_last_update_info.txt"
        self.tracked_path = STATE_DIR / f"{source}_tracked_document_names.json"
        self.init_path    = STATE_DIR / f"{source}_init_in_progress.json"

        self._tracked = self._read_json(self.tracked_path, default={})

    # -- persistence helpers --
    @staticmethod
    def _read_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! Could not read {path.name} ({exc}); treating as empty.")
            return default

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        """Write via a temp file + replace so a crash cannot leave torn JSON."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    # -- fingerprints --
    @property
    def fingerprints(self) -> dict[str, str]:
        return {key: entry["hash"] for key, entry in self._tracked.items()}

    def point_ids_for(self, key: str) -> list[str]:
        return list(self._tracked.get(key, {}).get("point_ids", []))

    def meta(self, key: str) -> dict[str, Any]:
        """Everything recorded for a key — extra fields passed to record()."""
        return dict(self._tracked.get(key, {}))

    def all_point_ids(self) -> set[str]:
        ids: set[str] = set()
        for entry in self._tracked.values():
            ids.update(entry.get("point_ids", []))
        return ids

    def record(self, key: str, digest: str, point_ids: list[str], **meta: Any) -> None:
        """Mark a key as indexed. Call after its points are safely upserted."""
        self._tracked[key] = {
            "hash"      : digest,
            "point_ids" : point_ids,
            "indexed_at": _now_iso(),
            **meta,
        }

    def forget(self, key: str) -> list[str]:
        """Drop a key from state and return the point IDs it owned."""
        entry = self._tracked.pop(key, {})
        return list(entry.get("point_ids", []))

    def save(self) -> None:
        self._write_json(self.tracked_path, self._tracked)

    def reset(self) -> None:
        """Forget everything — used when a full load starts from scratch."""
        self._tracked = {}

    @property
    def is_initialised(self) -> bool:
        """True when a previous run has indexed at least one unit."""
        return bool(self._tracked)

    # -- crash recovery --
    def begin_init(self, total: int) -> dict[str, str]:
        """
        Start (or resume) a full load. Returns the fingerprints already
        committed by a previous crashed attempt — empty on a clean start.
        """
        existing = self._read_json(self.init_path, default=None)

        if existing:
            done = existing.get("committed", {})
            print(
                f"  Found an interrupted full load from {existing.get('started_at')} "
                f"— resuming with {len(done)} unit(s) already uploaded."
            )
            return done

        self._write_json(self.init_path, {
            "started_at": _now_iso(),
            "total"     : total,
            "committed" : {},
        })
        return {}

    def commit_init(self, key: str, digest: str) -> None:
        """Record one unit as durably uploaded, for crash recovery."""
        state = self._read_json(self.init_path, default=None)
        if state is None:
            return
        state["committed"][key] = digest
        self._write_json(self.init_path, state)

    def end_init(self) -> None:
        self.init_path.unlink(missing_ok=True)

    @property
    def init_interrupted(self) -> bool:
        return self.init_path.exists()

    # -- run marker --
    def write_marker(self, mode: str, stats: dict[str, Any]) -> None:
        last_update = datetime.now(timezone.utc) - timedelta(hours=LAST_UPDATE_BUFFER_HOURS)
        lines = [
            f"source           = {self.source}",
            f"mode             = {mode}",
            f"finished_at      = {_now_iso()}",
            f"last_update_time = {last_update.replace(microsecond=0).isoformat()}",
            f"tracked_units    = {len(self._tracked)}",
        ]
        lines += [f"{key:<17}= {value}" for key, value in stats.items()]
        self.marker_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def read_last_update_time(self) -> str | None:
        if not self.marker_path.exists():
            return None
        for line in self.marker_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("last_update_time"):
                return line.split("=", 1)[1].strip()
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ── Console helpers ───────────────────────────────────────────────────────────
def banner(title: str, rows: Iterable[tuple[str, Any]] = ()) -> None:
    print("=" * 68)
    print(title)
    print("=" * 68)
    for label, value in rows:
        print(f"{label:<18}: {value}")
    print()
