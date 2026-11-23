"""
eval_cag.py — does the cached core make answers better, or only different?

Asks a fixed question set twice, with the core library (CAG-RAG) and without it
(plain RAG), and scores each answer on what can be checked without a judge:

  * hit       — the document that should answer the question was cited
  * facts     — the key facts expected in the answer are all present
  * rejected  — citations validation had to strip (the model invented them)
  * evidence  — the model claimed sufficient evidence
  * latency, prompt / cached tokens, and OpenRouter's billed cost

    python -m scripts.eval_cag                  # both modes, all questions
    python -m scripts.eval_cag --limit 3        # quick trial
    python -m scripts.eval_cag --out data/eval/run.json

Half the questions are answered by core documents, half by the long tail, so
the run shows both what the core adds and whether it distracts from RAG.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# (question, expected document_ids — any one cited is a hit, facts — all must appear)
CASES = [
    # ── Answered by core documents ──
    ("How long must batch documentation be retained under EU GMP?",
     ["EUDRALEX_EU_GMP_CHAPTER_4"], ["one year", "five years"]),
    ("What does 21 CFR 211 require for equipment cleaning and maintenance?",
     ["FDA_21_CFR_PART_211"], ["211.67"]),
    ("What are the two primary principles of quality risk management in ICH Q9?",
     ["ICH_ICH_Q9_R1"], ["patient", "commensurate"]),
    ("What does EU GMP require for self inspections?",
     ["EUDRALEX_EU_GMP_CHAPTER_9"], ["independent"]),
    ("What does EU GMP require when a product recall is decided?",
     ["EUDRALEX_EU_GMP_CHAPTER_8"], ["recall"]),
    ("What should management review cover according to ICH Q10?",
     ["ICH_ICH_Q10"], ["management review"]),
    # ── Answered by the long tail (RAG) ──
    ("What are WHO recommendations for hold time studies?",
     ["WHO_TRS_992_ANNEX_4"], ["hold time"]),
    ("How should a laboratory investigation of an OOS result be conducted per FDA?",
     ["FDA_FDA_INVESTIGATING_OOS_TEST_RESULTS"], ["laboratory investigation"]),
    ("What is a contamination control strategy under EU GMP Annex 1?",
     ["EUDRALEX_01_MANUFACTURE_OF_STERILE_MEDICINAL_PRODUCTS"],
     ["contamination control strategy"]),
    ("What does ALCOA mean in MHRA data integrity guidance?",
     ["MHRA_MHRA_GXP_DATA_INTEGRITY_GUIDANCE"], ["attributable"]),
    ("What does EU GMP Annex 11 require for audit trails?",
     ["EUDRALEX_09_COMPUTERISED_SYSTEMS"], ["audit trail"]),
    # ── Both: a core and a long-tail source ──
    ("Compare EU GMP and 21 CFR 211 requirements for laboratory records.",
     ["EUDRALEX_EU_GMP_CHAPTER_6", "FDA_21_CFR_PART_211"], ["211.194"]),
]


def score(response, expected_ids: list[str], facts: list[str]) -> dict:
    """What can be checked mechanically about one answer."""
    cited = {source.document_id for source in response.sources if source.cited}
    answer = response.answer.lower()
    return {
        "hit": bool(cited & set(expected_ids)),
        "facts": all(fact.lower() in answer for fact in facts),
        "rejected": len(response.rejected_citations),
        "evidence": response.has_sufficient_evidence,
        "cited": sorted(cited),
    }


def run(limit: int | None) -> list[dict]:
    from app.cag.core import CoreLibrary
    from app.chat import service as chat_service
    from app.chat.service import AskGMPService
    from app.retrieval.retriever import Retriever

    retriever = Retriever()
    modes = {
        "cag-rag": AskGMPService(retriever=retriever, core=CoreLibrary()),
        "rag":     AskGMPService(retriever=retriever, core=None),
    }

    rows = []
    for question, expected, facts in CASES[:limit]:
        for mode, service in modes.items():
            chat_service.last_usage = None
            started = time.perf_counter()
            response = service.answer(question)
            usage = chat_service.last_usage or {}
            rows.append({
                "mode": mode,
                "question": question,
                **score(response, expected, facts),
                "seconds": round(time.perf_counter() - started, 1),
                "prompt_tokens": usage.get("prompt_tokens"),
                "cached_tokens": usage.get("cached_tokens"),
                "cost": usage.get("cost"),
            })
            r = rows[-1]
            print(f"{mode:<8} hit={r['hit']!s:<5} facts={r['facts']!s:<5} "
                  f"rej={r['rejected']} {r['seconds']:>5}s "
                  f"cached={r['cached_tokens']} ${r['cost'] or 0:.4f}  {question[:60]}",
                  flush=True)
    return rows


def summarise(rows: list[dict]) -> None:
    print("\nmode      hit    facts  rejected  evidence  avg s   total $")
    for mode in ("cag-rag", "rag"):
        mine = [r for r in rows if r["mode"] == mode]
        if not mine:
            continue
        n = len(mine)
        print(f"{mode:<8} {sum(r['hit'] for r in mine):>2}/{n:<3} "
              f"{sum(r['facts'] for r in mine):>2}/{n:<3} "
              f"{sum(r['rejected'] for r in mine):>8}  "
              f"{sum(r['evidence'] for r in mine):>5}/{n:<3} "
              f"{sum(r['seconds'] for r in mine) / n:>5.1f}  "
              f"{sum(r['cost'] or 0 for r in mine):>8.4f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare CAG-RAG with plain RAG.")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--out", type=Path, help="write every row as JSON here")
    args = parser.parse_args(argv)

    rows = run(args.limit)
    summarise(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nRows written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
