"""
Evaluation harness.
Scores concept finders against gold set.
Must be built before accuracy claims can be made.

Three accuracy tiers measured separately:
  1. Retrieval recall   — did the right chunk appear in candidates?
  2. Selection accuracy — did top candidate match gold?
  3. Row accuracy       — did completed row match gold row? (Phase 2)
"""

from __future__ import annotations
import json
import csv
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime


# ── Gold set schema ───────────────────────────────────────────────────────────

@dataclass
class GoldRecord:
    """
    One ground-truth record from a completed spec.
    Manually created by reviewing completed spec + source protocol.
    """
    protocol_id: str
    concept: str

    # Ground truth evidence
    gold_snippet: str               # Exact text from protocol that governs this concept
    gold_section: str               # Section heading where it was found
    gold_page: Optional[int]        # Page number
    gold_explicit: str              # explicit | inferred | assumed | ambiguous

    # Ground truth spec row value
    gold_row_value: str             # What the programmer wrote in the spec

    # Metadata
    required_sponsor_clarification: bool = False   # Was sponsor contact needed?
    notes: str = ""
    created_by: str = ""
    created_date: str = ""


@dataclass
class EvalResult:
    protocol_id: str
    concept: str
    finder_version: str
    model_used: str
    prompt_version: str

    # Tier 1: Retrieval recall — did the right chunk appear?
    gold_in_candidates: bool = False        # Did any candidate contain gold snippet?
    gold_in_top3: bool = False              # Was it in top 3?
    top_candidate_matches_gold: bool = False  # Top-1 retrieval quality

    # Tier 2: Top candidate quality
    snippet_similarity: float = 0.0         # Simple overlap score 0-1
    section_match: bool = False
    page_match: bool = False
    explicit_type_correct: bool = False

    # Tier 3: Row accuracy (Phase 2)
    row_value_match: Optional[bool] = None  # Not yet implemented

    # Run metadata
    n_candidates: int = 0
    contradictions_detected: bool = False
    adjudicator_used: bool = False
    eval_date: str = ""


# ── Gold set builder ──────────────────────────────────────────────────────────

GOLD_SET_TEMPLATE_HEADERS = [
    "protocol_id",
    "concept",
    "gold_snippet",
    "gold_section",
    "gold_page",
    "gold_explicit",
    "gold_row_value",
    "required_sponsor_clarification",
    "notes",
    "created_by",
    "created_date",
]

def create_gold_set_template(
    output_path: str,
    protocol_ids: list[str],
    concepts: Optional[list[str]] = None,
):
    """
    Create a CSV template for manual gold set collection.
    One row per (protocol, concept) pair.
    Fill in manually by reviewing completed specs + source protocols.
    """
    concepts_to_collect = concepts or [
        "index_date",
        "follow_up_end",
        "primary_endpoint",
    ]

    rows = []
    for pid in protocol_ids:
        for concept in concepts_to_collect:
            rows.append({
                "protocol_id": pid,
                "concept": concept,
                "gold_snippet": "",         # FILL: exact text from protocol
                "gold_section": "",         # FILL: section heading
                "gold_page": "",            # FILL: page number
                "gold_explicit": "",        # FILL: explicit|inferred|assumed|ambiguous
                "gold_row_value": "",       # FILL: what programmer wrote in spec
                "required_sponsor_clarification": "",  # FILL: yes|no
                "notes": "",
                "created_by": "",
                "created_date": "",
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GOLD_SET_TEMPLATE_HEADERS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Gold set template created: {output_path}")
    print(f"Fill in {len(rows)} rows manually from completed specs + protocols.")


def load_gold_set(csv_path: str) -> list[GoldRecord]:
    records = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if not row["gold_snippet"]:
                continue    # Skip unfilled rows
            records.append(GoldRecord(
                protocol_id=row["protocol_id"],
                concept=row["concept"],
                gold_snippet=row["gold_snippet"],
                gold_section=row["gold_section"],
                gold_page=int(row["gold_page"]) if row["gold_page"] else None,
                gold_explicit=row["gold_explicit"],
                gold_row_value=row["gold_row_value"],
                required_sponsor_clarification=row.get("required_sponsor_clarification", "").lower() == "yes",
                notes=row.get("notes", ""),
                created_by=row.get("created_by", ""),
                created_date=row.get("created_date", ""),
            ))
    return records


# ── Evaluation ────────────────────────────────────────────────────────────────

def _snippet_overlap(pred: str, gold: str) -> float:
    """Simple word overlap score."""
    pred_words = set(pred.lower().split())
    gold_words = set(gold.lower().split())
    if not gold_words:
        return 0.0
    return len(pred_words & gold_words) / len(gold_words)


def evaluate_pack(
    pack_dict: dict,
    gold: GoldRecord,
) -> EvalResult:
    from ..schemas.evidence import EvidencePack
    pack = EvidencePack.model_validate(pack_dict)

    result = EvalResult(
        protocol_id=gold.protocol_id,
        concept=gold.concept,
        finder_version=pack.finder_version,
        model_used=pack.model_used,
        prompt_version=pack.prompt_version,
        n_candidates=len(pack.candidates),
        contradictions_detected=pack.contradictions_found,
        adjudicator_used=pack.adjudicator_used,
        eval_date=datetime.now().isoformat(),
    )

    if not pack.candidates:
        return result

    # Tier 1: retrieval recall
    for i, candidate in enumerate(pack.candidates):
        overlap = _snippet_overlap(candidate.snippet, gold.gold_snippet)
        if overlap > 0.5:
            result.gold_in_candidates = True
            if i < 3:
                result.gold_in_top3 = True
            if i == 0:
                result.top_candidate_matches_gold = True
            break

    # Tier 2: top candidate quality
    if pack.candidates:
        top = pack.candidates[0]
        result.snippet_similarity = _snippet_overlap(top.snippet, gold.gold_snippet)
        result.section_match = (
            gold.gold_section.lower() in (top.section_title or "").lower()
        )
        result.page_match = (top.page == gold.gold_page) if gold.gold_page else True
        result.explicit_type_correct = (top.explicit == gold.gold_explicit)

    return result


def run_evaluation(
    gold_csv: str,
    packs_dir: str,
    output_csv: str,
) -> dict:
    """
    Run full evaluation against gold set.
    packs_dir: directory containing {protocol_id}_evidence_packs.json files
    """
    gold_records = load_gold_set(gold_csv)
    results = []

    for gold in gold_records:
        pack_file = Path(packs_dir) / f"{gold.protocol_id}_evidence_packs.json"
        if not pack_file.exists():
            print(f"[Eval] No pack file for {gold.protocol_id}, skipping.")
            continue

        with open(pack_file) as f:
            run_output = json.load(f)

        packs_dict = run_output.get("evidence_packs", {})
        if gold.concept not in packs_dict:
            print(f"[Eval] Concept {gold.concept} not in packs for {gold.protocol_id}")
            continue

        result = evaluate_pack(packs_dict[gold.concept], gold)
        results.append(result)

    # Write results
    if results:
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))

    # Aggregate metrics
    if not results:
        return {}

    n = len(results)
    metrics = {
        "n_evaluated": n,
        "retrieval_recall": sum(r.gold_in_candidates for r in results) / n,
        "top3_recall": sum(r.gold_in_top3 for r in results) / n,
        "top1_retrieval_quality": sum(r.top_candidate_matches_gold for r in results) / n,
        "mean_snippet_similarity": sum(r.snippet_similarity for r in results) / n,
        "section_match_rate": sum(r.section_match for r in results) / n,
        "explicit_type_accuracy": sum(r.explicit_type_correct for r in results) / n,
        "contradiction_detection_rate": sum(r.contradictions_detected for r in results) / n,
    }

    print("\n=== EVALUATION RESULTS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.2%}" if isinstance(v, float) else f"  {k}: {v}")

    return metrics
