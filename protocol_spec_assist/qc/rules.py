"""
Deterministic QC engine.
Rule-based — no LLM involved.
Split into pre-review and post-review stages.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..schemas.evidence import EvidencePack


@dataclass
class QCResult:
    rule_id: str
    level: str              # error | warning | info
    concept: str
    message: str
    stage: str = "pre_review"   # pre_review | post_review
    detail: Optional[str] = None


# ── Pre-review QC (runs immediately after extraction) ────────────────────────

def qc_pre_review(packs: dict[str, EvidencePack]) -> list[QCResult]:
    """
    QC checks that make sense BEFORE human review.
    These flag issues the reviewer should know about.
    """
    results = []

    for concept, pack in packs.items():

        # No candidates at all — retrieval or extraction failed
        if not pack.candidates:
            results.append(QCResult(
                rule_id="QC-001",
                level="error",
                concept=concept,
                stage="pre_review",
                message=f"No evidence candidates found for {concept}.",
                detail="Check retrieval — relevant protocol sections may not be indexed."
            ))

        # Warn on low retrieval signal
        if pack.low_retrieval_signal:
            results.append(QCResult(
                rule_id="QC-002",
                level="warning",
                concept=concept,
                stage="pre_review",
                message=f"Low retrieval signal for {concept}.",
                detail="Fewer than 3 chunks retrieved. May be in appendix or non-standard section."
            ))

        # Warn on contradictions
        if pack.contradictions_found:
            results.append(QCResult(
                rule_id="QC-003",
                level="warning",
                concept=concept,
                stage="pre_review",
                message=f"Contradictions detected in {concept} definition.",
                detail=pack.contradiction_detail,
            ))

        # Check evidence has page refs
        if pack.candidates and all(c.page is None for c in pack.candidates):
            results.append(QCResult(
                rule_id="QC-005",
                level="info",
                concept=concept,
                stage="pre_review",
                message=f"No page references in {concept} evidence.",
                detail="Page numbers missing — traceability limited."
            ))

    return results


# ── Post-review QC (runs after human review) ─────────────────────────────────

def qc_post_review(packs: dict[str, EvidencePack]) -> list[QCResult]:
    """
    QC checks that only make sense AFTER human review.
    These validate that review was completed and results are consistent.
    """
    results = []

    for concept, pack in packs.items():

        # Unresolved packs — only meaningful after review
        if not pack.is_resolved:
            results.append(QCResult(
                rule_id="QC-004",
                level="warning",
                concept=concept,
                stage="post_review",
                message=f"{concept} not yet reviewed — no candidate selected.",
                detail="Requires human selection before row can be written."
            ))

    return results


def qc_cross_concept(packs: dict[str, EvidencePack], stage: str = "pre_review") -> list[QCResult]:
    """
    Cross-concept consistency checks. Stage-neutral — applies to both pre and post review.
    Dependencies:
    - follow_up_start should reference same anchor as index_date
    - follow_up_end should be after follow_up_start
    - baseline_window should be anchored to index_date
    """
    results = []

    # Check follow_up_end exists if follow_up_start exists
    if "follow_up_start" in packs and "follow_up_end" not in packs:
        results.append(QCResult(
            rule_id="QC-C001",
            level="error",
            concept="follow_up_end",
            stage=stage,
            message="follow_up_start defined but follow_up_end not found.",
            detail="Cannot define observation window without both start and end."
        ))

    # Check primary endpoint exists if study has follow-up
    if "follow_up_end" in packs and "primary_endpoint" not in packs:
        results.append(QCResult(
            rule_id="QC-C002",
            level="warning",
            concept="primary_endpoint",
            stage=stage,
            message="Follow-up defined but primary endpoint not found.",
        ))

    # Check index_date exists — it's foundational
    if "index_date" not in packs:
        results.append(QCResult(
            rule_id="QC-C003",
            level="error",
            concept="index_date",
            stage=stage,
            message="index_date not found. All time-anchored concepts depend on this.",
        ))

    return results


def qc_missing_concepts(
    packs: dict[str, EvidencePack],
    expected_concepts: list[str],
    implemented_concepts: Optional[list[str]] = None,
) -> list[QCResult]:
    """Flag expected concepts that weren't extracted.
    Only checks against implemented_concepts if provided, to avoid
    noisy warnings for concepts that haven't been built yet."""
    results = []
    found = set(packs.keys())

    for concept in expected_concepts:
        # Skip concepts that aren't implemented yet
        if implemented_concepts and concept not in implemented_concepts:
            continue
        if concept not in found:
            results.append(QCResult(
                rule_id="QC-M001",
                level="warning",
                concept=concept,
                stage="post_review",
                message=f"Expected concept '{concept}' not extracted.",
                detail="May need manual extraction or protocol does not address this concept."
            ))
    return results


def qc_quote_in_chunk(
    packs: dict[str, EvidencePack],
    chunk_lookup: Optional[dict[str, str]] = None,
) -> list[QCResult]:
    """Validate that candidate quoted_text actually appears in its source chunk.
    chunk_lookup: {chunk_id: chunk_text} from the indexed chunks.
    If not provided, this check is skipped."""
    if not chunk_lookup:
        return []

    results = []
    for concept, pack in packs.items():
        for candidate in pack.candidates:
            if not candidate.chunk_id or candidate.chunk_id not in chunk_lookup:
                continue
            chunk_text = chunk_lookup[candidate.chunk_id]
            # Normalize whitespace for comparison
            snippet_norm = " ".join(candidate.snippet.split()).lower()
            chunk_norm = " ".join(chunk_text.split()).lower()
            if snippet_norm not in chunk_norm:
                results.append(QCResult(
                    rule_id="QC-006",
                    level="warning",
                    concept=concept,
                    stage="pre_review",
                    message=f"Candidate {candidate.candidate_id}: quoted text not found in source chunk.",
                    detail=(
                        f"chunk_id={candidate.chunk_id}, "
                        f"snippet[:60]='{candidate.snippet[:60]}...'"
                    ),
                ))
    return results


# Implemented concepts
IMPLEMENTED_CONCEPTS = [
    "index_date", "follow_up_end", "primary_endpoint",
    "eligibility_inclusion", "eligibility_exclusion",
    "study_period", "censoring_rules",
]


def run_all_qc(
    packs: dict[str, EvidencePack],
    expected_concepts: Optional[list[str]] = None,
    stage: str = "pre_review",
) -> list[QCResult]:
    """Run QC appropriate to the current stage."""
    results = []

    if stage == "pre_review":
        results.extend(qc_pre_review(packs))
        results.extend(qc_cross_concept(packs, stage="pre_review"))
    elif stage == "post_review":
        results.extend(qc_post_review(packs))
        results.extend(qc_cross_concept(packs, stage="post_review"))
        if expected_concepts:
            results.extend(qc_missing_concepts(
                packs, expected_concepts,
                implemented_concepts=IMPLEMENTED_CONCEPTS,
            ))

    return results


def summarize_qc(results: list[QCResult]) -> str:
    errors = [r for r in results if r.level == "error"]
    warnings = [r for r in results if r.level == "warning"]
    infos = [r for r in results if r.level == "info"]

    lines = [
        f"=== QC Summary: {len(errors)} errors | {len(warnings)} warnings | {len(infos)} info ===",
    ]
    for r in results:
        icon = {"error": "X", "warning": "!", "info": "i"}.get(r.level, "")
        lines.append(f"[{icon}] [{r.rule_id}] ({r.stage}) {r.concept}: {r.message}")
        if r.detail:
            lines.append(f"   -> {r.detail}")
    return "\n".join(lines)
