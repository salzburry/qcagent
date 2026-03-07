"""
Deterministic QC engine.
Rule-based — no LLM involved.
Runs after evidence packs are resolved and rows are written.
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
    detail: Optional[str] = None


def qc_evidence_packs(packs: dict[str, EvidencePack]) -> list[QCResult]:
    """
    QC on evidence packs — before row writing.
    Checks: completeness, resolution state, evidence traceability.
    """
    results = []

    for concept, pack in packs.items():

        # Every pack must have at least one candidate
        if not pack.candidates:
            results.append(QCResult(
                rule_id="QC-001",
                level="error",
                concept=concept,
                message=f"No evidence candidates found for {concept}.",
                detail="Check retrieval — relevant protocol sections may not be indexed."
            ))

        # Warn on low retrieval signal
        if pack.low_retrieval_signal:
            results.append(QCResult(
                rule_id="QC-002",
                level="warning",
                concept=concept,
                message=f"Low retrieval signal for {concept}.",
                detail="Fewer than 3 chunks retrieved. May be in appendix or non-standard section."
            ))

        # Warn on contradictions
        if pack.contradictions_found:
            results.append(QCResult(
                rule_id="QC-003",
                level="warning",
                concept=concept,
                message=f"Contradictions detected in {concept} definition.",
                detail=pack.contradiction_detail,
            ))

        # Warn on unresolved packs
        if not pack.is_resolved:
            results.append(QCResult(
                rule_id="QC-004",
                level="warning",
                concept=concept,
                message=f"{concept} not yet reviewed — no candidate selected.",
                detail="Requires human selection before row can be written."
            ))

        # Check evidence has page refs
        if pack.candidates and all(c.page is None for c in pack.candidates):
            results.append(QCResult(
                rule_id="QC-005",
                level="info",
                concept=concept,
                message=f"No page references in {concept} evidence.",
                detail="Page numbers missing — traceability limited."
            ))

    return results


def qc_cross_concept(packs: dict[str, EvidencePack]) -> list[QCResult]:
    """
    Cross-concept consistency checks.
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
            message="follow_up_start defined but follow_up_end not found.",
            detail="Cannot define observation window without both start and end."
        ))

    # Check primary endpoint exists if study has follow-up
    if "follow_up_end" in packs and "primary_endpoint" not in packs:
        results.append(QCResult(
            rule_id="QC-C002",
            level="warning",
            concept="primary_endpoint",
            message="Follow-up defined but primary endpoint not found.",
        ))

    # Check index_date exists — it's foundational
    if "index_date" not in packs:
        results.append(QCResult(
            rule_id="QC-C003",
            level="error",
            concept="index_date",
            message="index_date not found. All time-anchored concepts depend on this.",
        ))

    return results


def qc_missing_concepts(
    packs: dict[str, EvidencePack],
    expected_concepts: list[str],
) -> list[QCResult]:
    """Flag expected concepts that weren't extracted."""
    results = []
    found = set(packs.keys())
    for concept in expected_concepts:
        if concept not in found:
            results.append(QCResult(
                rule_id="QC-M001",
                level="warning",
                concept=concept,
                message=f"Expected concept '{concept}' not extracted.",
                detail="May need manual extraction or protocol does not address this concept."
            ))
    return results


def run_all_qc(
    packs: dict[str, EvidencePack],
    expected_concepts: Optional[list[str]] = None,
) -> list[QCResult]:
    results = []
    results.extend(qc_evidence_packs(packs))
    results.extend(qc_cross_concept(packs))
    if expected_concepts:
        results.extend(qc_missing_concepts(packs, expected_concepts))
    return results


def summarize_qc(results: list[QCResult]) -> str:
    errors = [r for r in results if r.level == "error"]
    warnings = [r for r in results if r.level == "warning"]
    infos = [r for r in results if r.level == "info"]

    lines = [
        f"=== QC Summary: {len(errors)} errors | {len(warnings)} warnings | {len(infos)} info ===",
    ]
    for r in results:
        icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}.get(r.level, "")
        lines.append(f"{icon} [{r.rule_id}] {r.concept}: {r.message}")
        if r.detail:
            lines.append(f"   → {r.detail}")
    return "\n".join(lines)
