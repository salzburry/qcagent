"""
Deterministic evidence merger — resolves author + auditor into final candidates.

Rule-based, not generative. Never fabricates consensus.

Rules:
  - accept: keep candidate as-is
  - reject: drop candidate, log reason
  - repair: keep candidate but lower confidence, attach auditor note
  - no audit verdict: keep candidate (auditor failure is non-fatal)
  - unresolved contradictions: keep all candidates, mark for human review
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from ..schemas.evidence import EvidenceCandidate
from .evidence_auditor import AuditResult, CandidateAudit

logger = logging.getLogger(__name__)

# Confidence penalty applied to "repair" verdicts
REPAIR_CONFIDENCE_PENALTY = 0.2

# Floor confidence for repaired candidates (never go below this)
REPAIR_CONFIDENCE_FLOOR = 0.1


@dataclass
class MergeResult:
    """Output of the deterministic merger."""
    accepted: list[EvidenceCandidate]
    rejected: list[EvidenceCandidate]
    repaired: list[EvidenceCandidate]
    has_contradictions: bool
    auditor_notes: list[str]


def merge_candidates(
    candidates: list[EvidenceCandidate],
    audit: AuditResult,
    concept: str,
) -> MergeResult:
    """Merge author candidates with auditor verdicts.

    Deterministic rules only — no LLM call.

    Args:
        candidates: Original EvidenceCandidate list from the author/extractor
        audit: AuditResult from the evidence auditor
        concept: Concept name for logging

    Returns:
        MergeResult with accepted/rejected/repaired candidates.
    """
    # Build audit lookup by candidate index
    audit_by_idx: dict[int, CandidateAudit] = {}
    for a in audit.audits:
        audit_by_idx[a.candidate_index] = a

    accepted = []
    rejected = []
    repaired = []
    notes = []
    has_contradictions = False

    for i, candidate in enumerate(candidates):
        verdict_info = audit_by_idx.get(i)

        if verdict_info is None:
            # No audit verdict for this candidate — keep it (non-fatal)
            accepted.append(candidate)
            continue

        if verdict_info.contradicts_others:
            has_contradictions = True

        if verdict_info.verdict == "accept":
            accepted.append(candidate)

        elif verdict_info.verdict == "reject":
            rejected.append(candidate)
            reason = verdict_info.repair_note or "Auditor rejected"
            notes.append(
                f"Candidate {i} rejected: {reason} "
                f"(quote_found={verdict_info.quote_found}, "
                f"value_compatible={verdict_info.value_compatible})"
            )
            logger.info(f"[{concept}] Candidate {i} REJECTED: {reason}")

        elif verdict_info.verdict == "repair":
            # Lower confidence, attach note, keep candidate
            new_confidence = max(
                (candidate.llm_confidence or 0.5) - REPAIR_CONFIDENCE_PENALTY,
                REPAIR_CONFIDENCE_FLOOR,
            )
            # Create a new candidate with lowered confidence
            repaired_candidate = candidate.model_copy(update={
                "llm_confidence": new_confidence,
            })
            repaired.append(repaired_candidate)
            if verdict_info.repair_note:
                notes.append(
                    f"Candidate {i} repaired: {verdict_info.repair_note} "
                    f"(confidence {candidate.llm_confidence:.2f} → {new_confidence:.2f})"
                )
            logger.info(
                f"[{concept}] Candidate {i} REPAIRED: "
                f"confidence {candidate.llm_confidence:.2f} → {new_confidence:.2f}"
            )

        else:
            # Unknown verdict — keep candidate
            accepted.append(candidate)
            notes.append(f"Candidate {i}: unknown verdict '{verdict_info.verdict}', keeping")

    if audit.overall_note:
        notes.append(f"Auditor note: {audit.overall_note}")

    total = len(candidates)
    logger.info(
        f"[{concept}] Merge complete: "
        f"{len(accepted)} accepted, {len(repaired)} repaired, "
        f"{len(rejected)} rejected out of {total} | "
        f"contradictions={has_contradictions}"
    )

    return MergeResult(
        accepted=accepted,
        rejected=rejected,
        repaired=repaired,
        has_contradictions=has_contradictions,
        auditor_notes=notes,
    )
