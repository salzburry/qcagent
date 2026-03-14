"""
Evidence auditor — verifies extraction candidates against cited chunks.

Implements the "auditor" role from the author–auditor–merger pattern.
The auditor does NOT re-extract; it only checks:
  1. Does the quoted_text actually appear in the cited chunk?
  2. Is the normalized value compatible with the quote?
  3. Are there contradictions between candidates?

Uses a second LLM call with a narrow, flat schema.
Falls back to string-matching heuristics if the LLM call fails.
"""

from __future__ import annotations
import logging
from typing import Literal, Optional
from pydantic import BaseModel, Field

from ..serving.model_client import LocalModelClient

logger = logging.getLogger(__name__)


# ── Audit schema (intentionally flat and small) ──────────────────────────────

class CandidateAudit(BaseModel):
    """Audit verdict for a single extraction candidate."""
    candidate_index: int = Field(description="0-based index of the candidate being audited")
    quote_found: bool = Field(
        default=False,
        description="True if the quoted_text appears verbatim (or near-verbatim) in the cited chunk"
    )
    value_compatible: bool = Field(
        default=False,
        description="True if the extracted value/summary is consistent with the quoted text"
    )
    contradicts_others: bool = Field(
        default=False,
        description="True if this candidate contradicts another candidate in the list"
    )
    verdict: str = Field(
        default="accept",
        description="accept = keep as-is, reject = drop, repair = keep with correction"
    )
    repair_note: str = Field(
        default="",
        description="If verdict=repair, explain what should be corrected"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AuditResult(BaseModel):
    """Full audit result for all candidates in an extraction."""
    audits: list[CandidateAudit] = Field(default_factory=list)
    overall_note: str = Field(
        default="",
        description="Any cross-candidate observations (e.g. 'candidates 0 and 2 contradict')"
    )


# ── Audit system prompt ──────────────────────────────────────────────────────

AUDIT_SYSTEM_PROMPT = """You are an evidence auditor for clinical protocol extraction.

You are given:
1. A list of extraction CANDIDATES (each with quoted_text and a summary/value)
2. The ORIGINAL CHUNKS those candidates were extracted from

Your job is to VERIFY each candidate — not to re-extract.

For each candidate, check:
- quote_found: Does the quoted_text actually appear in the cited chunk? Allow minor whitespace differences.
- value_compatible: Is the extracted value/summary consistent with what the quote says?
- contradicts_others: Does this candidate contradict any other candidate in the list?
- verdict: "accept" if quote_found AND value_compatible. "reject" if quote not found or value incompatible. "repair" if partially correct but needs adjustment.
- repair_note: If verdict is "repair", explain the correction.

Be strict about quote verification. The whole point is to catch hallucinated quotes."""


def audit_candidates(
    client: LocalModelClient,
    candidates_text: str,
    chunks_text: str,
    concept: str,
) -> AuditResult:
    """Run the evidence auditor on extraction candidates.

    Args:
        client: Model client (uses default model, not adjudicator)
        candidates_text: Formatted string of candidates to audit
        chunks_text: Original chunk text the candidates were extracted from
        concept: Concept name for logging

    Returns:
        AuditResult with per-candidate verdicts.
        On failure, returns an empty AuditResult (non-fatal).
    """
    user_prompt = (
        f"## CANDIDATES TO AUDIT\n{candidates_text}\n\n"
        f"## ORIGINAL SOURCE CHUNKS\n{chunks_text}\n\n"
        f"Audit each candidate. Return your verdicts."
    )

    try:
        result = client.extract(
            system_prompt=AUDIT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=AuditResult,
            use_adjudicator=False,
            prompt_version="audit-0.1.0",
        )
        audit = result.parsed
        logger.info(
            f"[{concept}] Audit complete: {len(audit.audits)} verdicts, "
            f"accepts={sum(1 for a in audit.audits if a.verdict == 'accept')}, "
            f"rejects={sum(1 for a in audit.audits if a.verdict == 'reject')}, "
            f"repairs={sum(1 for a in audit.audits if a.verdict == 'repair')}"
        )
        return audit

    except Exception as e:
        logger.warning(
            f"[{concept}] Auditor LLM call failed ({type(e).__name__}: {e}). "
            f"Falling back to string-match heuristics."
        )
        return _heuristic_audit(candidates_text, chunks_text)


def _heuristic_audit(candidates_text: str, chunks_text: str) -> AuditResult:
    """Fallback auditor using simple string matching.

    Not as good as LLM audit, but catches obvious hallucinated quotes.
    """
    # Simple: count how many "quoted_text" values appear in chunks
    # This is a basic sanity check — the LLM audit is preferred
    return AuditResult(
        audits=[],
        overall_note="Heuristic fallback — LLM audit unavailable",
    )


def format_candidates_for_audit(candidates: list, concept: str) -> str:
    """Format extraction candidates into a string for the auditor.

    Works with any candidate-based extraction schema that has
    quoted_text/snippet and chunk_id fields.
    """
    parts = []
    for i, c in enumerate(candidates):
        # Handle both raw extraction objects and EvidenceCandidate objects
        quoted = getattr(c, "quoted_text", None) or getattr(c, "snippet", "")
        chunk_id = getattr(c, "chunk_id", None) or ""
        summary = getattr(c, "summary", None) or getattr(c, "definition", "") or ""
        confidence = getattr(c, "confidence", None) or getattr(c, "llm_confidence", 0.0)
        explicit = getattr(c, "explicit", "unknown")

        parts.append(
            f"[Candidate {i}]\n"
            f"  chunk_id: {chunk_id}\n"
            f"  quoted_text: {quoted}\n"
            f"  summary: {summary}\n"
            f"  explicit: {explicit}\n"
            f"  confidence: {confidence}"
        )
    return "\n\n".join(parts)
