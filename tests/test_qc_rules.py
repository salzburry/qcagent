"""Tests for QC rule engine staging."""
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.qc.rules import (
    qc_pre_review, qc_post_review, qc_missing_concepts,
    run_all_qc, PHASE1_CONCEPTS,
)


def _make_pack(concept: str, n_candidates: int = 3, **kwargs) -> EvidencePack:
    candidates = [
        EvidenceCandidate(candidate_id=f"c{i}", snippet=f"text {i}", page=i + 1)
        for i in range(n_candidates)
    ]
    return EvidencePack(
        protocol_id="P001",
        concept=concept,
        candidates=candidates,
        **kwargs,
    )


def test_pre_review_no_unresolved_warnings():
    """Pre-review QC should NOT warn about unresolved packs."""
    packs = {
        "index_date": _make_pack("index_date"),
        "follow_up_end": _make_pack("follow_up_end"),
    }
    results = qc_pre_review(packs)
    rule_ids = [r.rule_id for r in results]
    assert "QC-004" not in rule_ids  # QC-004 = unresolved, only post-review


def test_post_review_flags_unresolved():
    """Post-review QC should warn about unresolved packs."""
    packs = {
        "index_date": _make_pack("index_date"),
    }
    results = qc_post_review(packs)
    rule_ids = [r.rule_id for r in results]
    assert "QC-004" in rule_ids


def test_post_review_resolved_no_warning():
    """Post-review QC should not warn if candidate is selected."""
    pack = _make_pack("index_date")
    pack.select_candidate("c0")
    packs = {"index_date": pack}
    results = qc_post_review(packs)
    rule_ids = [r.rule_id for r in results]
    assert "QC-004" not in rule_ids


def test_pre_review_empty_candidates():
    packs = {"index_date": _make_pack("index_date", n_candidates=0)}
    results = qc_pre_review(packs)
    rule_ids = [r.rule_id for r in results]
    assert "QC-001" in rule_ids


def test_pre_review_contradictions():
    packs = {"index_date": _make_pack("index_date", contradictions_found=True)}
    results = qc_pre_review(packs)
    rule_ids = [r.rule_id for r in results]
    assert "QC-003" in rule_ids


def test_missing_concepts_only_checks_implemented():
    """Should not warn about concepts that aren't implemented yet."""
    packs = {"index_date": _make_pack("index_date")}
    expected = [
        "index_date", "follow_up_end", "primary_endpoint",
        "eligibility_inclusion",  # not in PHASE1_CONCEPTS
    ]
    results = qc_missing_concepts(
        packs, expected, implemented_concepts=PHASE1_CONCEPTS
    )
    concepts_warned = [r.concept for r in results]
    # Should warn about follow_up_end and primary_endpoint (implemented but missing)
    assert "follow_up_end" in concepts_warned
    assert "primary_endpoint" in concepts_warned
    # Should NOT warn about eligibility_inclusion (not implemented yet)
    assert "eligibility_inclusion" not in concepts_warned


def test_run_all_qc_pre_review_stage():
    packs = {
        "index_date": _make_pack("index_date"),
        "follow_up_end": _make_pack("follow_up_end"),
        "primary_endpoint": _make_pack("primary_endpoint"),
    }
    results = run_all_qc(packs, stage="pre_review")
    stages = {r.stage for r in results}
    assert "post_review" not in stages


def test_run_all_qc_post_review_stage():
    packs = {
        "index_date": _make_pack("index_date"),
    }
    results = run_all_qc(
        packs,
        expected_concepts=["index_date", "follow_up_end"],
        stage="post_review",
    )
    # Should include post-review checks
    assert any(r.stage == "post_review" for r in results)
