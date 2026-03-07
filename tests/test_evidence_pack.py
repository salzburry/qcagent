"""Tests for EvidencePack selection and resolution."""
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate


def _make_pack() -> EvidencePack:
    return EvidencePack(
        protocol_id="P001",
        concept="index_date",
        candidates=[
            EvidenceCandidate(candidate_id="abc123", snippet="first"),
            EvidenceCandidate(candidate_id="def456", snippet="second"),
            EvidenceCandidate(candidate_id="ghi789", snippet="third"),
        ],
    )


def test_not_resolved_initially():
    pack = _make_pack()
    assert not pack.is_resolved
    assert pack.selected_candidate is None
    assert pack.governing_text is None


def test_select_candidate_by_id():
    pack = _make_pack()
    assert pack.select_candidate("def456")
    assert pack.is_resolved
    assert pack.selected_candidate.snippet == "second"
    assert pack.selected_candidate_id == "def456"


def test_select_nonexistent_candidate():
    pack = _make_pack()
    assert not pack.select_candidate("nonexistent")
    assert not pack.is_resolved


def test_reviewer_override():
    pack = _make_pack()
    pack.reviewer_override = "custom definition from protocol"
    assert pack.is_resolved
    assert pack.governing_text == "custom definition from protocol"
    # override takes precedence even if candidate selected
    pack.select_candidate("abc123")
    assert pack.governing_text == "custom definition from protocol"


def test_selection_stable_after_reorder():
    """Selection by ID survives candidate list reordering."""
    pack = _make_pack()
    pack.select_candidate("def456")

    # Simulate reordering candidates
    pack.candidates = list(reversed(pack.candidates))

    # Still resolves to the same candidate by ID
    assert pack.selected_candidate.snippet == "second"
    assert pack.selected_candidate.candidate_id == "def456"
