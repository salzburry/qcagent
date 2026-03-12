"""Tests for all v0.2 review fixes."""
import json
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.ingest.parse_protocol import (
    _deterministic_chunk_id, _get_page, _table_to_text,
    ParsedSection, ParsedProtocol,
)
from protocol_spec_assist.qc.rules import (
    qc_pre_review, qc_post_review, qc_cross_concept,
    qc_quote_in_chunk, run_all_qc,
)
from protocol_spec_assist.serving.model_client import ExtractionResult


# ── Fix 1: pyproject.toml — tested by pip install -e . (not unit testable) ──


# ── Fix 2: is_resolved checks actual candidate match ────────────────────────

def test_is_resolved_false_with_stale_id():
    """Stale selected_candidate_id that matches no candidate → not resolved."""
    pack = EvidencePack(
        protocol_id="P001",
        concept="index_date",
        candidates=[
            EvidenceCandidate(candidate_id="abc", snippet="text"),
        ],
        selected_candidate_id="nonexistent_id",
    )
    # selected_candidate is None because ID doesn't match
    assert pack.selected_candidate is None
    # is_resolved should be False because no actual candidate matches
    assert not pack.is_resolved


def test_is_resolved_true_with_valid_id():
    pack = EvidencePack(
        protocol_id="P001",
        concept="index_date",
        candidates=[
            EvidenceCandidate(candidate_id="abc", snippet="text"),
        ],
    )
    pack.select_candidate("abc")
    assert pack.is_resolved
    assert pack.selected_candidate.candidate_id == "abc"


def test_is_resolved_with_multi_row_selection():
    """Multi-row concepts resolved via selected_candidate_ids."""
    pack = EvidencePack(
        protocol_id="P001",
        concept="eligibility_inclusion",
        candidates=[
            EvidenceCandidate(candidate_id="inc1", snippet="age >= 18"),
            EvidenceCandidate(candidate_id="inc2", snippet="confirmed diagnosis"),
        ],
        selected_candidate_ids=["inc1", "inc2"],
    )
    assert pack.is_resolved
    assert len(pack.selected_candidates) == 2


def test_is_resolved_with_reviewer_override_only():
    """Pack resolved via reviewer_override even without candidate selection."""
    pack = EvidencePack(
        protocol_id="P001",
        concept="index_date",
        candidates=[],
        reviewer_override="Custom definition from reviewer",
    )
    assert pack.is_resolved
    assert pack.governing_text == "Custom definition from reviewer"


def test_generation_mode_reviewed_with_multi_row():
    """generation_mode = reviewed when selected_candidate_ids is set."""
    from protocol_spec_assist.spec_output.spec_schema import build_program_spec
    packs = {
        "eligibility_inclusion": EvidencePack(
            protocol_id="P001",
            concept="eligibility_inclusion",
            candidates=[
                EvidenceCandidate(candidate_id="inc1", snippet="age >= 18"),
            ],
            selected_candidate_ids=["inc1"],
        ),
    }
    spec = build_program_spec(packs, protocol_id="P001")
    assert spec.generation_mode == "reviewed"


def test_generation_mode_reviewed_with_override():
    """generation_mode = reviewed when reviewer_override is set."""
    from protocol_spec_assist.spec_output.spec_schema import build_program_spec
    packs = {
        "index_date": EvidencePack(
            protocol_id="P001",
            concept="index_date",
            candidates=[],
            reviewer_override="Manual index date definition",
        ),
    }
    spec = build_program_spec(packs, protocol_id="P001")
    assert spec.generation_mode == "reviewed"
    assert spec.important_dates[0].definition == "Manual index date definition"


# ── Fix 3: Page tracking uses None, not 0 ───────────────────────────────────

def test_parsed_section_page_none():
    """Pages should be Optional[int], allowing None."""
    sec = ParsedSection(
        heading="Test", heading_level=1, text="body",
        page_start=None, page_end=None,
    )
    assert sec.page_start is None
    assert sec.page_end is None


# ── Fix 4: Chunk ID collisions with positional component ────────────────────

def test_chunk_id_different_position_same_content():
    """Identical text at different positions gets different IDs."""
    a = _deterministic_chunk_id("P001", "same text", heading="H", page=1, position=0)
    b = _deterministic_chunk_id("P001", "same text", heading="H", page=1, position=1)
    assert a != b


def test_chunk_id_none_page():
    """None page should be handled gracefully and produce a valid UUID."""
    import uuid
    cid = _deterministic_chunk_id("P001", "text", page=None)
    assert len(cid) == 36  # UUID string format: 8-4-4-4-12
    uuid.UUID(cid)  # raises ValueError if invalid


# ── Fix 5: Table headings preserve parent context ───────────────────────────

def test_to_chunks_skips_empty_sections():
    """Empty sections should not produce chunks."""
    proto = ParsedProtocol(
        protocol_id="P001",
        title="Test",
        sections=[
            ParsedSection(
                heading="Empty", heading_level=1, text="",
                page_start=1, page_end=1, source_type="table",
                table_data=[],
            ),
            ParsedSection(
                heading="Real", heading_level=1, text="real content here",
                page_start=2, page_end=2,
            ),
        ],
    )
    chunks = proto.to_chunks()
    assert len(chunks) == 1
    assert chunks[0]["heading"] == "Real"


# ── Fix 6: Empty chunks not indexed ─────────────────────────────────────────

def test_to_chunks_empty_table_no_data():
    """Table section with no data and no text produces zero chunks."""
    proto = ParsedProtocol(
        protocol_id="P001",
        title="Test",
        sections=[
            ParsedSection(
                heading="Table", heading_level=1, text="",
                page_start=1, page_end=1, source_type="table",
                table_data=[],
            ),
        ],
    )
    chunks = proto.to_chunks()
    assert len(chunks) == 0


# ── Fix 7: Retrieval score semantics ────────────────────────────────────────

def test_retrieved_chunk_score_names():
    """RetrievedChunk should have retrieval_score, not dense_score."""
    from protocol_spec_assist.retrieval.search import RetrievedChunk
    chunk = RetrievedChunk(
        text="t", heading="h", source_type="narrative",
        page=1, protocol_id="P", retrieval_score=0.5,
    )
    assert chunk.retrieval_score == 0.5
    assert chunk.score == 0.5  # falls back to retrieval_score when no rerank
    assert not hasattr(chunk, "dense_score")


# ── Fix 10: ExtractionResult ────────────────────────────────────────────────

def test_extraction_result_structure():
    result = ExtractionResult(
        parsed={"test": True},
        model_used="test-model",
        raw_response='{"test": true}',
        prompt_version="0.2.0",
    )
    assert result.model_used == "test-model"
    assert result.raw_response == '{"test": true}'
    assert result.prompt_version == "0.2.0"


# ── Fix 12: QC cross-concept stage labeling ─────────────────────────────────

def _make_pack(concept, **kwargs):
    return EvidencePack(
        protocol_id="P001", concept=concept,
        candidates=[EvidenceCandidate(candidate_id="c1", snippet="s")],
        **kwargs,
    )


def test_qc_cross_concept_stage_pre_review():
    packs = {"follow_up_end": _make_pack("follow_up_end")}
    results = qc_cross_concept(packs, stage="pre_review")
    for r in results:
        assert r.stage == "pre_review"


def test_qc_cross_concept_stage_post_review():
    packs = {"follow_up_end": _make_pack("follow_up_end")}
    results = qc_cross_concept(packs, stage="post_review")
    for r in results:
        assert r.stage == "post_review"


def test_run_all_qc_post_review_consistent_stages():
    """Post-review QC should not return pre_review stage labels."""
    packs = {
        "index_date": _make_pack("index_date"),
        "follow_up_end": _make_pack("follow_up_end"),
        "primary_endpoint": _make_pack("primary_endpoint"),
    }
    results = run_all_qc(packs, stage="post_review")
    for r in results:
        assert r.stage == "post_review", f"Got stage={r.stage} for rule {r.rule_id}"


# ── Fix 14: Quote-in-chunk validation ────────────────────────────────────────

def test_qc_quote_in_chunk_passes():
    pack = EvidencePack(
        protocol_id="P001",
        concept="index_date",
        candidates=[
            EvidenceCandidate(
                candidate_id="c1",
                chunk_id="chunk_abc",
                snippet="index date is defined as first dispensing",
            ),
        ],
    )
    chunk_lookup = {
        "chunk_abc": "The index date is defined as first dispensing of the study drug.",
    }
    results = qc_quote_in_chunk({"index_date": pack}, chunk_lookup)
    assert len(results) == 0


def test_qc_quote_in_chunk_fails():
    pack = EvidencePack(
        protocol_id="P001",
        concept="index_date",
        candidates=[
            EvidenceCandidate(
                candidate_id="c1",
                chunk_id="chunk_abc",
                snippet="completely different text that model hallucinated",
            ),
        ],
    )
    chunk_lookup = {
        "chunk_abc": "The index date is defined as first dispensing of the study drug.",
    }
    results = qc_quote_in_chunk({"index_date": pack}, chunk_lookup)
    assert len(results) == 1
    assert results[0].rule_id == "QC-006"


def test_qc_quote_in_chunk_no_lookup():
    """No chunk lookup → skip silently."""
    pack = EvidencePack(
        protocol_id="P001",
        concept="index_date",
        candidates=[
            EvidenceCandidate(candidate_id="c1", chunk_id="x", snippet="text"),
        ],
    )
    results = qc_quote_in_chunk({"index_date": pack}, None)
    assert len(results) == 0


# ── Fix 15: Eval metric label ───────────────────────────────────────────────

def test_eval_metric_name():
    """Contradiction metric should be flag_rate, not detection_rate."""
    from protocol_spec_assist.eval import harness
    import inspect
    source = inspect.getsource(harness.run_evaluation)
    assert "contradiction_flag_rate" in source
    assert "contradiction_detection_rate" not in source


# ── Smaller cleanups ────────────────────────────────────────────────────────

def test_candidate_id_is_required():
    """candidate_id should not have a default — it must be provided."""
    import pytest
    with pytest.raises(Exception):
        EvidenceCandidate(snippet="text")  # missing candidate_id


def test_finder_version_is_0_3():
    """Default finder_version should be 0.3.0."""
    pack = EvidencePack(protocol_id="P", concept="index_date")
    assert pack.finder_version == "0.3.0"
