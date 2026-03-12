"""Tests for spec output layer: schema, HTML renderer, Excel writer."""

import json
import pytest
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.spec_output.spec_schema import (
    ProgramSpec, build_program_spec, SpecEntry, CriterionEntry,
)
from protocol_spec_assist.spec_output.html_renderer import render_html


def _make_pack(concept: str, n_candidates: int = 2, **kwargs) -> EvidencePack:
    candidates = [
        EvidenceCandidate(
            candidate_id=f"c{i}",
            snippet=f"Evidence text {i} for {concept}",
            page=i + 1,
            sponsor_term=f"term_{i}",
            llm_confidence=0.9 - i * 0.1,
            explicit="explicit" if i == 0 else "inferred",
        )
        for i in range(n_candidates)
    ]
    return EvidencePack(
        protocol_id="P001",
        concept=concept,
        candidates=candidates,
        overall_confidence=0.85,
        **kwargs,
    )


# ── ProgramSpec schema tests ────────────────────────────────────────────────

def test_program_spec_defaults():
    spec = ProgramSpec()
    assert spec.protocol_id == ""
    assert spec.spec_version == "0.3.0"
    assert spec.generation_mode == "draft"
    assert spec.inclusion_criteria == []
    assert spec.exclusion_criteria == []
    assert spec.censoring_rules == []


def test_program_spec_serialization():
    spec = ProgramSpec(protocol_id="P001")
    data = spec.model_dump()
    assert data["protocol_id"] == "P001"
    # Roundtrip
    spec2 = ProgramSpec.model_validate(data)
    assert spec2.protocol_id == "P001"


# ── build_program_spec tests ────────────────────────────────────────────────

def test_build_spec_from_single_concept():
    packs = {"index_date": _make_pack("index_date")}
    spec = build_program_spec(packs, protocol_id="P001")
    assert spec.protocol_id == "P001"
    assert spec.generation_mode == "draft"
    assert "Evidence text 0" in spec.index_date.value


def test_build_spec_uses_selected_candidate():
    pack = _make_pack("index_date")
    pack.select_candidate("c1")
    packs = {"index_date": pack}
    spec = build_program_spec(packs, protocol_id="P001")
    assert spec.generation_mode == "reviewed"
    assert "Evidence text 1" in spec.index_date.value


def test_build_spec_with_all_concepts():
    packs = {
        "index_date": _make_pack("index_date"),
        "follow_up_end": _make_pack("follow_up_end"),
        "primary_endpoint": _make_pack("primary_endpoint"),
        "eligibility_inclusion": _make_pack("eligibility_inclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "demographic", "lookback_window": None, "operational_detail": None},
                "c1": {"domain": "clinical", "lookback_window": "12 months", "operational_detail": None},
            }}),
        "eligibility_exclusion": _make_pack("eligibility_exclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "treatment", "lookback_window": "6 months", "operational_detail": None},
            }}),
        "study_period": _make_pack("study_period",
            concept_metadata={
                "study_period_start": "2020-01-01",
                "study_period_end": "2023-12-31",
                "data_source": "Optum CDM",
                "data_source_version": "Q4 2023",
                "design_type": "retrospective_cohort",
            }),
        "censoring_rules": _make_pack("censoring_rules",
            concept_metadata={"per_candidate": {
                "c0": {"rule_type": "event_based", "applies_to": "primary endpoint"},
            }}),
    }
    spec = build_program_spec(packs, protocol_id="P001")

    assert spec.index_date.value != ""
    assert spec.follow_up_end.value != ""
    assert spec.primary_endpoint.value != ""
    assert len(spec.inclusion_criteria) == 2
    assert spec.inclusion_criteria[0].domain == "demographic"
    assert spec.inclusion_criteria[1].lookback_window == "12 months"
    assert len(spec.exclusion_criteria) >= 1
    assert spec.study_design.data_source.value == "Optum CDM"
    assert spec.study_design.design_type.value == "retrospective_cohort"
    assert len(spec.censoring_rules) >= 1


def test_build_spec_empty_packs():
    spec = build_program_spec({}, protocol_id="P001")
    assert spec.protocol_id == "P001"
    assert spec.index_date.value == ""


# ── HTML renderer tests ─────────────────────────────────────────────────────

def test_html_contains_protocol_id():
    spec = ProgramSpec(protocol_id="P001")
    html = render_html(spec)
    assert "P001" in html


def test_html_contains_key_sections():
    packs = {
        "index_date": _make_pack("index_date"),
        "eligibility_inclusion": _make_pack("eligibility_inclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "demographic", "lookback_window": None, "operational_detail": None},
            }}),
    }
    spec = build_program_spec(packs, protocol_id="P001")
    html = render_html(spec)

    assert "Study Design" in html
    assert "Key Concepts" in html
    assert "Inclusion Criteria" in html
    assert "Exclusion Criteria" in html
    assert "Censoring Rules" in html
    assert "Evidence text 0" in html


def test_html_qc_warnings():
    spec = ProgramSpec(
        protocol_id="P001",
        qc_warnings=["Low signal for follow_up_end"],
    )
    html = render_html(spec)
    assert "QC Warnings" in html
    assert "Low signal" in html


def test_html_draft_mode_label():
    spec = ProgramSpec(protocol_id="P001", generation_mode="draft")
    html = render_html(spec)
    assert "DRAFT" in html


# ── Excel writer tests ──────────────────────────────────────────────────────

def test_excel_import():
    """Just verify the excel_writer module can be imported."""
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl not installed")
    from protocol_spec_assist.spec_output.excel_writer import save_excel
    assert callable(save_excel)


def test_excel_save(tmp_path):
    """Test that save_excel creates a valid .xlsx file."""
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl not installed")

    from protocol_spec_assist.spec_output.excel_writer import save_excel

    packs = {
        "index_date": _make_pack("index_date"),
        "eligibility_inclusion": _make_pack("eligibility_inclusion",
            concept_metadata={"per_candidate": {
                "c0": {"domain": "demographic", "lookback_window": None, "operational_detail": None},
            }}),
        "censoring_rules": _make_pack("censoring_rules",
            concept_metadata={"per_candidate": {
                "c0": {"rule_type": "event_based", "applies_to": "all"},
            }}),
    }
    spec = build_program_spec(packs, protocol_id="P001")

    out_path = str(tmp_path / "test_spec.xlsx")
    result = save_excel(spec, out_path)
    assert result == out_path

    wb = openpyxl.load_workbook(out_path)
    assert "Overview" in wb.sheetnames
    assert "Inclusion Criteria" in wb.sheetnames
    assert "Censoring Rules" in wb.sheetnames
