"""
Regression tests for P0 fixes from the code review.

Tests cover:
  1. Demographics provenance leak: static-only packs must not emit explicit rows
  2. FU must not inherit raw FUED definition text
  3. Data Prep placeholders must use [UNRESOLVED] markers
  4. Parse-fail protocols should emit shell specs, not partial hallucinated specs
  5. Date candidate mining from chunks
  6. Extracted INDEX replaces placeholder INDEX (existing fix, regression guard)
"""
import pytest
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.spec_output.spec_schema import build_program_spec
from protocol_spec_assist.row_completion.demographics_writer import DemographicsWriter
from protocol_spec_assist.row_completion.data_prep_writer import expand_data_prep, _DATE_DEFAULTS


# ── 1. Demographics provenance leak ─────────────────────────────────────────

class TestDemographicsProvenanceLeak:
    """Static-only demographics packs must not produce visible evidence-looking rows."""

    def _make_static_only_pack(self):
        """Pack with no real evidence — candidates lack page and llm_confidence."""
        return EvidencePack(
            protocol_id="P001",
            concept="demographics",
            candidates=[
                EvidenceCandidate(
                    candidate_id="static1",
                    snippet="Age at index date",
                    page=None,
                    llm_confidence=None,
                    explicit="inferred",
                ),
            ],
        )

    def _make_evidence_backed_pack(self):
        """Pack with real extracted evidence — has page and confidence."""
        return EvidencePack(
            protocol_id="P001",
            concept="demographics",
            candidates=[
                EvidenceCandidate(
                    candidate_id="real1",
                    snippet="Patient age at index date",
                    page=5,
                    llm_confidence=0.92,
                    explicit="explicit",
                ),
            ],
        )

    def test_static_only_pack_marks_rows_as_unresolved(self):
        """Static-only pack should mark ALL rows as inferred with UNRESOLVED note."""
        writer = DemographicsWriter()
        pack = self._make_static_only_pack()
        rows = writer.expand(pack, data_source="generic")

        assert len(rows) > 0
        for row in rows:
            assert row.explicit == "inferred", f"{row.variable} should be inferred"
            assert row.source_page is None, f"{row.variable} should have no source_page"
            assert row.confidence is None, f"{row.variable} should have no confidence"
            assert "[UNRESOLVED]" in row.additional_notes, (
                f"{row.variable} should have [UNRESOLVED] in notes"
            )

    def test_static_only_pack_does_not_produce_explicit_rows(self):
        """No row from a static-only pack should be marked as explicit."""
        writer = DemographicsWriter()
        pack = self._make_static_only_pack()
        rows = writer.expand(pack, data_source="generic")

        explicit_rows = [r for r in rows if r.explicit == "explicit"]
        assert len(explicit_rows) == 0, (
            f"Static-only pack produced {len(explicit_rows)} explicit rows: "
            f"{[r.variable for r in explicit_rows]}"
        )

    def test_evidence_backed_pack_produces_explicit_core_rows(self):
        """Pack with real evidence should produce explicit core rows with provenance."""
        writer = DemographicsWriter()
        pack = self._make_evidence_backed_pack()
        rows = writer.expand(pack, data_source="generic")

        core_rows = [r for r in rows if r.variable in {"AGE", "SEX", "RACE", "ETH", "ETHN"}]
        assert len(core_rows) > 0
        for row in core_rows:
            assert row.source_page == 5
            assert row.confidence == 0.92
            assert "[UNRESOLVED]" not in row.additional_notes

    def test_empty_pack_marks_all_as_unresolved(self):
        """Pack with no candidates at all is static-only."""
        writer = DemographicsWriter()
        pack = EvidencePack(
            protocol_id="P001",
            concept="demographics",
            candidates=[],
        )
        rows = writer.expand(pack, data_source="generic")

        for row in rows:
            assert row.explicit == "inferred"
            assert "[UNRESOLVED]" in row.additional_notes

    def test_demographics_via_build_program_spec_static_only(self):
        """End-to-end: static-only demographics in build_program_spec."""
        pack = self._make_static_only_pack()
        spec = build_program_spec(
            {"demographics": pack}, protocol_id="P001",
        )
        for row in spec.demographics:
            assert row.explicit == "inferred"
            assert row.source_page is None


# ── 2. FU must not inherit raw FUED definition text ──────────────────────────

class TestFUDefinition:
    """FU time period must describe the period from INDEX to FUED, not just copy FUED text."""

    def test_fu_from_follow_up_end_is_a_period_definition(self):
        """When FU is auto-generated from follow_up_end, it must describe a period."""
        fued_text = "earliest of death, disenrollment, or data cutoff"
        packs = {
            "follow_up_end": EvidencePack(
                protocol_id="P001",
                concept="follow_up_end",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="fu1",
                        snippet=fued_text,
                        page=10,
                        llm_confidence=0.9,
                    ),
                ],
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")

        fu_periods = [p for p in spec.time_periods if p.time_period == "FU"]
        assert len(fu_periods) == 1
        fu = fu_periods[0]

        # FU definition should NOT be identical to the FUED text
        assert fu.definition != fued_text, (
            "FU definition should not be a raw copy of FUED text"
        )

        # FU definition should reference INDEX (it's a period from INDEX to FUED)
        assert "INDEX" in fu.definition.upper(), (
            "FU definition should reference INDEX date"
        )

        # FU definition should still reference the FUED content for context
        assert "death" in fu.definition.lower() or "FUED" in fu.definition, (
            "FU definition should reference the FUED end anchor"
        )

    def test_fu_not_overwritten_if_already_extracted(self):
        """If study_period already extracted FU, follow_up_end should not overwrite it."""
        packs = {
            "study_period": EvidencePack(
                protocol_id="P001",
                concept="study_period",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="sp1",
                        snippet="Follow-up from index to end of data",
                        page=3,
                        llm_confidence=0.85,
                        sponsor_term="FU",
                    ),
                ],
                concept_metadata={
                    "per_candidate": {
                        "sp1": {
                            "row_type": "time_period",
                            "time_period": "FU",
                            "label": "Follow-up period",
                            "definition": "INDEX to min(FUED, event)",
                        },
                    },
                },
            ),
            "follow_up_end": EvidencePack(
                protocol_id="P001",
                concept="follow_up_end",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="fu1",
                        snippet="end of observation",
                        page=10,
                        llm_confidence=0.9,
                    ),
                ],
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")

        fu_periods = [p for p in spec.time_periods if p.time_period == "FU"]
        assert len(fu_periods) == 1
        # Should keep the study_period-extracted definition, not overwrite
        assert fu_periods[0].definition == "INDEX to min(FUED, event)"


# ── 3. Data Prep placeholders must use [UNRESOLVED] markers ──────────────────

class TestDataPrepPlaceholders:
    """Unresolved Data Prep placeholders must not carry realistic definitions."""

    def test_default_definitions_contain_unresolved_marker(self):
        """All default date definitions must contain [UNRESOLVED]."""
        for var, defaults in _DATE_DEFAULTS.items():
            assert "[UNRESOLVED]" in defaults["definition"], (
                f"Default definition for {var} should contain [UNRESOLVED]: "
                f"got '{defaults['definition']}'"
            )

    def test_placeholder_dates_have_unresolved_notes(self):
        """Placeholder dates generated by expand_data_prep must have UNRESOLVED notes."""
        # Empty pack → all three required dates are placeholders
        pack = EvidencePack(
            protocol_id="P001",
            concept="study_period",
            candidates=[],
            concept_metadata={"per_candidate": {}},
        )
        _, dates, _ = expand_data_prep(pack)

        for date in dates:
            if date.variable in {"INIT", "INDEX", "FUED"}:
                assert "[UNRESOLVED" in date.additional_notes, (
                    f"Placeholder {date.variable} should have [UNRESOLVED] in notes: "
                    f"got '{date.additional_notes}'"
                )
                assert "[UNRESOLVED]" in date.definition, (
                    f"Placeholder {date.variable} should have [UNRESOLVED] in definition: "
                    f"got '{date.definition}'"
                )

    def test_extracted_dates_replace_unresolved_placeholders(self):
        """Extracted INDEX from index_date pack should replace placeholder."""
        packs = {
            "study_period": EvidencePack(
                protocol_id="P001",
                concept="study_period",
                candidates=[],
                concept_metadata={"per_candidate": {}},
            ),
            "index_date": EvidencePack(
                protocol_id="P001",
                concept="index_date",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="idx1",
                        snippet="First dispensing of study drug",
                        page=8,
                        llm_confidence=0.95,
                    ),
                ],
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")

        index_dates = [d for d in spec.important_dates if d.variable == "INDEX"]
        assert len(index_dates) == 1
        assert index_dates[0].definition == "First dispensing of study drug"
        assert "[UNRESOLVED]" not in index_dates[0].definition


# ── 4. Parse-fail protocols should emit shell specs ──────────────────────────

class TestParseFailShellSpec:
    """Parse-fail should produce a shell spec with QC warnings, not hallucinated content."""

    def test_shell_spec_has_no_evidence_rows(self):
        """Shell spec (empty packs) should have no data rows."""
        spec = build_program_spec(
            packs={}, protocol_id="FAIL_001",
            qc_warnings=["CRITICAL: PDF parse quality is FAIL"],
        )
        assert len(spec.demographics) == 0
        assert len(spec.inclusion_criteria) == 0
        assert len(spec.exclusion_criteria) == 0
        assert len(spec.important_dates) == 0
        assert len(spec.outcome_variables) == 0
        assert len(spec.qc_warnings) > 0
        assert "FAIL" in spec.qc_warnings[0]


# ── 5. Date candidate mining ────────────────────────────────────────────────

class TestDateCandidateMining:
    """Test the local date candidate mining (Pass 1 of two-pass extraction)."""

    def test_mine_date_candidates_finds_date_patterns(self):
        from protocol_spec_assist.concepts.study_design import _mine_date_candidates
        from protocol_spec_assist.retrieval.search import RetrievedChunk

        chunks = [
            RetrievedChunk(
                text="The index date is defined as the date of first dispensing of the study drug. "
                     "Patients must have 12 months of continuous enrollment prior to index date. "
                     "The study period spans from January 1, 2015 to December 31, 2022.",
                heading="Study Design",
                source_type="narrative",
                page=3,
                protocol_id="P001",
                retrieval_score=0.9,
                chunk_id="chunk_abc",
            ),
        ]

        candidates = _mine_date_candidates(chunks)
        assert len(candidates) > 0

        # Should find at least the date reference and the enrollment mention
        texts = " ".join(c["text"] for c in candidates).lower()
        assert "index date" in texts or "first dispensing" in texts
        assert any(c["chunk_id"] == "chunk_abc" for c in candidates)

    def test_mine_date_candidates_empty_chunks(self):
        from protocol_spec_assist.concepts.study_design import _mine_date_candidates
        assert _mine_date_candidates([]) == []


# ── 6. Regression guard: extracted INDEX replaces placeholder INDEX ──────────

class TestIndexReplacesPlaceholder:
    """Guard the existing fix: extracted INDEX must replace placeholder INDEX."""

    def test_extracted_index_wins_over_placeholder(self):
        packs = {
            "study_period": EvidencePack(
                protocol_id="P001",
                concept="study_period",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="sp_idx",
                        snippet="[UNRESOLVED] placeholder index",
                        page=None,
                        llm_confidence=None,
                        sponsor_term="INDEX",
                    ),
                ],
                concept_metadata={
                    "per_candidate": {
                        "sp_idx": {
                            "row_type": "important_date",
                            "variable": "INDEX",
                            "label": "Index date",
                            "definition": "[UNRESOLVED] Definition must be extracted",
                        },
                    },
                },
            ),
            "index_date": EvidencePack(
                protocol_id="P001",
                concept="index_date",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="real_idx",
                        snippet="Date of first line therapy initiation",
                        page=12,
                        llm_confidence=0.93,
                    ),
                ],
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")

        index_dates = [d for d in spec.important_dates if d.variable == "INDEX"]
        assert len(index_dates) == 1
        assert index_dates[0].definition == "Date of first line therapy initiation"
        assert index_dates[0].source_page == 12
        assert "[UNRESOLVED]" not in index_dates[0].definition


# ── 7. Parser multi-strategy ────────────────────────────────────────────────

class TestParserStrategies:
    """Test that the parser infrastructure supports multiple strategies."""

    def test_page_first_parser_exists(self):
        """Page-first parser function should exist and be callable."""
        from protocol_spec_assist.ingest.parse_protocol import _parse_with_pymupdf_page_first
        assert callable(_parse_with_pymupdf_page_first)

    def test_quality_score_grade_values(self):
        """Quality scoring should produce valid grade values."""
        from protocol_spec_assist.ingest.parse_protocol import (
            _quality_score, ParsedProtocol, ParsedSection,
        )
        # Good parse
        good = ParsedProtocol(
            protocol_id="P001", title="Test",
            sections=[
                ParsedSection(
                    heading=f"Section {i}", heading_level=1,
                    text="x" * 200, page_start=i, page_end=i,
                )
                for i in range(10)
            ],
        )
        assert _quality_score(good).grade == "pass"

        # Bad parse — many empty micro-sections
        bad = ParsedProtocol(
            protocol_id="P001", title="Test",
            sections=[
                ParsedSection(
                    heading="x", heading_level=1,
                    text="y", page_start=i, page_end=i,
                )
                for i in range(100)
            ],
        )
        assert _quality_score(bad).grade == "fail"

        # Empty parse
        empty = ParsedProtocol(protocol_id="P001", title=None, sections=[])
        assert _quality_score(empty).grade == "fail"
