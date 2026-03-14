"""
Regression tests for P1 features:
  1. Cohort definition extraction and row writing (Section B)
  2. Source-data-prep issue generation (Section D)
  3. Multi-channel source detection
  4. Outcome row naming closer to org contract
"""
import pytest
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.spec_output.spec_schema import build_program_spec


# ── 1. Cohort Definitions (StudyPop Section B) ──────────────────────────────

class TestCohortDefinitions:
    """Cohort definitions must populate StudyPop Section B when evidence exists."""

    def test_cohort_definitions_populate_section_b(self):
        """Cohort pack with candidates should produce CohortRow entries."""
        packs = {
            "cohort_definitions": EvidencePack(
                protocol_id="P001",
                concept="cohort_definitions",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="coh1",
                        snippet="Patients initiating drug A as first-line therapy",
                        page=15,
                        llm_confidence=0.88,
                        sponsor_term="Treatment cohort",
                    ),
                    EvidenceCandidate(
                        candidate_id="coh2",
                        snippet="Patients on standard-of-care chemotherapy",
                        page=16,
                        llm_confidence=0.85,
                        sponsor_term="Comparator cohort",
                    ),
                ],
                concept_metadata={
                    "per_candidate": {
                        "coh1": {
                            "cohort_label": "Treatment cohort - Drug A",
                            "cohort_variable": "COHORT",
                            "values": "1=Drug A",
                            "definition": "Patients initiating drug A as 1L therapy after INDEX",
                        },
                        "coh2": {
                            "cohort_label": "Comparator cohort - SOC",
                            "cohort_variable": "COHORT",
                            "values": "2=SOC",
                            "definition": "Patients on standard-of-care after INDEX",
                        },
                    },
                },
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")

        assert len(spec.cohort_definitions) == 2
        assert spec.cohort_definitions[0].variable == "COHORT"
        assert spec.cohort_definitions[0].values == "1=Drug A"
        assert spec.cohort_definitions[0].definition == "Patients initiating drug A as 1L therapy after INDEX"
        assert spec.cohort_definitions[0].source_page == 15

    def test_empty_cohort_pack_no_rows(self):
        """Empty cohort pack should produce no Section B rows."""
        packs = {
            "cohort_definitions": EvidencePack(
                protocol_id="P001",
                concept="cohort_definitions",
                candidates=[],
                concept_metadata={"per_candidate": {}},
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")
        assert len(spec.cohort_definitions) == 0

    def test_cohort_without_metadata_uses_snippet(self):
        """Cohort row falls back to snippet when definition metadata is missing."""
        packs = {
            "cohort_definitions": EvidencePack(
                protocol_id="P001",
                concept="cohort_definitions",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="coh1",
                        snippet="Exposed patients initiated on treatment X",
                        page=10,
                        llm_confidence=0.8,
                        sponsor_term="Exposed",
                    ),
                ],
                concept_metadata={"per_candidate": {}},
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")
        assert len(spec.cohort_definitions) == 1
        assert "Exposed" in spec.cohort_definitions[0].definition


# ── 2. Source Data Prep (Section D) ──────────────────────────────────────────

class TestSourceDataPrep:
    """Source data preparation issues must populate Section D."""

    def test_source_data_prep_populates_section_d(self):
        """Source data prep pack should produce SourceDataPrep rows."""
        packs = {
            "source_data_prep": EvidencePack(
                protocol_id="P001",
                concept="source_data_prep",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="sdp1",
                        snippet="Race not available in MarketScan",
                        page=None,
                        llm_confidence=0.9,
                        sponsor_term="RACE",
                    ),
                    EvidenceCandidate(
                        candidate_id="sdp2",
                        snippet="ECOG requires NLP extraction from notes",
                        page=8,
                        llm_confidence=0.75,
                        sponsor_term="ECOG",
                    ),
                ],
                concept_metadata={
                    "per_candidate": {
                        "sdp1": {
                            "source_table_variable": "ENROLLMENT.RACE",
                            "situation": "Race not available in MarketScan",
                            "action": "Set to Missing",
                            "reasoning": "MarketScan does not capture race data",
                        },
                        "sdp2": {
                            "source_table_variable": "CLINICAL_NOTES.ECOG",
                            "situation": "ECOG requires NLP extraction",
                            "action": "Apply validated NLP algorithm or set Missing",
                            "reasoning": "ECOG not in structured fields",
                        },
                    },
                },
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")

        assert len(spec.source_data_prep) == 2
        assert spec.source_data_prep[0].row_number == 1
        assert spec.source_data_prep[0].source_table_variable == "ENROLLMENT.RACE"
        assert spec.source_data_prep[0].action == "Set to Missing"
        assert spec.source_data_prep[1].row_number == 2

    def test_empty_source_data_prep(self):
        """Empty pack should produce no Section D rows."""
        packs = {
            "source_data_prep": EvidencePack(
                protocol_id="P001",
                concept="source_data_prep",
                candidates=[],
                concept_metadata={"per_candidate": {}},
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")
        assert len(spec.source_data_prep) == 0

    def test_source_limitation_pack_from_registry(self):
        """Registry-only source limitation pack should produce valid candidates."""
        from protocol_spec_assist.concepts.source_data_prep import _build_source_limitation_pack
        pack = _build_source_limitation_pack("P001", "marketscan")

        # MarketScan is missing biomarkers
        assert len(pack.candidates) > 0
        snippets = " ".join(c.snippet.lower() for c in pack.candidates)
        assert "not available" in snippets or "missing" in snippets.lower()

    def test_generic_source_no_limitations(self):
        """Generic source should produce no limitation issues."""
        from protocol_spec_assist.concepts.source_data_prep import _build_source_limitation_pack
        pack = _build_source_limitation_pack("P001", "generic")
        assert len(pack.candidates) == 0


# ── 3. Multi-channel Source Detection ────────────────────────────────────────

class TestMultiChannelSourceDetection:
    """Source detection should use multiple evidence channels."""

    def test_override_takes_priority(self):
        from protocol_spec_assist.data_sources.registry import detect_source_multi
        result = detect_source_multi(
            data_source_override="COTA",
            study_period_metadata={"data_source": "Flatiron"},
            protocol_title="A Flatiron Study",
        )
        assert result == "cota"

    def test_study_period_metadata_channel(self):
        from protocol_spec_assist.data_sources.registry import detect_source_multi
        result = detect_source_multi(
            study_period_metadata={"data_source": "Optum CDM"},
        )
        assert result == "optum_cdm"

    def test_protocol_title_channel(self):
        from protocol_spec_assist.data_sources.registry import detect_source_multi
        result = detect_source_multi(
            protocol_title="A Retrospective Study Using Flatiron Health EHR Data",
        )
        assert result == "flatiron"

    def test_protocol_text_channel(self):
        from protocol_spec_assist.data_sources.registry import detect_source_multi
        result = detect_source_multi(
            protocol_text_sample="This study will use IBM MarketScan Commercial Claims data...",
        )
        assert result == "marketscan"

    def test_fallback_to_generic(self):
        from protocol_spec_assist.data_sources.registry import detect_source_multi
        result = detect_source_multi(
            protocol_title="A Study of Treatment X in Patients with Disease Y",
        )
        assert result == "generic"

    def test_all_channels_empty(self):
        from protocol_spec_assist.data_sources.registry import detect_source_multi
        assert detect_source_multi() == "generic"

    def test_priority_order(self):
        """Override > study_period > title > text."""
        from protocol_spec_assist.data_sources.registry import detect_source_multi

        # Override wins over everything
        assert detect_source_multi(
            data_source_override="quest",
            study_period_metadata={"data_source": "COTA"},
            protocol_title="Flatiron study",
            protocol_text_sample="MarketScan data",
        ) == "quest"

        # study_period wins over title and text
        assert detect_source_multi(
            study_period_metadata={"data_source": "COTA"},
            protocol_title="Flatiron study",
            protocol_text_sample="MarketScan data",
        ) == "cota"

        # title wins over text
        assert detect_source_multi(
            protocol_title="Flatiron study",
            protocol_text_sample="MarketScan data",
        ) == "flatiron"


# ── 4. Outcome Row Naming ────────────────────────────────────────────────────

class TestOutcomeRowNaming:
    """Outcome rows should use sponsor-derived names, not generic PRIMARY_EP."""

    def test_endpoint_uses_sponsor_term_as_prefix(self):
        """Endpoint writer should derive variable prefix from sponsor term."""
        from protocol_spec_assist.row_completion.outcomes_writer import EndpointWriter

        pack = EvidencePack(
            protocol_id="P001",
            concept="primary_endpoint",
            candidates=[
                EvidenceCandidate(
                    candidate_id="ep1",
                    snippet="Overall survival defined as time from index to death",
                    page=20,
                    llm_confidence=0.95,
                    sponsor_term="Overall Survival",
                ),
            ],
            concept_metadata={
                "per_candidate": {
                    "ep1": {
                        "is_composite": False,
                        "components": [],
                        "time_to_event": True,
                    },
                },
            },
        )
        writer = EndpointWriter()
        rows = writer.expand(pack)

        # Should use "OS" prefix, not "PRIMARY_EP"
        var_names = [r.variable for r in rows]
        assert "OS" in var_names, f"Expected 'OS' in {var_names}"
        assert "OS_EVENTFL" in var_names
        assert "OS_EVENTDT" in var_names
        assert "OS_TTOEVENT" in var_names  # time_to_event is True
        assert "PRIMARY_EP" not in var_names

    def test_composite_endpoint_expands_components(self):
        """Composite endpoints should expand each component into sub-family."""
        from protocol_spec_assist.row_completion.outcomes_writer import EndpointWriter

        pack = EvidencePack(
            protocol_id="P001",
            concept="primary_endpoint",
            candidates=[
                EvidenceCandidate(
                    candidate_id="ep1",
                    snippet="MACE: composite of CV death, MI, and stroke",
                    page=22,
                    llm_confidence=0.9,
                    sponsor_term="Major Adverse Cardiovascular Events",
                ),
            ],
            concept_metadata={
                "per_candidate": {
                    "ep1": {
                        "is_composite": True,
                        "components": ["CV death", "MI", "stroke"],
                        "time_to_event": True,
                    },
                },
            },
        )
        writer = EndpointWriter()
        rows = writer.expand(pack)

        var_names = [r.variable for r in rows]
        assert "MACE" in var_names
        assert "MACE_EVENTFL" in var_names

        # Components should also be expanded (CV death → CV_DEATH or CVDEATH)
        assert any("CV" in v and "DEATH" in v for v in var_names), f"Expected CV death component in {var_names}"
        assert any("MI" in v for v in var_names), f"Expected MI component in {var_names}"

    def test_non_tte_endpoint_no_ttoevent(self):
        """Non-time-to-event endpoints should not have TTOEVENT row."""
        from protocol_spec_assist.row_completion.outcomes_writer import EndpointWriter

        pack = EvidencePack(
            protocol_id="P001",
            concept="primary_endpoint",
            candidates=[
                EvidenceCandidate(
                    candidate_id="ep1",
                    snippet="Objective response rate",
                    page=20,
                    llm_confidence=0.9,
                    sponsor_term="Objective Response Rate",
                ),
            ],
            concept_metadata={
                "per_candidate": {
                    "ep1": {
                        "is_composite": False,
                        "components": [],
                        "time_to_event": False,
                    },
                },
            },
        )
        writer = EndpointWriter()
        rows = writer.expand(pack)

        var_names = [r.variable for r in rows]
        assert "ORR" in var_names
        assert not any("TTOEVENT" in v for v in var_names)

    def test_censoring_uses_meaningful_names(self):
        """Censoring writer should derive meaningful names from rule labels."""
        from protocol_spec_assist.row_completion.outcomes_writer import CensoringWriter

        pack = EvidencePack(
            protocol_id="P001",
            concept="censoring_rules",
            candidates=[
                EvidenceCandidate(
                    candidate_id="cr1",
                    snippet="Patients censored at death if death is not the primary event",
                    page=25,
                    llm_confidence=0.9,
                    sponsor_term="Death censoring",
                ),
                EvidenceCandidate(
                    candidate_id="cr2",
                    snippet="Patients censored at disenrollment from the health plan",
                    page=25,
                    llm_confidence=0.85,
                    sponsor_term="Disenrollment",
                ),
            ],
            concept_metadata={
                "per_candidate": {
                    "cr1": {"rule_type": "event_based", "applies_to": "PFS"},
                    "cr2": {"rule_type": "event_based", "applies_to": "all"},
                },
            },
        )
        writer = CensoringWriter()
        rows = writer.expand(pack)

        var_names = [r.variable for r in rows]
        # Should use DEATH, DISENRL, not CENS01, CENS02
        assert "DEATH" in var_names, f"Expected 'DEATH' in {var_names}"
        assert "DISENRL" in var_names, f"Expected 'DISENRL' in {var_names}"
        assert "DEATH_FL" in var_names
        assert "DEATH_DT" in var_names
        assert "DISENRL_FL" in var_names

    def test_censoring_deduplicates_prefixes(self):
        """If two rules map to same prefix, second should get a suffix."""
        from protocol_spec_assist.row_completion.outcomes_writer import CensoringWriter

        pack = EvidencePack(
            protocol_id="P001",
            concept="censoring_rules",
            candidates=[
                EvidenceCandidate(
                    candidate_id="cr1",
                    snippet="Death from any cause",
                    page=25,
                    llm_confidence=0.9,
                    sponsor_term="Death from any cause",
                ),
                EvidenceCandidate(
                    candidate_id="cr2",
                    snippet="Death from cardiovascular cause",
                    page=25,
                    llm_confidence=0.85,
                    sponsor_term="Death from CV cause",
                ),
            ],
            concept_metadata={
                "per_candidate": {
                    "cr1": {"rule_type": "event_based", "applies_to": "all"},
                    "cr2": {"rule_type": "event_based", "applies_to": "OS"},
                },
            },
        )
        writer = CensoringWriter()
        rows = writer.expand(pack)

        # Both should map to DEATH-like prefix, but should be deduplicated
        def_row_vars = [r.variable for r in rows if "_FL" not in r.variable
                        and "_DT" not in r.variable and "_REAS" not in r.variable]
        assert len(def_row_vars) == len(set(def_row_vars)), (
            f"Duplicate definition row variables: {def_row_vars}"
        )

    def test_normalize_var_name_known_endpoints(self):
        """Known endpoint terms should map to standard abbreviations."""
        from protocol_spec_assist.row_completion.outcomes_writer import _normalize_var_name

        assert _normalize_var_name("Overall Survival") == "OS"
        assert _normalize_var_name("Progression-Free Survival") == "PFS"
        assert _normalize_var_name("Major Adverse Cardiovascular Events") == "MACE"
        assert _normalize_var_name("Time to Treatment Discontinuation") == "TTD"
        assert _normalize_var_name("Complete Response") == "CR"
        assert _normalize_var_name("Duration of Response") == "DOR"

    def test_normalize_var_name_unknown_terms(self):
        """Unknown terms should produce reasonable short variable names."""
        from protocol_spec_assist.row_completion.outcomes_writer import _normalize_var_name

        result = _normalize_var_name("Some Custom Study Endpoint")
        assert len(result) > 0
        assert len(result) <= 16
        assert result == result.upper()  # should be uppercase

    def test_fallback_to_positional_when_no_sponsor_term(self):
        """Without sponsor term, endpoint should fall back to EP01."""
        from protocol_spec_assist.row_completion.outcomes_writer import EndpointWriter

        pack = EvidencePack(
            protocol_id="P001",
            concept="primary_endpoint",
            candidates=[
                EvidenceCandidate(
                    candidate_id="ep1",
                    snippet="Primary endpoint definition",
                    page=20,
                    llm_confidence=0.9,
                    sponsor_term="",
                ),
            ],
            concept_metadata={"per_candidate": {}},
        )
        writer = EndpointWriter()
        rows = writer.expand(pack)

        var_names = [r.variable for r in rows]
        assert "EP01" in var_names


# ── 5. Concept name registration ─────────────────────────────────────────────

class TestConceptNameRegistration:
    """New concept names should be registered in the ConceptName literal."""

    def test_cohort_definitions_in_concept_names(self):
        pack = EvidencePack(
            protocol_id="P001",
            concept="cohort_definitions",
            candidates=[],
        )
        assert pack.concept == "cohort_definitions"

    def test_source_data_prep_in_concept_names(self):
        pack = EvidencePack(
            protocol_id="P001",
            concept="source_data_prep",
            candidates=[],
        )
        assert pack.concept == "source_data_prep"


# ── 6. End-to-end: all new sections populated ───────────────────────────────

class TestEndToEndNewSections:
    """build_program_spec should populate all new sections together."""

    def test_full_spec_with_all_new_sections(self):
        packs = {
            "cohort_definitions": EvidencePack(
                protocol_id="P001",
                concept="cohort_definitions",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="coh1",
                        snippet="Drug A cohort",
                        page=10,
                        llm_confidence=0.9,
                        sponsor_term="Treatment",
                    ),
                ],
                concept_metadata={
                    "per_candidate": {
                        "coh1": {
                            "cohort_label": "Treatment",
                            "cohort_variable": "COHORT",
                            "values": "1=Treated",
                            "definition": "Patients initiating Drug A",
                        },
                    },
                },
            ),
            "source_data_prep": EvidencePack(
                protocol_id="P001",
                concept="source_data_prep",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="sdp1",
                        snippet="Race missing",
                        page=None,
                        llm_confidence=0.9,
                        sponsor_term="RACE",
                    ),
                ],
                concept_metadata={
                    "per_candidate": {
                        "sdp1": {
                            "source_table_variable": "MEMBER.RACE",
                            "situation": "Race not available",
                            "action": "Set Missing",
                            "reasoning": "Claims limitation",
                        },
                    },
                },
            ),
            "primary_endpoint": EvidencePack(
                protocol_id="P001",
                concept="primary_endpoint",
                candidates=[
                    EvidenceCandidate(
                        candidate_id="ep1",
                        snippet="Time from index to death from any cause",
                        page=20,
                        llm_confidence=0.95,
                        sponsor_term="Overall Survival",
                    ),
                ],
                concept_metadata={
                    "per_candidate": {
                        "ep1": {
                            "is_composite": False,
                            "components": [],
                            "time_to_event": True,
                        },
                    },
                },
            ),
        }
        spec = build_program_spec(packs, protocol_id="P001")

        # Section B populated
        assert len(spec.cohort_definitions) == 1

        # Section D populated
        assert len(spec.source_data_prep) == 1

        # Outcomes use sponsor-derived names
        os_rows = [r for r in spec.outcome_variables if r.variable == "OS"]
        assert len(os_rows) == 1
