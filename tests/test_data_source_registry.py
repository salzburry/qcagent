"""Tests for data source definition registry."""
import pytest
from protocol_spec_assist.data_sources.registry import (
    detect_source,
    get_definition,
    is_variable_available,
    resolve_static_template,
    DEFINITIONS,
    SOURCE_AVAILABILITY,
)


# ── detect_source ────────────────────────────────────────────────────────────

class TestDetectSource:
    def test_cota(self):
        assert detect_source("COTA EHR") == "cota"
        assert detect_source("cota real-world data") == "cota"

    def test_flatiron(self):
        assert detect_source("Flatiron Health database") == "flatiron"

    def test_optum_cdm(self):
        assert detect_source("Optum Clinformatics") == "optum_cdm"
        assert detect_source("Optum CDM") == "optum_cdm"

    def test_optum_ehr(self):
        assert detect_source("Optum EHR database") == "optum_ehr"
        assert detect_source("Optum Electronic Health Records") == "optum_ehr"

    def test_marketscan(self):
        assert detect_source("IBM MarketScan") == "marketscan"
        assert detect_source("Truven database") == "marketscan"
        assert detect_source("Merative claims") == "marketscan"

    def test_inalon(self):
        assert detect_source("Inovalon data") == "inalon"
        assert detect_source("Inalon claims") == "inalon"

    def test_quest(self):
        assert detect_source("Quest Diagnostics lab data") == "quest"

    def test_generic_fallback(self):
        assert detect_source("Unknown database") == "generic"
        assert detect_source("") == "generic"
        assert detect_source("Some other source") == "generic"

    def test_none_input(self):
        assert detect_source("") == "generic"

    def test_case_insensitive(self):
        assert detect_source("MARKETSCAN") == "marketscan"
        assert detect_source("COTA") == "cota"


# ── get_definition ───────────────────────────────────────────────────────────

class TestGetDefinition:
    def test_source_specific_override(self):
        result = get_definition("cota", "AGE", "default age def")
        assert "COTA" in result or "date_of_birth" in result
        assert result != "default age def"

    def test_fallback_to_default(self):
        result = get_definition("cota", "NONEXISTENT_VAR", "my default")
        assert result == "my default"

    def test_generic_source_always_defaults(self):
        result = get_definition("generic", "AGE", "generic age def")
        assert result == "generic age def"

    def test_claims_sources_have_overrides(self):
        for source in ["optum_cdm", "marketscan"]:
            age_def = get_definition(source, "AGE", "default")
            assert age_def != "default"

    def test_quest_lab_focus(self):
        ldh = get_definition("quest", "LABLDH", "default")
        assert ldh != "default"
        assert "LAB_ORDER_RESULT" in ldh


# ── is_variable_available ────────────────────────────────────────────────────

class TestIsVariableAvailable:
    def test_ehr_sources_have_all(self):
        for source in ["cota", "flatiron"]:
            for concept in ["demographics", "clinical_characteristics", "biomarkers", "lab_variables", "treatment_variables"]:
                assert is_variable_available(source, concept, "") is True

    def test_claims_no_biomarkers(self):
        for source in ["optum_cdm", "marketscan", "inalon"]:
            assert is_variable_available(source, "biomarkers", "") is False

    def test_quest_no_treatment(self):
        assert is_variable_available("quest", "treatment_variables", "") is False
        assert is_variable_available("quest", "clinical_characteristics", "") is False

    def test_quest_has_labs(self):
        assert is_variable_available("quest", "lab_variables", "") is True
        assert is_variable_available("quest", "biomarkers", "") is True

    def test_generic_always_available(self):
        for concept in ["demographics", "clinical_characteristics", "biomarkers", "lab_variables", "treatment_variables"]:
            assert is_variable_available("generic", concept, "") is True

    def test_unknown_source_uses_generic(self):
        assert is_variable_available("nonexistent_source", "biomarkers", "") is True


# ── resolve_static_template ──────────────────────────────────────────────────

class TestResolveStaticTemplate:
    @pytest.fixture
    def sample_template(self):
        return [
            {"variable_name": "AGE", "definition": "Generic age def", "additional_notes": ""},
            {"variable_name": "SEX", "definition": "Generic sex def", "additional_notes": ""},
            {"variable_name": "WEIGHT", "definition": "Generic weight def", "additional_notes": "some note"},
        ]

    def test_generic_no_changes(self, sample_template):
        resolved = resolve_static_template(sample_template, "generic", "demographics")
        assert resolved[0]["definition"] == "Generic age def"
        assert resolved[1]["definition"] == "Generic sex def"

    def test_cota_overrides_definitions(self, sample_template):
        resolved = resolve_static_template(sample_template, "cota", "demographics")
        assert resolved[0]["definition"] != "Generic age def"
        assert "COTA" in resolved[0]["definition"] or "date_of_birth" in resolved[0]["definition"]

    def test_claims_overrides(self, sample_template):
        resolved = resolve_static_template(sample_template, "optum_cdm", "demographics")
        age_def = resolved[0]["definition"]
        assert "MEMBER_DETAIL" in age_def or "BIRTH_YR" in age_def

    def test_unavailable_concept_adds_note(self, sample_template):
        resolved = resolve_static_template(sample_template, "quest", "clinical_characteristics")
        for entry in resolved:
            assert "quest" in entry["additional_notes"].lower()

    def test_available_concept_no_extra_note(self, sample_template):
        resolved = resolve_static_template(sample_template, "cota", "demographics")
        assert "may not be available" not in resolved[0]["additional_notes"].lower()

    def test_preserves_existing_notes_when_unavailable(self, sample_template):
        resolved = resolve_static_template(sample_template, "quest", "clinical_characteristics")
        weight_entry = next(e for e in resolved if e["variable_name"] == "WEIGHT")
        assert "some note" in weight_entry["additional_notes"]
        assert "quest" in weight_entry["additional_notes"].lower()

    def test_original_template_unchanged(self, sample_template):
        original_def = sample_template[0]["definition"]
        resolve_static_template(sample_template, "cota", "demographics")
        assert sample_template[0]["definition"] == original_def


# ── Coverage of all sources ──────────────────────────────────────────────────

class TestSourceCoverage:
    def test_all_sources_in_availability(self):
        expected = {"cota", "flatiron", "optum_cdm", "optum_ehr", "marketscan", "inalon", "quest", "generic"}
        assert set(SOURCE_AVAILABILITY.keys()) == expected

    def test_all_defined_sources_have_availability(self):
        for source_key in DEFINITIONS:
            assert source_key in SOURCE_AVAILABILITY
