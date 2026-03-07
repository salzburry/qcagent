"""Tests for TA pack loader."""
from protocol_spec_assist.ta_packs.loader import (
    load_ta_pack, get_synonyms, get_hotspot_warning,
    get_section_priority, build_query_bank,
)


def test_load_oncology():
    pack = load_ta_pack("oncology")
    assert pack is not None
    assert pack.name == "oncology"
    assert "index_date" in pack.concept_synonyms
    assert len(pack.expected_concepts) > 0
    assert len(pack.ambiguity_hotspots) > 0


def test_load_cardiovascular():
    pack = load_ta_pack("cardiovascular")
    assert pack is not None
    assert pack.name == "cardiovascular"


def test_load_nonexistent():
    pack = load_ta_pack("nonexistent_ta")
    assert pack is None


def test_get_synonyms():
    pack = load_ta_pack("oncology")
    syns = get_synonyms(pack, "index_date")
    assert isinstance(syns, list)
    assert len(syns) > 0


def test_get_synonyms_no_pack():
    assert get_synonyms(None, "index_date") == []


def test_get_hotspot_warning():
    pack = load_ta_pack("oncology")
    warning = get_hotspot_warning(pack, "index_date")
    assert warning is not None
    assert "HIGH" in warning or "MEDIUM" in warning


def test_get_hotspot_no_match():
    pack = load_ta_pack("oncology")
    warning = get_hotspot_warning(pack, "nonexistent_concept")
    assert warning is None


def test_get_section_priority():
    pack = load_ta_pack("oncology")
    sections = get_section_priority(pack, "index_date")
    assert isinstance(sections, list)


def test_build_query_bank_with_pack():
    pack = load_ta_pack("oncology")
    queries = build_query_bank("index date definition", pack, "index_date")
    assert len(queries) >= 2  # base + at least one synonym
    assert queries[0] == "index date definition"


def test_build_query_bank_without_pack():
    queries = build_query_bank("index date definition", None, "index_date")
    assert queries == ["index date definition"]
