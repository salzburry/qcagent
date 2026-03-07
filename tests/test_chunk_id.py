"""Tests for deterministic chunk ID generation."""
from protocol_spec_assist.ingest.parse_protocol import _deterministic_chunk_id


def test_same_input_same_id():
    a = _deterministic_chunk_id("P001", "some text", heading="Section 1", page=5)
    b = _deterministic_chunk_id("P001", "some text", heading="Section 1", page=5)
    assert a == b


def test_different_page_different_id():
    """Identical text on different pages must produce different IDs."""
    a = _deterministic_chunk_id("P001", "some text", heading="Section 1", page=5)
    b = _deterministic_chunk_id("P001", "some text", heading="Section 1", page=6)
    assert a != b


def test_different_section_different_id():
    """Identical text under different headings must produce different IDs."""
    a = _deterministic_chunk_id("P001", "some text", heading="Section 1", page=5)
    b = _deterministic_chunk_id("P001", "some text", heading="Section 2", page=5)
    assert a != b


def test_different_protocol_different_id():
    a = _deterministic_chunk_id("P001", "some text", heading="Section 1", page=5)
    b = _deterministic_chunk_id("P002", "some text", heading="Section 1", page=5)
    assert a != b


def test_different_source_type_different_id():
    a = _deterministic_chunk_id("P001", "text", heading="H", page=1, source_type="narrative")
    b = _deterministic_chunk_id("P001", "text", heading="H", page=1, source_type="table")
    assert a != b


def test_id_is_16_chars():
    cid = _deterministic_chunk_id("P001", "text")
    assert len(cid) == 16
    assert cid.isalnum()
