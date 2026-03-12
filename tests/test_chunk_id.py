"""Tests for deterministic chunk ID generation."""
import uuid

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


def test_id_is_valid_uuid():
    """chunk_id must be a valid UUID string — required by Qdrant for point IDs."""
    cid = _deterministic_chunk_id("P001", "text")
    parsed = uuid.UUID(cid)  # raises ValueError if invalid
    assert str(parsed) == cid


def test_id_is_valid_uuid_various_inputs():
    """Multiple inputs all produce valid UUIDs."""
    inputs = [
        ("P001", "hello world", "Section 1", 1, "narrative", 0),
        ("P002", "table data", "Appendix A", 42, "table", 3),
        ("P003", "", "", None, "", 0),
    ]
    for pid, text, heading, page, stype, pos in inputs:
        cid = _deterministic_chunk_id(pid, text, heading=heading, page=page,
                                       source_type=stype, position=pos)
        parsed = uuid.UUID(cid)
        assert str(parsed) == cid
