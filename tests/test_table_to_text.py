"""Tests for table text serialization."""
from protocol_spec_assist.ingest.parse_protocol import _table_to_text


def test_empty_rows():
    assert _table_to_text([]) == ""


def test_single_row():
    rows = [{"col_a": "val1", "col_b": "val2"}]
    result = _table_to_text(rows)
    lines = result.split("\n")
    assert lines[0] == "col_a | col_b"
    assert lines[1] == "val1 | val2"


def test_multiple_rows():
    rows = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
    result = _table_to_text(rows)
    lines = result.split("\n")
    assert lines[0] == "name | age"
    assert lines[1] == "Alice | 30"
    assert lines[2] == "Bob | 25"


def test_column_names_not_characters():
    """Regression: str(dict_keys(...)) should NOT be character-split."""
    rows = [{"endpoint": "OS", "type": "primary"}]
    result = _table_to_text(rows)
    header = result.split("\n")[0]
    # Must contain actual column names, not 'd | i | c | t | ...'
    assert "endpoint" in header
    assert "type" in header
    assert "d | i | c | t" not in header
