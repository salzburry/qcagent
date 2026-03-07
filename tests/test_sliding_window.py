"""Tests for text chunking / sliding window."""
from protocol_spec_assist.ingest.parse_protocol import _sliding_window


def test_short_text_single_chunk():
    text = "This is a short sentence."
    chunks = _sliding_window(text, max_chars=1000)
    assert len(chunks) == 1
    assert "short sentence" in chunks[0]


def test_long_text_multiple_chunks():
    text = ". ".join([f"Sentence number {i}" for i in range(50)])
    chunks = _sliding_window(text, max_chars=100, overlap=20)
    assert len(chunks) > 1


def test_empty_text():
    chunks = _sliding_window("")
    assert len(chunks) == 1  # returns [text] as fallback


def test_bullet_splitting():
    """Bullets should be split points."""
    text = "Header text.\n - Item one details here\n - Item two details here\n - Item three details"
    chunks = _sliding_window(text, max_chars=50, overlap=10)
    assert len(chunks) >= 2


def test_overlap_present():
    """Chunks should share some text at boundaries."""
    text = "First sentence is here. Second sentence is here. Third sentence is here. Fourth sentence is here."
    chunks = _sliding_window(text, max_chars=60, overlap=20)
    if len(chunks) >= 2:
        # Some overlap expected between consecutive chunks
        words_0 = set(chunks[0].split())
        words_1 = set(chunks[1].split())
        assert len(words_0 & words_1) > 0
