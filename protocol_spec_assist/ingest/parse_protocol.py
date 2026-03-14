"""
Protocol ingestion via Docling.
Preserves: page layout, reading order, table structure, appendices, footnotes.
Output: structured dict ready for chunking and indexing.

Quality scoring: every parse result is graded (pass/warn/fail).
If Docling produces a "fail" grade, falls back to PyMuPDF automatically.
PyMuPDF fallback merges adjacent micro-sections to avoid chunk fragmentation.
"""

from __future__ import annotations
import hashlib
import json
import re
import statistics
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedSection:
    heading: str
    heading_level: int
    text: str
    page_start: Optional[int]
    page_end: Optional[int]
    source_type: str = "narrative"      # narrative | table | appendix | footnote
    table_data: Optional[list[dict]] = None   # parsed table rows if source_type == table
    raw_json: Optional[dict] = None


@dataclass
class ParseQuality:
    """Parse quality metrics — used to decide whether to accept or retry."""
    n_sections: int
    median_heading_len: float
    median_text_len: float
    table_count: int
    empty_ratio: float          # fraction of sections with text < 20 chars
    grade: str                  # "pass" | "warn" | "fail"

    def __str__(self) -> str:
        return (
            f"ParseQuality(grade={self.grade}, sections={self.n_sections}, "
            f"med_heading={self.median_heading_len:.0f}, "
            f"med_text={self.median_text_len:.0f}, "
            f"tables={self.table_count}, empty_ratio={self.empty_ratio:.2f})"
        )


@dataclass
class ParsedProtocol:
    protocol_id: str
    title: Optional[str]
    sections: list[ParsedSection] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    quality: Optional[ParseQuality] = None

    def sections_by_type(self, source_type: str) -> list[ParsedSection]:
        return [s for s in self.sections if s.source_type == source_type]

    def to_chunks(self) -> list[dict]:
        """Flatten to chunk dicts for indexing."""
        chunks = []
        for sec in self.sections:
            # Skip empty sections (e.g. failed table extraction)
            if not sec.text.strip() and not sec.table_data:
                continue

            base = {
                "protocol_id": self.protocol_id,
                "heading": sec.heading,
                "heading_level": sec.heading_level,
                "source_type": sec.source_type,
                "page_start": sec.page_start,
                "page_end": sec.page_end,
            }
            if sec.source_type == "table" and sec.table_data:
                # Each table row becomes its own chunk for precise retrieval
                for row_idx, row in enumerate(sec.table_data):
                    chunk_text = json.dumps(row)
                    chunk_id = _deterministic_chunk_id(
                        self.protocol_id, chunk_text,
                        heading=sec.heading, page=sec.page_start,
                        source_type=sec.source_type,
                        position=row_idx,
                    )
                    chunks.append({
                        **base,
                        "text": chunk_text,
                        "is_table_row": True,
                        "chunk_id": chunk_id,
                    })
            else:
                # Skip if text is blank (empty table with no table_data)
                if not sec.text.strip():
                    continue
                # Split long narrative sections into overlapping chunks
                for chunk_idx, chunk_text in enumerate(_sliding_window(sec.text)):
                    chunk_id = _deterministic_chunk_id(
                        self.protocol_id, chunk_text,
                        heading=sec.heading, page=sec.page_start,
                        source_type=sec.source_type,
                        position=chunk_idx,
                    )
                    chunks.append({
                        **base,
                        "text": chunk_text,
                        "is_table_row": False,
                        "chunk_id": chunk_id,
                    })
        return chunks


def _deterministic_chunk_id(
    protocol_id: str, text: str,
    heading: str = "", page: Optional[int] = None,
    source_type: str = "", position: int = 0,
) -> str:
    """Deterministic UUID from protocol_id + heading + page + source_type + position + content hash.
    Returns a valid UUID string — required by Qdrant for point IDs.
    Includes structural context and position so identical text in the same
    section/page (e.g. duplicate table rows) gets different IDs."""
    content = f"{protocol_id}:{heading}:{page}:{source_type}:{position}:{text}"
    hex32 = hashlib.sha256(content.encode()).hexdigest()[:32]
    return str(uuid.UUID(hex=hex32))


# ── Quality scoring ───────────────────────────────────────────────────────────

def _quality_score(parsed: ParsedProtocol) -> ParseQuality:
    """Score parse quality to decide accept vs fallback."""
    sections = parsed.sections
    if not sections:
        return ParseQuality(
            n_sections=0, median_heading_len=0, median_text_len=0,
            table_count=0, empty_ratio=1.0, grade="fail",
        )

    heading_lens = [len(s.heading) for s in sections]
    text_lens = [len(s.text) for s in sections]
    table_count = sum(1 for s in sections if s.source_type == "table")
    empty_count = sum(1 for s in sections if len(s.text.strip()) < 20 and s.source_type != "table")
    empty_ratio = empty_count / len(sections) if sections else 1.0

    med_heading = statistics.median(heading_lens)
    med_text = statistics.median(text_lens)

    # Grade logic:
    # - fail: median heading < 5 chars (noise headings) OR median text < 30 chars
    #         OR empty_ratio > 0.6 OR fewer than 3 sections
    # - warn: median text < 80 chars OR empty_ratio > 0.3
    # - pass: everything else
    if med_heading < 5 or med_text < 30 or empty_ratio > 0.6 or len(sections) < 3:
        grade = "fail"
    elif med_text < 80 or empty_ratio > 0.3:
        grade = "warn"
    else:
        grade = "pass"

    return ParseQuality(
        n_sections=len(sections),
        median_heading_len=med_heading,
        median_text_len=med_text,
        table_count=table_count,
        empty_ratio=empty_ratio,
        grade=grade,
    )


# ── Sliding window chunking (bullet-aware) ────────────────────────────────────

# Patterns that mark atomic list items — should not be split mid-item
_LIST_ITEM_RE = re.compile(
    r'\n\s*'                       # newline + optional whitespace
    r'(?:'
    r'\d{1,3}[.)]\s'               # numbered: "1. ", "12) "
    r'|[a-z][.)]\s'                # lettered: "a. ", "b) "
    r'|[ivxIVX]+[.)]\s'            # roman: "i. ", "iv) "
    r'|[-•●○▪▸]\s'                 # bullet markers
    r')'
)

def _sliding_window(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks, preserving list items as atomic units.

    Splits on:
      1. Blank lines (paragraph boundaries)
      2. Sentence-ending punctuation followed by whitespace
      3. List item markers (numbered, lettered, bulleted)

    Each list item is kept as one unit — never split mid-item.
    """
    text = text.strip()
    if not text:
        return [text]

    # Split into atomic parts: paragraphs, sentences, and list items
    # First split on blank lines
    paragraphs = re.split(r'\n\s*\n', text)

    parts = []
    for para in paragraphs:
        # Within each paragraph, split on list item boundaries
        items = _LIST_ITEM_RE.split(para)
        # Also split long non-list text on sentence boundaries
        for item in items:
            item = item.strip()
            if not item:
                continue
            if len(item) > max_chars:
                # Long prose block — split on sentence boundaries
                sentences = re.split(r'(?<=[.!?])\s+', item)
                parts.extend(s.strip() for s in sentences if s.strip())
            else:
                parts.append(item)

    if not parts:
        return [text]

    # Build overlapping chunks from parts
    chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) + 1 > max_chars and current:
            chunks.append(current.strip())
            # Overlap: take the tail of the current chunk
            current = current[-overlap:] + " " + part
        else:
            current = (current + " " + part) if current else part
    if current.strip():
        chunks.append(current.strip())

    return chunks or [text]


# ── Main entry point ──────────────────────────────────────────────────────────

def parse_protocol(pdf_path: str, protocol_id: Optional[str] = None) -> ParsedProtocol:
    """
    Parse a protocol PDF using a multi-strategy approach with quality gating.

    Strategy order:
    1. Docling (no OCR) — best table/layout preservation for clean PDFs.
    2. Docling (with OCR) — handles scanned or image-heavy PDFs.
    3. PyMuPDF heading-based — lightweight structural parse with micro-section merging.
    4. PyMuPDF page-first — last resort; treats each page as a section, no heading
       detection. Produces fewer, larger sections that avoid junk-heading fragmentation.

    All strategies are evaluated and scored. The best-scoring acceptable result
    is returned (not the first acceptable one). This ensures that a later
    strategy with grade "pass" wins over an earlier strategy with grade "warn".

    If all strategies produce "fail", the best-scoring result is returned so
    the downstream fail-closed gate in protocol_run can emit a shell spec.
    """
    path = Path(pdf_path)
    pid = protocol_id or path.stem

    best: Optional[ParsedProtocol] = None
    best_score: float = -1.0

    def _score_value(q: ParseQuality) -> float:
        """Numeric score for comparing quality across strategies.

        Components (max ~4.0):
          - grade:       pass=2.0, warn=1.0, fail=0.0
          - empty_ratio: up to 0.5 (lower empty ratio = higher score)
          - text_len:    up to 0.5 (longer median text = higher score)
          - tables:      up to 1.0 (more tables = higher score, capped at 10)

        Table extraction is heavily weighted because protocol tables contain
        critical structured data (eligibility criteria, endpoints, visit
        schedules) that narrative text alone cannot replace.
        """
        grade_map = {"pass": 2.0, "warn": 1.0, "fail": 0.0}
        return (
            grade_map.get(q.grade, 0.0)
            + (1.0 - q.empty_ratio) * 0.5
            + min(q.median_text_len / 200.0, 0.5)
            + min(q.table_count / 10.0, 1.0)
        )

    def _try_strategy(result: ParsedProtocol, label: str) -> None:
        """Score a parse result and update best if this is the highest-scoring."""
        nonlocal best, best_score
        quality = _quality_score(result)
        result.quality = quality
        sv = _score_value(quality)
        print(f"[Parser] {label}: {quality} (score={sv:.2f})")
        if sv > best_score:
            best = result
            best_score = sv

    # Strategy 1: Docling (no OCR)
    try:
        result = _parse_with_docling(path, pid, do_ocr=False)
        _try_strategy(result, "Docling (no OCR)")

        # Strategy 2: Docling (with OCR) — recovers scanned/image PDFs
        print("[Parser] Trying Docling with OCR enabled...")
        try:
            result_ocr = _parse_with_docling(path, pid, do_ocr=True)
            _try_strategy(result_ocr, "Docling (OCR)")
        except Exception as e:
            print(f"[Parser] Docling OCR failed ({type(e).__name__}: {e}), continuing to PyMuPDF.")

    except ImportError:
        print("[Parser] Docling not installed, falling back to PyMuPDF.")
        print("[Parser] Install: pip install docling")
    except Exception as e:
        print(f"[Parser] Docling failed ({type(e).__name__}: {e}), falling back to PyMuPDF.")

    # Strategy 3: PyMuPDF heading-based with micro-section merging
    try:
        fallback = _parse_with_pymupdf(path, pid)
        fallback = _merge_micro_sections(fallback)
        _try_strategy(fallback, "PyMuPDF heading-based (merged)")
    except Exception as e:
        print(f"[Parser] PyMuPDF heading-based failed ({type(e).__name__}: {e}).")

    # Strategy 4: PyMuPDF page-first — no heading detection, one section per page.
    # This avoids the junk-heading problem entirely for difficult PDFs.
    try:
        page_first = _parse_with_pymupdf_page_first(path, pid)
        _try_strategy(page_first, "PyMuPDF page-first")
    except Exception as e:
        print(f"[Parser] PyMuPDF page-first failed ({type(e).__name__}: {e}).")

    # Return the best result across all strategies
    if best is not None:
        print(f"[Parser] Selected: {best.quality} (score={best_score:.2f})")
        return best

    # Absolute fallback: empty protocol
    return ParsedProtocol(
        protocol_id=pid, title=None, sections=[],
        quality=ParseQuality(
            n_sections=0, median_heading_len=0, median_text_len=0,
            table_count=0, empty_ratio=1.0, grade="fail",
        ),
    )


# ── Docling parser ────────────────────────────────────────────────────────────

def _parse_with_docling(path: Path, protocol_id: str, do_ocr: bool = False) -> ParsedProtocol:
    """
    Docling parse — preserves tables, reading order, layout.
    pip install docling

    Args:
        do_ocr: Enable OCR for scanned/image-heavy PDFs. Slower but recovers
                text from non-selectable pages.
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    options = PdfPipelineOptions()
    options.do_ocr = do_ocr
    options.do_table_structure = True   # critical — preserves table structure

    # Force OCR to use CPU to avoid CUDA OOM when vLLM already occupies the GPU.
    # RapidOCR (used by Docling) defaults to GPU if available, which crashes
    # or degrades vLLM inference when both compete for VRAM on A100 40GB.
    _prev_cuda = None
    if do_ocr:
        import os
        os.environ["RAPIDOCR_DEVICE"] = "cpu"
        # Also hide CUDA from any other OCR sub-libraries that ignore RAPIDOCR_DEVICE
        _prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=options)
            }
        )
        result = converter.convert(str(path))
    finally:
        # Restore CUDA visibility so vLLM inference is not affected
        if do_ocr:
            if _prev_cuda is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = _prev_cuda
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    doc = result.document

    sections = []
    current_heading = "Preamble"
    current_level = 0
    current_text = ""
    current_page_start: Optional[int] = None
    current_page_end: Optional[int] = None
    current_source_type = "narrative"

    for item, level in doc.iterate_items():
        item_type = type(item).__name__

        if item_type == "SectionHeaderItem":
            # Flush accumulated text as a section
            if current_text.strip():
                sections.append(ParsedSection(
                    heading=current_heading,
                    heading_level=current_level,
                    text=current_text.strip(),
                    page_start=current_page_start,
                    page_end=current_page_end,
                    source_type=current_source_type,
                ))
            # Start new section
            current_heading = getattr(item, "text", "")[:120]
            current_level = level
            current_text = ""
            current_page_start = _get_page(item)
            current_page_end = current_page_start
            current_source_type = _classify_section(current_heading, "")

        elif item_type == "TextItem":
            # Append to current section instead of creating standalone
            text = getattr(item, "text", "")
            page = _get_page(item)
            # Initialize page_start on first text in a section
            if current_page_start is None and page is not None:
                current_page_start = page
            if page is not None:
                current_page_end = page
            current_text += " " + text

        elif item_type == "TableItem":
            # Flush accumulated text before table
            if current_text.strip():
                sections.append(ParsedSection(
                    heading=current_heading,
                    heading_level=current_level,
                    text=current_text.strip(),
                    page_start=current_page_start,
                    page_end=current_page_end,
                    source_type=current_source_type,
                ))
                current_text = ""

            table_data = _extract_table_data(item)
            page = _get_page(item)
            table_text = _table_to_text(table_data)

            # Skip empty tables entirely
            if not table_data and not table_text:
                # Reset section start tracking for post-table narrative
                current_page_start = page
                current_page_end = page
                continue

            # Preserve parent section context in table heading
            heading = f"{current_heading} | Table (p.{page})" if page else f"{current_heading} | Table"
            sections.append(ParsedSection(
                heading=heading,
                heading_level=level,
                text=table_text,
                page_start=page,
                page_end=page,
                source_type="table",
                table_data=table_data,
            ))

            # Reset section start tracking for post-table narrative
            current_page_start = page
            current_page_end = page

    # Flush final section
    if current_text.strip():
        sections.append(ParsedSection(
            heading=current_heading,
            heading_level=current_level,
            text=current_text.strip(),
            page_start=current_page_start,
            page_end=current_page_end,
            source_type=current_source_type,
        ))

    return ParsedProtocol(
        protocol_id=protocol_id,
        title=_extract_title(doc),
        sections=sections,
    )


# ── PyMuPDF fallback ──────────────────────────────────────────────────────────

def _parse_with_pymupdf(path: Path, protocol_id: str) -> ParsedProtocol:
    """
    PyMuPDF heading-based fallback — less accurate on tables but functional.
    pip install pymupdf

    Improved heading detection: a bold/large span is only treated as a heading if it:
      - has at least 5 characters (avoids single-word/number junk headings)
      - contains at least one letter (avoids bullet markers, page numbers)
      - is the dominant span in its line (not a bold word inside a paragraph)
    """
    import fitz

    doc = fitz.open(str(path))
    sections = []
    current_heading = "Preamble"
    current_level = 0
    current_text = ""
    current_page = 1

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                # Collect all spans in this line to check if heading-candidate
                # is the dominant span (not just an inline bold word)
                line_spans = [s for s in line["spans"] if s["text"].strip()]
                line_text_len = sum(len(s["text"].strip()) for s in line_spans)

                for span in line_spans:
                    text = span["text"].strip()
                    if not text:
                        continue
                    is_bold = "bold" in span["font"].lower()
                    is_large = span["size"] > 11

                    # Improved heading heuristics:
                    # 1. Must be bold or large
                    # 2. Must have >= 5 chars (avoid junk like "1.", "•", "A")
                    # 3. Must contain at least one letter
                    # 4. Must be < 120 chars
                    # 5. This span must be the majority of the line text
                    #    (avoids treating inline bold words as headings)
                    is_heading_candidate = (
                        (is_bold or is_large)
                        and 5 <= len(text) < 120
                        and any(c.isalpha() for c in text)
                        and (len(text) >= line_text_len * 0.7 or line_text_len < 80)
                    )

                    if is_heading_candidate:
                        if current_text.strip():
                            sections.append(ParsedSection(
                                heading=current_heading,
                                heading_level=current_level,
                                text=current_text.strip(),
                                page_start=current_page,
                                page_end=page_num,
                                source_type=_classify_section(current_heading, current_text),
                            ))
                        current_heading = text
                        current_level = 1 if is_large else 2
                        current_text = ""
                        current_page = page_num
                    else:
                        current_text += " " + text

    if current_text.strip():
        sections.append(ParsedSection(
            heading=current_heading,
            heading_level=current_level,
            text=current_text.strip(),
            page_start=current_page,
            page_end=current_page,
            source_type=_classify_section(current_heading, current_text),
        ))

    doc.close()
    return ParsedProtocol(protocol_id=protocol_id, title=None, sections=sections)


def _parse_with_pymupdf_page_first(path: Path, protocol_id: str) -> ParsedProtocol:
    """
    PyMuPDF page-first fallback — one section per page, no heading detection.

    This avoids the junk-heading fragmentation problem entirely. Each page
    becomes a single section with a synthetic heading ("Page N"). The result
    has fewer, larger sections that are better for retrieval even if structural
    headings are lost.

    Also attempts basic table extraction from each page.
    """
    import fitz

    doc = fitz.open(str(path))
    sections = []
    title = None

    for page_num, page in enumerate(doc, start=1):
        # Extract all text on the page
        page_text = page.get_text("text").strip()

        if not page_text:
            continue

        # Use the first non-trivial text line from page 1 as title
        if title is None and page_num == 1:
            for line in page_text.split("\n"):
                line = line.strip()
                if len(line) > 10 and any(c.isalpha() for c in line):
                    title = line[:120]
                    break

        # Try to extract tables from this page
        try:
            tables = page.find_tables()
            if tables and tables.tables:
                for tbl_idx, table in enumerate(tables.tables):
                    table_data = []
                    df_rows = table.extract()
                    if df_rows and len(df_rows) > 1:
                        headers = [str(h or f"col_{i}") for i, h in enumerate(df_rows[0])]
                        for row in df_rows[1:]:
                            table_data.append(dict(zip(headers, [str(v or "") for v in row])))

                    if table_data:
                        table_text = _table_to_text(table_data)
                        sections.append(ParsedSection(
                            heading=f"Page {page_num} | Table {tbl_idx + 1}",
                            heading_level=2,
                            text=table_text,
                            page_start=page_num,
                            page_end=page_num,
                            source_type="table",
                            table_data=table_data,
                        ))
        except Exception:
            pass  # table extraction is best-effort

        # Add narrative section for the page
        sections.append(ParsedSection(
            heading=f"Page {page_num}",
            heading_level=1,
            text=page_text,
            page_start=page_num,
            page_end=page_num,
            source_type="narrative",
        ))

    doc.close()
    return ParsedProtocol(protocol_id=protocol_id, title=title, sections=sections)


# ── Post-processing: merge micro-sections ─────────────────────────────────────

_MIN_SECTION_TEXT_LEN = 50

def _merge_micro_sections(parsed: ParsedProtocol) -> ParsedProtocol:
    """Merge adjacent narrative sections where the text body is too short.

    PyMuPDF often creates a new section for every bold span, resulting in
    hundreds of micro-sections (e.g. 552 sections with ~17 char text bodies).
    This merges them into their successor section, keeping the first heading.
    """
    if len(parsed.sections) < 2:
        return parsed

    merged: list[ParsedSection] = []
    pending: Optional[ParsedSection] = None

    for sec in parsed.sections:
        # Tables are never merged
        if sec.source_type == "table":
            if pending:
                merged.append(pending)
                pending = None
            merged.append(sec)
            continue

        if pending is None:
            pending = ParsedSection(
                heading=sec.heading,
                heading_level=sec.heading_level,
                text=sec.text,
                page_start=sec.page_start,
                page_end=sec.page_end,
                source_type=sec.source_type,
                table_data=sec.table_data,
            )
        else:
            # Decide: is the pending section too small to stand alone?
            if len(pending.text.strip()) < _MIN_SECTION_TEXT_LEN:
                # Merge into current: keep pending heading, append texts
                pending.text = pending.text + "\n" + sec.heading + "\n" + sec.text
                pending.page_end = sec.page_end
            else:
                # Pending is big enough — flush it, start new pending
                merged.append(pending)
                pending = ParsedSection(
                    heading=sec.heading,
                    heading_level=sec.heading_level,
                    text=sec.text,
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    source_type=sec.source_type,
                    table_data=sec.table_data,
                )

    if pending:
        merged.append(pending)

    parsed.sections = merged
    return parsed


# ── Helpers ───────────────────────────────────────────────────────────────────

APPENDIX_SIGNALS = ["appendix", "supplement", "annex", "schedule of assessments"]
FOOTNOTE_SIGNALS = ["footnote", "note:", "\u2020", "\u2021", "\u00a7", "*"]
TABLE_SIGNALS = ["table", "exhibit", "figure"]

def _classify_section(heading: str, text: str) -> str:
    h = heading.lower()
    t = text.lower()[:200]
    if any(s in h for s in APPENDIX_SIGNALS):
        return "appendix"
    if any(s in h or s in t for s in FOOTNOTE_SIGNALS):
        return "footnote"
    if any(s in h for s in TABLE_SIGNALS):
        return "table"
    return "narrative"

def _get_page(item) -> Optional[int]:
    """Return page number or None if unavailable."""
    try:
        if item.prov:
            return item.prov[0].page_no
    except Exception:
        pass
    return None

def _extract_table_data(item) -> list[dict]:
    try:
        df = item.export_to_dataframe()
        return df.to_dict(orient="records")
    except Exception:
        return []

def _table_to_text(rows: list[dict]) -> str:
    """Convert table rows to pipe-delimited text for indexing."""
    if not rows:
        return ""
    header = " | ".join(str(k) for k in rows[0].keys())
    body = "\n".join(" | ".join(str(v) for v in row.values()) for row in rows)
    return header + "\n" + body

def _extract_title(doc) -> Optional[str]:
    try:
        for item, _ in doc.iterate_items():
            if type(item).__name__ == "TitleItem":
                return item.text
    except Exception:
        pass
    return None
