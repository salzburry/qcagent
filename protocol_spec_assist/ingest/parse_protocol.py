"""
Protocol ingestion via Docling.
Preserves: page layout, reading order, table structure, appendices, footnotes.
Output: structured dict ready for chunking and indexing.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedSection:
    heading: str
    heading_level: int
    text: str
    page_start: int
    page_end: int
    source_type: str = "narrative"      # narrative | table | appendix | footnote
    table_data: Optional[list[dict]] = None   # parsed table rows if source_type == table
    raw_json: Optional[dict] = None


@dataclass
class ParsedProtocol:
    protocol_id: str
    title: Optional[str]
    sections: list[ParsedSection] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def sections_by_type(self, source_type: str) -> list[ParsedSection]:
        return [s for s in self.sections if s.source_type == source_type]

    def to_chunks(self) -> list[dict]:
        """Flatten to chunk dicts for indexing."""
        chunks = []
        for sec in self.sections:
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
                for row in sec.table_data:
                    chunk_text = json.dumps(row)
                    chunk_id = _deterministic_chunk_id(
                        self.protocol_id, chunk_text,
                        heading=sec.heading, page=sec.page_start,
                        source_type=sec.source_type,
                    )
                    chunks.append({
                        **base,
                        "text": chunk_text,
                        "is_table_row": True,
                        "chunk_id": chunk_id,
                    })
            else:
                # Split long narrative sections into overlapping chunks
                for chunk_text in _sliding_window(sec.text):
                    chunk_id = _deterministic_chunk_id(
                        self.protocol_id, chunk_text,
                        heading=sec.heading, page=sec.page_start,
                        source_type=sec.source_type,
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
    heading: str = "", page: int = 0, source_type: str = "",
) -> str:
    """Deterministic ID from protocol_id + heading + page + source_type + content hash.
    Includes structural context so identical text on different pages/sections
    gets different IDs."""
    content = f"{protocol_id}:{heading}:{page}:{source_type}:{text}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _sliding_window(text: str, max_chars: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks on sentence boundaries."""
    import re
    # Split on sentence-ending punctuation, bullet points, and blank lines
    parts = re.split(r'(?<=[.!?])\s+|(?:\n\s*\n)|(?:\n\s*[-•●]\s)', text.strip())
    chunks, current = [], ""
    for part in parts:
        if len(current) + len(part) > max_chars and current:
            chunks.append(current.strip())
            current = current[-overlap:] + " " + part
        else:
            current += " " + part
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def parse_protocol(pdf_path: str, protocol_id: Optional[str] = None) -> ParsedProtocol:
    """
    Parse a protocol PDF using Docling.
    Falls back to PyMuPDF if Docling is not installed.
    """
    path = Path(pdf_path)
    pid = protocol_id or path.stem

    try:
        return _parse_with_docling(path, pid)
    except ImportError:
        print("[Parser] Docling not installed, falling back to PyMuPDF.")
        print("[Parser] Install: pip install docling")
        return _parse_with_pymupdf(path, pid)


def _parse_with_docling(path: Path, protocol_id: str) -> ParsedProtocol:
    """
    Docling parse — preserves tables, reading order, layout.
    pip install docling
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions

    options = PdfPipelineOptions()
    options.do_ocr = False              # set True for scanned PDFs
    options.do_table_structure = True   # critical — preserves table structure

    # FIX: Actually pass options to DocumentConverter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=options)
        }
    )
    result = converter.convert(str(path))
    doc = result.document

    sections = []
    current_heading = "Preamble"
    current_level = 0
    current_text = ""
    current_page_start = 0
    current_page_end = 0
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
            current_text += " " + text
            current_page_end = _get_page(item)

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
            heading = f"Table (p.{page})"
            sections.append(ParsedSection(
                heading=heading,
                heading_level=level,
                text=_table_to_text(table_data),
                page_start=page,
                page_end=page,
                source_type="table",
                table_data=table_data,
            ))

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


def _parse_with_pymupdf(path: Path, protocol_id: str) -> ParsedProtocol:
    """
    PyMuPDF fallback — less accurate on tables but functional.
    pip install pymupdf
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
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    is_bold = "bold" in span["font"].lower()
                    is_large = span["size"] > 11
                    if (is_bold or is_large) and len(text) < 120:
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

def _get_page(item) -> int:
    try:
        return item.prov[0].page_no if item.prov else 0
    except Exception:
        return 0

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
    # FIX: join actual column names, not characters of str(dict_keys(...))
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
