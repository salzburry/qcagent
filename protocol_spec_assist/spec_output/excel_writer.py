"""
Excel workbook writer — formatted .xlsx output of draft program spec.
Requires openpyxl.
"""

from __future__ import annotations
from pathlib import Path

from .spec_schema import ProgramSpec

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False


# Color fills by confidence level
FILL_GREEN = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid") if OPENPYXL_AVAILABLE else None
FILL_YELLOW = PatternFill(start_color="FEF9C3", end_color="FEF9C3", fill_type="solid") if OPENPYXL_AVAILABLE else None
FILL_RED = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid") if OPENPYXL_AVAILABLE else None
FILL_INFERRED = PatternFill(start_color="E0E7FF", end_color="E0E7FF", fill_type="solid") if OPENPYXL_AVAILABLE else None
FILL_HEADER = PatternFill(start_color="1E40AF", end_color="1E40AF", fill_type="solid") if OPENPYXL_AVAILABLE else None
FONT_HEADER = Font(color="FFFFFF", bold=True, size=11) if OPENPYXL_AVAILABLE else None
FONT_NORMAL = Font(size=10) if OPENPYXL_AVAILABLE else None


def _confidence_fill(conf: float | None):
    if conf is None:
        return None
    if conf >= 0.8:
        return FILL_GREEN
    if conf >= 0.6:
        return FILL_YELLOW
    return FILL_RED


def _style_header(ws, row: int, ncols: int):
    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = FONT_HEADER
        cell.fill = FILL_HEADER
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _style_row(ws, row: int, ncols: int, confidence: float | None = None, explicit: str = "explicit"):
    fill = None
    if explicit == "inferred":
        fill = FILL_INFERRED
    elif confidence is not None:
        fill = _confidence_fill(confidence)

    for col in range(1, ncols + 1):
        cell = ws.cell(row=row, column=col)
        cell.font = FONT_NORMAL
        cell.alignment = Alignment(vertical="top", wrap_text=True)
        if fill:
            cell.fill = fill


def save_excel(spec: ProgramSpec, output_path: str) -> str:
    """Generate formatted Excel workbook from ProgramSpec."""
    if not OPENPYXL_AVAILABLE:
        raise ImportError("openpyxl required for Excel output. Run: pip install openpyxl")

    wb = Workbook()

    # ── Tab 1: Overview ──
    ws = wb.active
    ws.title = "Overview"
    headers = ["Field", "Value", "Confidence", "Status", "Page", "Notes"]
    ws.append(headers)
    _style_header(ws, 1, len(headers))

    overview_rows = [
        ("Protocol ID", spec.protocol_id, "", spec.generation_mode, "", ""),
        ("Design Type", spec.study_design.design_type.value, "", "", "", ""),
        ("Data Source", spec.study_design.data_source.value, "", "", "", spec.study_design.data_source.notes),
        ("Study Period Start", spec.study_design.study_period_start.value, "", "", "", ""),
        ("Study Period End", spec.study_design.study_period_end.value, "", "", "", ""),
        (
            "Index Date",
            spec.index_date.value,
            f"{spec.index_date.confidence:.0%}" if spec.index_date.confidence else "",
            spec.index_date.explicit,
            str(spec.index_date.page or ""),
            spec.index_date.notes,
        ),
        (
            "Follow-up End",
            spec.follow_up_end.value,
            f"{spec.follow_up_end.confidence:.0%}" if spec.follow_up_end.confidence else "",
            spec.follow_up_end.explicit,
            str(spec.follow_up_end.page or ""),
            spec.follow_up_end.notes,
        ),
        (
            "Primary Endpoint",
            spec.primary_endpoint.value,
            f"{spec.primary_endpoint.confidence:.0%}" if spec.primary_endpoint.confidence else "",
            spec.primary_endpoint.explicit,
            str(spec.primary_endpoint.page or ""),
            spec.primary_endpoint.notes,
        ),
    ]
    for r, row_data in enumerate(overview_rows, 2):
        for c, val in enumerate(row_data, 1):
            ws.cell(row=r, column=c, value=val)
        conf = None
        if row_data[2]:
            try:
                conf = float(row_data[2].strip("%")) / 100
            except (ValueError, AttributeError):
                pass
        _style_row(ws, r, len(headers), confidence=conf, explicit=row_data[3] if len(row_data) > 3 else "explicit")

    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 12
    ws.column_dimensions["D"].width = 12
    ws.column_dimensions["E"].width = 8
    ws.column_dimensions["F"].width = 30

    # ── Tab 2: Inclusion Criteria ──
    ws_inc = wb.create_sheet("Inclusion Criteria")
    inc_headers = ["Criterion", "Evidence", "Domain", "Lookback", "Confidence", "Status", "Page"]
    ws_inc.append(inc_headers)
    _style_header(ws_inc, 1, len(inc_headers))

    for r, crit in enumerate(spec.inclusion_criteria, 2):
        ws_inc.cell(row=r, column=1, value=crit.label)
        ws_inc.cell(row=r, column=2, value=crit.value)
        ws_inc.cell(row=r, column=3, value=crit.domain)
        ws_inc.cell(row=r, column=4, value=crit.lookback_window or "")
        ws_inc.cell(row=r, column=5, value=f"{crit.confidence:.0%}" if crit.confidence else "")
        ws_inc.cell(row=r, column=6, value=crit.explicit)
        ws_inc.cell(row=r, column=7, value=str(crit.page or ""))
        _style_row(ws_inc, r, len(inc_headers), confidence=crit.confidence, explicit=crit.explicit)

    ws_inc.column_dimensions["A"].width = 25
    ws_inc.column_dimensions["B"].width = 60
    ws_inc.column_dimensions["C"].width = 14
    ws_inc.column_dimensions["D"].width = 22
    ws_inc.column_dimensions["E"].width = 12

    # ── Tab 3: Exclusion Criteria ──
    ws_exc = wb.create_sheet("Exclusion Criteria")
    ws_exc.append(inc_headers)  # same headers
    _style_header(ws_exc, 1, len(inc_headers))

    for r, crit in enumerate(spec.exclusion_criteria, 2):
        ws_exc.cell(row=r, column=1, value=crit.label)
        ws_exc.cell(row=r, column=2, value=crit.value)
        ws_exc.cell(row=r, column=3, value=crit.domain)
        ws_exc.cell(row=r, column=4, value=crit.lookback_window or "")
        ws_exc.cell(row=r, column=5, value=f"{crit.confidence:.0%}" if crit.confidence else "")
        ws_exc.cell(row=r, column=6, value=crit.explicit)
        ws_exc.cell(row=r, column=7, value=str(crit.page or ""))
        _style_row(ws_exc, r, len(inc_headers), confidence=crit.confidence, explicit=crit.explicit)

    ws_exc.column_dimensions["A"].width = 25
    ws_exc.column_dimensions["B"].width = 60
    ws_exc.column_dimensions["C"].width = 14
    ws_exc.column_dimensions["D"].width = 22
    ws_exc.column_dimensions["E"].width = 12

    # ── Tab 4: Endpoints ──
    ws_ep = wb.create_sheet("Endpoints")
    ep_headers = ["Endpoint", "Evidence", "Confidence", "Status", "Page", "Notes"]
    ws_ep.append(ep_headers)
    _style_header(ws_ep, 1, len(ep_headers))

    for r, (label, entry) in enumerate([
        ("Primary Endpoint", spec.primary_endpoint),
    ], 2):
        ws_ep.cell(row=r, column=1, value=label)
        ws_ep.cell(row=r, column=2, value=entry.value)
        ws_ep.cell(row=r, column=3, value=f"{entry.confidence:.0%}" if entry.confidence else "")
        ws_ep.cell(row=r, column=4, value=entry.explicit)
        ws_ep.cell(row=r, column=5, value=str(entry.page or ""))
        ws_ep.cell(row=r, column=6, value=entry.notes)
        _style_row(ws_ep, r, len(ep_headers), confidence=entry.confidence, explicit=entry.explicit)

    ws_ep.column_dimensions["A"].width = 22
    ws_ep.column_dimensions["B"].width = 60

    # ── Tab 5: Censoring Rules ──
    ws_cr = wb.create_sheet("Censoring Rules")
    cr_headers = ["Rule", "Evidence", "Type", "Applies To", "Confidence", "Status", "Page"]
    ws_cr.append(cr_headers)
    _style_header(ws_cr, 1, len(cr_headers))

    for r, rule in enumerate(spec.censoring_rules, 2):
        ws_cr.cell(row=r, column=1, value=rule.label)
        ws_cr.cell(row=r, column=2, value=rule.value)
        ws_cr.cell(row=r, column=3, value=rule.rule_type)
        ws_cr.cell(row=r, column=4, value=rule.applies_to or "all")
        ws_cr.cell(row=r, column=5, value=f"{rule.confidence:.0%}" if rule.confidence else "")
        ws_cr.cell(row=r, column=6, value=rule.explicit)
        ws_cr.cell(row=r, column=7, value=str(rule.page or ""))
        _style_row(ws_cr, r, len(cr_headers), confidence=rule.confidence, explicit=rule.explicit)

    ws_cr.column_dimensions["A"].width = 22
    ws_cr.column_dimensions["B"].width = 60
    ws_cr.column_dimensions["C"].width = 16
    ws_cr.column_dimensions["D"].width = 18

    # Save
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(str(path))
    return str(path)
