"""
HTML preview renderer — Navidence-style clean output.
Generates a single self-contained HTML file from a ProgramSpec.
Reviewable in any browser. No dependencies beyond stdlib.
"""

from __future__ import annotations
from pathlib import Path
from .spec_schema import ProgramSpec


def _confidence_badge(conf: float) -> str:
    if conf >= 0.8:
        color, label = "#22c55e", "High"
    elif conf >= 0.6:
        color, label = "#f59e0b", "Medium"
    else:
        color, label = "#ef4444", "Low"
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;">{label} ({conf:.0%})</span>'


def _explicit_badge(explicit: str) -> str:
    colors = {
        "explicit": "#3b82f6",
        "inferred": "#f59e0b",
        "assumed": "#ef4444",
        "ambiguous": "#ef4444",
        "not_found": "#6b7280",
    }
    color = colors.get(explicit, "#6b7280")
    return f'<span style="background:{color};color:white;padding:2px 6px;border-radius:4px;font-size:0.75em;">{explicit}</span>'


def _page_ref(page) -> str:
    if page is not None:
        return f'<span style="color:#6b7280;font-size:0.85em;">p.{page}</span>'
    return ""


def render_html(spec: ProgramSpec) -> str:
    """Render a ProgramSpec as a self-contained HTML document."""

    # ── QC summary bar ────────────────────────────────────────────────
    qc_items = []
    if spec.concepts_with_low_signal:
        qc_items.append(f'<span style="color:#f59e0b;">Low signal: {", ".join(spec.concepts_with_low_signal)}</span>')
    if spec.concepts_with_contradictions:
        qc_items.append(f'<span style="color:#ef4444;">Contradictions: {", ".join(spec.concepts_with_contradictions)}</span>')
    if not qc_items:
        qc_items.append('<span style="color:#22c55e;">No QC flags</span>')
    qc_bar = " | ".join(qc_items)

    # ── Study design section ──────────────────────────────────────────
    design_rows = []
    if spec.design_type:
        design_rows.append(f"<tr><td><strong>Study Design</strong></td><td>{_esc(spec.design_type)}</td></tr>")
    if spec.study_period_start or spec.study_period_end:
        period = f"{_esc(spec.study_period_start or '?')} to {_esc(spec.study_period_end or '?')}"
        design_rows.append(f"<tr><td><strong>Study Period</strong></td><td>{period}</td></tr>")
    if spec.data_source:
        ds = _esc(spec.data_source)
        if spec.data_source_version:
            ds += f" ({_esc(spec.data_source_version)})"
        design_rows.append(f"<tr><td><strong>Data Source</strong></td><td>{ds}</td></tr>")
    design_html = _table_wrap(design_rows) if design_rows else '<p style="color:#6b7280;">Not extracted</p>'

    # ── Index date section ────────────────────────────────────────────
    if spec.index_date_definition:
        idx_html = f"""
        <blockquote style="border-left:3px solid #3b82f6;padding-left:12px;margin:8px 0;">
            {_esc(spec.index_date_definition)}
        </blockquote>
        <p>{_confidence_badge(spec.index_date_confidence)}
        {_page_ref(spec.index_date_page)}
        {f' | Sponsor term: <em>{_esc(spec.index_date_sponsor_term)}</em>' if spec.index_date_sponsor_term else ''}
        </p>"""
    else:
        idx_html = '<p style="color:#6b7280;">Not extracted</p>'

    # ── Follow-up end section ─────────────────────────────────────────
    if spec.follow_up_end_definition:
        fue_html = f"""
        <blockquote style="border-left:3px solid #3b82f6;padding-left:12px;margin:8px 0;">
            {_esc(spec.follow_up_end_definition)}
        </blockquote>
        <p>{_confidence_badge(spec.follow_up_end_confidence)}</p>"""
    else:
        fue_html = '<p style="color:#6b7280;">Not extracted</p>'

    # ── Inclusion criteria section ────────────────────────────────────
    inc_rows = []
    for c in spec.inclusion_criteria:
        op_def = f"<br><em style='color:#6b7280;font-size:0.9em;'>{_esc(c.operational_definition)}</em>" if c.operational_definition else ""
        lookback = f"<br><span style='color:#6b7280;font-size:0.85em;'>Lookback: {_esc(c.lookback_window)}</span>" if c.lookback_window else ""
        inc_rows.append(f"""<tr>
            <td><strong>{_esc(c.criterion_id)}</strong></td>
            <td><span style="background:#e0e7ff;padding:2px 6px;border-radius:3px;font-size:0.8em;">{_esc(c.domain)}</span></td>
            <td>{_esc(c.criterion_label)}</td>
            <td>{_esc(c.description)}{op_def}{lookback}</td>
            <td>{_confidence_badge(c.confidence)} {_explicit_badge(c.explicit)} {_page_ref(c.page)}</td>
        </tr>""")
    inc_html = _table_wrap(inc_rows, headers=["ID", "Domain", "Label", "Description", "Confidence"]) if inc_rows else '<p style="color:#6b7280;">Not extracted</p>'

    # ── Exclusion criteria section ────────────────────────────────────
    exc_rows = []
    for c in spec.exclusion_criteria:
        op_def = f"<br><em style='color:#6b7280;font-size:0.9em;'>{_esc(c.operational_definition)}</em>" if c.operational_definition else ""
        lookback = f"<br><span style='color:#6b7280;font-size:0.85em;'>Lookback: {_esc(c.lookback_window)}</span>" if c.lookback_window else ""
        exc_rows.append(f"""<tr>
            <td><strong>{_esc(c.criterion_id)}</strong></td>
            <td><span style="background:#fce7f3;padding:2px 6px;border-radius:3px;font-size:0.8em;">{_esc(c.domain)}</span></td>
            <td>{_esc(c.criterion_label)}</td>
            <td>{_esc(c.description)}{op_def}{lookback}</td>
            <td>{_confidence_badge(c.confidence)} {_explicit_badge(c.explicit)} {_page_ref(c.page)}</td>
        </tr>""")
    exc_html = _table_wrap(exc_rows, headers=["ID", "Domain", "Label", "Description", "Confidence"]) if exc_rows else '<p style="color:#6b7280;">Not extracted</p>'

    # ── Endpoints section ─────────────────────────────────────────────
    ep_rows = []
    for ep in spec.endpoints:
        components = ""
        if ep.is_composite and ep.components:
            items = "".join(f"<li>{_esc(c)}</li>" for c in ep.components)
            components = f"<br><em>Components:</em><ul style='margin:4px 0;'>{items}</ul>"
        tte = " | Time-to-event" if ep.time_to_event else ""
        ep_rows.append(f"""<tr>
            <td><strong>{_esc(ep.endpoint_id)}</strong></td>
            <td>{_esc(ep.type)}</td>
            <td>{_esc(ep.label)}</td>
            <td>{_esc(ep.description)}{components}</td>
            <td>{_confidence_badge(ep.confidence)}{tte} {_page_ref(ep.page)}</td>
        </tr>""")
    ep_html = _table_wrap(ep_rows, headers=["ID", "Type", "Label", "Description", "Confidence"]) if ep_rows else '<p style="color:#6b7280;">Not extracted</p>'

    # ── Censoring rules section ───────────────────────────────────────
    cr_rows = []
    for cr in spec.censoring_rules:
        applies = f"<br><span style='color:#6b7280;font-size:0.85em;'>Applies to: {_esc(cr.applies_to)}</span>" if cr.applies_to else ""
        cr_rows.append(f"""<tr>
            <td><strong>{_esc(cr.rule_id)}</strong></td>
            <td><span style="background:#fef3c7;padding:2px 6px;border-radius:3px;font-size:0.8em;">{_esc(cr.rule_type)}</span></td>
            <td>{_esc(cr.rule_label)}</td>
            <td>{_esc(cr.description)}{applies}</td>
            <td>{_confidence_badge(cr.confidence)} {_page_ref(cr.page)}</td>
        </tr>""")
    cr_html = _table_wrap(cr_rows, headers=["ID", "Type", "Label", "Description", "Confidence"]) if cr_rows else '<p style="color:#6b7280;">Not extracted</p>'

    # ── Assemble full HTML ────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Program Spec — {_esc(spec.protocol_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #1f2937; line-height: 1.5; }}
  h1 {{ border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
  h2 {{ color: #1e40af; margin-top: 32px; border-bottom: 1px solid #e5e7eb; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 8px 0; }}
  th {{ background: #f3f4f6; text-align: left; padding: 8px 12px; font-size: 0.85em; border-bottom: 2px solid #d1d5db; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e5e7eb; vertical-align: top; }}
  tr:hover {{ background: #f9fafb; }}
  blockquote {{ background: #f0f7ff; margin: 8px 0; padding: 8px 12px; border-radius: 4px; }}
  .header-meta {{ color: #6b7280; font-size: 0.9em; margin-bottom: 16px; }}
  .qc-bar {{ background: #f9fafb; border: 1px solid #e5e7eb; padding: 8px 16px; border-radius: 6px; margin: 12px 0; }}
  .draft-badge {{ background: #fbbf24; color: #1f2937; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }}
</style>
</head>
<body>

<h1>Program Spec <span class="draft-badge">DRAFT — AUTO-GENERATED</span></h1>
<div class="header-meta">
    <strong>Protocol:</strong> {_esc(spec.protocol_id)}
    {f' | <strong>Title:</strong> {_esc(spec.protocol_title)}' if spec.protocol_title else ''}
    <br>
    <strong>Generated:</strong> {spec.generated_at}
    | <strong>Concepts extracted:</strong> {len(spec.concepts_extracted)}
</div>

<div class="qc-bar">{qc_bar}</div>

<h2>Study Design</h2>
{design_html}

<h2>Index Date</h2>
{idx_html}

<h2>Follow-up End</h2>
{fue_html}

<h2>Inclusion Criteria ({len(spec.inclusion_criteria)})</h2>
{inc_html}

<h2>Exclusion Criteria ({len(spec.exclusion_criteria)})</h2>
{exc_html}

<h2>Endpoints ({len(spec.endpoints)})</h2>
{ep_html}

<h2>Censoring Rules ({len(spec.censoring_rules)})</h2>
{cr_html}

<hr style="margin-top:40px;">
<p style="color:#9ca3af;font-size:0.8em;">
    Auto-generated by Protocol Spec Assist v{spec.generator_version}.
    This is a draft for review — not a validated specification.
    All extracted content requires human verification before use.
</p>

</body>
</html>"""

    return html


def save_html(spec: ProgramSpec, output_path: str) -> str:
    """Render and save HTML to file. Returns the output path."""
    html = render_html(spec)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    print(f"[HTML] Spec preview saved: {path}")
    return str(path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _esc(text) -> str:
    """HTML-escape a string."""
    if text is None:
        return ""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _table_wrap(rows: list[str], headers: list[str] | None = None) -> str:
    """Wrap row HTML in a table element."""
    header_html = ""
    if headers:
        ths = "".join(f"<th>{h}</th>" for h in headers)
        header_html = f"<thead><tr>{ths}</tr></thead>"
    body = "".join(rows)
    return f"<table>{header_html}<tbody>{body}</tbody></table>"
