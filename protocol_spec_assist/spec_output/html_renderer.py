"""
HTML renderer — self-contained HTML preview of draft program spec.
No external CSS/JS dependencies.
"""

from __future__ import annotations
from pathlib import Path

from .spec_schema import ProgramSpec


def _confidence_badge(conf: float | None) -> str:
    if conf is None:
        return '<span class="badge badge-grey">N/A</span>'
    if conf >= 0.8:
        return f'<span class="badge badge-green">{conf:.0%}</span>'
    if conf >= 0.6:
        return f'<span class="badge badge-yellow">{conf:.0%}</span>'
    return f'<span class="badge badge-red">{conf:.0%}</span>'


def _explicit_badge(explicit: str) -> str:
    if explicit == "explicit":
        return '<span class="badge badge-green">explicit</span>'
    if explicit == "inferred":
        return '<span class="badge badge-yellow">inferred</span>'
    return f'<span class="badge badge-grey">{explicit}</span>'


def _page_ref(page: int | None) -> str:
    if page is None:
        return ""
    return f' <span class="page-ref">p.{page}</span>'


def render_html(spec: ProgramSpec) -> str:
    """Generate self-contained HTML preview of draft program spec."""

    # QC warning bar
    qc_html = ""
    if spec.qc_warnings:
        items = "".join(f"<li>{w}</li>" for w in spec.qc_warnings)
        qc_html = f'<div class="qc-bar"><strong>QC Warnings:</strong><ul>{items}</ul></div>'

    mode_label = "DRAFT" if spec.generation_mode == "draft" else "REVIEWED"
    mode_class = "mode-draft" if spec.generation_mode == "draft" else "mode-reviewed"

    # Study design section
    sd = spec.study_design
    study_design_rows = ""
    for label, entry in [
        ("Design Type", sd.design_type),
        ("Data Source", sd.data_source),
        ("Study Period Start", sd.study_period_start),
        ("Study Period End", sd.study_period_end),
    ]:
        if entry.value:
            study_design_rows += (
                f"<tr><td>{label}</td><td>{entry.value}</td>"
                f"<td>{entry.notes or ''}</td></tr>\n"
            )

    # Single-value concepts
    single_concepts = ""
    for label, entry in [
        ("Index Date", spec.index_date),
        ("Follow-up End", spec.follow_up_end),
        ("Primary Endpoint", spec.primary_endpoint),
    ]:
        if entry.value:
            single_concepts += (
                f"<tr><td>{label}</td>"
                f"<td>{entry.value}</td>"
                f"<td>{_confidence_badge(entry.confidence)} "
                f"{_explicit_badge(entry.explicit)}"
                f"{_page_ref(entry.page)}</td></tr>\n"
            )

    # Inclusion criteria
    inc_rows = ""
    for c in spec.inclusion_criteria:
        inc_rows += (
            f"<tr><td>{c.label}</td>"
            f"<td>{c.value}</td>"
            f"<td>{c.domain}</td>"
            f"<td>{c.lookback_window or ''}</td>"
            f"<td>{_confidence_badge(c.confidence)} "
            f"{_explicit_badge(c.explicit)}"
            f"{_page_ref(c.page)}</td></tr>\n"
        )

    # Exclusion criteria
    exc_rows = ""
    for c in spec.exclusion_criteria:
        exc_rows += (
            f"<tr><td>{c.label}</td>"
            f"<td>{c.value}</td>"
            f"<td>{c.domain}</td>"
            f"<td>{c.lookback_window or ''}</td>"
            f"<td>{_confidence_badge(c.confidence)} "
            f"{_explicit_badge(c.explicit)}"
            f"{_page_ref(c.page)}</td></tr>\n"
        )

    # Censoring rules
    cens_rows = ""
    for r in spec.censoring_rules:
        cens_rows += (
            f"<tr><td>{r.label}</td>"
            f"<td>{r.value}</td>"
            f"<td>{r.rule_type}</td>"
            f"<td>{r.applies_to or 'all'}</td>"
            f"<td>{_confidence_badge(r.confidence)} "
            f"{_explicit_badge(r.explicit)}"
            f"{_page_ref(r.page)}</td></tr>\n"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Program Spec — {spec.protocol_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; color: #333; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
h2 {{ color: #1e40af; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; vertical-align: top; }}
th {{ background: #f3f4f6; font-weight: 600; }}
.badge {{ padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: 500; }}
.badge-green {{ background: #dcfce7; color: #166534; }}
.badge-yellow {{ background: #fef9c3; color: #854d0e; }}
.badge-red {{ background: #fee2e2; color: #991b1b; }}
.badge-grey {{ background: #f3f4f6; color: #6b7280; }}
.page-ref {{ color: #6b7280; font-size: 0.85em; }}
.qc-bar {{ background: #fef3c7; border: 1px solid #f59e0b; border-radius: 6px; padding: 12px 16px; margin: 16px 0; }}
.qc-bar ul {{ margin: 4px 0 0 0; padding-left: 20px; }}
.mode-draft {{ background: #fef9c3; color: #854d0e; padding: 4px 12px; border-radius: 4px; font-weight: 600; }}
.mode-reviewed {{ background: #dcfce7; color: #166534; padding: 4px 12px; border-radius: 4px; font-weight: 600; }}
</style>
</head>
<body>
<h1>Program Spec — {spec.protocol_id} <span class="{mode_class}">{mode_label}</span></h1>
{qc_html}

<h2>Study Design</h2>
<table>
<tr><th>Field</th><th>Value</th><th>Notes</th></tr>
{study_design_rows if study_design_rows else '<tr><td colspan="3">No study design information extracted.</td></tr>'}
</table>

<h2>Key Concepts</h2>
<table>
<tr><th>Concept</th><th>Evidence</th><th>Quality</th></tr>
{single_concepts if single_concepts else '<tr><td colspan="3">No key concepts extracted.</td></tr>'}
</table>

<h2>Inclusion Criteria</h2>
<table>
<tr><th>Criterion</th><th>Evidence</th><th>Domain</th><th>Lookback</th><th>Quality</th></tr>
{inc_rows if inc_rows else '<tr><td colspan="5">No inclusion criteria extracted.</td></tr>'}
</table>

<h2>Exclusion Criteria</h2>
<table>
<tr><th>Criterion</th><th>Evidence</th><th>Domain</th><th>Lookback</th><th>Quality</th></tr>
{exc_rows if exc_rows else '<tr><td colspan="5">No exclusion criteria extracted.</td></tr>'}
</table>

<h2>Censoring Rules</h2>
<table>
<tr><th>Rule</th><th>Evidence</th><th>Type</th><th>Applies To</th><th>Quality</th></tr>
{cens_rows if cens_rows else '<tr><td colspan="5">No censoring rules extracted.</td></tr>'}
</table>

<footer style="margin-top: 40px; color: #9ca3af; font-size: 0.85em;">
Generated by Protocol Spec Assist v{spec.spec_version} | Mode: {mode_label}
</footer>
</body>
</html>"""


def save_html(spec: ProgramSpec, output_path: str) -> str:
    """Render and save HTML to file."""
    html = render_html(spec)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path)
