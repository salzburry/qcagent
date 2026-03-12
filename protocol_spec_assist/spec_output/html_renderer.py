"""
HTML renderer — self-contained HTML preview of program spec (9-tab layout).
No external CSS/JS dependencies.

Generates a tabbed view matching the Excel structure:
  1.Cover | 2.QC Review | 3.Data Prep | 4.StudyPop |
  5A.Demos | 5B.ClinChars | 5C.BioVars | 5D.LabVars |
  6.TreatVars | 7.Outcomes
"""

from __future__ import annotations
from pathlib import Path

from .spec_schema import ProgramSpec, VariableRow


def _confidence_badge(conf: float | None) -> str:
    if conf is None:
        return '<span class="badge badge-grey">N/A</span>'
    if conf >= 0.8:
        return f'<span class="badge badge-green">{conf:.0%}</span>'
    if conf >= 0.6:
        return f'<span class="badge badge-yellow">{conf:.0%}</span>'
    return f'<span class="badge badge-red">{conf:.0%}</span>'


def _page_ref(page: int | None) -> str:
    if page is None:
        return ""
    return f' <span class="page-ref">p.{page}</span>'


def _variable_table(rows: list[VariableRow], empty_msg: str = "No variables extracted.") -> str:
    """Render a variable tab as an HTML table with the standardised column layout."""
    if not rows:
        return f'<p class="empty">{empty_msg}</p>'

    header = (
        "<tr>"
        "<th>Time Period</th><th>Variable</th><th>Label</th>"
        "<th>Values</th><th>Definition</th><th>Code Lists</th>"
        "<th>Additional Notes</th><th>Confidence</th>"
        "</tr>"
    )
    body = ""
    for v in rows:
        body += (
            f"<tr>"
            f"<td>{v.time_period}</td>"
            f"<td><code>{v.variable}</code></td>"
            f"<td>{v.label}</td>"
            f"<td>{v.values}</td>"
            f"<td>{v.definition}</td>"
            f"<td>{v.code_lists_group}</td>"
            f"<td>{v.additional_notes}</td>"
            f"<td>{_confidence_badge(v.confidence)}{_page_ref(v.source_page)}</td>"
            f"</tr>\n"
        )
    return f"<table>{header}{body}</table>"


def render_html(spec: ProgramSpec) -> str:
    """Generate self-contained HTML preview of program spec (9-tab layout)."""

    mode_label = "DRAFT" if spec.generation_mode == "draft" else "REVIEWED"
    mode_class = "mode-draft" if spec.generation_mode == "draft" else "mode-reviewed"

    # QC warnings
    qc_html = ""
    if spec.qc_warnings:
        items = "".join(f"<li>{w}</li>" for w in spec.qc_warnings)
        qc_html = f'<div class="qc-bar"><strong>QC Warnings:</strong><ul>{items}</ul></div>'

    # ── Cover tab content ──
    study_info_rows = ""
    for label, val in [
        ("Study ID", spec.study_info.study_id),
        ("Asset", spec.study_info.asset),
        ("Indication", spec.study_info.indication),
        ("Study Title", spec.study_info.study_title),
        ("Study Status", spec.study_info.study_status),
    ]:
        if val:
            study_info_rows += f"<tr><td><strong>{label}</strong></td><td>{val}</td></tr>\n"

    tab_status_rows = ""
    for ts in spec.tab_statuses:
        tab_status_rows += (
            f"<tr><td>{ts.tab}</td><td>{ts.purpose}</td>"
            f"<td>{ts.status}</td></tr>\n"
        )

    # ── Data Prep content ──
    data_source_html = ""
    if spec.data_source.data_source:
        data_source_html = (
            f"<h3>Data Source</h3>"
            f"<table><tr><th>Source</th><th>Population Subset</th><th>Version</th></tr>"
            f"<tr><td>{spec.data_source.data_source}</td>"
            f"<td>{spec.data_source.population_subset}</td>"
            f"<td>{spec.data_source.version}</td></tr></table>"
        )

    important_dates_rows = ""
    for dt in spec.important_dates:
        important_dates_rows += (
            f"<tr><td><code>{dt.variable}</code></td><td>{dt.label}</td>"
            f"<td>{dt.definition}</td><td>{dt.additional_notes}</td>"
            f"<td>{_confidence_badge(dt.confidence)}{_page_ref(dt.source_page)}</td></tr>\n"
        )

    time_period_rows = ""
    for tp in spec.time_periods:
        time_period_rows += (
            f"<tr><td><code>{tp.time_period}</code></td><td>{tp.label}</td>"
            f"<td>{tp.definition}</td><td>{tp.additional_notes}</td>"
            f"<td>{_confidence_badge(tp.confidence)}{_page_ref(tp.source_page)}</td></tr>\n"
        )

    # ── StudyPop content ──
    def _criterion_rows(criteria):
        rows = ""
        for c in criteria:
            rows += (
                f"<tr><td>{c.time_period}</td>"
                f"<td><code>{c.variable}</code></td>"
                f"<td>{c.label}</td><td>{c.values}</td>"
                f"<td>{c.definition}</td>"
                f"<td>{c.additional_notes}</td>"
                f"<td>{_confidence_badge(c.confidence)}{_page_ref(c.source_page)}</td></tr>\n"
            )
        return rows

    inclusion_rows = _criterion_rows(spec.inclusion_criteria)
    exclusion_rows = _criterion_rows(spec.exclusion_criteria)

    cohort_rows = ""
    for coh in spec.cohort_definitions:
        cohort_rows += (
            f"<tr><td><code>{coh.variable}</code></td><td>{coh.label}</td>"
            f"<td>{coh.values}</td><td>{coh.definition}</td>"
            f"<td>{coh.additional_notes}</td></tr>\n"
        )

    # ── Variable tab sections ──
    demos_table = _variable_table(spec.demographics, "No demographic variables extracted.")
    clin_table = _variable_table(spec.clinical_characteristics, "No clinical characteristics extracted.")
    bio_table = _variable_table(spec.biomarkers, "No biomarker variables extracted.")
    lab_table = _variable_table(spec.lab_variables, "No laboratory variables extracted.")
    treat_table = _variable_table(spec.treatment_variables, "No treatment variables extracted.")
    outcome_table = _variable_table(spec.outcome_variables, "No outcome variables extracted.")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Program Spec — {spec.protocol_id}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
h1 {{ border-bottom: 2px solid #2563eb; padding-bottom: 8px; }}
h2 {{ color: #1e40af; margin-top: 32px; border-bottom: 1px solid #93c5fd; padding-bottom: 4px; }}
h3 {{ color: #374151; margin-top: 20px; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
th, td {{ border: 1px solid #d1d5db; padding: 8px 12px; text-align: left; vertical-align: top; }}
th {{ background: #c00000; color: white; font-weight: 600; }}
code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
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
.empty {{ color: #9ca3af; font-style: italic; }}
.tab-nav {{ display: flex; flex-wrap: wrap; gap: 4px; margin: 16px 0; }}
.tab-nav a {{ background: #006100; color: white; padding: 6px 14px; border-radius: 4px 4px 0 0; text-decoration: none; font-size: 0.9em; font-weight: 500; }}
.tab-nav a:hover {{ background: #004d00; }}
.section-header {{ background: #006100; color: white; padding: 8px 12px; font-weight: bold; margin-top: 24px; border-radius: 4px; }}
</style>
</head>
<body>
<h1>Program Spec — {spec.protocol_id} <span class="{mode_class}">{mode_label}</span></h1>
{qc_html}

<nav class="tab-nav">
<a href="#cover">1.Cover</a>
<a href="#qcreview">2.QC Review</a>
<a href="#dataprep">3.Data Prep</a>
<a href="#studypop">4.StudyPop</a>
<a href="#demos">5A.Demos</a>
<a href="#clinchars">5B.ClinChars</a>
<a href="#biovars">5C.BioVars</a>
<a href="#labvars">5D.LabVars</a>
<a href="#treatvars">6.TreatVars</a>
<a href="#outcomes">7.Outcomes</a>
</nav>

<!-- 1. Cover -->
<h2 id="cover">1. Cover</h2>
<h3>Study Information</h3>
<table>
<tr><th>Field</th><th>Value</th></tr>
{study_info_rows if study_info_rows else '<tr><td colspan="2">No study information available.</td></tr>'}
</table>

<h3>Programming Specification Tabs</h3>
<table>
<tr><th>Tab</th><th>Purpose</th><th>Status</th></tr>
{tab_status_rows if tab_status_rows else '<tr><td colspan="3">No tab statuses defined.</td></tr>'}
</table>

<!-- 2. QC Review -->
<h2 id="qcreview">2. QC Review</h2>
{qc_html if qc_html else '<p class="empty">No QC comments.</p>'}

<!-- 3. Data Prep -->
<h2 id="dataprep">3. Data Prep</h2>
{data_source_html}

<h3>Important Dates</h3>
<table>
<tr><th>Variable</th><th>Label</th><th>Definition</th><th>Additional Notes</th><th>Confidence</th></tr>
{important_dates_rows if important_dates_rows else '<tr><td colspan="5">No important dates extracted.</td></tr>'}
</table>

<h3>Study Time Periods</h3>
<table>
<tr><th>Time Period</th><th>Label</th><th>Definition</th><th>Additional Notes</th><th>Confidence</th></tr>
{time_period_rows if time_period_rows else '<tr><td colspan="5">No time periods extracted.</td></tr>'}
</table>

<!-- 4. StudyPop -->
<h2 id="studypop">4. Study Population</h2>
<h3>Inclusion Criteria</h3>
<table>
<tr><th>Time Period</th><th>Variable</th><th>Label</th><th>Values</th><th>Definition</th><th>Additional Notes</th><th>Confidence</th></tr>
{inclusion_rows if inclusion_rows else '<tr><td colspan="7">No inclusion criteria extracted.</td></tr>'}
</table>

<h3>Exclusion Criteria</h3>
<table>
<tr><th>Time Period</th><th>Variable</th><th>Label</th><th>Values</th><th>Definition</th><th>Additional Notes</th><th>Confidence</th></tr>
{exclusion_rows if exclusion_rows else '<tr><td colspan="7">No exclusion criteria extracted.</td></tr>'}
</table>

<h3>Cohort Definitions</h3>
<table>
<tr><th>Variable</th><th>Label</th><th>Values</th><th>Definition</th><th>Additional Notes</th></tr>
{cohort_rows if cohort_rows else '<tr><td colspan="5">No cohort definitions extracted.</td></tr>'}
</table>

<!-- 5A. Demos -->
<h2 id="demos">5A. Demographics</h2>
{demos_table}

<!-- 5B. ClinChars -->
<h2 id="clinchars">5B. Clinical Characteristics</h2>
{clin_table}

<!-- 5C. BioVars -->
<h2 id="biovars">5C. Biomarker Variables</h2>
{bio_table}

<!-- 5D. LabVars -->
<h2 id="labvars">5D. Laboratory Variables</h2>
{lab_table}

<!-- 6. TreatVars -->
<h2 id="treatvars">6. Treatment Variables</h2>
{treat_table}

<!-- 7. Outcomes -->
<h2 id="outcomes">7. Outcomes</h2>
{outcome_table}

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
