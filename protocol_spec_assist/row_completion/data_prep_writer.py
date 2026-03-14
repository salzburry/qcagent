"""
Data Prep row writer — converts DataPrepExtraction evidence into
ImportantDate and TimePeriod rows for the Data Prep tab.

This writer consumes the new structured study_period evidence pack
(produced by find_data_prep_dates) and outputs typed rows rather than
the old lossy flat metadata conversion.
"""

from __future__ import annotations
from typing import Optional

from ..schemas.evidence import EvidencePack
from ..spec_output.spec_schema import (
    ImportantDate, TimePeriod, DataSourceEntry,
)
from ..data_sources.registry import get_definition, detect_source


# Canonical important date variables and their default labels
_DATE_DEFAULTS = {
    "INIT": {
        "label": "Initial diagnosis date",
        "definition": "Date of first qualifying diagnosis",
    },
    "INDEX": {
        "label": "Index date",
        "definition": "Date of cohort entry / treatment initiation",
    },
    "FUED": {
        "label": "Follow-up end date",
        "definition": "Last date of data availability or end of observation",
    },
    "CENSDT": {
        "label": "Censoring date",
        "definition": "Date of censoring (if not event)",
    },
    "ENROLLDT": {
        "label": "Enrollment start date",
        "definition": "Start of continuous enrollment period",
    },
}

# Canonical time periods and their default labels
_PERIOD_DEFAULTS = {
    "STUDY_PD": {
        "label": "Full study period",
    },
    "PRE_INT": {
        "label": "Pre-index period",
    },
    "FU": {
        "label": "Follow-up period",
    },
    "BASELINE": {
        "label": "Baseline assessment window",
    },
    "WASHOUT": {
        "label": "Treatment washout window",
    },
}


def expand_data_prep(
    pack: EvidencePack,
    data_source_key: str = "generic",
) -> tuple[DataSourceEntry, list[ImportantDate], list[TimePeriod]]:
    """Convert a study_period/data_prep_dates EvidencePack into typed Data Prep rows.

    Returns:
        (data_source_entry, important_dates, time_periods)
    """
    meta = pack.concept_metadata or {}
    per_candidate = meta.get("per_candidate", {})

    # ── Data source entry ──
    ds_name = meta.get("data_source") or ""
    ds_version = meta.get("data_source_version") or ""
    data_source = DataSourceEntry(
        data_source=ds_name,
        population_subset="",
        version=ds_version,
    )

    # ── Important dates ──
    important_dates: list[ImportantDate] = []
    seen_variables: set[str] = set()

    candidates = pack.selected_candidates if pack.selected_candidates is not None else pack.candidates

    for cand in candidates:
        cm = per_candidate.get(cand.candidate_id, {})
        if cm.get("row_type") != "important_date":
            continue

        variable = cm.get("variable", "").upper()
        if variable in seen_variables:
            continue
        seen_variables.add(variable)

        defaults = _DATE_DEFAULTS.get(variable, {})
        label = cm.get("label") or defaults.get("label", variable)

        # Prefer extracted definition, fall back to source-specific, then default
        definition = cm.get("definition") or ""
        if not definition:
            definition = get_definition(data_source_key, variable, defaults.get("definition", ""))

        important_dates.append(ImportantDate(
            variable=variable,
            label=label,
            definition=definition,
            additional_notes=f"sponsor_term: {cand.sponsor_term or 'n/a'}",
            source_page=cand.page,
            confidence=cand.llm_confidence,
        ))

    # Ensure minimum required dates exist (INIT, INDEX, FUED)
    for required_var in ["INIT", "INDEX", "FUED"]:
        if required_var not in seen_variables:
            defaults = _DATE_DEFAULTS[required_var]
            definition = get_definition(data_source_key, required_var, defaults["definition"])
            important_dates.append(ImportantDate(
                variable=required_var,
                label=defaults["label"],
                definition=definition,
                additional_notes="auto-generated — not found in protocol text",
                confidence=None,
            ))

    # Sort by canonical order
    date_order = ["INIT", "INDEX", "FUED", "CENSDT", "ENROLLDT"]
    important_dates.sort(key=lambda d: (
        date_order.index(d.variable) if d.variable in date_order else 99
    ))

    # ── Time periods ──
    time_periods: list[TimePeriod] = []
    seen_periods: set[str] = set()

    for cand in candidates:
        cm = per_candidate.get(cand.candidate_id, {})
        if cm.get("row_type") != "time_period":
            continue

        period = cm.get("time_period", "").upper()
        if period in seen_periods:
            continue
        seen_periods.add(period)

        defaults = _PERIOD_DEFAULTS.get(period, {})
        label = cm.get("label") or defaults.get("label", period)
        definition = cm.get("definition") or ""

        time_periods.append(TimePeriod(
            time_period=period,
            label=label,
            definition=definition,
            additional_notes="",
            source_page=cand.page,
            confidence=cand.llm_confidence,
        ))

    # Sort by canonical order
    period_order = ["STUDY_PD", "PRE_INT", "BASELINE", "WASHOUT", "FU"]
    time_periods.sort(key=lambda p: (
        period_order.index(p.time_period) if p.time_period in period_order else 99
    ))

    return data_source, important_dates, time_periods
