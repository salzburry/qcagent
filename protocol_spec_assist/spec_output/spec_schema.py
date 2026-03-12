"""
Program spec schema — structured to match the real 9-tab program spec format.

Tab layout (matching sample PROG_SPEC_RWDAP template):
  1. Cover       — study info, team, tab status
  2. QC Review   — QC comments (placeholder for human QC)
  3. Data Prep   — data source, important dates, time periods, source prep
  4. StudyPop    — inclusion/exclusion criteria, cohort definitions
  5A. Demos      — demographic variables
  5B. ClinChars  — clinical characteristics
  5C. BioVars    — biomarker variables
  5D. LabVars    — laboratory variables
  6. TreatVars   — treatment-related variables
  7. Outcomes    — outcome/endpoint variables

Each variable tab uses a standardised row schema:
  Time Period | Variable | Label | Values | Definition | Code Lists Group |
  Additional Notes | Date Modified | QC Reviewed
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field

from ..schemas.evidence import EvidencePack


# ── Shared row model (matches column layout in variable tabs) ──────────────

class VariableRow(BaseModel):
    """One variable definition row — the unit of every variable tab."""
    time_period: str = ""
    variable: str = ""
    label: str = ""
    values: str = ""
    definition: str = ""
    code_lists_group: str = ""
    additional_notes: str = ""
    date_modified: str = ""
    qc_reviewed: str = ""
    # Provenance fields from evidence extraction (not in Excel output columns,
    # but kept for traceability)
    source_page: Optional[int] = None
    confidence: Optional[float] = None
    explicit: str = "explicit"


# ── Cover tab models ───────────────────────────────────────────────────────

class StudyInfo(BaseModel):
    """Section A of Cover tab: study-level metadata."""
    study_id: str = ""
    asset: str = ""
    indication: str = ""
    study_title: str = ""
    study_status: str = "In Progress"
    study_closed_date: str = ""


class TeamMember(BaseModel):
    """One row of Section B on Cover tab."""
    role: str = ""
    group: str = ""
    name: str = ""


class TabStatus(BaseModel):
    """One row of Section C on Cover tab — tab completion status."""
    tab: str = ""
    purpose: str = ""
    status: str = "Not started"
    notes: str = ""
    deadlines: str = ""


# ── Data Prep tab models ───────────────────────────────────────────────────

class DataSourceEntry(BaseModel):
    """Section A of Data Prep: data source selection."""
    data_source: str = ""
    population_subset: str = ""
    version: str = ""


class ImportantDate(BaseModel):
    """Section B of Data Prep: important dates (INIT, INDEX, etc.)."""
    variable: str = ""
    label: str = ""
    date_format: str = "Date"
    definition: str = ""
    additional_notes: str = ""
    date_modified: str = ""
    qc_reviewed: str = ""
    source_page: Optional[int] = None
    confidence: Optional[float] = None


class TimePeriod(BaseModel):
    """Section C of Data Prep: study time periods (STUDY_PD, PRE_INT, FU, etc.)."""
    time_period: str = ""
    label: str = ""
    definition: str = ""
    additional_notes: str = ""
    date_modified: str = ""
    qc_reviewed: str = ""
    source_page: Optional[int] = None
    confidence: Optional[float] = None


class SourceDataPrep(BaseModel):
    """Section D of Data Prep: source data preparation decisions."""
    row_number: int = 0
    source_table_variable: str = ""
    situation: str = ""
    action: str = ""
    reasoning: str = ""
    date_modified: str = ""
    qc_reviewed: str = ""


# ── StudyPop tab models ────────────────────────────────────────────────────

class CriterionRow(BaseModel):
    """One inclusion/exclusion criterion row in StudyPop Section A."""
    time_period: str = ""
    variable: str = ""
    label: str = ""
    values: str = ""
    definition: str = ""
    additional_notes: str = ""
    code_lists_group: str = ""
    date_modified: str = ""
    qc_reviewed: str = ""
    # Extraction provenance
    source_page: Optional[int] = None
    confidence: Optional[float] = None
    explicit: str = "explicit"
    domain: str = "other"
    lookback_window: Optional[str] = None


class CohortRow(BaseModel):
    """Cohort definition row in StudyPop Section B."""
    variable: str = ""
    label: str = ""
    values: str = ""
    definition: str = ""
    additional_notes: str = ""
    code_lists_group: str = ""
    date_modified: str = ""
    qc_reviewed: str = ""
    source_page: Optional[int] = None
    confidence: Optional[float] = None


# ── Top-level ProgramSpec ──────────────────────────────────────────────────

class ProgramSpec(BaseModel):
    """Full program spec matching the 9-tab Excel template."""
    protocol_id: str = ""
    spec_version: str = "1.0.0"
    generation_mode: str = "draft"  # draft | reviewed

    # Tab 1: Cover
    study_info: StudyInfo = Field(default_factory=StudyInfo)
    team_members: list[TeamMember] = Field(default_factory=list)
    tab_statuses: list[TabStatus] = Field(default_factory=list)

    # Tab 2: QC Review (placeholder — populated during human review)
    qc_warnings: list[str] = Field(default_factory=list)

    # Tab 3: Data Prep
    data_source: DataSourceEntry = Field(default_factory=DataSourceEntry)
    important_dates: list[ImportantDate] = Field(default_factory=list)
    time_periods: list[TimePeriod] = Field(default_factory=list)
    source_data_prep: list[SourceDataPrep] = Field(default_factory=list)

    # Tab 4: StudyPop
    inclusion_criteria: list[CriterionRow] = Field(default_factory=list)
    exclusion_criteria: list[CriterionRow] = Field(default_factory=list)
    cohort_definitions: list[CohortRow] = Field(default_factory=list)

    # Tab 5A: Demos
    demographics: list[VariableRow] = Field(default_factory=list)

    # Tab 5B: ClinChars
    clinical_characteristics: list[VariableRow] = Field(default_factory=list)

    # Tab 5C: BioVars
    biomarkers: list[VariableRow] = Field(default_factory=list)

    # Tab 5D: LabVars
    lab_variables: list[VariableRow] = Field(default_factory=list)

    # Tab 6: TreatVars
    treatment_variables: list[VariableRow] = Field(default_factory=list)

    # Tab 7: Outcomes
    outcome_variables: list[VariableRow] = Field(default_factory=list)

    # Unmapped variables — LLM found these but they don't match any static template.
    # Review-only: not written to production spec tabs.
    unmapped_variables: list[VariableRow] = Field(default_factory=list)


# ── Legacy compat aliases (kept so existing imports don't break) ───────────

class SpecEntry(BaseModel):
    """A single spec entry with value, provenance, and confidence."""
    value: str = ""
    source_snippet: str = ""
    page: Optional[int] = None
    confidence: Optional[float] = None
    explicit: str = "explicit"
    notes: str = ""


CriterionEntry = CriterionRow   # alias for backwards compat
CensoringRuleEntry = VariableRow  # alias — censoring now lives in Outcomes tab


# ── Builder helpers ────────────────────────────────────────────────────────

def _get_governing_text(pack: EvidencePack) -> str | None:
    """Get the governing text: reviewer override > selected candidate > top-ranked."""
    if pack.reviewer_override is not None:
        return pack.reviewer_override
    if pack.selected_candidate is not None:
        return pack.selected_candidate.snippet
    if pack.candidates:
        return pack.candidates[0].snippet
    return None


def _get_governing_candidate(pack: EvidencePack):
    """Get the governing candidate: selected by reviewer, or top-ranked if draft."""
    if pack.selected_candidate is not None:
        return pack.selected_candidate
    if pack.candidates:
        return pack.candidates[0]
    return None


def _default_tab_statuses() -> list[TabStatus]:
    """Return the standard 9-tab status rows for the Cover tab."""
    tabs = [
        ("1.Cover", "This tab defines the basic information of the programming specification template"),
        ("2.QC Review", "This tab details the QC review comments and suggested updates"),
        ("3.Data Prep", "This tab details any analytic decisions that need to be applied to the source data before analytic file build"),
        ("4.StudyPop", "This tab defines how to select the eligible patient population from the RWDAP"),
        ("5A.Demos", "Demographic characteristics"),
        ("5B.ClinChars", "Clinical characteristics (excluding biomarkers and labs)"),
        ("5C.BioVars", "Biomarker variables"),
        ("5D.LabVars", "Laboratory variables"),
        ("6.TreatVars", "Treatment related variables"),
        ("7.Outcomes", "This tab defines variables needed for outcomes analyses"),
    ]
    return [
        TabStatus(tab=name, purpose=purpose, status="In progress")
        for name, purpose in tabs
    ]


def build_program_spec(
    packs: dict[str, EvidencePack],
    protocol_id: str = "",
    qc_warnings: Optional[list[str]] = None,
) -> ProgramSpec:
    """
    Translate evidence packs -> ProgramSpec (9-tab layout).

    Maps existing concepts to the correct tabs:
      - study_period      -> Data Prep (data source, important dates, time periods)
      - index_date        -> Data Prep Section B (important dates)
      - follow_up_end     -> Data Prep Section B + C (important dates, time periods)
      - eligibility_*     -> StudyPop Section A (inclusion/exclusion criteria)
      - primary_endpoint  -> Outcomes tab
      - censoring_rules   -> Outcomes tab
    """
    spec = ProgramSpec(
        protocol_id=protocol_id,
        qc_warnings=qc_warnings or [],
    )

    # Determine generation mode — reviewed if any pack has review state set
    any_reviewed = any(
        p.selected_candidate_id is not None
        or bool(p.selected_candidate_ids)
        or p.reviewer_override is not None
        for p in packs.values()
    )
    spec.generation_mode = "reviewed" if any_reviewed else "draft"

    # ── Cover tab ──
    spec.study_info.study_id = protocol_id
    spec.tab_statuses = _default_tab_statuses()

    # ── Data Prep: data source and study design ──
    if "study_period" in packs:
        sp = packs["study_period"]
        meta = sp.concept_metadata or {}

        spec.data_source = DataSourceEntry(
            data_source=meta.get("data_source") or "",
            population_subset="",
            version=meta.get("data_source_version") or "",
        )

        # Important dates — initial diagnosis (label derived from protocol metadata)
        if meta.get("study_period_start") or meta.get("study_period_end"):
            indication = meta.get("indication") or spec.study_info.indication or ""
            init_label = f"Initial {indication} diagnosis date" if indication else "Initial diagnosis date"
            init_def = meta.get("diagnosis_definition") or "Date of first qualifying diagnosis"
            spec.important_dates.append(ImportantDate(
                variable="INIT",
                label=init_label,
                definition=init_def,
                additional_notes="",
            ))

        # Time periods
        sp_start = meta.get("study_period_start") or ""
        sp_end = meta.get("study_period_end") or ""
        if sp_start or sp_end:
            spec.time_periods.append(TimePeriod(
                time_period="STUDY_PD",
                label="Full study period",
                definition=f"{sp_start} through {sp_end}" if sp_start and sp_end else sp_start or sp_end,
            ))

    # ── Data Prep: index date ──
    if "index_date" in packs:
        idx_pack = packs["index_date"]
        gov_text = _get_governing_text(idx_pack)
        cand = _get_governing_candidate(idx_pack)
        if gov_text:
            spec.important_dates.append(ImportantDate(
                variable="INDEX",
                label="Index date",
                definition=gov_text,
                additional_notes=f"sponsor_term: {cand.sponsor_term or 'n/a'}" if cand else "",
                source_page=cand.page if cand else None,
                confidence=cand.llm_confidence if cand else None,
            ))

    # ── Data Prep: follow-up end ──
    if "follow_up_end" in packs:
        fu_pack = packs["follow_up_end"]
        gov_text = _get_governing_text(fu_pack)
        cand = _get_governing_candidate(fu_pack)
        if gov_text:
            spec.important_dates.append(ImportantDate(
                variable="FUED",
                label="End of follow-up date",
                definition=gov_text,
                additional_notes=f"sponsor_term: {cand.sponsor_term or 'n/a'}" if cand else "",
                source_page=cand.page if cand else None,
                confidence=cand.llm_confidence if cand else None,
            ))
            spec.time_periods.append(TimePeriod(
                time_period="FU",
                label="Follow-up period",
                definition=gov_text,
                source_page=cand.page if cand else None,
                confidence=cand.llm_confidence if cand else None,
            ))

    # ── StudyPop: inclusion criteria ──
    if "eligibility_inclusion" in packs:
        inc_pack = packs["eligibility_inclusion"]
        meta = (inc_pack.concept_metadata or {}).get("per_candidate", {})
        candidates = inc_pack.selected_candidates if inc_pack.selected_candidates is not None else inc_pack.candidates
        for cand in candidates:
            cm = meta.get(cand.candidate_id, {})
            spec.inclusion_criteria.append(CriterionRow(
                time_period="STUDY_PD",
                variable=cand.sponsor_term or "",
                label=cand.sponsor_term or "",
                values="",
                definition=cand.snippet,
                domain=cm.get("domain", "other"),
                lookback_window=cm.get("lookback_window"),
                additional_notes=cm.get("operational_detail") or "",
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
                source_page=cand.page,
            ))

    # ── StudyPop: exclusion criteria ──
    if "eligibility_exclusion" in packs:
        exc_pack = packs["eligibility_exclusion"]
        meta = (exc_pack.concept_metadata or {}).get("per_candidate", {})
        candidates = exc_pack.selected_candidates if exc_pack.selected_candidates is not None else exc_pack.candidates
        for cand in candidates:
            cm = meta.get(cand.candidate_id, {})
            spec.exclusion_criteria.append(CriterionRow(
                time_period="STUDY_PD",
                variable=cand.sponsor_term or "",
                label=cand.sponsor_term or "",
                values="",
                definition=cand.snippet,
                domain=cm.get("domain", "other"),
                lookback_window=cm.get("lookback_window"),
                additional_notes=cm.get("operational_detail") or "",
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
                source_page=cand.page,
            ))

    # ── Outcomes: primary endpoint ──
    if "primary_endpoint" in packs:
        ep_pack = packs["primary_endpoint"]
        gov_text = _get_governing_text(ep_pack)
        cand = _get_governing_candidate(ep_pack)
        if gov_text:
            spec.outcome_variables.append(VariableRow(
                time_period="FU",
                variable="PRIMARY_EP",
                label="Primary Endpoint",
                values="",
                definition=gov_text,
                additional_notes=f"sponsor_term: {cand.sponsor_term or 'n/a'}" if cand else "",
                source_page=cand.page if cand else None,
                confidence=cand.llm_confidence if cand else None,
                explicit=cand.explicit if cand else "explicit",
            ))

    # ── Outcomes: censoring rules ──
    if "censoring_rules" in packs:
        cr_pack = packs["censoring_rules"]
        meta = (cr_pack.concept_metadata or {}).get("per_candidate", {})
        candidates = cr_pack.selected_candidates if cr_pack.selected_candidates is not None else cr_pack.candidates
        for cand in candidates:
            cm = meta.get(cand.candidate_id, {})
            spec.outcome_variables.append(VariableRow(
                time_period="FU",
                variable=cand.sponsor_term or "CENSORING",
                label=cand.sponsor_term or "Censoring Rule",
                values="",
                definition=cand.snippet,
                additional_notes=f"type: {cm.get('rule_type', 'event_based')}; applies_to: {cm.get('applies_to', 'all')}",
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
            ))

    # ── Variable tabs (5A–6): demographics, clinical chars, biomarkers, labs, treatment ──
    _VARIABLE_TAB_CONCEPTS = {
        "demographics": "demographics",
        "clinical_characteristics": "clinical_characteristics",
        "biomarkers": "biomarkers",
        "lab_variables": "lab_variables",
        "treatment_variables": "treatment_variables",
    }
    for concept_key, spec_field in _VARIABLE_TAB_CONCEPTS.items():
        if concept_key not in packs:
            continue
        var_pack = packs[concept_key]
        meta = (var_pack.concept_metadata or {}).get("per_candidate", {})
        candidates = var_pack.selected_candidates if var_pack.selected_candidates is not None else var_pack.candidates
        target_list = getattr(spec, spec_field)
        for cand in candidates:
            cm = meta.get(cand.candidate_id, {})
            target_list.append(VariableRow(
                time_period=cm.get("time_period", ""),
                variable=cm.get("variable_name", cand.canonical_term or ""),
                label=cm.get("label", ""),
                values=cm.get("values", ""),
                definition=cand.snippet,
                code_lists_group=cm.get("code_lists_group", ""),
                additional_notes=cm.get("additional_notes", ""),
                source_page=cand.page,
                confidence=cand.llm_confidence,
                explicit=cand.explicit,
            ))

    # ── Unmapped variables: LLM-found variables not in any static template ──
    for concept_key in _VARIABLE_TAB_CONCEPTS:
        if concept_key not in packs:
            continue
        var_pack = packs[concept_key]
        meta = var_pack.concept_metadata or {}
        for uv in meta.get("unmapped_variables", []):
            spec.unmapped_variables.append(VariableRow(
                variable=uv.get("variable_name", ""),
                label=uv.get("label", ""),
                definition=uv.get("definition", ""),
                confidence=uv.get("confidence"),
                additional_notes=f"Source tab: {concept_key} (review required)",
            ))

    return spec
