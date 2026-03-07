"""
Spec row schemas.
These are what get written to the workbook.
Produced by row_completion writers, NOT by concept finders.
Each row carries a reference back to its governing EvidencePack.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel


RowStatus = Literal["auto_filled", "human_confirmed", "human_edited", "flagged", "empty"]


class RowBase(BaseModel):
    """Every spec row carries provenance back to evidence."""
    concept: str
    protocol_id: str
    evidence_pack_concept: str          # which EvidencePack produced this
    governing_snippet: Optional[str]    # exact text that governed this row
    governing_section: Optional[str]
    governing_page: Optional[int]
    explicit_type: str = "explicit"
    confidence: float = 0.0
    status: RowStatus = "auto_filled"
    reviewer_flag: bool = False
    reviewer_note: Optional[str] = None


# ── Study Population rows ─────────────────────────────────────────────────────

class StudyDesignRow(RowBase):
    design_type: Optional[str] = None
    study_period_start: Optional[str] = None
    study_period_end: Optional[str] = None
    enrollment_start: Optional[str] = None
    enrollment_end: Optional[str] = None
    follow_up_type: Optional[str] = None
    follow_up_description: Optional[str] = None


class IndexDateRow(RowBase):
    label: Optional[str] = None
    event_description: Optional[str] = None
    operational_definition: Optional[str] = None
    source_table: Optional[str] = None
    lookback_days: Optional[int] = None
    washout_required: Optional[bool] = None
    washout_days: Optional[int] = None


class ObservationPeriodRow(RowBase):
    period_type: Optional[str] = None       # baseline | follow_up | washout
    duration_value: Optional[int] = None
    duration_unit: Optional[str] = None
    anchor: Optional[str] = None
    continuous_enrollment_required: Optional[bool] = None


class DataSourceRow(RowBase):
    database_name: Optional[str] = None
    version: Optional[str] = None
    data_vintage: Optional[str] = None
    primary: Optional[bool] = None
    role: Optional[str] = None


# ── I/E Criteria rows ─────────────────────────────────────────────────────────

class EligibilityRow(RowBase):
    criterion_id: Optional[str] = None
    type: Optional[str] = None              # inclusion | exclusion
    domain: Optional[str] = None
    description: Optional[str] = None
    operational_definition: Optional[str] = None
    codelist_ref: Optional[str] = None
    lookback_days: Optional[int] = None


# ── Endpoint rows ─────────────────────────────────────────────────────────────

class EndpointRow(RowBase):
    endpoint_id: Optional[str] = None
    type: Optional[str] = None              # primary | secondary | exploratory
    label: Optional[str] = None
    operational_definition: Optional[str] = None
    time_to_event: Optional[bool] = None
    censoring_rules: Optional[str] = None
    codelist_ref: Optional[str] = None


# ── Covariate rows ────────────────────────────────────────────────────────────

class CovariateRow(RowBase):
    covariate_id: Optional[str] = None
    category: Optional[str] = None
    label: Optional[str] = None
    operational_definition: Optional[str] = None
    measurement_window: Optional[str] = None
    variable_type: Optional[str] = None
    codelist_ref: Optional[str] = None
