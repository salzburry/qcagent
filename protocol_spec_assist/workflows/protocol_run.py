"""
Prefect workflow — protocol_run.
Fixed path: ingest → index → find concepts → QC → spec generation → save.
Workflow, not agent. Every step is predetermined.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

try:
    from prefect import flow, task, get_run_logger
    from prefect.cache_policies import NONE as NO_CACHE
    from prefect.states import Paused
    PREFECT_AVAILABLE = True
except ImportError:
    # Graceful fallback if Prefect not installed
    PREFECT_AVAILABLE = False
    NO_CACHE = None
    def flow(fn=None, **kwargs):
        return fn if fn else lambda f: f
    def task(fn=None, **kwargs):
        return fn if fn else lambda f: f
    def get_run_logger():
        import logging
        return logging.getLogger("protocol_run")

from ..ingest.parse_protocol import parse_protocol
from ..retrieval.search import ProtocolIndex
from ..serving.model_client import LocalModelClient
from ..ta_packs.loader import load_ta_pack
from ..concepts.index_date import find_index_date
from ..concepts.endpoints import find_follow_up_end, find_primary_endpoint
from ..concepts.eligibility import find_inclusion_criteria, find_exclusion_criteria
from ..concepts.study_design import find_study_period, find_data_prep_dates, find_censoring_rules
from ..concepts.cohort_definitions import find_cohort_definitions
from ..concepts.source_data_prep import find_source_data_prep
from ..concepts.demographics import find_demographics
from ..concepts.clinical_characteristics import find_clinical_characteristics
from ..concepts.biomarkers import find_biomarkers
from ..concepts.lab_variables import find_lab_variables
from ..concepts.treatment_variables import find_treatment_variables
from ..data_sources.registry import detect_source, detect_source_multi
from ..schemas.evidence import EvidencePack
from ..qc.rules import run_all_qc, summarize_qc
from ..spec_output.spec_schema import build_program_spec
from ..spec_output.html_renderer import save_html
try:
    from ..spec_output.excel_writer import save_excel
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False


# ── Tasks (each is a named, retriable step) ───────────────────────────────────

@task(name="parse-protocol", retries=1)
def task_parse_protocol(pdf_path: str, protocol_id: str) -> dict:
    """Parse protocol PDF → structured sections dict with quality metrics."""
    parsed = parse_protocol(pdf_path, protocol_id)
    chunks = parsed.to_chunks()
    quality = None
    if parsed.quality:
        quality = {
            "grade": parsed.quality.grade,
            "n_sections": parsed.quality.n_sections,
            "median_heading_len": parsed.quality.median_heading_len,
            "median_text_len": parsed.quality.median_text_len,
            "table_count": parsed.quality.table_count,
            "empty_ratio": parsed.quality.empty_ratio,
        }
    return {
        "chunks": chunks,
        "title": parsed.title,
        "n_sections": len(parsed.sections),
        "parse_quality": quality,
    }


@task(name="index-protocol", retries=1)
def task_index_protocol(
    chunks: list[dict],
    protocol_id: str,
    index: ProtocolIndex,
) -> None:
    """Embed and index protocol chunks into Qdrant."""
    index.index_protocol(chunks, protocol_id)


def _run_concept_finder(
    finder_fn,
    protocol_id: str,
    index: ProtocolIndex,
    client: LocalModelClient,
    ta_pack,
    **kwargs,
) -> dict:
    """Run a concept finder and return serialized EvidencePack."""
    pack = finder_fn(protocol_id, index, client, ta_pack, **kwargs)
    return pack.model_dump()


@task(name="find-index-date", cache_policy=NO_CACHE)
def task_find_index_date(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_index_date, pid, index, client, ta_pack)


@task(name="find-follow-up-end", cache_policy=NO_CACHE)
def task_find_follow_up_end(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_follow_up_end, pid, index, client, ta_pack)


@task(name="find-primary-endpoint", cache_policy=NO_CACHE)
def task_find_primary_endpoint(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_primary_endpoint, pid, index, client, ta_pack)


@task(name="find-inclusion-criteria", cache_policy=NO_CACHE)
def task_find_inclusion_criteria(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_inclusion_criteria, pid, index, client, ta_pack)


@task(name="find-exclusion-criteria", cache_policy=NO_CACHE)
def task_find_exclusion_criteria(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_exclusion_criteria, pid, index, client, ta_pack)


@task(name="find-study-period", cache_policy=NO_CACHE)
def task_find_study_period(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_study_period, pid, index, client, ta_pack)


@task(name="find-censoring-rules", cache_policy=NO_CACHE)
def task_find_censoring_rules(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_censoring_rules, pid, index, client, ta_pack)


@task(name="find-cohort-definitions", cache_policy=NO_CACHE)
def task_find_cohort_definitions(pid, index, client, ta_pack) -> dict:
    return _run_concept_finder(find_cohort_definitions, pid, index, client, ta_pack)


@task(name="find-source-data-prep", cache_policy=NO_CACHE)
def task_find_source_data_prep(pid, index, client, ta_pack, data_source="generic") -> dict:
    return _run_concept_finder(find_source_data_prep, pid, index, client, ta_pack, data_source=data_source)


@task(name="find-demographics", cache_policy=NO_CACHE)
def task_find_demographics(pid, index, client, ta_pack, data_source="generic") -> dict:
    return _run_concept_finder(find_demographics, pid, index, client, ta_pack, data_source=data_source)


@task(name="find-clinical-characteristics", cache_policy=NO_CACHE)
def task_find_clinical_characteristics(pid, index, client, ta_pack, data_source="generic") -> dict:
    return _run_concept_finder(find_clinical_characteristics, pid, index, client, ta_pack, data_source=data_source)


@task(name="find-biomarkers", cache_policy=NO_CACHE)
def task_find_biomarkers(pid, index, client, ta_pack, data_source="generic") -> dict:
    return _run_concept_finder(find_biomarkers, pid, index, client, ta_pack, data_source=data_source)


@task(name="find-lab-variables", cache_policy=NO_CACHE)
def task_find_lab_variables(pid, index, client, ta_pack, data_source="generic") -> dict:
    return _run_concept_finder(find_lab_variables, pid, index, client, ta_pack, data_source=data_source)


@task(name="find-treatment-variables", cache_policy=NO_CACHE)
def task_find_treatment_variables(pid, index, client, ta_pack, data_source="generic") -> dict:
    return _run_concept_finder(find_treatment_variables, pid, index, client, ta_pack, data_source=data_source)


@task(name="run-qc")
def task_run_qc(
    packs_dict: dict[str, dict],
    expected_concepts: Optional[list[str]],
    chunk_lookup: Optional[dict[str, str]] = None,
) -> list[dict]:
    packs = {k: EvidencePack.model_validate(v) for k, v in packs_dict.items()}
    # Run pre-review QC only — post-review QC runs after human selection
    results = run_all_qc(packs, expected_concepts, stage="pre_review", chunk_lookup=chunk_lookup)
    print(summarize_qc(results))
    return [vars(r) for r in results]


@task(name="generate-spec")
def task_generate_spec(
    packs_dict: dict[str, dict],
    qc_results: list[dict],
    output_dir: str,
    protocol_id: str,
    data_source: str = "generic",
) -> dict:
    """Generate draft spec: JSON + HTML + Excel (if available)."""
    packs = {k: EvidencePack.model_validate(v) for k, v in packs_dict.items()}
    qc_warnings = [r["message"] for r in qc_results if r.get("level") in ("error", "warning")]

    spec = build_program_spec(
        packs, protocol_id=protocol_id, qc_warnings=qc_warnings, data_source=data_source,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save spec JSON
    spec_json_path = out_dir / f"{protocol_id}_spec.json"
    with open(spec_json_path, "w") as f:
        json.dump(spec.model_dump(), f, indent=2, default=str)

    # Save HTML preview
    html_path = save_html(spec, str(out_dir / f"{protocol_id}_spec.html"))

    # Save Excel (if openpyxl available)
    excel_path = None
    if EXCEL_AVAILABLE:
        try:
            excel_path = save_excel(spec, str(out_dir / f"{protocol_id}_spec.xlsx"))
        except Exception as e:
            print(f"[Spec] Excel generation failed: {e}")

    outputs = {
        "spec_json": str(spec_json_path),
        "spec_html": html_path,
        "spec_excel": excel_path,
    }
    print(f"[Spec] Generated: {', '.join(k for k, v in outputs.items() if v)}")
    return outputs


@task(name="save-evidence-packs")
def task_save_packs(
    packs_dict: dict[str, dict],
    qc_results: list[dict],
    output_dir: str,
    protocol_id: str,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "protocol_id": protocol_id,
        "evidence_packs": packs_dict,
        "qc_results": qc_results,
    }

    out_path = out_dir / f"{protocol_id}_evidence_packs.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"[Workflow] Evidence packs saved: {out_path}")
    return str(out_path)


# ── Main flow ─────────────────────────────────────────────────────────────────

@flow(name="protocol-spec-run", log_prints=True)
def protocol_run(
    pdf_path: str,
    protocol_id: Optional[str] = None,
    ta_name: Optional[str] = None,
    data_source_override: Optional[str] = None,
    index_dir: str = "data/index",
    output_dir: str = "data/outputs",
    skip_indexing: bool = False,
):
    """
    Main workflow: protocol PDF → evidence packs → QC → draft spec.

    Concepts: index_date, follow_up_end, primary_endpoint,
    eligibility_inclusion, eligibility_exclusion, study_period, censoring_rules,
    demographics, clinical_characteristics, biomarkers, lab_variables, treatment_variables.

    Human review happens after this flow completes — via UI.
    """
    logger = get_run_logger()
    pid = protocol_id or Path(pdf_path).stem

    logger.info(f"Starting protocol run: {pid} | TA: {ta_name or 'none'}")

    # Step 0: Preflight — verify model servers are reachable
    client = LocalModelClient()
    if not client.check_model_available(use_adjudicator=False):
        raise RuntimeError(
            f"Default model server not available at {client.config.default_base_url}. "
            f"Start vLLM before running the pipeline."
        )
    logger.info("Model preflight passed — default model available.")

    # Step 1: Parse (with quality scoring)
    parse_result = task_parse_protocol(pdf_path, pid)
    quality = parse_result.get("parse_quality") or {}
    parse_grade = quality.get("grade", "n/a")
    logger.info(
        f"Parsed {parse_result['n_sections']} sections | "
        f"quality={parse_grade} | "
        f"med_text={quality.get('median_text_len', 0):.0f} | "
        f"tables={quality.get('table_count', 0)}"
    )

    # GATE: fail closed on bad parses — do not feed garbage into extraction
    if parse_grade == "fail":
        logger.error(
            "Parse quality is FAIL — extraction would produce unreliable results. "
            "Generating shell spec with QC issues instead."
        )
        shell_warnings = [
            f"CRITICAL: PDF parse quality is FAIL (median_text={quality.get('median_text_len', 0):.0f}, "
            f"empty_ratio={quality.get('empty_ratio', 0):.2f}). "
            f"All extracted rows are unreliable. Re-parse with OCR or provide a cleaner PDF.",
        ]
        # Generate a shell spec with no evidence, only QC warnings
        shell_spec = build_program_spec(
            packs={}, protocol_id=pid, qc_warnings=shell_warnings,
        )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        spec_json_path = out_dir / f"{pid}_spec.json"
        with open(spec_json_path, "w") as f:
            json.dump(shell_spec.model_dump(), f, indent=2, default=str)
        html_path = save_html(shell_spec, str(out_dir / f"{pid}_spec.html"))
        logger.info(f"Shell spec saved: {spec_json_path}")
        return str(spec_json_path)

    # Step 2: Index (skip if already indexed)
    # Share one ProtocolIndex instance across all finders to avoid reloading models
    index = ProtocolIndex(index_dir=index_dir)
    if not skip_indexing:
        task_index_protocol(parse_result["chunks"], pid, index)
    else:
        logger.info("Skipping indexing — using existing index.")

    # Step 3: Find concepts — all share the same index and client instances
    ta_pack = load_ta_pack(ta_name) if ta_name else None

    index_date_pack = task_find_index_date(pid, index, client, ta_pack)
    follow_up_end_pack = task_find_follow_up_end(pid, index, client, ta_pack)
    primary_endpoint_pack = task_find_primary_endpoint(pid, index, client, ta_pack)
    inclusion_pack = task_find_inclusion_criteria(pid, index, client, ta_pack)
    exclusion_pack = task_find_exclusion_criteria(pid, index, client, ta_pack)
    study_period_pack = task_find_study_period(pid, index, client, ta_pack)
    censoring_rules_pack = task_find_censoring_rules(pid, index, client, ta_pack)

    # Step 3b: Cohort definitions + censoring rules (no source dependency)
    cohort_pack = task_find_cohort_definitions(pid, index, client, ta_pack)

    # Step 3c: Multi-channel data source detection
    # Uses: explicit override > study_period extraction > protocol title > protocol text
    sp_meta = study_period_pack.get("concept_metadata") or {}
    protocol_title = parse_result.get("title") or ""
    # Sample broadly from chunks for source keyword detection.
    # Use more chunks (up to 20) and longer snippets (up to 500 chars) to
    # increase the chance of catching source-identifying keywords that may
    # appear deeper in the protocol (e.g. data source descriptions in methods).
    protocol_text_sample = " ".join(
        c.get("text", "")[:500] for c in parse_result["chunks"][:20]
    )
    detected_source = detect_source_multi(
        data_source_override=data_source_override or "",
        study_period_metadata=sp_meta,
        protocol_title=protocol_title,
        protocol_text_sample=protocol_text_sample,
    )
    logger.info(f"Data source detected: {detected_source} "
                f"(override={data_source_override or 'none'}, "
                f"sp_meta={sp_meta.get('data_source', 'none')}, "
                f"title={'matched' if detect_source(protocol_title) != 'generic' else 'no match'})")

    # Step 3d: Source-dependent concept finders
    demographics_pack = task_find_demographics(pid, index, client, ta_pack, detected_source)
    clinical_chars_pack = task_find_clinical_characteristics(pid, index, client, ta_pack, detected_source)
    biomarkers_pack = task_find_biomarkers(pid, index, client, ta_pack, detected_source)
    lab_variables_pack = task_find_lab_variables(pid, index, client, ta_pack, detected_source)
    treatment_vars_pack = task_find_treatment_variables(pid, index, client, ta_pack, detected_source)

    # Step 3e: Source data preparation issues (needs detected source)
    # Wrapped in try/except because this finder can timeout on large prompts
    # or when GPU memory is constrained. Falls back to registry-only issues.
    try:
        source_data_prep_pack = task_find_source_data_prep(pid, index, client, ta_pack, detected_source)
    except Exception as e:
        logger.warning(f"source_data_prep finder failed ({type(e).__name__}: {e}). "
                       f"Falling back to registry-only source limitations.")
        from .concepts.source_data_prep import _build_source_limitation_pack
        fallback_pack = _build_source_limitation_pack(pid, detected_source)
        source_data_prep_pack = fallback_pack.model_dump()

    packs = {
        "index_date": index_date_pack,
        "follow_up_end": follow_up_end_pack,
        "primary_endpoint": primary_endpoint_pack,
        "eligibility_inclusion": inclusion_pack,
        "eligibility_exclusion": exclusion_pack,
        "study_period": study_period_pack,
        "censoring_rules": censoring_rules_pack,
        "cohort_definitions": cohort_pack,
        "source_data_prep": source_data_prep_pack,
        "demographics": demographics_pack,
        "clinical_characteristics": clinical_chars_pack,
        "biomarkers": biomarkers_pack,
        "lab_variables": lab_variables_pack,
        "treatment_variables": treatment_vars_pack,
    }

    # Step 4: Pre-review QC
    # Build chunk_lookup so qc_quote_in_chunk can validate snippets against source text
    chunk_lookup = {c["chunk_id"]: c["text"] for c in parse_result["chunks"] if c.get("chunk_id")}
    expected = ta_pack.expected_concepts if ta_pack else None
    qc_results = task_run_qc(packs, expected, chunk_lookup=chunk_lookup)

    # Step 5: Save evidence packs
    output_path = task_save_packs(packs, qc_results, output_dir, pid)

    # Step 6: Generate draft spec (HTML + Excel + JSON)
    spec_outputs = task_generate_spec(packs, qc_results, output_dir, pid, data_source=detected_source)

    logger.info(f"Run complete. Evidence packs at: {output_path}")
    logger.info(f"Draft spec: {spec_outputs.get('spec_html', 'n/a')}")
    logger.info("Next step: open review UI to select governing evidence per concept.")

    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Protocol Spec Assist — Evidence Extraction")
    parser.add_argument("pdf", help="Protocol PDF path")
    parser.add_argument("--protocol-id", help="Protocol identifier (default: filename)")
    parser.add_argument("--ta", help="Therapeutic area: oncology | cardiovascular | respiratory | immunology | vaccines")
    parser.add_argument("--data-source", help="Data source override: cota | flatiron | optum_cdm | optum_ehr | marketscan | inalon | quest")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--output-dir", default="data/outputs")
    parser.add_argument("--skip-indexing", action="store_true")
    args = parser.parse_args()

    protocol_run(
        pdf_path=args.pdf,
        protocol_id=args.protocol_id,
        ta_name=args.ta,
        data_source_override=args.data_source,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        skip_indexing=args.skip_indexing,
    )
