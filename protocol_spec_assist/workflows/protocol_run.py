"""
Prefect workflow — protocol_run.
Fixed path: ingest → index → find concepts → QC → auto-translate to spec → save outputs.
Workflow, not agent. Every step is predetermined.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

try:
    from prefect import flow, task, get_run_logger
    from prefect.states import Paused
    PREFECT_AVAILABLE = True
except ImportError:
    # Graceful fallback if Prefect not installed
    PREFECT_AVAILABLE = False
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
from ..concepts.study_design import find_study_period, find_censoring_rules
from ..schemas.evidence import EvidencePack
from ..qc.rules import run_all_qc, summarize_qc, IMPLEMENTED_CONCEPTS
from ..spec_output.spec_schema import build_program_spec
from ..spec_output.html_renderer import save_html
from ..spec_output.excel_writer import save_excel


# ── Tasks (each is a named, retriable step) ───────────────────────────────────

@task(name="parse-protocol", retries=1)
def task_parse_protocol(pdf_path: str, protocol_id: str) -> dict:
    """Parse protocol PDF → structured sections dict."""
    parsed = parse_protocol(pdf_path, protocol_id)
    chunks = parsed.to_chunks()
    return {"chunks": chunks, "title": parsed.title, "n_sections": len(parsed.sections)}


@task(name="index-protocol", retries=1)
def task_index_protocol(
    chunks: list[dict],
    protocol_id: str,
    index_dir: str,
) -> str:
    """Embed and index protocol chunks into Qdrant."""
    index = ProtocolIndex(index_dir=index_dir)
    index.index_protocol(chunks, protocol_id)
    return index_dir


def _run_concept_finder(finder_fn, protocol_id, index_dir, ta_name):
    """Helper to run a concept finder with fresh client/index instances."""
    index = ProtocolIndex(index_dir=index_dir)
    client = LocalModelClient()
    ta_pack = load_ta_pack(ta_name) if ta_name else None
    pack = finder_fn(protocol_id, index, client, ta_pack)
    return pack.model_dump()


@task(name="find-index-date")
def task_find_index_date(protocol_id: str, index_dir: str, ta_name: Optional[str]) -> dict:
    return _run_concept_finder(find_index_date, protocol_id, index_dir, ta_name)


@task(name="find-follow-up-end")
def task_find_follow_up_end(protocol_id: str, index_dir: str, ta_name: Optional[str]) -> dict:
    return _run_concept_finder(find_follow_up_end, protocol_id, index_dir, ta_name)


@task(name="find-primary-endpoint")
def task_find_primary_endpoint(protocol_id: str, index_dir: str, ta_name: Optional[str]) -> dict:
    return _run_concept_finder(find_primary_endpoint, protocol_id, index_dir, ta_name)


@task(name="find-inclusion-criteria")
def task_find_inclusion_criteria(protocol_id: str, index_dir: str, ta_name: Optional[str]) -> dict:
    return _run_concept_finder(find_inclusion_criteria, protocol_id, index_dir, ta_name)


@task(name="find-exclusion-criteria")
def task_find_exclusion_criteria(protocol_id: str, index_dir: str, ta_name: Optional[str]) -> dict:
    return _run_concept_finder(find_exclusion_criteria, protocol_id, index_dir, ta_name)


@task(name="find-study-period")
def task_find_study_period(protocol_id: str, index_dir: str, ta_name: Optional[str]) -> dict:
    return _run_concept_finder(find_study_period, protocol_id, index_dir, ta_name)


@task(name="find-censoring-rules")
def task_find_censoring_rules(protocol_id: str, index_dir: str, ta_name: Optional[str]) -> dict:
    return _run_concept_finder(find_censoring_rules, protocol_id, index_dir, ta_name)


@task(name="run-qc")
def task_run_qc(
    packs_dict: dict[str, dict],
    expected_concepts: Optional[list[str]],
) -> list[dict]:
    packs = {k: EvidencePack.model_validate(v) for k, v in packs_dict.items()}
    # Run pre-review QC only — post-review QC runs after human selection
    results = run_all_qc(packs, expected_concepts, stage="pre_review")
    print(summarize_qc(results))
    return [vars(r) for r in results]


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


@task(name="generate-spec-outputs")
def task_generate_spec(
    packs_dict: dict[str, dict],
    output_dir: str,
    protocol_id: str,
    protocol_title: Optional[str] = None,
) -> dict:
    """Auto-translate evidence packs → program spec → HTML + Excel."""
    spec = build_program_spec(protocol_id, packs_dict, protocol_title)

    out_dir = Path(output_dir)
    html_path = save_html(spec, str(out_dir / f"{protocol_id}_spec.html"))
    excel_path = save_excel(spec, str(out_dir / f"{protocol_id}_spec.xlsx"))

    # Also save spec JSON
    spec_json_path = out_dir / f"{protocol_id}_spec.json"
    with open(spec_json_path, "w") as f:
        json.dump(spec.model_dump(), f, indent=2, default=str)

    return {
        "html_path": html_path,
        "excel_path": excel_path,
        "json_path": str(spec_json_path),
    }


# ── Main flow ─────────────────────────────────────────────────────────────────

@flow(name="protocol-spec-run", log_prints=True)
def protocol_run(
    pdf_path: str,
    protocol_id: Optional[str] = None,
    ta_name: Optional[str] = None,
    index_dir: str = "data/index",
    output_dir: str = "data/outputs",
    skip_indexing: bool = False,
):
    """
    Main workflow: protocol PDF → evidence packs → QC → program spec.

    Concepts extracted:
      - index_date, follow_up_end, primary_endpoint  (Phase 1)
      - eligibility_inclusion, eligibility_exclusion  (Phase 1.5)
      - study_period, censoring_rules                 (Phase 1.5)

    Outputs:
      - evidence_packs.json     — raw extraction results
      - _spec.html              — HTML preview for review
      - _spec.xlsx              — Excel program spec draft
      - _spec.json              — structured spec as JSON
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

    # Step 1: Parse
    parse_result = task_parse_protocol(pdf_path, pid)
    logger.info(f"Parsed {parse_result['n_sections']} sections")

    # Step 2: Index (skip if already indexed)
    if not skip_indexing:
        task_index_protocol(parse_result["chunks"], pid, index_dir)
    else:
        logger.info("Skipping indexing — using existing index.")

    # Step 3: Find concepts — all 7 finders run sequentially
    packs = {}

    packs["index_date"] = task_find_index_date(pid, index_dir, ta_name)
    packs["follow_up_end"] = task_find_follow_up_end(pid, index_dir, ta_name)
    packs["primary_endpoint"] = task_find_primary_endpoint(pid, index_dir, ta_name)
    packs["eligibility_inclusion"] = task_find_inclusion_criteria(pid, index_dir, ta_name)
    packs["eligibility_exclusion"] = task_find_exclusion_criteria(pid, index_dir, ta_name)
    packs["study_period"] = task_find_study_period(pid, index_dir, ta_name)
    packs["censoring_rules"] = task_find_censoring_rules(pid, index_dir, ta_name)

    # Step 4: Pre-review QC
    ta_pack = load_ta_pack(ta_name) if ta_name else None
    expected = ta_pack.expected_concepts if ta_pack else None
    qc_results = task_run_qc(packs, expected)

    # Step 5: Save evidence packs
    output_path = task_save_packs(packs, qc_results, output_dir, pid)

    # Step 6: Auto-translate to program spec (HTML + Excel + JSON)
    spec_paths = task_generate_spec(
        packs, output_dir, pid,
        protocol_title=parse_result.get("title"),
    )

    logger.info(f"Run complete.")
    logger.info(f"  Evidence packs: {output_path}")
    logger.info(f"  HTML spec:      {spec_paths['html_path']}")
    logger.info(f"  Excel spec:     {spec_paths['excel_path']}")
    logger.info("Next step: review the HTML/Excel spec and correct as needed.")

    return output_path


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Protocol Spec Assist — Evidence Extraction + Spec Generation")
    parser.add_argument("pdf", help="Protocol PDF path")
    parser.add_argument("--protocol-id", help="Protocol identifier (default: filename)")
    parser.add_argument("--ta", help="Therapeutic area: oncology | cardiovascular")
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--output-dir", default="data/outputs")
    parser.add_argument("--skip-indexing", action="store_true")
    args = parser.parse_args()

    protocol_run(
        pdf_path=args.pdf,
        protocol_id=args.protocol_id,
        ta_name=args.ta,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        skip_indexing=args.skip_indexing,
    )
