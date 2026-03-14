# Protocol Spec Assist

Fully local AI-assisted protocol → programming spec authoring tool.
No cloud. No external API calls after model download. Complete data control.

---

## Architecture

```
Protocol PDF
    │
    ▼
[Docling Parser]
    Preserves: layout, reading order, tables, appendices, footnotes
    Fallback: PyMuPDF for simple parsing
    │
    ▼
[BGE-M3 + Qdrant Hybrid Index]
    Dense + sparse vectors, RRF fusion, persistent local store
    │
    ▼
[Concept Finders]  ← Fixed workflow nodes, not agents
    │
    ├── index_date              ← v0.3
    ├── follow_up_end           ← v0.3
    ├── primary_endpoint        ← v0.3
    ├── eligibility_inclusion   ← v0.3
    ├── eligibility_exclusion   ← v0.3
    ├── study_period            ← v0.3
    ├── censoring_rules         ← v0.3
    ├── follow_up_start         ← planned
    ├── key_covariate           ← planned
    └── ...
    │
    Each finder does the same fixed sequence:
      1. Build query bank (base query + TA pack synonyms)
      2. Hybrid retrieval (dense + sparse + RRF fusion)
      3. BGE reranker
      4. vLLM structured output → EvidencePack (schema-constrained)
      5. Confidence router → adjudicator model if needed
      6. Return EvidencePack with chunk_id provenance
    │
    ▼
[EvidencePacks]
    concept | candidates | contradictions | confidence | provenance | concept_metadata
    │
    ▼
[QC Engine]  ← Deterministic, no LLM
    Pre-review: completeness, retrieval signal, contradictions, page refs, quote-in-chunk
    Post-review: unresolved packs, cross-concept consistency, missing concepts
    │
    ▼
[Draft Spec Generator]  ← v0.3
    EvidencePacks → ProgramSpec (Pydantic schema)
    Outputs: JSON + self-contained HTML preview + formatted Excel workbook
    Uses selected_candidate_id if reviewed, top-ranked candidate if draft
    │
    ▼
[Review UI]  ← Planned
    Programmer selects governing evidence per concept
    Flags contradictions, adds notes, overrides if needed
```

---

## Key Design Decisions

**Workflow not agent.** Concept finders are fixed-path automation nodes.
The LLM reasons inside a bounded box. System behavior is deterministic.

**Concept-first not tab-first.** `index_date` is a concept.
`Study Population` tab row is an output artifact. One concept can populate multiple tabs.

**EvidencePack as handoff.** Every concept produces one EvidencePack.
Every row writer consumes one. QC runs on packs. Clean separation.

**Provenance via chunk_id.** Every candidate carries its `chunk_id` back to the indexed chunk.
No fuzzy snippet matching — deterministic provenance from retrieval through to review.

**Concept metadata preserved.** Concept-specific fields (e.g. `rule_type`, `components`, `time_to_event`)
are carried through to downstream consumers via `concept_metadata` on EvidencePack.

**TA packs for synonym expansion.** Not sponsor-specific — TA-level priors.
Oncology and CV packs included. Add more as needed.

**Tiered model setup.** Two supported tiers:
- `colab_a100` — Qwen3-14B (base) on A100 40GB. Good extraction quality (~70%). Budget-friendly for Colab.
- `h100` — Qwen3-235B-A22B-FP8 (MoE, ~22B active params) on H100 80GB. Best quality.
  Note: BF16 weights are ~470GB (118 shards). Use the FP8 quantized variant
  or multi-GPU tensor parallelism. See TEST_RUN_GUIDE.md for details.
Set `MODEL_TIER=colab_a100` or `MODEL_TIER=h100` to auto-configure. Both extraction and adjudication
run on the same server — no separate adjudicator needed.

**Model-agnostic client.** LocalModelClient uses OpenAI-compatible interface.
Swap to GPT-4o = change env vars, zero code changes.

**QC staged correctly.** Pre-review QC flags issues for the reviewer (including quote-in-chunk validation).
Post-review QC validates completeness after human selection. No false warnings.

**Device-aware retrieval.** Embedding and reranker models auto-detect GPU/CPU.
Override with `RETRIEVAL_DEVICE` and `RETRIEVAL_FP16` env vars for explicit control.

---

## Folder Structure

```
protocol_spec_assist/
│
├── __init__.py
│
├── ingest/
│   ├── __init__.py
│   └── parse_protocol.py       # Docling parser + PyMuPDF fallback
│
├── retrieval/
│   ├── __init__.py
│   └── search.py               # BGE-M3 + Qdrant hybrid (RRF) + reranker
│
├── schemas/
│   ├── __init__.py
│   ├── evidence.py             # EvidencePack, EvidenceCandidate — core data model
│   └── rows.py                 # Spec row schemas per tab
│
├── ta_packs/
│   ├── __init__.py
│   ├── oncology.yaml           # Synonyms, hotspots, expected concepts
│   ├── cardiovascular.yaml
│   └── loader.py
│
├── concepts/
│   ├── __init__.py
│   ├── index_date.py           # Index date finder
│   ├── endpoints.py            # follow_up_end + primary_endpoint
│   ├── eligibility.py          # eligibility_inclusion + eligibility_exclusion
│   └── study_design.py         # study_period + censoring_rules
│
├── spec_output/
│   ├── __init__.py
│   ├── spec_schema.py          # ProgramSpec Pydantic model + build_program_spec()
│   ├── html_renderer.py        # Self-contained HTML preview with confidence badges
│   └── excel_writer.py         # Formatted Excel workbook (openpyxl)
│
├── qc/
│   ├── __init__.py
│   └── rules.py                # Deterministic QC: pre-review + post-review stages
│
├── workflows/
│   ├── __init__.py
│   └── protocol_run.py         # Prefect flow — wires everything together
│
├── ui/                         # Planned — Streamlit review app
│   └── __init__.py
│
├── serving/
│   ├── __init__.py
│   └── model_client.py         # vLLM client (OpenAI-compatible, dual endpoints)
│
├── eval/
│   ├── __init__.py
│   └── harness.py              # Gold set + evaluation harness
│
data/
├── protocols/                  # Drop PDFs here
├── gold_set/                   # Manual ground truth CSV
├── index/                      # Qdrant persistent store
└── outputs/                    # EvidencePack JSON + spec outputs

pyproject.toml                  # Package config — pip install -e .
requirements.txt                # Dependencies with lower-bound versions
```

---

## Setup

### Quick start — Colab A100 40GB

```bash
# 1. Install
pip install -e .

# 2. Download models (Colab: saves to Google Drive)
python colab_setup.py --download-models --tier colab_a100

# 3. Start vLLM (auto-detects GPU, picks Qwen3-14B for 40GB)
python setup_vllm.py --set-env

# 4. Run on a protocol
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/PROTOCOL.pdf --ta oncology
```

### Full setup — H100 80GB

```bash
# 1. Install
pip install -e .

# 2. Download models
hf download Qwen/Qwen3-235B-A22B-FP8           # MoE LLM, FP8 quantized (~120 GB)
hf download BAAI/bge-m3
hf download BAAI/bge-reranker-v2-m3

# 3. Prefetch Docling models for offline use
python -c "from docling.utils.model_downloader import download_models; download_models()"

# 4. Start vLLM server
python setup_vllm.py --tier h100 --set-env

# 5. Run on a protocol
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/PROTOCOL.pdf --ta oncology --output-dir data/outputs/
```

---

## Environment Variables

```bash
# Easiest: set the tier and everything else auto-configures
MODEL_TIER=colab_a100                           # colab_a100 | colab_a100_single | h100

# Or override individually:
VLLM_BASE_URL=http://localhost:8000/v1         # Default model server
ADJUDICATOR_BASE_URL=http://localhost:8000/v1   # Same server (single-model setup)
VLLM_API_KEY=local                              # API key (default: local)
DEFAULT_MODEL=Qwen/Qwen3-14B                   # Base extractor (or Qwen3-235B-A22B-FP8)
ADJUDICATOR_MODEL=Qwen/Qwen3-14B              # Adjudicator

# Retrieval device control (auto-detected by default)
RETRIEVAL_DEVICE=cpu                            # Force CPU for embeddings/reranker
RETRIEVAL_FP16=false                            # Disable fp16 (required for CPU)
```

---

## Pipeline Outputs (v0.3)

Each run produces 4 artifacts in `data/outputs/`:

| File | Content |
|------|---------|
| `{protocol_id}_evidence_packs.json` | Raw evidence packs + QC results |
| `{protocol_id}_spec.json` | Structured ProgramSpec (machine-readable) |
| `{protocol_id}_spec.html` | Self-contained HTML preview with confidence badges |
| `{protocol_id}_spec.xlsx` | Formatted Excel workbook (10 sheets, color-coded) |

---

## v0.3 Changes

### New features
1. **9 new concept finders** — eligibility_inclusion, eligibility_exclusion, study_period, censoring_rules, demographics, clinical_characteristics, biomarkers, lab_variables, treatment_variables
2. **Draft spec generation** — EvidencePacks → ProgramSpec with JSON + HTML + Excel outputs
3. **HTML preview** — self-contained HTML with confidence badges, explicit/inferred markers, QC warnings
4. **Excel workbook** — 10 sheets (1.Cover, 2.QC Review, 3.Data Prep, 4.StudyPop, 5A.Demos, 5B.ClinChars, 5C.BioVars, 5D.LabVars, 6.TreatVars, 7.Outcomes), color-coded by confidence
5. **Spec uses selected candidate** — uses `selected_candidate_id` when reviewed, top-ranked when draft
6. **qc_quote_in_chunk wired** — validates that candidate snippets appear in source chunks
7. **Device-aware retrieval** — auto-detects GPU/CPU, respects `RETRIEVAL_DEVICE` and `RETRIEVAL_FP16` env vars
8. **Shared ProtocolIndex** — single index instance shared across all finders (no repeated model loading)

### Fixes
9. **Version consistency** — all finders, schemas, and spec at v0.3.0
10. **Default model** — upgraded to `Qwen/Qwen3-235B-A22B-FP8` MoE for H100
11. **HuggingFace CLI** — updated to `hf download` (was `huggingface-cli download`)
12. **Model tiers** — `MODEL_TIER` env var for one-line GPU config (`colab_a100` / `h100`)
13. **Colab setup** — `colab_setup.py` handles Drive mounting, model download, compute budgeting
14. **GPU auto-detection** — `setup_vllm.py` auto-selects model based on VRAM (14B for 40GB, 235B for 80GB)
15. **Google Drive integration** — models persist on Drive across Colab sessions (download once)

---

## v0.2 Changes (from flat prototype)

### Fixes applied
1. **Package structure** — proper `protocol_spec_assist/` package with `__init__.py` files and working relative imports
2. **Docling options** — `PdfPipelineOptions` now actually passed to `DocumentConverter`
3. **True hybrid retrieval** — Qdrant `query_points` with dense+sparse prefetch and RRF fusion
4. **Table serialization** — `_table_to_text()` joins column names correctly (not characters of `str(dict_keys(...))`)
5. **Provenance via chunk_id** — LLM returns `chunk_id` from input chunks instead of fuzzy snippet matching
6. **Adjudicator endpoint** — separate `default_base_url` and `adjudicator_base_url` for dual-server setup
7. **QC staged** — pre-review QC (before human selection) vs post-review QC (after), no false "unresolved" warnings
8. **Concept metadata preserved** — `rule_type`, `components`, `time_to_event`, `llm_confidence` survive handoff
9. **Deterministic chunk IDs** — based on `protocol_id + content_hash`, prevents duplicates on re-index
10. **Section aggregation** — Docling parser accumulates text under headings instead of creating one section per item
11. **Re-index safety** — existing chunks deleted before re-indexing a protocol
12. **Eval harness** — `created_date` in template, `writerow` instead of `writerows([row])`, configurable concept list
