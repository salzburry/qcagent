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
    Fallback: PyMuPDF with micro-section merging
    Quality scoring: pass / warn / fail (fail = pipeline stops)
    │
    ▼
[BGE-M3 + Qdrant Hybrid Index]
    Dense + sparse vectors, RRF fusion, persistent local store
    │
    ▼
[Concept Finders]  ← Fixed workflow nodes, not agents
    │
    ├── index_date              ← v0.4
    ├── follow_up_end           ← v0.4
    ├── primary_endpoint        ← v0.4
    ├── eligibility_inclusion   ← v0.4 (two-pass: inventory + detail)
    ├── eligibility_exclusion   ← v0.4 (two-pass: inventory + detail)
    ├── study_period            ← v0.4 (DataPrepExtraction schema)
    ├── censoring_rules         ← v0.4
    ├── demographics            ← v0.4
    ├── clinical_characteristics← v0.4
    ├── biomarkers              ← v0.4
    ├── lab_variables           ← v0.4
    └── treatment_variables     ← v0.4
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
[Row Writers]  ← v0.4 — deterministic row-family expansion
    DemographicsWriter: AGE → AGE, AGEN, AGEGR, AGEGRN
    DataPrepWriter:     evidence → ImportantDate + TimePeriod rows
    EndpointWriter:     endpoint → EVENTFL, EVENTDT, TTOEVENT
    CensoringWriter:    rules → CENS01, CENS01FL, CENS01DT
    │
    ▼
[QC Engine]  ← Deterministic, no LLM
    Pre-review: completeness, retrieval signal, contradictions, page refs,
                quote-in-chunk, Data Prep dates, demographics minimum
    Post-review: unresolved packs, cross-concept consistency, missing concepts
    │
    ▼
[Draft Spec Generator]  ← v0.4
    EvidencePacks → Row Writers → ProgramSpec (Pydantic schema)
    Outputs: JSON + self-contained HTML preview + formatted Excel workbook
    Excel: hidden provenance columns (J-L) + hidden _Provenance sheet
    Stakeholder-facing tabs have NO confidence coloring
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

**Row writers for deterministic expansion.** Variable families (AGE → AGE/AGEN/AGEGR/AGEGRN)
are expanded deterministically by row writers, not by the LLM. Source-specific definitions
come from `data_sources/registry.py`.

**TA packs for synonym expansion.** Not sponsor-specific — TA-level priors.
Oncology and CV packs included. Add more as needed.

**Single model.** Qwen3-14B on A100 40GB via vLLM. Same model handles both
extraction and adjudication — no separate server needed.

**Model-agnostic client.** LocalModelClient uses OpenAI-compatible interface.
Swap to GPT-4o = change env vars, zero code changes.

**QC staged correctly.** Pre-review QC flags issues for the reviewer (including quote-in-chunk validation).
Post-review QC validates completeness after human selection. No false warnings.

**Parse-fail gate.** If PDF parse quality is FAIL, the pipeline stops and produces a
shell spec with a CRITICAL QC warning instead of feeding garbage into extraction.

**Extracted evidence beats placeholders.** Auto-generated placeholder rows (INIT, INDEX, FUED)
are always replaced when real extracted evidence is available from concept finders.

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
│   └── parse_protocol.py       # Docling parser + PyMuPDF fallback + quality scoring
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
│   ├── eligibility.py          # Two-pass: inventory → per-criterion detail
│   ├── study_design.py         # DataPrepExtraction (dates + periods) + censoring_rules
│   ├── demographics.py         # Demographics finder (static template + LLM enrichment)
│   ├── clinical_characteristics.py
│   ├── biomarkers.py
│   ├── lab_variables.py
│   └── treatment_variables.py
│
├── row_completion/
│   ├── __init__.py
│   ├── base.py                 # RowWriter base class
│   ├── demographics_writer.py  # AGE/SEX/RACE/ETH family expansion
│   ├── data_prep_writer.py     # ImportantDate + TimePeriod from evidence
│   └── outcomes_writer.py      # Endpoint + censoring variable families
│
├── data_sources/
│   ├── __init__.py
│   └── registry.py             # Source-specific definitions (COTA, Flatiron, Optum, etc.)
│
├── spec_output/
│   ├── __init__.py
│   ├── spec_schema.py          # ProgramSpec Pydantic model + build_program_spec()
│   ├── html_renderer.py        # Self-contained HTML preview with confidence badges
│   └── excel_writer.py         # Excel workbook + hidden provenance sheet
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
│   └── model_client.py         # vLLM client (OpenAI-compatible, per-call max_tokens)
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

## Quick Start

```bash
pip install -e .                                    # CPU — no GPU needed
pytest tests/ -v                                    # CPU — verify install
python colab_setup.py --download-models             # CPU — download ~33GB of models
python setup_vllm.py --set-env                      # GPU — requires A100 40GB
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/PROTOCOL.pdf --ta oncology       # GPU — run extraction
```

See **[TEST_RUN_GUIDE.md](TEST_RUN_GUIDE.md)** for full step-by-step instructions,
Colab setup, environment variables, pipeline outputs, and troubleshooting.

---

## v0.4 Changes

### New features
1. **Row completion layer** — `row_completion/` module with deterministic row-family expansion (DemographicsWriter, DataPrepWriter, EndpointWriter, CensoringWriter)
2. **DataPrepExtraction schema** — replaces monolithic study_period with named operational fields (INIT, INDEX, FUED, CENSDT, STUDY_PD, PRE_INT, FU, etc.)
3. **Two-pass eligibility** — inventory pass (lightweight list) then per-criterion detail pass to avoid token overflow on 20+ criteria
4. **Parse quality gate** — pipeline stops on FAIL grade, produces shell spec with CRITICAL QC warning
5. **Placeholder precedence fix** — extracted evidence from index_date/follow_up_end always replaces auto-generated placeholders
6. **Hidden provenance** — hidden columns J-L on variable tabs + hidden _Provenance sheet; NO confidence coloring on stakeholder-facing tabs
7. **Bullet-aware chunking** — sliding window preserves numbered/lettered/bulleted list items as atomic units
8. **PyMuPDF micro-section merging** — merges adjacent micro-sections (reduces 552 → 142 sections on noisy PDFs)
9. **Per-call max_tokens** — model client supports per-call override (default raised from 1536 to 4096)
10. **StudyPop operational definitions** — definition field uses operational detail; protocol quote moved to additional_notes as provenance
11. **Document-order chunk neighborhoods** — eligibility detail pass uses page-order neighbors, not rank-order
12. **Contradiction detection restored** — eligibility inventory prompts detect and propagate contradictions
13. **ETH/ETHN variable names** — aligned with data_sources/registry.py (was ETHNIC/ETHNICN)
14. **QC expanded** — QC-007 (Data Prep completeness), QC-008 (demographics minimum), all 12 concept finders tracked

### Fixes
15. **Broadened Docling fallback** — catches all exceptions, not just ImportError
16. **max_tokens raised** — from 1536 to 4096 for multi-row extractions

---

## v0.3 Changes

### New features
1. **9 new concept finders** — eligibility_inclusion, eligibility_exclusion, study_period, censoring_rules, demographics, clinical_characteristics, biomarkers, lab_variables, treatment_variables
2. **Draft spec generation** — EvidencePacks → ProgramSpec with JSON + HTML + Excel outputs
3. **HTML preview** — self-contained HTML with confidence badges, explicit/inferred markers, QC warnings
4. **Excel workbook** — 10 sheets (1.Cover, 2.QC Review, 3.Data Prep, 4.StudyPop, 5A.Demos, 5B.ClinChars, 5C.BioVars, 5D.LabVars, 6.TreatVars, 7.Outcomes)
5. **Spec uses selected candidate** — uses `selected_candidate_id` when reviewed, top-ranked when draft
6. **qc_quote_in_chunk wired** — validates that candidate snippets appear in source chunks
7. **Device-aware retrieval** — auto-detects GPU/CPU, respects `RETRIEVAL_DEVICE` and `RETRIEVAL_FP16` env vars
8. **Shared ProtocolIndex** — single index instance shared across all finders (no repeated model loading)

### Fixes
9. **Version consistency** — all finders, schemas, and spec at v0.3.0
10. **Default model** — Qwen3-14B on A100 40GB
11. **HuggingFace CLI** — updated to `hf download` (was `huggingface-cli download`)
12. **Colab setup** — `colab_setup.py` handles Drive mounting, model download, compute budgeting
13. **GPU auto-detection** — `setup_vllm.py` detects GPU and applies workarounds
14. **Google Drive integration** — models persist on Drive across Colab sessions (download once)
15. **Retrieval on CPU** — embeddings/reranker default to CPU on single-GPU machines

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
