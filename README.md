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
    ├── index_date          ← Phase 1
    ├── follow_up_end       ← Phase 1
    ├── primary_endpoint    ← Phase 1
    ├── follow_up_start     ← Phase 2
    ├── eligibility_*       ← Phase 2
    ├── censoring_rules     ← Phase 2
    ├── key_covariate       ← Phase 2
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
    Pre-review: completeness, retrieval signal, contradictions, page refs
    Post-review: unresolved packs, cross-concept consistency, missing concepts
    │
    ▼
[Review UI]  ← Human in the loop
    Programmer selects governing evidence per concept
    Flags contradictions, adds notes, overrides if needed
    │
    ▼
[Row Completion]  ← Phase 2
    LLM writes spec row from selected evidence
    │
    ▼
[Excel Workbook Writer]  ← Phase 2
    Canonical rows → formatted workbook
    Color-coded: explicit | inferred | flagged
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

**Adjudicator routing.** Low confidence → larger model on separate endpoint for second pass.
Qwen3-8B default (port 8000), Qwen3-30B-A3B adjudicator (port 8001). Same interface, different model.

**Model-agnostic client.** LocalModelClient uses OpenAI-compatible interface.
Swap to GPT-4o = change env vars, zero code changes.

**QC staged correctly.** Pre-review QC flags issues for the reviewer.
Post-review QC validates completeness after human selection. No false warnings.

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
│   ├── index_date.py           # Phase 1
│   └── endpoints.py            # follow_up_end + primary_endpoint (Phase 1)
│
├── row_completion/             # Phase 2 — row writers per tab
│   └── __init__.py
│
├── qc/
│   ├── __init__.py
│   └── rules.py                # Deterministic QC: pre-review + post-review stages
│
├── workflows/
│   ├── __init__.py
│   └── protocol_run.py         # Prefect flow — wires everything together
│
├── ui/                         # Phase 2 — Streamlit review app
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
└── outputs/                    # EvidencePack JSON + workbooks

pyproject.toml                  # Package config — pip install -e .
requirements.txt                # Pinned dependencies
```

---

## Setup

```bash
# 1. Install as editable package
pip install -e .

# 2. Download models (one-time, ~20GB total)
huggingface-cli download Qwen/Qwen3-8B-Instruct
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct  # adjudicator
huggingface-cli download BAAI/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3

# 3. Prefetch Docling models for offline use
docling-tools models download

# 4. Start vLLM servers
# Default model (port 8000):
vllm serve Qwen/Qwen3-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --enable-prefix-caching

# Adjudicator model (port 8001):
vllm serve Qwen/Qwen3-30B-A3B-Instruct \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 32768 \
    --enable-prefix-caching

# 5. Run on a protocol
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/PROTOCOL.pdf \
    --ta oncology \
    --output-dir data/outputs/
```

---

## MVP Build Order

| MVP | What | Why first |
|-----|------|-----------|
| 0 | Gold set (5-10 specs) + eval harness | Validates everything. No accuracy claims without this. |
| 1 | Retrieval + 3 concept finders | Core value — find/surface evidence |
| 2 | TA pack (oncology) | Synonym expansion, ambiguity warnings |
| 3 | Review UI | Human selection — product lives or dies here |
| 4 | Row completion from selected evidence | Copilot-style completion once evidence is confirmed |
| 5 | Deterministic QC | Trust layer |
| 6 | Workbook writer | Output artifact |

---

## Evaluation Metrics

Three tiers, measured separately:

| Tier | Metric | Target |
|------|--------|--------|
| Retrieval recall | Gold snippet in top-10 candidates | >= 80% |
| Top-1 retrieval quality | Top candidate matches gold | >= 65% |
| Row accuracy | Completed row matches gold row | >= 70% (Phase 2) |

Build eval harness first. Run on 5 protocols. Fix retrieval before fixing prompts.

---

## Environment Variables

```bash
VLLM_BASE_URL=http://localhost:8000/v1         # Default model server
ADJUDICATOR_BASE_URL=http://localhost:8001/v1   # Adjudicator model server
VLLM_API_KEY=local                              # API key (default: local)
DEFAULT_MODEL=Qwen/Qwen3-8B-Instruct           # Main extractor
ADJUDICATOR_MODEL=Qwen/Qwen3-30B-A3B-Instruct  # Hard cases
```

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
