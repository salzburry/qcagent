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
      2. Hybrid retrieval (dense + sparse)
      3. BGE reranker
      4. vLLM structured output → EvidencePack (schema-constrained)
      5. Confidence router → adjudicator model if needed
      6. Return EvidencePack
    │
    ▼
[EvidencePacks]
    concept | candidates | contradictions | confidence | provenance
    │
    ▼
[QC Engine]  ← Deterministic, no LLM
    Level 1: Completeness + evidence presence
    Level 2: Cross-concept consistency
    Level 3: Missing expected concepts (from TA pack)
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

**Provenance on every candidate.** Snippet + section + page + explicit_type.  
Every auto-filled cell is traceable to a protocol passage.

**TA packs for synonym expansion.** Not sponsor-specific — TA-level priors.  
Oncology and CV packs included. Add more as needed.

**Adjudicator routing.** Low confidence → larger model for second pass.  
Qwen3-8B default, Qwen3-30B-A3B adjudicator. Same interface, different model.

**Model-agnostic client.** LocalModelClient uses OpenAI-compatible interface.  
Swap to GPT-4o = change two env vars, zero code changes.

---

## Folder Structure

```
protocol_spec_assist/
│
├── ingest/
│   └── parse_protocol.py       # Docling parser + PyMuPDF fallback
│
├── retrieval/
│   └── search.py               # BGE-M3 + Qdrant hybrid + reranker
│
├── schemas/
│   ├── evidence.py             # EvidencePack, EvidenceCandidate — core data model
│   └── rows.py                 # Spec row schemas per tab
│
├── ta_packs/
│   ├── oncology.yaml           # Synonyms, hotspots, expected concepts
│   ├── cardiovascular.yaml
│   └── loader.py
│
├── concepts/
│   ├── index_date.py           # Phase 1
│   └── endpoints.py            # follow_up_end + primary_endpoint (Phase 1)
│
├── row_completion/             # Phase 2 — row writers per tab
│
├── qc/
│   └── rules.py                # Deterministic QC: completeness + cross-concept
│
├── workflows/
│   └── protocol_run.py         # Prefect flow — wires everything together
│
├── ui/                         # Phase 2 — Streamlit review app
│
├── serving/
│   └── model_client.py         # vLLM client (OpenAI-compatible)
│
├── eval/
│   └── harness.py              # Gold set + evaluation harness
│
└── data/
    ├── protocols/              # Drop PDFs here
    ├── gold_set/               # Manual ground truth CSV
    ├── index/                  # Qdrant persistent store
    └── outputs/                # EvidencePack JSON + workbooks
```

---

## Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Download models (one-time, ~20GB total)
huggingface-cli download Qwen/Qwen3-8B-Instruct
huggingface-cli download BAAI/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3

# 3. Start vLLM server
vllm serve Qwen/Qwen3-8B-Instruct \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --enable-prefix-caching

# 4. Run on a protocol
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
| Retrieval recall | Gold snippet in top-10 candidates | ≥ 80% |
| Selection accuracy | Top candidate matches gold | ≥ 65% |
| Row accuracy | Completed row matches gold row | ≥ 70% (Phase 2) |

Build eval harness first. Run on 5 protocols. Fix retrieval before fixing prompts.

---

## Environment Variables

```bash
VLLM_BASE_URL=http://localhost:8000/v1   # vLLM server
DEFAULT_MODEL=Qwen/Qwen3-8B-Instruct     # Main extractor
ADJUDICATOR_MODEL=Qwen/Qwen3-30B-A3B-Instruct  # Hard cases
```
