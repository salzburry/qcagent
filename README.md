# Protocol Spec Assist

Fully local AI-assisted protocol → programming spec authoring tool.
No cloud. No external API calls after model download. Complete data control.

---

## Architecture

```
Protocol PDF
    │
    ▼
[Multi-Strategy Parser]
    Strategy 1: Docling (no OCR) — layout, reading order, tables
    Strategy 2: Docling (with OCR) — for scanned/mixed PDFs
    Strategy 3: PyMuPDF heading-based — heading detection + section merging
    Strategy 4: PyMuPDF page-first — one section per page, no heading detection
    Each strategy quality-scored; best result accepted
    Quality scoring: pass / warn / fail (fail = pipeline stops)
    │
    ▼
[BGE-M3 + Qdrant Hybrid Index]
    Dense + sparse vectors, RRF fusion, persistent local store
    │
    ▼
[Concept Finders]  ← Fixed workflow nodes, not agents
    │
    ├── index_date
    ├── follow_up_end
    ├── primary_endpoint
    ├── eligibility_inclusion   (two-pass: inventory + detail)
    ├── eligibility_exclusion   (two-pass: inventory + detail)
    ├── study_period            (two-pass: regex mining + LLM classification)
    ├── censoring_rules
    ├── demographics
    ├── clinical_characteristics
    ├── biomarkers
    ├── lab_variables
    ├── treatment_variables
    ├── cohort_definitions      (Section B: treatment arms, comparators, analysis populations)
    └── source_data_prep        (Section D: protocol-specific + source-known issues)
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
[Row Writers]  ← Deterministic row-family expansion
    DemographicsWriter: AGE → AGE, AGEN, AGEGR, AGEGRN (provenance-gated)
    DataPrepWriter:     evidence → ImportantDate + TimePeriod rows ([UNRESOLVED] if missing)
    EndpointWriter:     sponsor term → OS_EVENTFL, PFS_TTOEVENT (org naming)
    CensoringWriter:    rules → DEATH_FL, LTFU_DT, DISENRL_REAS (meaningful prefixes)
    │
    ▼
[QC Engine]  ← Deterministic, no LLM
    Pre-review: completeness, retrieval signal, contradictions, page refs,
                quote-in-chunk, Data Prep dates, demographics minimum
    Post-review: unresolved packs, cross-concept consistency, missing concepts
    │
    ▼
[Draft Spec Generator]
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

**Schema flattening for vLLM.** Pydantic generates JSON schemas with `$ref`, `$defs`,
and `anyOf` (for Optional fields). vLLM's xgrammar backend silently mishandles these,
producing empty/garbage output instead of raising errors. `_flatten_schema()` inlines
all references and simplifies nullable unions before sending to vLLM via `guided_json`.

**Chain-of-thought before constraints.** All extraction schemas place a `chain_of_thought`
field as the FIRST field, and `reasoning` as the first field in nested sub-models.
This follows the "Think Before Constraining" approach (arXiv:2601.07525) — letting the
model reason freely before committing to structured fields improves accuracy by up to 27%
on constrained decoding tasks with smaller models.

**Outlines guided decoding.** vLLM is configured with outlines as the guided decoding
backend instead of the default xgrammar. Outlines has broader JSON schema coverage per
JSONSchemaBench benchmarks and handles complex schemas more reliably. `setup_vllm.py`
auto-detects the correct CLI flag (`--guided-decoding-backend` on older vLLM,
`--structured-outputs-config.backend` on newer versions).

**Model-agnostic client.** LocalModelClient uses OpenAI-compatible interface.
Swap to GPT-4o = change env vars, zero code changes.

**QC staged correctly.** Pre-review QC flags issues for the reviewer (including quote-in-chunk validation).
Post-review QC validates completeness after human selection. No false warnings.

**Best-of-all parser selection.** All four parser strategies are evaluated and scored.
The best-scoring acceptable result is returned — not the first acceptable one.
This ensures a page-first parse with grade "pass" wins over a heading-based parse
with grade "warn", even though heading-based runs first.

**Parse-fail gate.** If all parser strategies produce grade FAIL, the pipeline stops
and produces a shell spec with a CRITICAL QC warning instead of feeding garbage into extraction.

**[UNRESOLVED] markers, not plausible fakes.** Placeholder rows use explicit `[UNRESOLVED]` markers
instead of realistic-sounding default definitions. This prevents reviewers from mistaking
auto-generated text for extracted protocol language.

**Extracted evidence beats placeholders.** Auto-generated placeholder rows (INIT, INDEX, FUED)
are always replaced when real extracted evidence is available from concept finders.

**Multi-channel source detection.** Data source is detected via a 4-channel priority cascade:
explicit override > study_period metadata > protocol title > protocol text sample.

**Two-pass Data Prep extraction.** Pass 1 mines date-like candidates locally via regex.
Pass 2 sends pre-mined candidates to the LLM for classification — cheaper and more reliable
than one-shot extraction.

**Sponsor-derived outcome naming.** Endpoint and censoring variable prefixes derive from
sponsor terms (OS, PFS, MACE, DEATH, LTFU) instead of positional names (EP01, CENS01).
20+ known abbreviation mappings included.

**Demographics provenance gate.** Static-only demographics packs (no source page provenance)
are marked as `explicit="inferred"` with `[UNRESOLVED]` notes, preventing false provenance claims.
Optional demographic families (BMI, weight, height, smoking) are only included when mentioned
in candidates with actual page-level provenance — not from static template snippets.

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
│   └── parse_protocol.py       # Multi-strategy parser (4 strategies, quality-scored)
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
│   ├── study_design.py         # Two-pass DataPrepExtraction (regex mine + LLM classify) + censoring
│   ├── demographics.py         # Demographics finder (static template + LLM enrichment)
│   ├── clinical_characteristics.py
│   ├── biomarkers.py
│   ├── lab_variables.py
│   ├── treatment_variables.py
│   ├── cohort_definitions.py   # Section B: treatment arms, comparators, analysis populations
│   └── source_data_prep.py     # Section D: protocol + source-known data prep issues
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
│   └── registry.py             # Source-specific definitions + multi-channel detection
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

## Hardware Requirements

| Component | Minimum |
|-----------|---------|
| **GPU** | 1x A100 40 GB (only needed for LLM inference) |
| **RAM** | 32 GB |
| **Disk** | 35 GB free (+ Google Drive if using Colab) |
| **CPU** | 8 cores |

### Model sizes on disk

| Model | Size | Purpose |
|-------|------|---------|
| `Qwen/Qwen3-14B` | ~28 GB | Base extractor + adjudicator |
| `BAAI/bge-m3` | ~2.3 GB | Embeddings (dense + sparse) |
| `BAAI/bge-reranker-v2-m3` | ~2.3 GB | Reranker |
| Docling models | ~360 MB (auto-downloaded) | PDF table/layout parsing |

**First-time download: ~33 GB.** Download on CPU — costs zero compute units on Colab.

---

## IMPORTANT: CPU-First Workflow

**Do as much as possible on CPU before switching to GPU.**

GPU time is expensive (A100 = ~8.3 Colab units/hr). Only LLM inference
requires a GPU. Everything else — install, tests, model downloads,
code development, PDF preparation — runs on CPU for free.

```
┌─────────────────────────────────────────────────────┐
│  CPU (FREE)                                         │
│                                                     │
│  ✓ Install the package          (Step 1)            │
│  ✓ Run unit tests               (Step 2)            │
│  ✓ Download models              (Step 3)            │
│  ✓ Prepare your protocol PDF    (Step 4)            │
│  ✓ Debug / develop pipeline code                    │
│  ✓ Review outputs from previous runs                │
│                                                     │
├─────────────────────────────────────────────────────┤
│  A100 GPU (COSTS COMPUTE UNITS)                     │
│                                                     │
│  ✓ Start the LLM server        (Step 5)             │
│  ✓ Run the pipeline            (Step 6)             │
│                                                     │
│  → Disconnect GPU when done to stop billing         │
└─────────────────────────────────────────────────────┘
```

---

## Setup

Pick your environment below. After environment setup, all remaining steps (1–7) are **identical**.

### Option A: Google Colab

**Cost:** Requires Colab Pro/Pro+ with A100 access. Free tier T4 does **not** have enough VRAM.

**Recommended GPU: A100 40GB** (~8.3 units/hr, Qwen3-14B fits comfortably).

#### Compute Unit Budget (150 units = ~18 hours on A100)

| Task | Runtime | GPU | Units |
|------|---------|-----|-------|
| Install + tests + download models | CPU (free) | None | **0** |
| Debug/develop pipeline code | CPU (free) | None | **0** |
| Prepare PDFs + review outputs | CPU (free) | None | **0** |
| Pipeline dev & testing | A100 | ~6 hrs | **~50** |
| Extraction runs | A100 | ~10 hrs | **~85** |
| Buffer | — | — | **~15** |

**Key rule: Do ALL setup on FREE CPU runtime. Switch to A100 ONLY for inference (Steps 5–6).**

#### A1. Mount Google Drive (CRITICAL)

> **WARNING:** Colab's VM disk is only ~100GB and is wiped every session.
> You MUST save models to Google Drive at `/content/drive/MyDrive/`, NOT `/drive/`.
> Using `/drive` creates a local folder on the VM — your 2TB is at `/content/drive/MyDrive/`.

```python
# Cell 1 — Mount Drive (follow the auth prompt)
from google.colab import drive
drive.mount('/content/drive')

# Verify — should show your Drive folders
import os
print(os.listdir('/content/drive/MyDrive/'))
```

#### A2. Clone the repo

```python
!git clone https://github.com/salzburry/qcagent.git
%cd qcagent
```

Then continue to **Step 1** below.

### Option B: Local Linux / Cloud VM

#### B1. Requirements

- Python 3.10 or 3.11
- NVIDIA A100 40GB (or any GPU with >= 30GB VRAM) — only needed for Steps 5–6
- CUDA drivers installed (`nvidia-smi` should work)

#### B2. Clone and enter the repo

```bash
git clone https://github.com/salzburry/qcagent.git
cd qcagent
```

#### B3. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

#### B4. Place your protocol PDF

```bash
mkdir -p data/protocols
cp /path/to/your_protocol.pdf data/protocols/
```

Then continue to **Step 1** below.

---

## Step 1: Install the Package (CPU)

**No GPU needed.**

```bash
pip install -e ".[dev]"
pip install vllm

# Verify install
python -c "from protocol_spec_assist.schemas.evidence import EvidencePack; print('OK')"
python -m protocol_spec_assist.workflows.protocol_run --help
```

### Troubleshooting: Docling install

Docling depends on PyTorch. If you're on a CPU-only machine or hit issues:

```bash
pip install docling --extra-index-url https://download.pytorch.org/whl/cpu
```

If Docling fails to install entirely, the pipeline falls back to PyMuPDF automatically (less accurate table extraction, but functional).

---

## Step 2: Run Unit Tests (CPU)

**No GPU, no vLLM, no models needed.** Verifies all code logic is correct.

```bash
pytest tests/ -v
```

You should see **~160 passed, 2 skipped**.

### Quick smoke test

```bash
python -c "
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.qc.rules import run_all_qc, qc_quote_in_chunk
from protocol_spec_assist.ta_packs.loader import load_ta_pack
from protocol_spec_assist.serving.model_client import LocalModelClient, ExtractionResult
from protocol_spec_assist.row_completion import DemographicsWriter, expand_data_prep
print('All imports OK')
"
```

---

## Step 3: Download Models (CPU)

**No GPU needed.** Downloads run on CPU and cost zero compute units.

### 3a. Embedding + Reranker

These download automatically from HuggingFace when first loaded.
To pre-download (recommended to avoid timeout during pipeline run):

```bash
pip install huggingface-hub

# BGE-M3 embeddings (~2.3 GB)
huggingface-cli download BAAI/bge-m3

# BGE reranker (~2.3 GB)
huggingface-cli download BAAI/bge-reranker-v2-m3
```

### 3b. LLM model

```bash
# Base extractor + adjudicator (~28 GB)
huggingface-cli download Qwen/Qwen3-14B
```

**Or use the Colab helper** (downloads to Google Drive automatically):
```bash
python colab_setup.py --download-models
```

### 3c. Docling models

```bash
python -c "from docling.utils.model_downloader import download_models; download_models()"
```

Without this, Docling auto-downloads on first PDF parse (~30s extra).

---

## Step 4: Prepare Your Protocol PDF (CPU)

**No GPU needed.**

### Text-based (digital) PDF

Just place it in `data/protocols/`:
```bash
cp your_protocol.pdf data/protocols/
```

### Scanned (image-only) PDF

You must OCR it first:

```bash
pip install ocrmypdf
# May also need: apt install tesseract-ocr (Linux) or brew install tesseract (Mac)
ocrmypdf scanned_protocol.pdf data/protocols/protocol_ocr.pdf
```

Verify text was extracted:
```bash
python -c "
import fitz
doc = fitz.open('data/protocols/your_protocol.pdf')
text = doc[0].get_text()
print(f'Page 1 text length: {len(text)} chars')
print(text[:500])
doc.close()
"
```

### Colab: Upload your PDF

```python
import os
from google.colab import files
os.makedirs("data/protocols", exist_ok=True)
uploaded = files.upload()
for filename in uploaded:
    os.rename(filename, f"data/protocols/{filename}")
    print(f"Uploaded: data/protocols/{filename}")
```

---

## Step 5: Start the LLM Server (GPU)

**This step requires an A100 40GB GPU.** If on Colab, switch to A100 runtime now.

### Colab: Switch to A100

**Runtime → Change runtime type → A100 GPU**

```python
# Re-mount Drive (new VM after runtime change)
from google.colab import drive
drive.mount('/content/drive')

%cd qcagent
!pip install -e .   # Re-install (new VM)

# Verify GPU
!nvidia-smi
```

You should see `A100-SXM4-40GB` or similar.

### 5a. Start vLLM

**Recommended: use the setup script** (auto-detects GPU):

```bash
python setup_vllm.py --set-env
```

**What `setup_vllm.py` does automatically:**
- Kills stale GPU processes from previous crashed runs
- Checks free GPU memory before launching
- Applies T4/sm_75 flashinfer workaround
- Uses `--enforce-eager` and `VLLM_USE_V1=0` to avoid V1 engine core crashes
- Auto-detects and sets the outlines guided decoding backend (version-aware flag detection)
- Adds `--served-model-name` when loading from a local path (ensures client model ID matches)
- Dumps last 80 lines of stderr on failure for diagnosis

Options:
```bash
python setup_vllm.py --port 8001                    # custom port
python setup_vllm.py --max-model-len 8192            # reduce context if tight
python setup_vllm.py --model /path/to/local/model    # explicit model path
```

**Manual start:**

```bash
vllm serve Qwen/Qwen3-14B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 16384 \
    --enable-prefix-caching \
    --dtype auto \
    --gpu-memory-utilization 0.95 \
    --enforce-eager \
    --structured-outputs-config.backend outlines  # older vLLM: --guided-decoding-backend outlines
```

**In a Colab notebook** (background process):

```python
import subprocess

vllm_proc = subprocess.Popen(
    [
        "vllm", "serve", "Qwen/Qwen3-14B",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "16384",
        "--enable-prefix-caching",
        "--dtype", "auto",
        "--gpu-memory-utilization", "0.95",
        "--enforce-eager",
        "--structured-outputs-config.backend", "outlines",
    ],
    stdout=open("vllm_stdout.log", "w"),
    stderr=open("vllm_stderr.log", "w"),
)
print(f"vLLM starting (PID: {vllm_proc.pid})...")
```

### 5b. Wait for vLLM to be ready (2-3 minutes)

```python
import urllib.request, time

for i in range(60):
    try:
        resp = urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5)
        if resp.status == 200:
            print("vLLM is ready!")
            break
    except Exception:
        pass
    if i % 6 == 0:
        print(f"  Waiting... ({i * 5}s elapsed)")
    time.sleep(5)
else:
    print("vLLM failed to start. Check: tail -30 vllm_stderr.log")
```

Or from the terminal:
```bash
curl http://localhost:8000/v1/models
```

### 5c. Set environment variables

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export ADJUDICATOR_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local
export DEFAULT_MODEL=Qwen/Qwen3-14B
export ADJUDICATOR_MODEL=Qwen/Qwen3-14B
```

In a notebook:
```python
import os
os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"
os.environ["ADJUDICATOR_BASE_URL"] = "http://localhost:8000/v1"
os.environ["VLLM_API_KEY"] = "local"
os.environ["DEFAULT_MODEL"] = "Qwen/Qwen3-14B"
os.environ["ADJUDICATOR_MODEL"] = "Qwen/Qwen3-14B"
```

### 5d. Ollama alternative (for local testing without vLLM)

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:14b

export VLLM_BASE_URL=http://localhost:11434/v1
export ADJUDICATOR_BASE_URL=http://localhost:11434/v1
export DEFAULT_MODEL=qwen3:14b
export ADJUDICATOR_MODEL=qwen3:14b
```

**Note:** Ollama supports structured outputs but has not been fully validated for strict schema-constrained extraction. vLLM is the recommended setup.

---

## Step 6: Run the Pipeline (GPU)

**Requires vLLM running (Step 5).**

```bash
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/your_protocol.pdf \
    --protocol-id your_protocol_id
```

### Options

```bash
# With therapeutic area pack (synonym expansion + section priorities)
--ta oncology              # oncology | cardiovascular | respiratory | immunology | vaccines

# With data source override
--data-source cota         # cota | flatiron | optum_cdm | optum_ehr | marketscan | inalon | quest

# Reuse existing index (skip re-parsing and re-indexing)
--skip-indexing

# Custom output directory
--output-dir data/outputs
```

### Full example

```bash
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/my_protocol.pdf \
    --protocol-id MY_PROTOCOL_001 \
    --ta oncology \
    --data-source flatiron
```

### What happens during the run

```
Step 0: Model preflight      — verifies vLLM is reachable
Step 1: Parse protocol        — Multi-strategy pipeline (Docling, Docling+OCR, PyMuPDF heading,
         │                       PyMuPDF page-first). Best quality score wins (~30-120s)
         ├── Quality scoring  — grades each strategy: pass / warn / fail
         └── FAIL gate        — if all strategies fail, stops here with shell spec + QC warning
Step 2: Index protocol        — BGE-M3 embeds chunks, Qdrant indexes (~20-60s)
Step 3: Find concepts         — 14 concept finders run sequentially:
        ├── index_date
        ├── follow_up_end
        ├── primary_endpoint
        ├── eligibility_inclusion   (two-pass: inventory → detail)
        ├── eligibility_exclusion   (two-pass: inventory → detail)
        ├── study_period            (two-pass: regex mining → LLM classification)
        ├── censoring_rules
        ├── demographics
        ├── clinical_characteristics
        ├── biomarkers
        ├── lab_variables
        ├── treatment_variables
        ├── cohort_definitions      (Section B: arms, comparators, populations)
        └── source_data_prep        (Section D: protocol + source-known issues)
Step 4: Pre-review QC         — deterministic rule checks (12 rules)
Step 5: Save evidence packs   — JSON output
Step 6: Generate draft spec   — Row writers expand → JSON + HTML + Excel
```

**Expected runtime:** 10-30 minutes depending on GPU and protocol length.

---

## Step 7: Inspect the Output (CPU)

**No GPU needed.** You can disconnect the GPU runtime and review on CPU.

Output files appear in `data/outputs/`:

| File | Description |
|------|-------------|
| `{id}_evidence_packs.json` | Raw evidence with ranked candidates and QC results |
| `{id}_spec.json` | Structured ProgramSpec (machine-readable) |
| `{id}_spec.html` | Self-contained HTML preview with confidence badges |
| `{id}_spec.xlsx` | Excel workbook (10 visible sheets + hidden _Provenance sheet) |

### View from terminal

```bash
python -m json.tool data/outputs/MY_PROTOCOL_001_evidence_packs.json | head -80
```

### View HTML in a notebook

```python
from IPython.display import HTML, display

with open("data/outputs/MY_PROTOCOL_001_spec.html") as f:
    display(HTML(f.read()))
```

### Download from Colab

```python
from google.colab import files
files.download("data/outputs/MY_PROTOCOL_001_spec.xlsx")
files.download("data/outputs/MY_PROTOCOL_001_spec.html")
files.download("data/outputs/MY_PROTOCOL_001_evidence_packs.json")
```

### Save outputs to Drive (before Colab session ends)

```python
import shutil
shutil.copytree("data/outputs", "/content/drive/MyDrive/qcagent_outputs", dirs_exist_ok=True)
print("Outputs saved to Drive — safe from session timeout.")
```

### What's in the evidence packs

Each concept has ranked candidates with:
- `snippet` — exact quoted text from the protocol
- `chunk_id` — UUID linking back to the indexed chunk
- `page` — page number (or null)
- `section_title` — heading from the document
- `retrieval_score` / `rerank_score` — retrieval pipeline scores
- `llm_confidence` — model's self-reported confidence

### What's in the Excel workbook

- **Visible tabs (10):** 1.Cover, 2.QC Review, 3.Data Prep, 4.StudyPop, 5A.Demos, 5B.ClinChars, 5C.BioVars, 5D.LabVars, 6.TreatVars, 7.Outcomes
- **Hidden columns (J-L)** on variable tabs: Confidence, Source Page, Explicit
- **Hidden _Provenance sheet:** Row-level provenance summary with confidence-based coloring
- **No confidence coloring** on stakeholder-facing tabs

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `RuntimeError: Default model server not available` | vLLM isn't running. Start it (Step 5), then check `curl http://localhost:8000/v1/models` |
| `ImportError: No module named 'pydantic'` | Run `pip install -e ".[dev]"` |
| `ImportError: docling` | `pip install docling` — or ignore, pipeline auto-falls-back to PyMuPDF |
| `CUDA out of memory` | Reduce `--max-model-len` (see below) |
| `Free memory on device cuda:0 ... less than desired` | Stale GPU processes. Run `nvidia-smi` then `kill -9 <pid>` |
| `Engine core initialization failed. Failed core proc(s): {}` | V1 engine crash. Run `export VLLM_USE_V1=0` before starting vLLM |
| `Empty response content from model` | Auto-retries 2x. If persistent, check `tail -30 vllm_stderr.log` |
| Empty text from parsed PDF | Your PDF is scanned. OCR it first (Step 4) |
| Parse quality is FAIL | Pipeline stops and generates shell spec. Provide a cleaner PDF or OCR first |
| `No evidence candidates found` | Expected for some concepts. Check QC output |
| All extractions return 0 candidates / low confidence | Schema issue — vLLM's xgrammar mishandles `$ref`/`anyOf`. Use `--structured-outputs-config.backend outlines` or ensure `setup_vllm.py` is used |
| Adjudicator unavailable warning | Safe to ignore. Both endpoints point to same model |

### vLLM crashes on T4 (flashinfer)

If vLLM crashes on T4 with `DSLRuntimeError: ICE` or `NVVM Compilation Error`, this is a known
incompatibility between flashinfer's CUTLASS DSL and the T4's sm_75 architecture (vLLM >=0.17).

**Fix:** Uninstall flashinfer before starting vLLM:

```bash
pip uninstall -y flashinfer flashinfer-python
```

vLLM will automatically fall back to a compatible attention backend. The `setup_vllm.py` script
does this automatically when it detects a T4 or other sm_75 GPU.

### vLLM "Engine core initialization failed"

**1. Stale GPU processes from a previous run** — the most common cause.

```bash
nvidia-smi
kill -9 <pid>
```

`setup_vllm.py` does this automatically before every launch.

**2. V1 engine multiprocess crash:**

```bash
export VLLM_USE_V1=0
vllm serve Qwen/Qwen3-14B --enforce-eager ...
```

`setup_vllm.py` sets both of these automatically.

### vLLM out of memory

Qwen3-14B needs ~28 GB VRAM. If you're running low:

1. Reduce `--max-model-len 8192` if 16384 doesn't fit
2. Ensure no stale GPU processes are running (`nvidia-smi`, then `kill -9 <pid>`)
3. Use `--enforce-eager` to avoid torch.compile memory spikes
4. T4 (16GB) is too small for this model. Use A100 40GB minimum
