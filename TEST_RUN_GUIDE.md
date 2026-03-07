# Protocol Spec Assist — Test Run Guide

Step-by-step instructions to run the full pipeline end-to-end.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU (vLLM)** | 1x GPU with 18 GB VRAM (Qwen3-8B fp16) | 1x 24 GB GPU (e.g. RTX 3090/4090/A5000) |
| **GPU (adjudicator)** | Optional — 18 GB VRAM on second GPU or same GPU | 1x 24 GB GPU on port 8001 |
| **RAM** | 16 GB | 32 GB (BGE-M3 + reranker load into RAM before GPU transfer) |
| **Disk** | 40 GB free | 80 GB free |
| **CPU** | 4 cores | 8+ cores (Docling parsing is CPU-heavy) |

### Model sizes on disk

| Model | Size | Purpose |
|-------|------|---------|
| `Qwen/Qwen3-8B-Instruct` | ~16 GB (fp16 weights) | Default LLM extractor |
| `Qwen/Qwen3-30B-A3B-Instruct` | ~18 GB (Q4) / ~58 GB (fp16) | Adjudicator (optional for test) |
| `BAAI/bge-m3` | ~2.3 GB | Embeddings (dense + sparse) |
| `BAAI/bge-reranker-v2-m3` | ~1.1 GB (568M params) | Cross-encoder reranker |
| Docling models | ~1 GB (auto-downloaded) | PDF table/layout parsing |

**Total first-time download: ~20 GB** (without adjudicator) or **~40 GB** (with adjudicator at fp16).

### No GPU? Alternatives

- **Ollama** can serve Qwen3-8B quantized on CPU (slow but works). See Step 3b below.
- **vLLM CPU mode** exists but is experimental. See: https://docs.vllm.ai/en/latest/getting_started/installation/cpu/
- BGE-M3 and the reranker both run on CPU (slower, but functional).

---

## Step 0: Prerequisites

```bash
# Python 3.10+ required
python --version   # must be >= 3.10

# CUDA toolkit (for GPU)
nvidia-smi         # verify GPU visible
nvcc --version     # CUDA compiler

# Git LFS (for HuggingFace model downloads)
sudo apt install git-lfs   # Ubuntu/Debian
git lfs install
```

---

## Step 1: Install the package

```bash
cd /path/to/qcagent

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate

# Install protocol-spec-assist + all dependencies
pip install -e ".[dev]"

# Verify install
python -c "from protocol_spec_assist.schemas.evidence import EvidencePack; print('OK')"
python -m protocol_spec_assist.workflows.protocol_run --help
```

### Troubleshooting: Docling install

Docling depends on PyTorch. If you're on a CPU-only machine:

```bash
pip install docling --extra-index-url https://download.pytorch.org/whl/cpu
```

On macOS Intel:
```bash
pip install "docling[mac_intel]"
```

If Docling fails to install entirely, the pipeline will fall back to PyMuPDF automatically (less accurate table extraction, but functional).

---

## Step 2: Download models

### 2a. Embedding + Reranker models (auto-downloaded on first use)

These download automatically from HuggingFace when first loaded by FlagEmbedding.
To pre-download for offline use:

```bash
pip install huggingface-hub

# BGE-M3 embeddings (~2.3 GB)
huggingface-cli download BAAI/bge-m3

# BGE reranker (~1.1 GB)
huggingface-cli download BAAI/bge-reranker-v2-m3
```

### 2b. LLM model for vLLM

```bash
# Default extractor model (~16 GB)
huggingface-cli download Qwen/Qwen3-8B-Instruct

# (Optional) Adjudicator model — only needed if you want the confidence-router second pass
# This is an MoE model: 30B total params, only 3.3B active per token
# At fp16: ~58 GB on disk. At Q4: ~18 GB.
huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct
```

### 2c. Docling models (auto-downloaded on first use)

Docling downloads its layout/table models automatically on first PDF conversion.
No manual step needed. First parse will be slower (~30s extra).

---

## Step 3: Start the LLM server

### 3a. vLLM (recommended — requires CUDA GPU)

```bash
pip install vllm

# Start default model on port 8000
vllm serve Qwen/Qwen3-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --dtype auto

# Verify it's running (in another terminal)
curl http://localhost:8000/v1/models
```

Expected output: JSON listing `Qwen/Qwen3-8B-Instruct` as available.

**VRAM note:** Qwen3-8B at fp16 needs ~16 GB for weights + ~2-6 GB for KV cache depending on context length. A 24 GB GPU fits comfortably. A 16 GB GPU works with `--max-model-len 8192`.

```bash
# (Optional) Start adjudicator on port 8001 — needs a second GPU or enough VRAM
vllm serve Qwen/Qwen3-30B-A3B-Instruct \
    --host 0.0.0.0 \
    --port 8001 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --dtype auto
```

### 3b. Ollama alternative (no CUDA required, slower)

```bash
# Install Ollama: https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a Qwen3 model
ollama pull qwen3:8b

# Ollama serves on port 11434 by default
# Set env to redirect the pipeline:
export VLLM_BASE_URL=http://localhost:11434/v1
export DEFAULT_MODEL=qwen3:8b
```

**Warning:** Ollama does not support `response_format: json_schema` with `strict: true`.
Structured outputs will be less reliable. Expect occasional parse failures that trigger retries.

---

## Step 4: Get a test protocol PDF

Download a real observational study protocol from ClinicalTrials.gov:

```bash
mkdir -p data/protocols

# Option A: MS registry observational study (~50 pages)
curl -L -o data/protocols/NCT01013350.pdf \
    "https://cdn.clinicaltrials.gov/large-docs/50/NCT01013350/Prot_000.pdf"

# Option B: Cladribine tablets clinical study (~100+ pages)
curl -L -o data/protocols/NCT03961204.pdf \
    "https://cdn.clinicaltrials.gov/large-docs/04/NCT03961204/Prot_000.pdf"

# Option C: Surgical pain observational study
curl -L -o data/protocols/NCT06417528.pdf \
    "https://cdn.clinicaltrials.gov/large-docs/28/NCT06417528/Prot_SAP_000.pdf"

# Verify download
ls -lh data/protocols/
file data/protocols/*.pdf   # should say "PDF document"
```

---

## Step 5: Run the pipeline

```bash
# Make sure vLLM is running (Step 3) before this

# Basic run — no TA pack
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/NCT01013350.pdf \
    --protocol-id NCT01013350

# With oncology TA pack (synonym expansion + section priorities)
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/NCT01013350.pdf \
    --protocol-id NCT01013350 \
    --ta oncology

# With cardiovascular TA pack
python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/NCT03961204.pdf \
    --protocol-id NCT03961204 \
    --ta cardiovascular
```

### What happens during the run

```
Step 0: Model preflight      — verifies vLLM is reachable and model is loaded
Step 1: Parse protocol        — Docling extracts sections, tables, pages (~30-120s)
Step 2: Index protocol        — BGE-M3 embeds chunks, Qdrant indexes (~20-60s)
Step 3: Find concepts         — 3 concept finders run sequentially:
        ├── index_date        — hybrid retrieval → rerank → LLM extract
        ├── follow_up_end     — same workflow, different queries/schema
        └── primary_endpoint  — same workflow, different queries/schema
Step 4: Pre-review QC         — deterministic rule checks
Step 5: Save evidence packs   — JSON output to data/outputs/
```

### Expected output

```
[Parser] Docling loaded. Parsing NCT01013350...
[Index] Indexing 47 chunks for NCT01013350...
[Index] Done. 47 points indexed.
[IndexDateFinder] Done. 3 candidates | confidence=0.82 | contradictions=False
[FollowUpEndFinder] Done. 2 candidates | confidence=0.75 | contradictions=False
[PrimaryEndpointFinder] Done. 4 candidates | confidence=0.88 | contradictions=False
=== QC Summary: 0 errors | 1 warnings | 0 info ===
[!] [QC-002] (pre_review) follow_up_end: Low retrieval signal for follow_up_end.
[Workflow] Evidence packs saved: data/outputs/NCT01013350_evidence_packs.json
```

---

## Step 6: Inspect the output

```bash
# View the evidence packs JSON
python -m json.tool data/outputs/NCT01013350_evidence_packs.json | head -80

# Or use jq if installed
jq '.evidence_packs.index_date.candidates[0]' data/outputs/NCT01013350_evidence_packs.json
```

The output contains:
- `evidence_packs`: one EvidencePack per concept, each with ranked candidates
- `qc_results`: list of QC findings (errors, warnings, info)

Each candidate has:
- `snippet`: exact quoted text from the protocol
- `chunk_id`: link back to indexed chunk
- `page`: page number (or null)
- `section_title`: section heading from the document
- `retrieval_score` / `rerank_score`: retrieval pipeline scores
- `llm_confidence`: model's confidence in this candidate

---

## Step 7: Run unit tests (no GPU needed)

```bash
# All 56 tests — fast, no external dependencies
pytest tests/ -v

# Just the v0.2 fix tests
pytest tests/test_v02_fixes.py -v
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: Default model server not available` | vLLM not running or wrong port | Start vLLM (Step 3), check `curl http://localhost:8000/v1/models` |
| `ImportError: FlagEmbedding not installed` | Missing embedding dependency | `pip install FlagEmbedding` |
| `ImportError: docling` | Docling not installed | `pip install docling` — pipeline will auto-fallback to PyMuPDF |
| `CUDA out of memory` | Not enough VRAM for Qwen3-8B | Use `--max-model-len 8192` or use Ollama with quantized model |
| `Empty response content from model` | Model returned blank | Automatic retry (up to 2x). If persistent, check vLLM logs |
| `BackendUnavailable` on pip install | Wrong setuptools version | `pip install --upgrade setuptools wheel` then `pip install -e .` |
| Slow first run | Downloading BGE-M3 + Docling models | One-time download. Subsequent runs are faster |
| `No evidence candidates found` | Protocol doesn't discuss this concept | Expected for some protocols. Check QC output. |

---

## Environment Variables (optional overrides)

```bash
export VLLM_BASE_URL=http://localhost:8000/v1         # Default model endpoint
export ADJUDICATOR_BASE_URL=http://localhost:8001/v1   # Adjudicator endpoint
export VLLM_API_KEY=local                              # API key (default: "local")
export DEFAULT_MODEL=Qwen/Qwen3-8B-Instruct           # Model name on vLLM
export ADJUDICATOR_MODEL=Qwen/Qwen3-30B-A3B-Instruct  # Adjudicator model name
```

---

## Quick smoke test (no GPU, no vLLM)

If you just want to verify the code works without running the full pipeline:

```bash
# 1. Install
pip install -e ".[dev]" --no-deps
pip install pydantic pyyaml pytest

# 2. Run unit tests
pytest tests/ -v

# 3. Verify imports
python -c "
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.qc.rules import run_all_qc, qc_quote_in_chunk
from protocol_spec_assist.ta_packs.loader import load_ta_pack
from protocol_spec_assist.serving.model_client import LocalModelClient, ExtractionResult
print('All imports OK')
"

# 4. Verify CLI
python -m protocol_spec_assist.workflows.protocol_run --help
```
