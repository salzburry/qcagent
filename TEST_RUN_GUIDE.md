# Protocol Spec Assist — Test Run Guide

Step-by-step instructions to run the full pipeline end-to-end.
Works on **Google Colab**, **Databricks**, or **any Linux machine with a GPU**.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Environment Setup](#environment-setup) — pick your environment
   - [Google Colab](#option-a-google-colab)
   - [Databricks](#option-b-databricks)
   - [Local Linux / Cloud VM](#option-c-local-linux--cloud-vm)
3. [Install the Package](#step-1-install-the-package)
4. [Download Models](#step-2-download-models)
5. [Start the LLM Server](#step-3-start-the-llm-server)
6. [Prepare Your Protocol PDF](#step-4-prepare-your-protocol-pdf)
7. [Run the Pipeline](#step-5-run-the-pipeline)
8. [Inspect Output](#step-6-inspect-the-output)
9. [Run Unit Tests](#step-7-run-unit-tests)
10. [Troubleshooting](#troubleshooting)
11. [Environment Variables](#environment-variables)

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU (vLLM)** | 1x H100 80 GB | 1x H100 80 GB |
| **RAM** | 32 GB | 64 GB |
| **Disk** | 100 GB free | 150 GB free |
| **CPU** | 8 cores | 16+ cores |

### Model sizes on disk

| Model | Size | Purpose | Required? |
|-------|------|---------|-----------|
| `Qwen/Qwen3-235B-A22B` | ~60 GB | LLM extractor + adjudicator (MoE) | Yes |
| `BAAI/bge-m3` | ~2.3 GB | Embeddings (dense + sparse) | Yes |
| `BAAI/bge-reranker-v2-m3` | ~2.3 GB | Reranker | Yes |
| Docling models | ~360 MB (auto-downloaded) | PDF table/layout parsing | Yes |

**Total first-time download: ~65 GB.**

---

## Environment Setup

Pick your environment below. After setup, all remaining steps (1–7) are **identical** regardless of environment.

---

### Option A: Google Colab

**Cost:** Requires Colab Pro+ or Enterprise with A100/H100 access. Free tier T4 will not fit this model.

#### A1. Create a new notebook

Go to [colab.research.google.com](https://colab.research.google.com) → **New Notebook**.

#### A2. Enable GPU

**Runtime → Change runtime type → A100 or H100 GPU** (Pro+/Enterprise).

Verify in a cell:
```python
!nvidia-smi
```
You should see an A100 (40/80 GB) or H100 (80 GB).

#### A3. Clone the repo

```python
!git clone https://github.com/salzburry/qcagent.git
%cd qcagent
```

#### A4. Upload your protocol PDF

```python
import os
from google.colab import files

os.makedirs("data/protocols", exist_ok=True)
uploaded = files.upload()  # file picker appears

for filename in uploaded:
    os.rename(filename, f"data/protocols/{filename}")
    print(f"Uploaded: data/protocols/{filename}")
```

#### A5. Notes for Colab

- Colab Pro+ sessions timeout after idle. Max runtime varies by plan.
- Qwen3-235B-A22B requires ~60 GB VRAM. A100 80GB or H100 80GB recommended.
- All files are lost when the session ends. Download your outputs before disconnecting (see Step 6).

**Now continue to [Step 1: Install the Package](#step-1-install-the-package).** All commands below run in Colab cells with a `!` prefix (e.g., `!pip install ...`).

---

### Option B: Databricks

#### B1. Cluster requirements

Create or use a cluster with:
- **Runtime:** Databricks Runtime 14.x+ ML (includes Python 3.10+, PyTorch, CUDA)
- **Node type:** GPU instance with 80 GB VRAM (e.g., `Standard_NC24ads_A100_v4` on Azure, `p4d.24xlarge` on AWS, `a2-highgpu-1g` on GCP)
- **Single node** is fine for testing

#### B2. Clone the repo

In a notebook cell:
```python
%sh git clone https://github.com/salzburry/qcagent.git /tmp/qcagent
```

Or use Databricks Repos: **Repos → Add Repo → paste the GitHub URL**.

#### B3. Set working directory

```python
%cd /tmp/qcagent
# Or if using Repos:
# %cd /Workspace/Repos/<your-username>/qcagent
```

#### B4. Upload your protocol PDF

Use the Databricks UI: **Data → DBFS → Upload** → place in a known path.

Then copy it:
```python
%sh mkdir -p data/protocols
%sh cp /dbfs/FileStore/your_protocol.pdf data/protocols/
```

Or upload directly from a notebook cell:
```python
import shutil, os
os.makedirs("data/protocols", exist_ok=True)
# If using DBFS:
shutil.copy("/dbfs/FileStore/your_protocol.pdf", "data/protocols/your_protocol.pdf")
```

#### B5. Notes for Databricks

- Databricks ML Runtime comes with PyTorch + CUDA pre-installed — vLLM installs faster.
- If your cluster has multiple GPUs, vLLM will auto-detect and use them.
- For production, consider serving the model via Databricks Model Serving instead of vLLM in a notebook.
- Cluster auto-terminates after idle timeout — download outputs or save to DBFS.

**Now continue to [Step 1: Install the Package](#step-1-install-the-package).** All commands below run in notebook cells with `%sh` prefix.

---

### Option C: Local Linux / Cloud VM

#### C1. Requirements

- Python 3.10 or 3.11
- NVIDIA H100 or A100 with 80 GB VRAM
- CUDA drivers installed (`nvidia-smi` should work)

#### C2. Clone and enter the repo

```bash
git clone https://github.com/salzburry/qcagent.git
cd qcagent
```

#### C3. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

#### C4. Place your protocol PDF

```bash
mkdir -p data/protocols
cp /path/to/your_protocol.pdf data/protocols/
```

**Now continue to [Step 1: Install the Package](#step-1-install-the-package).**

---

## Step 1: Install the Package

```bash
# Install the project + dependencies
pip install -e ".[dev]"

# Install vLLM (the LLM server)
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

## Step 2: Download Models

### 2a. Embedding + Reranker (auto-download on first use)

These download automatically from HuggingFace when first loaded.
To pre-download (recommended to avoid timeout during pipeline run):

```bash
pip install huggingface-hub

# BGE-M3 embeddings (~2.3 GB)
huggingface-cli download BAAI/bge-m3

# BGE reranker (~2.3 GB)
huggingface-cli download BAAI/bge-reranker-v2-m3
```

### 2b. LLM model

```bash
# MoE extractor + adjudicator (~60 GB)
huggingface-cli download Qwen/Qwen3-235B-A22B
```

Single model handles both extraction and adjudication — no separate server needed.

### 2c. Docling models

```bash
python -c "from docling.utils.model_downloader import download_models; download_models()"
```

Without this, Docling auto-downloads on first PDF parse (~30s extra).

---

## Step 3: Start the LLM Server

### 3a. Pick the right vLLM settings for your GPU

| GPU | VRAM | max-model-len | Notes |
|-----|------|---------------|-------|
| H100 | 80 GB | `32768` | Full context, fastest inference |
| A100 | 80 GB | `32768` | Full context, slightly slower |
| A100 | 40 GB | `16384` | Tight — may need `--gpu-memory-utilization 0.95` |

### 3b. Start vLLM

**Recommended: use the setup script** (handles T4 compatibility automatically):

```bash
python setup_vllm.py
```

This auto-detects your GPU, applies workarounds, picks `--max-model-len`, and waits for the server to be ready.

**What `setup_vllm.py` does automatically:**
- Kills stale GPU processes from previous crashed runs (prevents "not enough free memory" errors)
- Checks free GPU memory before launching and aborts early with a clear message if insufficient
- Applies T4/sm_75 flashinfer workaround (see [T4 troubleshooting](#troubleshooting-vllm-crashes-on-t4-flashinfer))
- Uses `--enforce-eager` and `VLLM_USE_V1=0` to avoid V1 engine core crashes
- Dumps last 80 lines of stderr on failure for diagnosis

Options:
```bash
python setup_vllm.py --port 8001                                    # custom port
python setup_vllm.py --max-model-len 16384                         # reduce context if needed
python setup_vllm.py --set-env                                      # auto-set env vars
```

**Manual start** (separate terminal or notebook background process):

```bash
vllm serve Qwen/Qwen3-235B-A22B \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --dtype auto \
    --gpu-memory-utilization 0.90
```

Adjust `--max-model-len` per the table above.

**In a notebook** (Colab or Databricks), start as a background process:

```python
import subprocess

vllm_proc = subprocess.Popen(
    [
        "vllm", "serve", "Qwen/Qwen3-235B-A22B",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--max-model-len", "32768",       # H100 can handle full context
        "--enable-prefix-caching",
        "--dtype", "auto",
        "--gpu-memory-utilization", "0.90",
    ],
    stdout=open("vllm_stdout.log", "w"),
    stderr=open("vllm_stderr.log", "w"),
)
print(f"vLLM starting (PID: {vllm_proc.pid})...")
```

### 3c. Wait for vLLM to be ready (2-3 minutes)

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
# Poll until ready
curl http://localhost:8000/v1/models
```

### 3d. Set environment variables

```bash
export VLLM_BASE_URL=http://localhost:8000/v1
export ADJUDICATOR_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local
export DEFAULT_MODEL=Qwen/Qwen3-235B-A22B
export ADJUDICATOR_MODEL=Qwen/Qwen3-235B-A22B
```

In a notebook:
```python
import os
os.environ["VLLM_BASE_URL"] = "http://localhost:8000/v1"
os.environ["ADJUDICATOR_BASE_URL"] = "http://localhost:8000/v1"
os.environ["VLLM_API_KEY"] = "local"
os.environ["DEFAULT_MODEL"] = "Qwen/Qwen3-235B-A22B"
os.environ["ADJUDICATOR_MODEL"] = "Qwen/Qwen3-235B-A22B"
```

### 3e. Ollama alternative (not recommended for 235B model)

The Qwen3-235B-A22B model is too large for Ollama on most setups. If you need a lighter
alternative for testing without an H100, consider running a smaller model via Ollama:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:32b

export VLLM_BASE_URL=http://localhost:11434/v1
export ADJUDICATOR_BASE_URL=http://localhost:11434/v1
export DEFAULT_MODEL=qwen3:32b
export ADJUDICATOR_MODEL=qwen3:32b
```

**Note:** Ollama supports structured outputs but has not been fully validated for strict schema-constrained extraction. vLLM with the full 235B MoE model is the recommended setup.

---

## Step 4: Prepare Your Protocol PDF

### If your PDF is already text-based (digital)

Just place it in `data/protocols/`:
```bash
cp your_protocol.pdf data/protocols/
```

### If your PDF is scanned (images only)

You must OCR it first so the parser can extract text. Options:

- **Adobe Acrobat:** File → Save As Other → Searchable PDF (recommended)
- **ocrmypdf (free, command-line):**
  ```bash
  pip install ocrmypdf
  # May also need: apt install tesseract-ocr (Linux) or brew install tesseract (Mac)
  ocrmypdf scanned_protocol.pdf data/protocols/protocol_ocr.pdf
  ```

After OCR, verify text was extracted:
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

If the output shows text, you're good. If it's empty or garbled, the OCR quality is too low.

---

## Step 5: Run the Pipeline

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

### In a notebook

```python
!python -m protocol_spec_assist.workflows.protocol_run \
    data/protocols/my_protocol.pdf \
    --protocol-id MY_PROTOCOL_001 \
    --ta oncology
```

### What happens during the run

```
Step 0: Model preflight      — verifies vLLM is reachable
Step 1: Parse protocol        — Docling extracts sections, tables, pages (~30-120s)
Step 2: Index protocol        — BGE-M3 embeds chunks, Qdrant indexes (~20-60s)
Step 3: Find concepts         — 12 concept finders run sequentially:
        ├── index_date
        ├── follow_up_end
        ├── primary_endpoint
        ├── eligibility_inclusion
        ├── eligibility_exclusion
        ├── study_period
        ├── censoring_rules
        ├── demographics
        ├── clinical_characteristics
        ├── biomarkers
        ├── lab_variables
        └── treatment_variables
Step 4: Pre-review QC         — deterministic rule checks
Step 5: Save evidence packs   — JSON output
Step 6: Generate draft spec   — JSON + HTML + Excel
```

**Expected runtime:** 10-30 minutes depending on GPU and protocol length.

---

## Step 6: Inspect the Output

Output files appear in `data/outputs/`:

| File | Description |
|------|-------------|
| `{id}_evidence_packs.json` | Raw evidence with ranked candidates and QC results |
| `{id}_spec.json` | Structured ProgramSpec (machine-readable) |
| `{id}_spec.html` | Self-contained HTML preview with confidence badges |
| `{id}_spec.xlsx` | Excel workbook (10 sheets matching program spec template) |

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

### Save to DBFS from Databricks

```python
import shutil
shutil.copytree("data/outputs", "/dbfs/FileStore/protocol_outputs", dirs_exist_ok=True)
```

### What's in the evidence packs

Each concept has ranked candidates with:
- `snippet` — exact quoted text from the protocol
- `chunk_id` — UUID linking back to the indexed chunk
- `page` — page number (or null)
- `section_title` — heading from the document
- `retrieval_score` / `rerank_score` — retrieval pipeline scores
- `llm_confidence` — model's self-reported confidence

---

## Step 7: Run Unit Tests

No GPU, no vLLM, no models needed. Just verifies the code logic is correct.

```bash
# All 122 tests
pytest tests/ -v

# Individual test files
pytest tests/test_qc_rules.py -v        # QC engine
pytest tests/test_evidence_pack.py -v    # Evidence pack schema
pytest tests/test_spec_output.py -v      # Spec generation + Excel
pytest tests/test_ta_loader.py -v        # TA pack loading
pytest tests/test_v02_fixes.py -v        # Regression tests
```

### Quick smoke test (no GPU, no vLLM)

Verify everything imports and the CLI works:

```bash
pip install -e ".[dev]"

pytest tests/ -v

python -c "
from protocol_spec_assist.schemas.evidence import EvidencePack, EvidenceCandidate
from protocol_spec_assist.qc.rules import run_all_qc, qc_quote_in_chunk
from protocol_spec_assist.ta_packs.loader import load_ta_pack
from protocol_spec_assist.serving.model_client import LocalModelClient, ExtractionResult
print('All imports OK')
"

python -m protocol_spec_assist.workflows.protocol_run --help
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `RuntimeError: Default model server not available` | vLLM isn't running. Start it (Step 3), then check `curl http://localhost:8000/v1/models` |
| `ImportError: No module named 'pydantic'` | Run `pip install -e ".[dev]"` |
| `ImportError: docling` | `pip install docling` — or ignore, pipeline auto-falls-back to PyMuPDF |
| `CUDA out of memory` | Reduce `--max-model-len` or use quantized model (see below) |
| `Free memory on device cuda:0 ... less than desired` | Stale GPU processes from a previous run. `setup_vllm.py` auto-kills these; for manual runs: `nvidia-smi` then `kill -9 <pid>` |
| `Engine core initialization failed. Failed core proc(s): {}` | V1 engine crash. `setup_vllm.py` avoids this with `VLLM_USE_V1=0`. For manual runs: `export VLLM_USE_V1=0` before starting vLLM |
| `Empty response content from model` | Auto-retries 2x. If persistent, check `tail -30 vllm_stderr.log` |
| Empty text from parsed PDF | Your PDF is scanned. OCR it first (Step 4) |
| Slow first run | One-time model downloads. Subsequent runs are faster |
| `No evidence candidates found` | Expected for some concepts. Check QC output |
| Adjudicator unavailable warning | Safe to ignore. Both endpoints point to same model |

### Troubleshooting: vLLM crashes on T4 (flashinfer)

If vLLM crashes on T4 with `DSLRuntimeError: ICE` or `NVVM Compilation Error`, this is a known
incompatibility between flashinfer's CUTLASS DSL and the T4's sm_75 architecture (vLLM >=0.17).

**Fix:** Uninstall flashinfer before starting vLLM:

```bash
pip uninstall -y flashinfer flashinfer-python
```

vLLM will automatically fall back to a compatible attention backend. The `setup_vllm.py` script
does this automatically when it detects a T4 or other sm_75 GPU.

### Troubleshooting: vLLM "Engine core initialization failed" (H100/A100/any GPU)

If vLLM crashes with `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`,
this is typically caused by one of two things:

**1. Stale GPU processes from a previous run** — the most common cause.

Check with `nvidia-smi`. If you see python/vllm processes using GPU memory, kill them:
```bash
# See what's on the GPU
nvidia-smi

# Kill stale processes
kill -9 <pid>
```

`setup_vllm.py` does this automatically before every launch.

**2. V1 engine multiprocess crash** — the vLLM V1 engine spawns a subprocess that can die
before registering, producing the unhelpful `{}` error.

Fix: disable V1 and use eager mode:
```bash
export VLLM_USE_V1=0
vllm serve Qwen/Qwen3-235B-A22B --enforce-eager ...
```

`setup_vllm.py` sets both of these automatically.

### Troubleshooting: vLLM out of memory

Qwen3-235B-A22B requires ~60 GB VRAM. If you're running low:

1. Reduce `--gpu-memory-utilization 0.95` and `--max-model-len 16384`
2. Ensure no stale GPU processes are running (`nvidia-smi`, then `kill -9 <pid>`)
3. If using A100 40GB, the model will not fit — use an 80GB GPU

---

## Environment Variables

All optional. Sensible defaults are built in.

```bash
# LLM endpoints
export VLLM_BASE_URL=http://localhost:8000/v1          # Default model
export ADJUDICATOR_BASE_URL=http://localhost:8000/v1   # Same server (single-model setup)
export VLLM_API_KEY=local                               # API key
export DEFAULT_MODEL=Qwen/Qwen3-235B-A22B              # MoE extractor + adjudicator
export ADJUDICATOR_MODEL=Qwen/Qwen3-235B-A22B          # Same model

# Retrieval (auto-detected, override if needed)
export RETRIEVAL_DEVICE=cpu                             # Force CPU for embeddings
export RETRIEVAL_FP16=false                             # Disable fp16 (required for CPU)
```

The Qwen3-235B-A22B model is powerful enough for both extraction and adjudication.
No dual-model setup is needed.
