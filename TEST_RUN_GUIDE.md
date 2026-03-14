# Protocol Spec Assist — Test Run Guide

Step-by-step instructions to run the full pipeline end-to-end.
Works on **Google Colab** or **any Linux machine with a GPU**.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Environment Setup](#environment-setup)
   - [Google Colab](#option-a-google-colab)
   - [Local Linux / Cloud VM](#option-b-local-linux--cloud-vm)
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

| Component | Minimum |
|-----------|---------|
| **GPU** | 1x A100 40 GB |
| **RAM** | 32 GB |
| **Disk** | 35 GB free (+ Google Drive for Colab) |
| **CPU** | 8 cores |

### Model sizes on disk

| Model | Size | Purpose |
|-------|------|---------|
| `Qwen/Qwen3-14B` | ~28 GB | Base extractor + adjudicator |
| `BAAI/bge-m3` | ~2.3 GB | Embeddings (dense + sparse) |
| `BAAI/bge-reranker-v2-m3` | ~2.3 GB | Reranker |
| Docling models | ~360 MB (auto-downloaded) | PDF table/layout parsing |

**First-time download: ~33 GB.**

---

## Environment Setup

Pick your environment below. After setup, all remaining steps (1–7) are **identical**.

---

### Option A: Google Colab

**Cost:** Requires Colab Pro/Pro+ with A100 access. Free tier T4 does **not** have enough VRAM.

**Recommended GPU: A100 40GB** (~8.3 units/hr, Qwen3-14B fits comfortably).

#### Compute Unit Budget (150 units = ~18 hours on A100)

| Task | Runtime | GPU | Units |
|------|---------|-----|-------|
| Download models to Drive | CPU (free) | None | **0** |
| Debug/develop pipeline code | CPU (free) | None | **0** |
| Pipeline dev & testing | A100 | ~6 hrs | **~50** |
| Extraction runs | A100 | ~10 hrs | **~85** |
| Buffer | — | — | **~15** |

**Key rule: Do ALL setup on FREE CPU runtime. Switch to A100 only for inference.**

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

#### A3. Download models to Drive (run on FREE CPU runtime — costs 0 units)

```python
# This downloads ~33GB to your Drive. Takes 10-20 min. Costs ZERO compute units.
!python colab_setup.py --download-models
```

This saves models to `/content/drive/MyDrive/qcagent_models/`. They persist across sessions — **you only download once**.

#### A4. Switch to A100 and start inference

**Runtime → Change runtime type → A100 GPU** (starts burning compute units now).

```python
# Re-mount Drive (new VM after runtime change)
from google.colab import drive
drive.mount('/content/drive')

%cd qcagent

# Verify GPU
!nvidia-smi
```

You should see `A100-SXM4-40GB` or similar.

#### A5. Upload your protocol PDF

```python
import os

# Option 1: Upload from local machine
from google.colab import files
os.makedirs("data/protocols", exist_ok=True)
uploaded = files.upload()
for filename in uploaded:
    os.rename(filename, f"data/protocols/{filename}")
    print(f"Uploaded: data/protocols/{filename}")

# Option 2: Copy from Google Drive
# import shutil
# shutil.copy("/content/drive/MyDrive/my_protocol.pdf", "data/protocols/")
```

#### A6. Save outputs to Drive (before session ends)

```python
import shutil
shutil.copytree("data/outputs", "/content/drive/MyDrive/qcagent_outputs", dirs_exist_ok=True)
print("Outputs saved to Drive — safe from session timeout.")
```

#### A7. Notes for Colab

- **Disconnect when idle** — Colab keeps billing even when you walk away.
- Models on Drive load instantly on subsequent sessions — no re-download.
- A100 40GB fits Qwen3-14B in full FP16 (~28GB VRAM). No quantization needed.

**Now continue to [Step 1: Install the Package](#step-1-install-the-package).** All commands below run in Colab cells with a `!` prefix (e.g., `!pip install ...`).

---

### Option B: Local Linux / Cloud VM

#### B1. Requirements

- Python 3.10 or 3.11
- NVIDIA A100 40GB (or any GPU with >= 30GB VRAM)
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
# Base extractor + adjudicator (~28 GB)
huggingface-cli download Qwen/Qwen3-14B
```

**Or use the Colab helper** (downloads to Google Drive automatically):
```bash
python colab_setup.py --download-models
```

### 2c. Docling models

```bash
python -c "from docling.utils.model_downloader import download_models; download_models()"
```

Without this, Docling auto-downloads on first PDF parse (~30s extra).

---

## Step 3: Start the LLM Server

### 3a. Start vLLM

**Recommended: use the setup script** (auto-detects GPU):

```bash
python setup_vllm.py --set-env
```

**What `setup_vllm.py` does automatically:**
- Kills stale GPU processes from previous crashed runs
- Checks free GPU memory before launching
- Applies T4/sm_75 flashinfer workaround
- Uses `--enforce-eager` and `VLLM_USE_V1=0` to avoid V1 engine core crashes
- Dumps last 80 lines of stderr on failure for diagnosis

Options:
```bash
python setup_vllm.py --port 8001                                    # custom port
python setup_vllm.py --max-model-len 8192                          # reduce context if tight
python setup_vllm.py --model /path/to/local/model                  # explicit model path
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
    --enforce-eager
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
    ],
    stdout=open("vllm_stdout.log", "w"),
    stderr=open("vllm_stderr.log", "w"),
)
print(f"vLLM starting (PID: {vllm_proc.pid})...")
```

### 3b. Wait for vLLM to be ready (2-3 minutes)

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

### 3c. Set environment variables

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

### 3d. Ollama alternative (for local testing without vLLM)

For quick local testing without setting up vLLM:

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
pytest tests/ -v
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
| `CUDA out of memory` | Reduce `--max-model-len` (see below) |
| `Free memory on device cuda:0 ... less than desired` | Stale GPU processes. Run `nvidia-smi` then `kill -9 <pid>` |
| `Engine core initialization failed. Failed core proc(s): {}` | V1 engine crash. Run `export VLLM_USE_V1=0` before starting vLLM |
| `Empty response content from model` | Auto-retries 2x. If persistent, check `tail -30 vllm_stderr.log` |
| Empty text from parsed PDF | Your PDF is scanned. OCR it first (Step 4) |
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

### Troubleshooting: vLLM "Engine core initialization failed"

If vLLM crashes with `RuntimeError: Engine core initialization failed. Failed core proc(s): {}`:

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

### Troubleshooting: vLLM out of memory

Qwen3-14B needs ~28 GB VRAM. If you're running low:

1. Reduce `--max-model-len 8192` if 16384 doesn't fit
2. Ensure no stale GPU processes are running (`nvidia-smi`, then `kill -9 <pid>`)
3. Use `--enforce-eager` to avoid torch.compile memory spikes
4. T4 (16GB) is too small for this model. Use A100 40GB minimum

---

## Environment Variables

All optional. Sensible defaults are built in.

```bash
# LLM endpoints
export VLLM_BASE_URL=http://localhost:8000/v1
export ADJUDICATOR_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local
export DEFAULT_MODEL=Qwen/Qwen3-14B
export ADJUDICATOR_MODEL=Qwen/Qwen3-14B

# Retrieval (auto-detected, override if needed)
export RETRIEVAL_DEVICE=cpu                             # Force CPU for embeddings
export RETRIEVAL_FP16=false                             # Disable fp16 (required for CPU)
```
