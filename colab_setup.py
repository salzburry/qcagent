#!/usr/bin/env python3
"""
colab_setup.py — One-click Colab setup for Protocol Spec Assist.

Handles the full Colab workflow:
  1. Mount Google Drive (models persist across sessions — download once)
  2. Download models to Drive (not the 100GB VM disk)
  3. Start vLLM with Qwen3-14B
  4. Set environment variables

Usage in a Colab cell:
    # First time (downloads models — takes 10-20 min):
    !python colab_setup.py --download-models

    # Subsequent sessions (models already on Drive):
    !python colab_setup.py

Compute unit budget (150 units):
    GPU       | Units/hr | Hours available
    A100 40GB | ~8.3     | ~18 hrs  ← recommended

Strategy: Do all setup/download on FREE CPU runtime. Switch to A100 only for inference.
"""

import argparse
import os
import shutil
import subprocess
import sys


# ── Google Drive paths ────────────────────────────────────────────────────────

DRIVE_MOUNT = "/content/drive"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_MODELS = "/content/drive/MyDrive/qcagent_models"
DRIVE_OUTPUTS = "/content/drive/MyDrive/qcagent_outputs"
DRIVE_HF_CACHE = "/content/drive/MyDrive/qcagent_models/.hf_cache"

# ── Model definitions ────────────────────────────────────────────────────────

MODELS = [
    {"repo": "Qwen/Qwen3-14B", "size_gb": 28, "purpose": "Base extractor + adjudicator"},
    {"repo": "BAAI/bge-m3", "size_gb": 2.3, "purpose": "Embeddings"},
    {"repo": "BAAI/bge-reranker-v2-m3", "size_gb": 2.3, "purpose": "Reranker"},
]


def is_colab():
    """Check if running in Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive():
    """Mount Google Drive at /content/drive. Must be called from Colab."""
    if not is_colab():
        print("[colab_setup] Not running in Colab — skipping Drive mount.")
        return False

    if os.path.ismount(DRIVE_MOUNT) or os.path.exists(DRIVE_ROOT):
        print(f"[colab_setup] Google Drive already mounted at {DRIVE_MOUNT}")
        return True

    print("[colab_setup] Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount(DRIVE_MOUNT)
        print(f"[colab_setup] Drive mounted. Your storage is at: {DRIVE_ROOT}")
        return True
    except Exception as e:
        print(f"[colab_setup] ERROR mounting Drive: {e}")
        print("[colab_setup] Follow the authentication prompt in Colab.")
        return False


def setup_directories():
    """Create model and output directories on Google Drive."""
    for path in [DRIVE_MODELS, DRIVE_OUTPUTS]:
        os.makedirs(path, exist_ok=True)
        print(f"[colab_setup] Directory ready: {path}")


def _model_already_downloaded(model_dir):
    """Check if a model directory contains weight files (recursive).

    Looks for common model weight extensions anywhere in the directory tree.
    This handles both flat downloads (--local-dir) and nested structures.
    """
    if not os.path.isdir(model_dir):
        return False
    weight_extensions = (".safetensors", ".bin", ".model", ".onnx", ".gguf")
    for root, _dirs, files in os.walk(model_dir):
        for f in files:
            if f.endswith(weight_extensions):
                return True
    return False


def get_model_root():
    """Return the model storage root — Drive if available, else local fallback."""
    if os.path.isdir(DRIVE_MODELS):
        return DRIVE_MODELS
    # Non-Colab or Drive not mounted: use local path
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "models")
    os.makedirs(local, exist_ok=True)
    return local


def download_models():
    """Download models to Google Drive (persists across sessions)."""
    model_root = get_model_root()
    total_gb = sum(m["size_gb"] for m in MODELS)
    print(f"\n[colab_setup] Downloading {len(MODELS)} models (~{total_gb:.0f} GB total)")
    print(f"[colab_setup] Destination: {model_root}")
    if model_root == DRIVE_MODELS:
        print(f"[colab_setup] Models persist on Drive — only need to download once.\n")
    else:
        print(f"[colab_setup] Using local storage (Drive not available).\n")

    for model in MODELS:
        repo = model["repo"]
        model_dir = os.path.join(model_root, repo.replace("/", "--"))

        # Check if already downloaded (recursive check for weight files)
        if _model_already_downloaded(model_dir):
            print(f"  [skip] {repo} — already downloaded ({model['purpose']})")
            continue

        print(f"  [download] {repo} (~{model['size_gb']} GB) — {model['purpose']}")
        cmd = [
            sys.executable, "-m", "huggingface_hub", "download",
            repo,
            "--local-dir", model_dir,
        ]
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            # Fallback to huggingface-cli
            cmd2 = ["huggingface-cli", "download", repo, "--local-dir", model_dir]
            subprocess.run(cmd2, capture_output=False)

    print(f"\n[colab_setup] All models downloaded to {model_root}")
    if model_root == DRIVE_MODELS:
        print("[colab_setup] They will persist across Colab sessions.")


def redirect_hf_cache_to_drive():
    """Redirect ALL HuggingFace downloads to Google Drive.

    Sets HF_HOME and HF_HUB_CACHE env vars so that huggingface_hub writes
    its blob cache to Drive instead of the VM disk at ~/.cache/huggingface/.
    Also replaces any existing VM-disk cache dir with a symlink to Drive.
    """
    os.makedirs(DRIVE_HF_CACHE, exist_ok=True)

    # 1. Set env vars — this is the primary mechanism.
    #    huggingface_hub respects these before touching ~/.cache.
    os.environ["HF_HOME"] = DRIVE_HF_CACHE
    os.environ["HF_HUB_CACHE"] = os.path.join(DRIVE_HF_CACHE, "hub")
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    print(f"[colab_setup] HF_HOME → {DRIVE_HF_CACHE}  (all downloads go to Drive)")

    # 2. Replace existing VM-disk cache with a symlink (belt-and-suspenders).
    hf_cache = os.path.expanduser("~/.cache/huggingface")
    if os.path.islink(hf_cache):
        return  # already a symlink — nothing to do
    if os.path.isdir(hf_cache):
        # Move any existing cached files to Drive so nothing is lost
        for item in os.listdir(hf_cache):
            src = os.path.join(hf_cache, item)
            dst = os.path.join(DRIVE_HF_CACHE, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(hf_cache)
    os.makedirs(os.path.dirname(hf_cache), exist_ok=True)
    try:
        os.symlink(DRIVE_HF_CACHE, hf_cache)
        print(f"[colab_setup] Symlinked: {hf_cache} → {DRIVE_HF_CACHE}")
    except OSError:
        pass  # env vars will handle it


def print_budget_info():
    """Print compute unit budget guidance."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  COMPUTE UNIT BUDGET GUIDE (150 units)                         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  A100 40GB: ~8.3 units/hr → ~18 hours total                   ║
║                                                                ║
║  Recommended budget split:                                     ║
║    Download models to Drive    │ CPU (free)  │   0 units       ║
║    Pipeline dev & testing      │ A100        │  ~50 units      ║
║    Extraction runs             │ A100        │  ~85 units      ║
║    Buffer                      │             │  ~15 units      ║
║                                                                ║
║  TIPS:                                                         ║
║  • Download models on FREE CPU runtime (costs 0 units)         ║
║  • Switch to A100 ONLY when ready to run inference             ║
║  • Disconnect when idle — Colab bills even when you walk away  ║
║  • Models on Drive load instantly — no re-download needed      ║
║                                                                ║
╚══════════════════════════════════════════════════════════════════╝
""")


def start_vllm(port=8000):
    """Start vLLM server with Qwen3-14B."""
    print("\n[colab_setup] Starting vLLM server...")

    # Resolve model path (Drive or local)
    llm_repo = "Qwen/Qwen3-14B"
    model_root = get_model_root()
    local_model_path = os.path.join(model_root, llm_repo.replace("/", "--"))

    model_path = llm_repo
    if os.path.isdir(local_model_path) and _model_already_downloaded(local_model_path):
        model_path = local_model_path
        print(f"[colab_setup] Loading model from: {local_model_path}")
    else:
        print(f"[colab_setup] Downloaded model not found at {local_model_path}")
        print(f"[colab_setup] vLLM will download {llm_repo} from HuggingFace (slow)")

    cmd = [
        sys.executable, "setup_vllm.py",
        "--model", model_path,
        "--port", str(port),
        "--set-env",
    ]
    os.execv(sys.executable, cmd)


def main():
    parser = argparse.ArgumentParser(
        description="One-click Colab setup for Protocol Spec Assist"
    )
    parser.add_argument("--download-models", action="store_true",
                        help="Download models to Google Drive (run on FREE CPU runtime)")
    parser.add_argument("--start-vllm", action="store_true",
                        help="Start vLLM server after setup")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--budget", action="store_true",
                        help="Show compute unit budget guide")
    args = parser.parse_args()

    if args.budget:
        print_budget_info()
        return

    print("=" * 64)
    print("  Protocol Spec Assist — Colab Setup")
    print("  Model: Qwen3-14B on A100 40GB")
    print("=" * 64)

    # 1. Mount Drive
    drive_ok = mount_drive()

    # 2. Create directories and redirect HF cache to Drive
    if drive_ok:
        setup_directories()
        redirect_hf_cache_to_drive()

    # 3. Download models (if requested)
    if args.download_models:
        if not drive_ok:
            print("[colab_setup] WARNING: Drive not mounted.")
            print("[colab_setup] Models will download to data/models/ (local storage).")
            if is_colab():
                print("[colab_setup] They will be LOST when the Colab session ends.")
                print("[colab_setup] Mount Drive first for persistent storage.")
        download_models()
        print_budget_info()
        print("[colab_setup] DONE. Now switch to A100 runtime and run:")
        print("[colab_setup]   !python colab_setup.py --start-vllm")
        return

    # 4. Set environment variables for the pipeline
    base_url = f"http://localhost:{args.port}/v1"
    os.environ["VLLM_BASE_URL"] = base_url
    os.environ["ADJUDICATOR_BASE_URL"] = base_url
    os.environ["VLLM_API_KEY"] = "local"
    os.environ["DEFAULT_MODEL"] = "Qwen/Qwen3-14B"
    os.environ["ADJUDICATOR_MODEL"] = "Qwen/Qwen3-14B"

    print(f"\n[colab_setup] Environment configured:")
    print(f"  DEFAULT_MODEL    = Qwen/Qwen3-14B")
    print(f"  ADJUDICATOR_MODEL= Qwen/Qwen3-14B")
    print(f"  VLLM_BASE_URL    = {base_url}")

    if drive_ok:
        print(f"  Model storage    = {DRIVE_MODELS}")
        print(f"  Output storage   = {DRIVE_OUTPUTS}")

    # 5. Start vLLM (if requested)
    if args.start_vllm:
        start_vllm(args.port)
    else:
        print(f"\n[colab_setup] Ready. Next steps:")
        print(f"  1. Start vLLM:  !python setup_vllm.py --set-env")
        print(f"  2. Run pipeline: !python -m protocol_spec_assist.workflows.protocol_run \\")
        print(f"       data/protocols/YOUR_PROTOCOL.pdf --ta oncology")


if __name__ == "__main__":
    main()
