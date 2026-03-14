#!/usr/bin/env python3
"""
colab_setup.py — One-click Colab setup for Protocol Spec Assist.

Handles the full Colab workflow:
  1. Mount Google Drive (models persist across sessions — download once)
  2. Download models to Drive (not the 100GB VM disk)
  3. Start vLLM with the right model for your GPU
  4. Set environment variables

Usage in a Colab cell:
    # First time (downloads models — takes 10-20 min):
    !python colab_setup.py --download-models

    # Subsequent sessions (models already on Drive):
    !python colab_setup.py

    # With explicit tier:
    !python colab_setup.py --tier colab_a100

Compute unit budget (150 units):
    GPU       | Units/hr | Hours available
    T4        | ~1.7     | ~88 hrs  (but 8B OOMs on T4)
    L4        | ~2.5     | ~60 hrs
    A100 40GB | ~8.3     | ~18 hrs  ← recommended
    A100 80GB | ~11.2    | ~13 hrs

Strategy: Do all setup/download on FREE CPU runtime. Switch to A100 only for inference.
"""

import argparse
import os
import subprocess
import sys
import time


# ── Google Drive paths ────────────────────────────────────────────────────────

DRIVE_MOUNT = "/content/drive"
DRIVE_ROOT = "/content/drive/MyDrive"
DRIVE_MODELS = "/content/drive/MyDrive/qcagent_models"
DRIVE_OUTPUTS = "/content/drive/MyDrive/qcagent_outputs"

# ── Model definitions ────────────────────────────────────────────────────────

MODELS = {
    "colab_a100": {
        "llm": [
            {"repo": "Qwen/Qwen3-14B", "size_gb": 28, "purpose": "Base extractor (A100 40GB)"},
        ],
        "support": [
            {"repo": "BAAI/bge-m3", "size_gb": 2.3, "purpose": "Embeddings"},
            {"repo": "BAAI/bge-reranker-v2-m3", "size_gb": 2.3, "purpose": "Reranker"},
        ],
    },
    "h100": {
        "llm": [
            {"repo": "Qwen/Qwen3-235B-A22B-FP8", "size_gb": 120, "purpose": "MoE extractor FP8 (H100)"},
        ],
        "support": [
            {"repo": "BAAI/bge-m3", "size_gb": 2.3, "purpose": "Embeddings"},
            {"repo": "BAAI/bge-reranker-v2-m3", "size_gb": 2.3, "purpose": "Reranker"},
        ],
    },
}


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

    if os.path.ismount(DRIVE_MOUNT) or os.path.exists(f"{DRIVE_ROOT}"):
        print(f"[colab_setup] Google Drive already mounted at {DRIVE_MOUNT}")
        return True

    print("[colab_setup] Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount(DRIVE_MOUNT)
        print(f"[colab_setup] Drive mounted. Your 2TB storage is at: {DRIVE_ROOT}")
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


def download_models(tier="colab_a100"):
    """Download models to Google Drive (persists across sessions)."""
    tier_models = MODELS.get(tier, MODELS["colab_a100"])

    all_models = tier_models["llm"] + tier_models["support"]
    total_gb = sum(m["size_gb"] for m in all_models)
    print(f"\n[colab_setup] Downloading {len(all_models)} models (~{total_gb:.0f} GB total)")
    print(f"[colab_setup] Destination: {DRIVE_MODELS}")
    print(f"[colab_setup] Models persist on Drive — only need to download once.\n")

    for model in all_models:
        repo = model["repo"]
        model_dir = os.path.join(DRIVE_MODELS, repo.replace("/", "--"))

        # Check if already downloaded
        if os.path.isdir(model_dir) and any(
            f.endswith((".safetensors", ".bin", ".model"))
            for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))
        ):
            print(f"  [skip] {repo} — already on Drive ({model['purpose']})")
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

    print(f"\n[colab_setup] All models downloaded to {DRIVE_MODELS}")
    print("[colab_setup] They will persist across Colab sessions.")


def symlink_hf_cache():
    """Create symlink so HuggingFace cache points to Drive models."""
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.islink(hf_cache) or os.path.isdir(hf_cache):
        return  # already set up

    os.makedirs(os.path.dirname(hf_cache), exist_ok=True)
    os.makedirs(DRIVE_MODELS, exist_ok=True)
    try:
        os.symlink(DRIVE_MODELS, hf_cache)
        print(f"[colab_setup] HF cache symlinked: {hf_cache} → {DRIVE_MODELS}")
    except OSError:
        print(f"[colab_setup] Could not symlink HF cache. Models will use Drive path directly.")


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


def start_vllm(tier="colab_a100", port=8000):
    """Start vLLM server with tier-appropriate settings.

    If models were downloaded to Drive, passes the Drive path as --model
    so vLLM loads from there instead of re-downloading from HuggingFace.
    The --model flag overrides the tier's default model name (see setup_vllm.py).
    """
    print(f"\n[colab_setup] Starting vLLM server (tier: {tier})...")

    # Resolve Drive model path
    tier_models = MODELS.get(tier, MODELS["colab_a100"])
    llm_repo = tier_models["llm"][0]["repo"]
    drive_model_path = os.path.join(DRIVE_MODELS, llm_repo.replace("/", "--"))

    cmd = [
        sys.executable, "setup_vllm.py",
        "--tier", tier,
        "--port", str(port),
        "--set-env",
    ]

    # --model after --tier overrides the tier's default (setup_vllm.py fix)
    if os.path.isdir(drive_model_path):
        cmd.extend(["--model", drive_model_path])
        print(f"[colab_setup] Loading model from Drive: {drive_model_path}")
    else:
        print(f"[colab_setup] Drive model not found at {drive_model_path}")
        print(f"[colab_setup] vLLM will download {llm_repo} from HuggingFace (slow)")

    # Also set MODEL_TIER so retrieval auto-routes to CPU
    os.environ["MODEL_TIER"] = tier

    os.execv(sys.executable, cmd)


def main():
    parser = argparse.ArgumentParser(
        description="One-click Colab setup for Protocol Spec Assist"
    )
    parser.add_argument("--tier", default="colab_a100",
                        choices=["colab_a100", "colab_a100_single", "h100"],
                        help="Model tier (default: colab_a100)")
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
    print(f"  Tier: {args.tier}")
    print("=" * 64)

    # 1. Mount Drive
    drive_ok = mount_drive()

    # 2. Create directories
    if drive_ok:
        setup_directories()
        symlink_hf_cache()

    # 3. Download models (if requested)
    if args.download_models:
        if not drive_ok:
            print("[colab_setup] WARNING: Drive not mounted. Models will download to VM disk.")
            print("[colab_setup] They will be LOST when the session ends.")
        download_models(args.tier)
        print_budget_info()
        print("[colab_setup] DONE. Now switch to A100 runtime and run:")
        print("[colab_setup]   !python colab_setup.py --start-vllm")
        return

    # 4. Set environment variables for the pipeline
    os.environ["MODEL_TIER"] = args.tier

    from protocol_spec_assist.serving.model_client import MODEL_TIERS
    tier = MODEL_TIERS.get(args.tier, MODEL_TIERS["colab_a100"])
    base_url = f"http://localhost:{args.port}/v1"
    os.environ["VLLM_BASE_URL"] = base_url
    os.environ["ADJUDICATOR_BASE_URL"] = base_url
    os.environ["VLLM_API_KEY"] = "local"
    os.environ["DEFAULT_MODEL"] = tier["default_model"]
    os.environ["ADJUDICATOR_MODEL"] = tier["adjudicator_model"]

    print(f"\n[colab_setup] Environment configured:")
    print(f"  MODEL_TIER       = {args.tier}")
    print(f"  DEFAULT_MODEL    = {tier['default_model']}")
    print(f"  ADJUDICATOR_MODEL= {tier['adjudicator_model']}")
    print(f"  VLLM_BASE_URL    = {base_url}")

    if drive_ok:
        print(f"  Model storage    = {DRIVE_MODELS}")
        print(f"  Output storage   = {DRIVE_OUTPUTS}")

    # 5. Start vLLM (if requested)
    if args.start_vllm:
        start_vllm(args.tier, args.port)
    else:
        print(f"\n[colab_setup] Ready. Next steps:")
        print(f"  1. Start vLLM:  !python setup_vllm.py --tier {args.tier} --set-env")
        print(f"  2. Run pipeline: !python -m protocol_spec_assist.workflows.protocol_run \\")
        print(f"       data/protocols/YOUR_PROTOCOL.pdf --ta oncology")


if __name__ == "__main__":
    main()
