#!/usr/bin/env python3
"""
setup_vllm.py — Start a vLLM server with automatic GPU compatibility handling.

Detects GPU compute capability and works around known issues:
- Tesla T4 (sm_75): flashinfer CUTLASS DSL crash on vLLM >=0.17
  Fix: uninstalls flashinfer, falls back to compatible attention backend
- Ampere+ (sm_80+): no workaround needed, all backends supported

Default model: Qwen3-14B (fits A100 40GB in FP16, ~28GB VRAM).

Usage (in a notebook cell or terminal):
    python setup_vllm.py                     # auto-detect GPU
    python setup_vllm.py --port 8001         # custom port
    python setup_vllm.py --model Qwen/Qwen3-14B  # explicit model
    python setup_vllm.py --set-env           # auto-set env vars
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request


def get_gpu_info():
    """Return (gpu_name, compute_capability_major) or (None, None)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        return name, cap[0]
    except ImportError:
        pass
    # Fallback: parse nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip().split("\n")[0]
        parts = out.split(",")
        name = parts[0].strip()
        cap_major = int(parts[1].strip().split(".")[0])
        return name, cap_major
    except Exception:
        return None, None


def fix_flashinfer_for_t4():
    """
    Uninstall flashinfer to avoid CUTLASS DSL crash on sm_75 (T4).

    vLLM >=0.17 tries to import flashinfer during attention backend
    enumeration. On T4, flashinfer's CUTLASS DSL init crashes with
    'NVVM Compilation Error' because the chip target is empty.
    Removing flashinfer lets vLLM fall back to a compatible backend.
    """
    print("[setup_vllm] T4 detected (compute capability 7.x)")
    print("[setup_vllm] Removing flashinfer to avoid CUTLASS DSL crash on sm_75...")
    for pkg in ["flashinfer", "flashinfer-python"]:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", pkg],
            capture_output=True,
        )
    print("[setup_vllm] flashinfer removed. vLLM will use a compatible attention backend.")


def kill_stale_gpu_processes():
    """Kill leftover vLLM / python processes occupying the GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
    except Exception:
        return
    if not out:
        print("[setup_vllm] GPU is free — no stale processes.")
        return
    for line in out.split("\n"):
        parts = line.split(",")
        if len(parts) < 2:
            continue
        pid = parts[0].strip()
        name = parts[1].strip()
        print(f"[setup_vllm] Killing stale GPU process: PID {pid} ({name})")
        try:
            subprocess.run(["kill", "-9", pid], capture_output=True)
        except Exception:
            pass
    # Give GPU a moment to reclaim memory
    time.sleep(3)
    print("[setup_vllm] Stale GPU processes cleaned up.")


def get_gpu_free_memory_gb():
    """Return free GPU memory in GiB, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        ).strip().split("\n")[0]
        return int(out.strip()) / 1024  # MiB → GiB
    except Exception:
        return None


def pick_max_model_len(gpu_name, cap_major):
    """Choose max-model-len based on GPU VRAM tier."""
    name_lower = (gpu_name or "").lower()
    if "a100" in name_lower:
        return 16384
    if "v100" in name_lower or "a10" in name_lower or "rtx" in name_lower:
        return 16384
    if "t4" in name_lower or cap_major == 7:
        return 8192
    return 16384  # conservative default


def wait_for_server(port, timeout=300, interval=5):
    """Poll vLLM /v1/models endpoint until ready or timeout."""
    url = f"http://localhost:{port}/v1/models"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                elapsed = int(time.time() - start)
                print(f"[setup_vllm] vLLM is ready! (took {elapsed}s)")
                return True
        except Exception:
            pass
        elapsed = int(time.time() - start)
        if elapsed % 30 < interval:
            print(f"[setup_vllm]   Waiting... ({elapsed}s elapsed)")
        time.sleep(interval)
    return False


def main():
    parser = argparse.ArgumentParser(description="Start vLLM with GPU-aware defaults")
    parser.add_argument("--model", default="Qwen/Qwen3-14B",
                        help="HuggingFace model ID (default: Qwen/Qwen3-14B)")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max sequence length (auto-detected from GPU if omitted)")
    parser.add_argument("--quantization", default=None, help="Quantization method (e.g. awq)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for server to be ready")
    parser.add_argument("--set-env", action="store_true",
                        help="Set VLLM_BASE_URL and related env vars in current process")
    args = parser.parse_args()

    # ── Detect GPU ──────────────────────────────────────────────────
    gpu_name, cap_major = get_gpu_info()
    if gpu_name is None:
        print("[setup_vllm] ERROR: No NVIDIA GPU detected. vLLM requires a CUDA GPU.")
        print("[setup_vllm] Alternative: use Ollama (see README.md)")
        sys.exit(1)

    print(f"[setup_vllm] GPU: {gpu_name} (compute capability {cap_major}.x)")
    print(f"[setup_vllm] Model: {args.model}")

    # ── Apply T4 workaround ─────────────────────────────────────────
    if cap_major < 8:
        fix_flashinfer_for_t4()

    # ── Kill stale GPU processes from previous runs ────────────────
    kill_stale_gpu_processes()

    # ── Verify GPU memory is available ──────────────────────────────
    free_gb = get_gpu_free_memory_gb()
    if free_gb is not None:
        print(f"[setup_vllm] Free GPU memory: {free_gb:.1f} GiB")
        if free_gb < 10:
            print(f"[setup_vllm] ERROR: Only {free_gb:.1f} GiB free. Need at least ~16 GiB.")
            print("[setup_vllm] Run: nvidia-smi  — to see what's using GPU memory.")
            sys.exit(1)

    # ── Choose max-model-len ────────────────────────────────────────
    max_model_len = args.max_model_len or pick_max_model_len(gpu_name, cap_major)
    print(f"[setup_vllm] Using --max-model-len {max_model_len}")

    # ── Determine canonical model name ──────────────────────────────
    # When loading from a local path (e.g. /content/drive/MyDrive/.../Qwen--Qwen3-14B),
    # vLLM advertises that local path as the model ID. The client expects the
    # canonical HF name (e.g. Qwen/Qwen3-14B). Use --served-model-name to align.
    canonical_model = args.model
    served_model_name = None
    if "/" not in args.model or args.model.startswith("/"):
        # Looks like a local path, not an HF model ID.
        # Try to recover the canonical name from the path components.
        # Common pattern: .../Qwen--Qwen3-14B or .../Qwen/Qwen3-14B
        parts = Path(args.model).parts
        if len(parts) >= 2:
            org, name = parts[-2], parts[-1]
            # HuggingFace download cache uses "--" as separator
            if "--" in name:
                served_model_name = name.replace("--", "/")
            else:
                served_model_name = f"{org}/{name}"
        if served_model_name:
            canonical_model = served_model_name
            print(f"[setup_vllm] Local model path detected. Will serve as: {served_model_name}")

    # ── Build vLLM command ──────────────────────────────────────────
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--max-model-len", str(max_model_len),
        "--enable-prefix-caching",
        "--dtype", "auto",
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--enforce-eager",  # skip torch.compile — avoids Dynamo crash in V1 engine
    ]
    if served_model_name:
        cmd.extend(["--served-model-name", served_model_name])
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    # Disable V1 multiprocess engine — its core subprocess crashes on many
    # setups with "Failed core proc(s): {}".  V0 is single-process and stable.
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "0"

    print(f"[setup_vllm] Starting: {' '.join(cmd)}")

    # ── Launch (stderr tee'd to file + console) ─────────────────────
    log_out = open("vllm_stdout.log", "w")
    log_err = open("vllm_stderr.log", "w")
    proc = subprocess.Popen(
        cmd, stdout=log_out, stderr=log_err, env=env,
    )
    print(f"[setup_vllm] vLLM starting (PID: {proc.pid})")
    print(f"[setup_vllm] Logs: tail -f vllm_stderr.log")

    # ── Wait for ready ──────────────────────────────────────────────
    if not args.no_wait:
        if not wait_for_server(args.port):
            print("[setup_vllm] ERROR: vLLM failed to start within 5 minutes.")
            # Dump last 80 lines of stderr for diagnosis
            print("[setup_vllm] === Last 80 lines of vllm_stderr.log ===")
            try:
                log_err.flush()
                with open("vllm_stderr.log") as f:
                    lines = f.readlines()
                    for line in lines[-80:]:
                        print(f"  {line.rstrip()}")
            except Exception:
                print("[setup_vllm] (could not read log)")
            # Check if process died
            retcode = proc.poll()
            if retcode is not None:
                print(f"[setup_vllm] Process exited with code {retcode}")
            sys.exit(1)

    # ── Set env vars ────────────────────────────────────────────────
    if args.set_env:
        base_url = f"http://localhost:{args.port}/v1"
        os.environ["VLLM_BASE_URL"] = base_url
        os.environ["ADJUDICATOR_BASE_URL"] = base_url
        os.environ["VLLM_API_KEY"] = "local"
        os.environ["DEFAULT_MODEL"] = canonical_model
        os.environ["ADJUDICATOR_MODEL"] = canonical_model
        print(f"[setup_vllm] Environment variables set (VLLM_BASE_URL={base_url}, model={canonical_model})")

    return proc


if __name__ == "__main__":
    main()
