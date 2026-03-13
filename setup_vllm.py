#!/usr/bin/env python3
"""
setup_vllm.py — Start a vLLM server with automatic GPU compatibility handling.

Detects GPU compute capability and works around known issues:
- Tesla T4 (sm_75): flashinfer CUTLASS DSL crash on vLLM >=0.17
  Fix: uninstalls flashinfer, falls back to compatible attention backend
- Ampere+ (sm_80+): no workaround needed, all backends supported

Usage (in a notebook cell or terminal):
    python setup_vllm.py                     # auto-detect everything
    python setup_vllm.py --port 8001         # custom port
    python setup_vllm.py --model Qwen/Qwen3-8B-AWQ --quantization awq  # quantized
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


def pick_max_model_len(gpu_name, cap_major):
    """Choose max-model-len based on GPU VRAM tier."""
    name_lower = (gpu_name or "").lower()
    if "a100" in name_lower or "h100" in name_lower:
        return 32768
    if "v100" in name_lower or "a10" in name_lower or "rtx" in name_lower:
        return 24576
    if "t4" in name_lower or cap_major == 7:
        return 16384
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
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="HuggingFace model ID")
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
        print("[setup_vllm] Alternative: use Ollama (see TEST_RUN_GUIDE.md, Step 3e)")
        sys.exit(1)

    print(f"[setup_vllm] GPU: {gpu_name} (compute capability {cap_major}.x)")

    # ── Apply T4 workaround ─────────────────────────────────────────
    if cap_major < 8:
        fix_flashinfer_for_t4()

    # ── Choose max-model-len ────────────────────────────────────────
    max_model_len = args.max_model_len or pick_max_model_len(gpu_name, cap_major)
    print(f"[setup_vllm] Using --max-model-len {max_model_len}")

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
    if args.quantization:
        cmd.extend(["--quantization", args.quantization])

    # Disable torch.compile which causes V1 engine core crashes
    env = os.environ.copy()
    env["VLLM_USE_V1"] = "1"
    env["VLLM_TORCH_COMPILE_LEVEL"] = "0"

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
            # Dump last 30 lines of stderr for diagnosis
            print("[setup_vllm] === Last 30 lines of vllm_stderr.log ===")
            try:
                log_err.flush()
                with open("vllm_stderr.log") as f:
                    lines = f.readlines()
                    for line in lines[-30:]:
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
        os.environ["DEFAULT_MODEL"] = args.model
        os.environ["ADJUDICATOR_MODEL"] = args.model
        print(f"[setup_vllm] Environment variables set (VLLM_BASE_URL={base_url})")

    return proc


if __name__ == "__main__":
    main()
