"""Auto-restart wrapper for run_study_a_entropy.py.

Repeatedly calls the entropy script until all 60 questions are cached.
Kills stale python processes holding VRAM before each restart.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "study_a" / "inference_cache.jsonl"
SCRIPT = ROOT / "experiments" / "run_study_a_entropy.py"
PYTHON = sys.executable
TARGET = 60


def count_cached() -> int:
    """Count unique (model, task, question_id) entries — ignores duplicate lines."""
    if not CACHE.exists():
        return 0
    try:
        seen: set = set()
        for line in CACHE.open(encoding="utf-8"):
            row = json.loads(line)
            seen.add((row["model"], row["task"], str(row["question_id"])))
        return len(seen)
    except Exception:
        return 0


def vram_free_gb() -> float:
    try:
        import torch
        free, _ = torch.cuda.mem_get_info(0)
        return free / 1e9
    except Exception:
        return -1.0


def kill_stale_python(my_pid: int) -> None:
    """Kill stale inference processes via PowerShell command-line filter."""
    try:
        ps_cmd = (
            "Get-CimInstance Win32_Process "
            "| Where-Object { $_.Name -eq 'python.exe' -and "
            "$_.CommandLine -like '*run_study_a_entropy*' } "
            "| ForEach-Object { "
            "Write-Host \"  [cleanup] killing pid=$($_.ProcessId)\"; "
            "Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            capture_output=False, text=True
        )
    except Exception as e:
        print(f"  [cleanup] warning: {e}", flush=True)


def main():
    my_pid = os.getpid()
    run = 0
    while True:
        # Kill stale processes holding VRAM before loading model
        kill_stale_python(my_pid)
        time.sleep(3)

        cached = count_cached()
        print(f"\n[restart {run}] cached={cached}/{TARGET}  VRAM_free={vram_free_gb():.1f}GB",
              flush=True)

        if cached >= TARGET:
            print("All questions cached. Done.", flush=True)
            break

        env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}
        proc = subprocess.run(
            [PYTHON, "-u", str(SCRIPT)],
            env=env,
        )
        print(f"[restart {run}] process exited with code {proc.returncode}", flush=True)

        # Short pause to let CUDA/OS release resources
        time.sleep(5)
        run += 1

    # Print final cache summary
    rows = [json.loads(l) for l in CACHE.open(encoding="utf-8")]
    from collections import defaultdict
    by: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["model"]][r["task"]].append(r["entropy_bits"])
    print("\n=== FINAL CACHE ===")
    for m in sorted(by):
        for t in sorted(by[m]):
            hs = by[m][t]
            print(f"  {m}|{t}: {len(hs)} Qs  mean_H={sum(hs)/len(hs):.3f}")
    print(f"Total: {len(rows)}/{TARGET}")


if __name__ == "__main__":
    main()
