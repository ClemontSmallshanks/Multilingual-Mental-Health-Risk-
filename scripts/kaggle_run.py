"""Run experiment batches on a Kaggle GPU from this machine (no browser needed).

Needs Kaggle API credentials: KAGGLE_API_TOKEN or ~/.kaggle/access_token (or legacy KAGGLE_USERNAME/KAGGLE_KEY).
kaggle.com -> Settings -> API -> Create New Token. Never commit them.

  python scripts/kaggle_run.py push  --batch r2 configs/R2_muril_multitask_baseline.json
  python scripts/kaggle_run.py status --batch r2
  python scripts/kaggle_run.py fetch  --batch r2        # downloads runs/ into ./runs (merging, never overwriting)

The kernel clones the repository at the current pushed commit, so commit + push before `push`.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def kaggle_user() -> str:
    """Username from $KAGGLE_USERNAME, else from the CLI config (works with KAGGLE_API_TOKEN / ~/.kaggle/access_token)."""
    user = os.environ.get("KAGGLE_USERNAME")
    if user:
        return user
    out = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if line.strip().startswith("- username:"):
            u = line.split(":", 1)[1].strip()
            if u and u != "None":
                return u
    sys.exit("No Kaggle credentials found (set KAGGLE_API_TOKEN, ~/.kaggle/access_token, or KAGGLE_USERNAME/KAGGLE_KEY).")


def kernel_id(batch: str) -> str:
    return f"{kaggle_user()}/mhrisk-{batch}"


def push(batch: str, configs: list[str], commit: str | None, cpu: bool = False, seeds: list[int] | None = None) -> None:
    commit = commit or subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    remote = subprocess.run(["git", "branch", "-r", "--contains", commit], cwd=ROOT, capture_output=True, text=True).stdout
    if not remote.strip():
        sys.exit(f"commit {commit[:8]} is not on any pushed branch; the kernel could not clone it. Push first.")
    for c in configs:
        if not (ROOT / c).exists():
            sys.exit(f"missing config {c}")
    entries = [c + (" --seeds " + " ".join(map(str, seeds)) if seeds else "") for c in configs]
    src = (ROOT / "kaggle" / "kernel_template.py").read_text()
    src = src.replace("__COMMIT__", commit).replace("__CONFIGS__", json.dumps(entries)).replace("__ALLOW_CPU__", str(cpu))
    d = Path(tempfile.mkdtemp(prefix="kaggle_"))
    (d / "run.py").write_text(src)
    kid = kernel_id(batch)
    meta = {"id": kid, "title": f"mhrisk-{batch}", "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": "true", "enable_gpu": "false" if cpu else "true", "enable_internet": "true",
            "machine_shape": "" if cpu else "NvidiaTeslaT4",
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    cmd = ["kaggle", "kernels", "push", "-p", str(d)] + ([] if cpu else ["--accelerator", "NvidiaTeslaT4"])
    subprocess.run(cmd, check=True)
    shutil.rmtree(d, ignore_errors=True)
    print(f"pushed {kid} @ {commit[:8]} on {'CPU' if cpu else 'GPU'}: {entries}")


def status(batch: str) -> None:
    subprocess.run(["kaggle", "kernels", "status", kernel_id(batch)], check=True)


def fetch(batch: str) -> None:
    d = Path(tempfile.mkdtemp(prefix="kaggle_out_"))
    subprocess.run(["kaggle", "kernels", "output", kernel_id(batch), "-p", str(d)], check=True)
    runs_src, runs_dst = d / "runs", ROOT / "runs"
    n = 0
    for f in sorted(runs_src.rglob("*")) if runs_src.exists() else []:
        if f.is_file() and "checkpoint" not in f.parts:
            dst = runs_dst / f.relative_to(runs_src)
            if dst.exists():
                continue  # never overwrite an existing run
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst); n += 1
    out = ROOT / "reports" / "results" / "kaggle_logs"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("batch_status.json", f"mhrisk-{batch}.log", "run.log"):
        if (d / name).exists():
            shutil.copy2(d / name, out / f"{batch}_{name}")
    print(f"fetched {n} new files into {runs_dst}; logs in {out}")
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["push", "status", "fetch"])
    ap.add_argument("--batch", required=True, help="short name, e.g. r2, e1, e3, e2, grid")
    ap.add_argument("configs", nargs="*")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--cpu", action="store_true", help="run on a Kaggle CPU machine (fp32; ~2 h per MuRIL multi-task seed)")
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="override config seeds (e.g. one seed per kernel)")
    a = ap.parse_args()
    {"push": lambda: push(a.batch, a.configs, a.commit, a.cpu, a.seeds), "status": lambda: status(a.batch), "fetch": lambda: fetch(a.batch)}[a.action]()
