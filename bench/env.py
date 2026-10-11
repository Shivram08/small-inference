"""Environment record for a run folder: software versions, git state, trace hash, and nvidia-smi snapshots."""

import hashlib
import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

GPU_FIELDS = [
    "name", "driver_version", "pstate", "clocks.sm", "clocks.mem", "clocks.max.sm", "clocks.max.mem",
    "temperature.gpu", "power.draw", "power.limit", "clocks_event_reasons.active",
]
REPO = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> str | None:
    """Return a command's stdout, or None if it is missing or fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_snapshot() -> dict:
    """Clocks, temperature, power, and throttle reasons right now.

    `performance_text` is the full `nvidia-smi -q -d PERFORMANCE` output. Its "Clocks Event Reasons Counters"
    are cumulative µs per throttle reason, so after minus before is the throttled time during the run.
    """
    snap = {"time_utc": datetime.now(timezone.utc).isoformat()}
    csv = _run(["nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits"])
    if csv is None:
        snap["error"] = "nvidia-smi unavailable"
        return snap
    values = [v.strip() for v in csv.strip().splitlines()[0].split(",")]
    snap["query"] = dict(zip(GPU_FIELDS, values))
    snap["performance_text"] = _run(["nvidia-smi", "-q", "-d", "PERFORMANCE"])
    return snap


def static_env(trace_path: Path) -> dict:
    """Everything about the run's setup that does not change during the run."""
    def version(pkg: str) -> str | None:
        try:
            return importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            return None

    import torch  # deferred: only needed here, and slow to import

    commit = _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    status = _run(["git", "-C", str(REPO), "status", "--porcelain"])
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ["torch", "triton", "transformers", "numpy"]},
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(status.strip()) if status is not None else None,
        "trace_path": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
    }
