"""Background telemetry during a run: GPU state from nvidia-smi and host CPU clocks from Windows performance counters.

Each sampler is a separate child process writing CSV to the run folder, so the runner itself stays single-threaded.
On exit, `telemetry.json` summarizes both files. Sampling is a measurement change: nvidia-smi polling may keep the
dGPU from idling down between requests (the Phase 0 cold-start prefill effect), and both samplers use a little CPU.

GPU (`telemetry_gpu.csv`): nvidia-smi's "Clocks Event Reasons Counters" are cumulative µs per throttle reason, but they
reset to zero when the dGPU powers off in an idle gap, so before-minus-after is unreliable at low load. Sampling them
every second and summing only the increases (`counter_increase`) bounds the loss to one interval per reset. The
active-reason bitmask in each sample gives the fraction of busy samples each reason was active in.

CPU (`telemetry_cpu.csv`): the host is a hybrid Core Ultra 7 155H, and WSL2 vCPUs float over its cores. Windows reports
per logical CPU a nominal frequency (1400 MHz on P-cores, 900 on E-cores, 700 on LP E-cores), "% Processor
Performance" (effective clock = nominal x performance / 100), and "% Processor Utility". The busiest core in each
sample is usually the one running the decode thread, so its class tells whether the thread sat on a P- or E-core.
"""

import json
import shutil
import statistics
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path

GPU_SAMPLE_FIELDS = [
    "timestamp", "pstate", "temperature.gpu", "clocks.sm", "clocks.mem", "power.draw",
    "clocks_event_reasons.active",
    "clocks_event_reasons_counters.sw_power_cap",
    "clocks_event_reasons_counters.sw_thermal_slowdown",
    "clocks_event_reasons_counters.hw_thermal_slowdown",
    "clocks_event_reasons_counters.hw_power_brake_slowdown",
    "clocks_event_reasons_counters.sync_boost",
]
THROTTLE_COUNTERS = [f for f in GPU_SAMPLE_FIELDS if f.startswith("clocks_event_reasons_counters.")]
# NVML clocks-event-reason bits. The SW power-cap and SW thermal counters also run while the GPU is idle (active mask
# 0x24 at idle on this laptop), so counter increases overstate throttling; the per-sample mask restricted to busy
# samples (idle bit clear) is the cleaner signal. Note that "gpu_idle" is the driver's judgment, not ours: during
# launch-bound decode it can report idle and hold ~1,200 MHz (17 W) for tens of seconds before boosting to ~1,940 MHz.
REASON_BITS = {"gpu_idle": 0x1, "applications_clocks_setting": 0x2, "sw_power_cap": 0x4, "hw_slowdown": 0x8,
               "sync_boost": 0x10, "sw_thermal_slowdown": 0x20, "hw_thermal_slowdown": 0x40,
               "hw_power_brake_slowdown": 0x80, "display_clock_setting": 0x100,
               "reliability": 0x400}  # 0x400 identified 2026-10-19: active at full boost during decode
CORE_CLASSES = ["P", "E", "LPE"]  # by nominal frequency rank, highest first
POWERSHELL = "powershell.exe"
_PS_COUNTERS = ("'\\Processor Information(*)\\Processor Frequency',"
                "'\\Processor Information(*)\\% Processor Performance',"
                "'\\Processor Information(*)\\% Processor Utility'")


def cpu_counter_script(interval_s: int | None) -> str:
    """PowerShell that prints one CSV line per logical CPU per sample: unix_ms,instance,nominal_mhz,perf_pct,util_pct.

    `interval_s=None` takes a single sample; otherwise it samples continuously until killed.
    """
    sampling = "-MaxSamples 1" if interval_s is None else f"-SampleInterval {interval_s} -Continuous"
    return (
        "$ErrorActionPreference = 'Stop'; $inv = [cultureinfo]::InvariantCulture; "
        f"Get-Counter -Counter {_PS_COUNTERS} {sampling} | ForEach-Object {{ "
        "$t = [DateTimeOffset]::new($_.Timestamp).ToUnixTimeMilliseconds(); $rows = @{}; "
        "foreach ($s in $_.CounterSamples) { "
        "if ($s.InstanceName -match '_total') { continue }; "
        "if (-not $rows.ContainsKey($s.InstanceName)) { $rows[$s.InstanceName] = @{} }; "
        "$rows[$s.InstanceName][$s.Path.Split('\\')[-1]] = $s.CookedValue.ToString($inv) }; "
        "foreach ($k in $rows.Keys) { $r = $rows[$k]; "
        "[Console]::Out.WriteLine(('{0},{1},{2},{3},{4}' -f $t, ($k -replace ',', ':'), "
        "$r['processor frequency'], $r['% processor performance'], $r['% processor utility'])) }; "
        "[Console]::Out.Flush() }"
    )


def counter_increase(values: list[float]) -> tuple[float, int]:
    """Total increase of a cumulative counter that may reset to zero, and the number of resets seen.

    A drop means the counter restarted from zero, so the new value is all increase since the reset. Whatever accrued
    between the previous sample and the reset is lost, at most one sampling interval's worth.
    """
    total, resets = 0.0, 0
    for prev, cur in zip(values, values[1:]):
        if cur >= prev:
            total += cur - prev
        else:
            total += cur
            resets += 1
    return total, resets


def _num(s: str) -> float | None:
    """Parse an nvidia-smi or counter value; None for N/A and other non-numbers."""
    try:
        return float(s)
    except ValueError:
        return None


def _stats(xs: list[float]) -> dict | None:
    """min / p50 / mean / max of a non-empty list, else None."""
    if not xs:
        return None
    return {"min": min(xs), "p50": statistics.median(xs), "mean": statistics.fmean(xs), "max": max(xs)}


def parse_gpu_csv(text: str) -> list[dict]:
    """Rows of `nvidia-smi --query-gpu=GPU_SAMPLE_FIELDS --format=csv,noheader,nounits` output as dicts."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(GPU_SAMPLE_FIELDS):
            continue  # a partial last line from killing the sampler mid-write
        row = dict(zip(GPU_SAMPLE_FIELDS, parts))
        try:
            row["unix_s"] = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        rows.append(row)
    return rows


def summarize_gpu(rows: list[dict]) -> dict:
    """Temperature, clocks, power, P-state mix, and throttled time with counter resets handled."""
    if not rows:
        return {"n_samples": 0}
    span_s = rows[-1]["unix_s"] - rows[0]["unix_s"]
    throttle = {}
    for field in THROTTLE_COUNTERS:
        values = [v for v in (_num(r[field]) for r in rows) if v is not None]
        us, resets = counter_increase(values)
        throttle[field.split(".", 1)[1]] = {"us": us, "resets": resets,
                                            "fraction_of_span": us / (span_s * 1e6) if span_s > 0 else None}
    col = lambda f: [v for v in (_num(r[f]) for r in rows) if v is not None]  # noqa: E731
    pstates = Counter(r["pstate"] for r in rows)
    masks = []
    for r in rows:
        try:
            masks.append(int(r["clocks_event_reasons.active"], 16))
        except ValueError:
            pass
    busy = [m for m in masks if not m & REASON_BITS["gpu_idle"]]
    return {
        "n_samples": len(rows),
        "span_s": span_s,
        "busy_fraction": len(busy) / len(masks) if masks else None,
        "reason_active_fraction_while_busy": {
            name: sum(bool(m & bit) for m in busy) / len(busy)
            for name, bit in REASON_BITS.items() if name != "gpu_idle"} if busy else None,
        "temperature_c": _stats(col("temperature.gpu")),
        "sm_clock_mhz": _stats(col("clocks.sm")),
        "mem_clock_mhz": _stats(col("clocks.mem")),
        "power_w": _stats(col("power.draw")),
        "pstate_fraction": {p: n / len(rows) for p, n in sorted(pstates.items())},
        "throttle": throttle,
    }


def parse_cpu_csv(text: str) -> dict[int, dict[str, tuple[float, float, float]]]:
    """Counter CSV -> {unix_ms: {instance: (nominal_mhz, perf_pct, util_pct)}}."""
    samples: dict[int, dict[str, tuple[float, float, float]]] = {}
    for line in text.splitlines():
        parts = line.strip().split(",")
        if len(parts) != 5:
            continue
        t, inst, *vals = parts
        nums = [_num(v) for v in vals]
        if not t.isdigit() or any(v is None for v in nums):
            continue
        samples.setdefault(int(t), {})[inst] = tuple(nums)
    return samples


def core_classes(sample: dict[str, tuple[float, float, float]]) -> dict[str, str]:
    """Map each logical CPU to P / E / LPE by its nominal frequency rank (highest nominal = P)."""
    nominal = sorted({v[0] for v in sample.values()}, reverse=True)
    label = {f: CORE_CLASSES[i] if i < len(CORE_CLASSES) else f"C{i}" for i, f in enumerate(nominal)}
    return {inst: label[v[0]] for inst, v in sample.items()}


def summarize_cpu_sample(sample: dict[str, tuple[float, float, float]]) -> dict:
    """Mean effective MHz per core class, and the busiest core's class, clock, and utility."""
    classes = core_classes(sample)
    by_class: dict[str, list[float]] = {}
    for inst, (nominal, perf, _) in sample.items():
        by_class.setdefault(classes[inst], []).append(nominal * perf / 100)
    busiest = max(sample, key=lambda inst: sample[inst][2])
    nominal, perf, util = sample[busiest]
    return {
        "effective_mhz_by_class": {c: statistics.fmean(v) for c, v in sorted(by_class.items())},
        "busiest": {"instance": busiest, "class": classes[busiest], "effective_mhz": nominal * perf / 100,
                    "util_pct": util},
    }


def summarize_cpu(samples: dict[int, dict[str, tuple[float, float, float]]]) -> dict:
    """Over all samples: which class the busiest core was in, its clock, and how often that class switched."""
    if not samples:
        return {"n_samples": 0}
    per = [summarize_cpu_sample(samples[t]) for t in sorted(samples)]
    busiest_classes = [p["busiest"]["class"] for p in per]
    counts = Counter(busiest_classes)
    classes = sorted({c for p in per for c in p["effective_mhz_by_class"]})
    return {
        "n_samples": len(per),
        "span_s": (max(samples) - min(samples)) / 1e3,
        "busiest_class_fraction": {c: counts[c] / len(per) for c in sorted(counts)},
        "busiest_class_switches": sum(a != b for a, b in zip(busiest_classes, busiest_classes[1:])),
        "busiest_effective_mhz": _stats([p["busiest"]["effective_mhz"] for p in per]),
        "effective_mhz_by_class": {c: _stats([p["effective_mhz_by_class"][c] for p in per
                                              if c in p["effective_mhz_by_class"]]) for c in classes},
    }


def cpu_snapshot() -> dict:
    """One sample of host CPU clocks, summarized per core class. {"error": ...} outside WSL or on failure."""
    if shutil.which(POWERSHELL) is None:
        return {"error": f"{POWERSHELL} not found"}
    try:
        out = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", cpu_counter_script(None)],
                             capture_output=True, text=True, check=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    samples = parse_cpu_csv(out)
    if not samples:
        return {"error": "no counter samples parsed"}
    t = max(samples)
    return {"unix_ms": t, **summarize_cpu_sample(samples[t])}


class Telemetry:
    """Context manager: start both samplers on enter; on exit stop them and write telemetry.json.

    A sampler whose tool is missing is skipped and noted in the summary rather than failing the run.
    """

    def __init__(self, run_dir: Path, gpu_interval_s: float = 1.0, cpu_interval_s: int = 2):
        """Sampling intervals: nvidia-smi takes milliseconds, Get-Counter whole seconds."""
        self.run_dir = Path(run_dir)
        self.gpu_interval_s, self.cpu_interval_s = gpu_interval_s, cpu_interval_s
        self.gpu_path, self.cpu_path = self.run_dir / "telemetry_gpu.csv", self.run_dir / "telemetry_cpu.csv"
        self.procs: dict[str, subprocess.Popen] = {}
        self.notes: dict[str, str] = {}

    def _start(self, name: str, cmd: list[str], path: Path) -> None:
        """Launch one sampler with stdout going to `path`."""
        if shutil.which(cmd[0]) is None:
            self.notes[name] = f"{cmd[0]} not found"
            return
        with open(path, "w") as f:
            self.procs[name] = subprocess.Popen(cmd, stdout=f, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)

    def __enter__(self) -> "Telemetry":
        """Start sampling."""
        self._start("gpu", ["nvidia-smi", f"--query-gpu={','.join(GPU_SAMPLE_FIELDS)}",
                            "--format=csv,noheader,nounits", f"-lms={round(self.gpu_interval_s * 1000)}"],
                    self.gpu_path)
        self._start("cpu", [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command",
                            cpu_counter_script(self.cpu_interval_s)], self.cpu_path)
        return self

    def stop(self) -> dict:
        """Stop the samplers, summarize what they wrote, and write telemetry.json. Safe to call twice."""
        for name, proc in self.procs.items():
            if proc.poll() is not None and name not in self.notes:
                self.notes[name] = f"sampler exited early with code {proc.returncode}"
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        self.procs = {}
        summary = {
            "gpu_interval_s": self.gpu_interval_s,
            "cpu_interval_s": self.cpu_interval_s,
            "notes": self.notes,
            "gpu": summarize_gpu(parse_gpu_csv(self.gpu_path.read_text())) if self.gpu_path.exists() else None,
            "cpu": summarize_cpu(parse_cpu_csv(self.cpu_path.read_text())) if self.cpu_path.exists() else None,
        }
        (self.run_dir / "telemetry.json").write_text(json.dumps(summary, indent=2))
        return summary

    def __exit__(self, *exc) -> None:
        """Stop sampling, even if the run raised."""
        self.stop()
