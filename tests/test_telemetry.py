"""Telemetry parsing and summaries, campaign run order and cooldown, and a short live sampler check."""

import json
import shutil
import time
from pathlib import Path

import pytest

from bench.baseline import cool_down, rep_orders
from bench.telemetry import (GPU_SAMPLE_FIELDS, POWERSHELL, Telemetry, core_classes, counter_increase,
                             parse_cpu_csv, parse_gpu_csv, summarize_cpu, summarize_gpu)


def test_counter_increase_handles_resets():
    """Increases sum normally; a drop means a reset to zero, so the new value is all increase since then."""
    assert counter_increase([]) == (0, 0)
    assert counter_increase([5]) == (0, 0)
    assert counter_increase([0, 10, 10, 25]) == (25, 0)
    # 100 -> 130 (+30), reset, 0 -> 7 (+7), 7 -> 20 (+13). The part accrued before the reset is unknowable.
    assert counter_increase([100, 130, 7, 20]) == (50, 1)
    assert counter_increase([50, 3, 1, 4]) == (3 + 1 + 3, 2)


def gpu_line(ts, pstate, temp, sm, mask, counters):
    """One nvidia-smi CSV row in GPU_SAMPLE_FIELDS order."""
    vals = [ts, pstate, temp, sm, "5501", "40.1", mask, *map(str, counters), "0"]
    assert len(vals) == len(GPU_SAMPLE_FIELDS)
    return ", ".join(vals)


def test_gpu_summary():
    """Throttle time across a reset, busy-sample reason fractions, and P-state mix from three samples."""
    text = "\n".join([
        gpu_line("2026/10/19 12:00:00.000", "P0", "60", "1800", "0x0000000000000004", [1000, 0, 0, 0]),
        gpu_line("2026/10/19 12:00:01.000", "P0", "65", "1700", "0x0000000000000000", [1500, 0, 0, 0]),
        gpu_line("2026/10/19 12:00:02.000", "P8", "62", "210", "0x0000000000000025", [200, 0, 0, 0]),  # reset
        "2026/10/19 12:00:03.000, P0, 6",  # partial line from killing the sampler
    ])
    rows = parse_gpu_csv(text)
    assert len(rows) == 3
    s = summarize_gpu(rows)
    assert s["span_s"] == pytest.approx(2.0)
    assert s["throttle"]["sw_power_cap"]["us"] == 500 + 200 and s["throttle"]["sw_power_cap"]["resets"] == 1
    assert s["busy_fraction"] == pytest.approx(2 / 3)  # the third sample has the idle bit
    assert s["reason_active_fraction_while_busy"]["sw_power_cap"] == pytest.approx(0.5)
    assert s["reason_active_fraction_while_busy"]["sw_thermal_slowdown"] == 0
    assert s["temperature_c"]["max"] == 65 and s["pstate_fraction"] == {"P0": pytest.approx(2 / 3),
                                                                        "P8": pytest.approx(1 / 3)}


def cpu_rows(t, util_by_inst):
    """Counter CSV rows for one sample of a mini hybrid CPU: 0:0-0:1 P (1400), 0:2 E (900), 0:3 LPE (700)."""
    nominal = {"0:0": 1400, "0:1": 1400, "0:2": 900, "0:3": 700}
    return [f"{t},{inst},{nominal[inst]},200,{util_by_inst.get(inst, 1.0)}" for inst in nominal]


def test_core_classes_by_nominal_frequency():
    """Highest nominal frequency is P, then E, then LPE."""
    sample = parse_cpu_csv("\n".join(cpu_rows(1000, {})))[1000]
    assert core_classes(sample) == {"0:0": "P", "0:1": "P", "0:2": "E", "0:3": "LPE"}


def test_cpu_summary_tracks_busiest_class():
    """Busiest core per sample: P, P, E, P gives 3/4 P, 1/4 E, and two class switches."""
    lines = (cpu_rows(1000, {"0:0": 95}) + cpu_rows(3000, {"0:1": 90}) + cpu_rows(5000, {"0:2": 99})
             + cpu_rows(7000, {"0:0": 97}) + ["garbage,line", "8000,0:0,N/A,1,1"])
    s = summarize_cpu(parse_cpu_csv("\n".join(lines)))
    assert s["n_samples"] == 4 and s["span_s"] == 6
    assert s["busiest_class_fraction"] == {"E": 0.25, "P": 0.75}
    assert s["busiest_class_switches"] == 2
    assert s["effective_mhz_by_class"]["P"]["p50"] == 2800  # 1400 nominal x 200%
    assert s["busiest_effective_mhz"]["min"] == 1800  # the E-core sample, 900 x 200%


def test_rep_orders():
    """Each repetition is a permutation of the (engine, trace) pairs; repetition 1 omits what already ran; the seed
    fixes the order; and positions really vary across repetitions."""
    burst, rates = Path("burst"), [Path("r1"), Path("r2"), Path("r3")]
    items = [(e, t) for e in ("hf", "contiguous") for t in [burst, *rates]]
    orders = rep_orders(items, reps=3, seed=0, already_run=(("hf", burst),))
    assert sorted(orders[0]) == sorted(x for x in items if x != ("hf", burst))
    assert all(sorted(o) == sorted(items) for o in orders[1:])
    assert orders == rep_orders(items, reps=3, seed=0, already_run=(("hf", burst),))
    many = rep_orders(items, reps=30, seed=1)
    assert len({tuple(o) for o in many}) > 1
    assert {o.index(("hf", burst)) for o in many} == set(range(len(items)))  # not always first


class FakeClock:
    """Simulated time and a GPU that cools 2 °C per second of sleep."""

    def __init__(self, temp):
        """Start at `temp` °C, t = 0."""
        self.t, self.temp = 0.0, temp

    def sleep(self, s):
        """Advance time and cool."""
        self.t += s
        self.temp = max(40.0, self.temp - 2 * s)

    def read(self):
        """Current temperature."""
        return self.temp


def test_cool_down_waits_for_target():
    """From 120 °C, cooling 2 °C/s to a 50 °C target takes 35 s; min wait 30 s, then 5 s polls."""
    c = FakeClock(120.0)
    r = cool_down(50.0, c.read, c.sleep, lambda: c.t, min_s=30, max_s=600, poll_s=5)
    assert r["waited_s"] == 35 and r["end_c"] == 50 and not r["timed_out"]


def test_cool_down_minimum_and_timeout():
    """Already cool: just the minimum wait. Unreachable target: gives up at max_s. No reading: plain minimum wait."""
    c = FakeClock(45.0)
    assert cool_down(50.0, c.read, c.sleep, lambda: c.t, min_s=30)["waited_s"] == 30
    c = FakeClock(90.0)
    r = cool_down(10.0, c.read, c.sleep, lambda: c.t, min_s=30, max_s=100, poll_s=5)
    assert r["timed_out"] and r["waited_s"] == 100
    c = FakeClock(90.0)
    r = cool_down(50.0, lambda: None, c.sleep, lambda: c.t, min_s=30)
    assert r["waited_s"] == 30 and not r["timed_out"]


@pytest.mark.skipif(shutil.which("nvidia-smi") is None or shutil.which(POWERSHELL) is None,
                    reason="needs nvidia-smi and Windows interop")
def test_live_samplers(tmp_path):
    """Both samplers start, write parseable CSV for ~5 s, stop cleanly, and leave a summary (about 10 s)."""
    with Telemetry(tmp_path, gpu_interval_s=0.5, cpu_interval_s=1) as t:
        time.sleep(5)
    assert t.procs == {}
    summary = json.loads((tmp_path / "telemetry.json").read_text())
    assert summary["notes"] == {}
    assert summary["gpu"]["n_samples"] >= 5
    assert summary["cpu"]["n_samples"] >= 2
    assert set(summary["cpu"]["effective_mhz_by_class"]) == {"P", "E", "LPE"}  # Core Ultra 7 155H
