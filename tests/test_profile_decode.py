"""Decode-step profiler accounting: interval union and per-step breakdown on synthetic profiler events."""

from dataclasses import dataclass

from torch.autograd import DeviceType
from torch.autograd.profiler_util import Interval

from bench.profile_decode import STEP_LABEL, breakdown, interval_union_us


@dataclass
class Event:
    """The FunctionEvent fields breakdown() reads."""
    name: str
    device_type: DeviceType
    time_range: Interval


def ev(name, start, end, device=DeviceType.CPU):
    """Build a synthetic event over [start, end] µs."""
    return Event(name, device, Interval(start, end))


def test_interval_union():
    """Overlaps and touching intervals count once; gaps don't count."""
    assert interval_union_us([]) == 0
    assert interval_union_us([(0, 10)]) == 10
    assert interval_union_us([(0, 10), (5, 15)]) == 15  # overlap
    assert interval_union_us([(20, 30), (0, 10)]) == 20  # unsorted, gap
    assert interval_union_us([(0, 10), (10, 12), (2, 3)]) == 12  # touching, nested


def test_breakdown_assigns_by_step_and_ignores_gpu_annotation():
    """Kernels and API calls land in the step containing their start. The GPU-side step annotation, which spans
    every kernel, must not count as GPU busy time (it did in the first version)."""
    events = [
        ev(STEP_LABEL, 0, 100), ev(STEP_LABEL, 200, 260),
        ev(STEP_LABEL, 1, 99, DeviceType.CUDA),  # GPU-side annotation for step 0
        ev("cudaLaunchKernel", 5, 8), ev("cudaLaunchKernel", 10, 12), ev("cudaMemcpyAsync", 60, 95),
        ev("gemv", 20, 40, DeviceType.CUDA), ev("rmsnorm", 45, 50, DeviceType.CUDA),
        ev("cudaLaunchKernel", 205, 209), ev("gemv", 215, 235, DeviceType.CUDA),
        ev("cudaLaunchKernel", 150, 151),  # between steps: ignored
    ]
    s0, s1 = breakdown(events)
    assert s0 == {"wall_us": 100, "launch_calls": 2, "launch_api_us": 5, "sync_us": 35, "n_kernels": 2,
                  "gpu_busy_us": 25, "other_cpu_us": 60}
    assert (s1["n_kernels"], s1["gpu_busy_us"], s1["launch_calls"], s1["sync_us"]) == (1, 20, 1, 0)
