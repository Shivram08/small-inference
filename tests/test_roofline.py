"""M0.6: roofline formulas and small-size smoke runs of each measurement."""

import pytest
import torch

from bench import roofline

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_intensity_matches_hand_calculation():
    """I = 2MKN / (2(MK + KN + MN)): about 1 FLOP/B at M = 1 and about 59 at M = 64 for K = 896, N = 4864."""
    assert roofline.intensity(1, 896, 4864) == pytest.approx(2 * 896 * 4864 / (2 * (896 + 896 * 4864 + 4864)))
    assert roofline.intensity(1, 896, 4864) == pytest.approx(0.9987, abs=1e-4)
    assert roofline.intensity(64, 896, 4864) == pytest.approx(59.0, abs=0.1)


def test_time_cuda_sees_gpu_work():
    """Event timing scales with work: a 4096^3 matmul takes clearly longer than a 512^3 one."""
    small = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    big = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    t_small = roofline.time_cuda(lambda: small @ small, reps=5, warmup=2)
    t_big = roofline.time_cuda(lambda: big @ big, reps=5, warmup=2)
    assert len(t_big) == 5 and (t_small > 0).all()
    assert t_big.min() > 20 * t_small.max()  # 512x more FLOPs


def test_small_measurements_are_plausible():
    """Each measurement runs at small size and gives physically plausible numbers for a laptop RTX 3050."""
    bw = roofline.measure_bandwidth(64 * 2**20)["gb_s_at_median"]
    assert 20 < bw < 400
    sweep = roofline.decode_sweep(ms=[1, 64])
    assert sweep[1]["tflop_s"] > 10 * sweep[0]["tflop_s"]  # batching raises achieved FLOP/s
    us = roofline.launch_overhead(n=200, reps=3)["us_per_launch"]["median"]
    assert 0.5 < us < 200
