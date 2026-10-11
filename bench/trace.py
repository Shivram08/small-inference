"""Build trace files: JSONL with one request per row, {"id", "arrival_s", "prompt_ids", "output_len"}.

M0.4 uses the arrival generator and writer with synthetic prompts; M0.5 adds the ShareGPT request sample.
"""

import json
from pathlib import Path

import numpy as np


def poisson_arrivals(rate: float, n: int, seed: int) -> np.ndarray:
    """n Poisson arrival times (seconds) at `rate` req/s: cumulative sum of exponential gaps with mean 1/rate."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.exponential(1 / rate, n))


def write_trace(path: Path, arrivals, prompts: list[list[int]], output_lens: list[int]) -> Path:
    """Write requests in arrival order with ids 0..n-1."""
    if not len(arrivals) == len(prompts) == len(output_lens):
        raise ValueError("arrivals, prompts and output_lens must have the same length")
    with open(path, "w") as f:
        for i, (a, p, n) in enumerate(zip(arrivals, prompts, output_lens)):
            f.write(json.dumps({"id": i, "arrival_s": float(a), "prompt_ids": list(p), "output_len": int(n)}) + "\n")
    return Path(path)
