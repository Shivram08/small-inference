"""Engine adapters for the harness, and one place that builds them by name."""


def load_engine(name: str, **fake_timing: float) -> tuple[object, dict]:
    """(engine, config recorded in run.json) for "fake", "hf", or "contiguous". Imports lazily: torch is slow."""
    if name == "hf":
        from bench.engines.hf import MODEL, HFEngine
        return HFEngine(), {"model": MODEL, "dtype": "bfloat16", "attn": "sdpa", "batch": 1}
    if name == "contiguous":
        from engine.engine import MAX_LEN, ContiguousEngine
        from engine.loader import DEFAULT_MODEL
        return ContiguousEngine(), {"model": DEFAULT_MODEL, "dtype": "bfloat16", "attn": "matmul-softmax",
                                    "kv": "contiguous", "max_len": MAX_LEN, "fused_matmuls": True, "batch": 1}
    if name == "fake":
        from bench.engines.fake import FakeEngine
        return FakeEngine(**fake_timing), dict(fake_timing)
    raise ValueError(f"unknown engine {name!r}")
