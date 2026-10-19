"""Engine adapters for the harness, and one place that builds them by name."""

CONTINUOUS_KV_BYTES = 3 * 2**29  # 1.5 GiB: 8,192 blocks of 16 tokens for 0.5B bf16, room for 64 x 2,048 tokens
_shared_models: dict = {}


def _model():
    """One bf16 0.5B model shared by every engine of this engine family in a process (a campaign may load several
    variants; each would otherwise hold its own 1 GB of weights)."""
    if "bf16" not in _shared_models:
        from engine.model import Qwen2Model
        _shared_models["bf16"] = Qwen2Model.from_pretrained()
    return _shared_models["bf16"]


def load_engine(name: str, **fake_timing: float) -> tuple[object, dict]:
    """(engine, config recorded in run.json) for "fake", "hf", "contiguous", "continuous", or
    "continuous-nochunk". Imports lazily: torch is slow."""
    if name == "hf":
        from bench.engines.hf import MODEL, HFEngine
        return HFEngine(), {"model": MODEL, "dtype": "bfloat16", "attn": "sdpa", "batch": 1}
    if name == "contiguous":
        from engine.engine import MAX_LEN, ContiguousEngine
        from engine.loader import DEFAULT_MODEL
        return ContiguousEngine(_model()), {"model": DEFAULT_MODEL, "dtype": "bfloat16", "attn": "matmul-softmax",
                                            "kv": "contiguous", "max_len": MAX_LEN, "fused_matmuls": True,
                                            "batch": 1}
    if name in ("continuous", "continuous-nochunk"):
        from dataclasses import asdict

        from engine.continuous import ContinuousEngine
        from engine.loader import DEFAULT_MODEL
        from engine.scheduler import SchedulerConfig
        sched = (SchedulerConfig() if name == "continuous"
                 else SchedulerConfig(chunked_prefill=False, max_num_batched_tokens=2048))
        eng = ContinuousEngine(_model(), kv_bytes=CONTINUOUS_KV_BYTES, config=sched)
        return eng, {"model": DEFAULT_MODEL, "dtype": "bfloat16", "kv": "paged", "block_size": 16,
                     "num_blocks": eng.manager.num_blocks, "attn": "triton decode + gather prefill",
                     "scheduler": asdict(sched)}
    if name == "fake":
        from bench.engines.fake import FakeEngine
        return FakeEngine(**fake_timing), dict(fake_timing)
    raise ValueError(f"unknown engine {name!r}")
