"""Engine adapters for the harness, and one place that builds them by name."""

CONTINUOUS_KV_BYTES = 3 * 2**29  # 1.5 GiB: 8,192 blocks of 16 tokens for 0.5B bf16, room for 64 x 2,048 tokens
CONTINUOUS = {  # engine name -> (KV format, chunked prefill)
    "continuous": ("bf16", True),
    "continuous-nochunk": ("bf16", False),
    "continuous-int8": ("int8", True),
    "continuous-int8k": ("int8-kchannel", True),
}
_shared_models: dict = {}


def _model(model_id: str | None = None):
    """One bf16 model per checkpoint shared by every engine in a process (a campaign may load several variants;
    each would otherwise hold its own weights: 1 GB for 0.5B, 3.1 GB for 1.5B)."""
    from engine.loader import DEFAULT_MODEL
    key = model_id or DEFAULT_MODEL
    if key not in _shared_models:
        from engine.model import Qwen2Model
        _shared_models[key] = Qwen2Model.from_pretrained(key)
    return _shared_models[key]


def load_engine(name: str, model_id: str | None = None, kv_bytes: int = CONTINUOUS_KV_BYTES,
                **fake_timing: float) -> tuple[object, dict]:
    """(engine, config recorded in run.json) for "fake", "hf", "contiguous", or a continuous variant (CONTINUOUS).
    model_id and kv_bytes apply to the continuous engines; hf and contiguous are the 0.5B baselines."""
    from engine.loader import DEFAULT_MODEL
    if name in ("hf", "contiguous") and model_id not in (None, DEFAULT_MODEL):
        raise ValueError(f"{name} runs only the default model {DEFAULT_MODEL}")
    if name == "hf":
        from bench.engines.hf import MODEL, HFEngine
        return HFEngine(), {"model": MODEL, "dtype": "bfloat16", "attn": "sdpa", "batch": 1}
    if name == "contiguous":
        from engine.engine import MAX_LEN, ContiguousEngine
        return ContiguousEngine(_model()), {"model": DEFAULT_MODEL, "dtype": "bfloat16", "attn": "matmul-softmax",
                                            "kv": "contiguous", "max_len": MAX_LEN, "fused_matmuls": True,
                                            "batch": 1}
    if name in CONTINUOUS:
        from dataclasses import asdict

        from engine.continuous import ContinuousEngine
        from engine.scheduler import SchedulerConfig
        fmt, chunked = CONTINUOUS[name]
        sched = SchedulerConfig() if chunked else SchedulerConfig(chunked_prefill=False, max_num_batched_tokens=2048)
        eng = ContinuousEngine(_model(model_id), kv_bytes=kv_bytes, config=sched, kv_format=fmt)
        return eng, {"model": model_id or DEFAULT_MODEL, "dtype": "bfloat16", "kv": "paged", "kv_format": fmt,
                     "kv_bytes": kv_bytes, "block_size": 16, "num_blocks": eng.manager.num_blocks,
                     "attn": "triton decode + gather prefill", "scheduler": asdict(sched)}
    if name == "fake":
        from bench.engines.fake import FakeEngine
        return FakeEngine(**fake_timing), dict(fake_timing)
    raise ValueError(f"unknown engine {name!r}")
