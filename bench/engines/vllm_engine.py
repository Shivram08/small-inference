"""vLLM adapter for the Engine protocol (Phase 6). Runs only in the separate vLLM venv (vLLM pins its own torch).

Same workload contract as every other engine: token-id prompts, greedy (temperature 0), EOS ignored, exactly
output_len tokens (min_tokens = max_tokens = output_len, ignore_eos), no prefix caching (none of this engine's
formats share prefixes). Each step() is one LLMEngine.step(); new tokens are found by diffing each request's
cumulative output length, so a step that emits several tokens for a request (it should not, without speculative
decoding) would still be reported correctly, in order.

Fairness (CLAUDE.md): match the KV budget to this engine's pool by choosing gpu_memory_utilization so vLLM's KV
cache comes out close to it, and record both sizes; compare both with CUDA graphs ("vllm") and eager ("vllm-eager").
Built against vLLM's public API only, not its source (the project rule until comparisons begin).
"""

import os

from engine.loader import DEFAULT_MODEL

MAX_MODEL_LEN = 2048  # the trace filter: prompt + output <= 2,048


class VLLMEngine:
    def __init__(self, model: str = DEFAULT_MODEL, enforce_eager: bool = False, gpu_memory_utilization: float = 0.6,
                 max_num_seqs: int = 64, max_num_batched_tokens: int = 512):
        """Start a vLLM LLMEngine with this engine's scheduler limits (64 sequences, 512-token budget)."""
        from vllm import EngineArgs, LLMEngine

        self.args = dict(model=model, dtype="bfloat16", enforce_eager=enforce_eager,
                         gpu_memory_utilization=gpu_memory_utilization, max_model_len=MAX_MODEL_LEN,
                         max_num_seqs=max_num_seqs, max_num_batched_tokens=max_num_batched_tokens,
                         enable_prefix_caching=False, enable_chunked_prefill=True, seed=0)
        self.engine = LLMEngine.from_engine_args(EngineArgs(**self.args))
        self._emitted: dict[str, int] = {}
        self._want: dict[str, int] = {}

    def add_request(self, req_id: int, prompt_ids: list[int], output_len: int) -> None:
        """Queue a request: greedy, EOS ignored, exactly output_len tokens."""
        from vllm import SamplingParams

        if not prompt_ids:
            raise ValueError("prompt_ids must be non-empty")
        if output_len < 2:
            raise ValueError(f"output_len must be >= 2, got {output_len}")
        params = SamplingParams(temperature=0.0, max_tokens=output_len, min_tokens=output_len, ignore_eos=True)
        key = str(req_id)
        self.engine.add_request(key, {"prompt_token_ids": list(prompt_ids)}, params)
        self._emitted[key], self._want[key] = 0, output_len

    def step(self) -> list[tuple[int, int, bool]]:
        """One vLLM engine step; the tokens it produced, with finished on each request's last token."""
        out = []
        for ro in self.engine.step():
            key = ro.request_id
            if key not in self._emitted:
                continue
            tokens = list(ro.outputs[0].token_ids)
            new = tokens[self._emitted[key]:]
            self._emitted[key] = len(tokens)
            for j, tok in enumerate(new):
                last = j == len(new) - 1
                out.append((int(key), int(tok), bool(ro.finished and last)))
            if ro.finished:
                if len(tokens) != self._want[key]:
                    raise RuntimeError(f"request {key}: vLLM produced {len(tokens)} tokens, wanted {self._want[key]}")
                del self._emitted[key], self._want[key]
        return out

    def has_unfinished(self) -> bool:
        """True while vLLM holds unfinished requests."""
        return self.engine.has_unfinished_requests()

    def config(self) -> dict:
        """What run.json records: the engine arguments, the vLLM version, and the KV cache size if exposed."""
        import vllm

        cfg = {"engine": "vllm", "vllm_version": vllm.__version__, **self.args,
               "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")}
        for path in ("vllm_config.cache_config", "cache_config"):
            obj = self.engine
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            cfg["num_gpu_blocks"] = getattr(obj, "num_gpu_blocks", None)
            cfg["block_size"] = getattr(obj, "block_size", None)
            break
        return cfg
