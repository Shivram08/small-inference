"""Build trace files: JSONL with one request per row, {"id", "arrival_s", "prompt_ids", "output_len"}, plus a
<name>.meta.json sidecar recording how the trace was made. Traces are committed and never edited or overwritten.

Requests come from ShareGPT V3: conversations whose first two turns are human -> gpt. The prompt is the human turn
in the Qwen chat template (default system prompt, generation prompt appended; adds 29 tokens), tokenized once.
output_len is the token length of the gpt reply; engines generate exactly that many tokens and ignore EOS.
One seeded request sample is shared by every trace built from the same (n, seed). Each rate gets Poisson arrivals
from the same arrival seed, so arrival times at different rates are exact rescalings of each other.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bench.env import sha256_file

DEFAULT_SOURCE = Path("data/raw/ShareGPT_V3_unfiltered_cleaned_split.json")
SOURCE_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"
FILTERS = {"max_prompt": 1024, "min_output": 4, "max_output": 512, "max_total": 2048}


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


def load_tokenizer(name: str = TOKENIZER):
    """The Qwen tokenizer from the local HF cache."""
    from transformers import AutoTokenizer  # deferred: slow import, not needed for arrivals
    return AutoTokenizer.from_pretrained(name, local_files_only=True)


def sample_requests(conversations: list[dict], tokenizer, n: int, seed: int,
                    filters: dict = FILTERS) -> tuple[list[tuple[list[int], int]], dict]:
    """Shuffle with `seed`, then tokenize until n requests pass the filters. Returns (prompt_ids, output_len) pairs
    in sample order and counts of why candidates were skipped."""
    stats = {"conversations": len(conversations), "not_human_gpt": 0, "empty_text": 0, "prompt_too_long": 0,
             "output_too_short": 0, "output_too_long": 0, "total_too_long": 0, "scanned": 0, "kept": 0}
    sample = []
    for idx in np.random.default_rng(seed).permutation(len(conversations)):
        if len(sample) == n:
            break
        stats["scanned"] += 1
        turns = conversations[idx]["conversations"]
        if len(turns) < 2 or turns[0]["from"] != "human" or turns[1]["from"] != "gpt":
            stats["not_human_gpt"] += 1
            continue
        prompt, reply = turns[0]["value"], turns[1]["value"]
        if not prompt.strip() or not reply.strip():
            stats["empty_text"] += 1
            continue
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        prompt_ids = tokenizer(text, add_special_tokens=False).input_ids
        output_len = len(tokenizer(reply, add_special_tokens=False).input_ids)
        if len(prompt_ids) > filters["max_prompt"]:
            stats["prompt_too_long"] += 1
        elif output_len < filters["min_output"]:
            stats["output_too_short"] += 1
        elif output_len > filters["max_output"]:
            stats["output_too_long"] += 1
        elif len(prompt_ids) + output_len > filters["max_total"]:
            stats["total_too_long"] += 1
        else:
            sample.append((prompt_ids, output_len))
    stats["kept"] = len(sample)
    if len(sample) < n:
        raise ValueError(f"only {len(sample)} of {n} requests pass the filters: {stats}")
    return sample, stats


def length_stats(values) -> dict:
    """Mean, percentiles and extremes of a list of token counts."""
    x = np.asarray(values)
    return {"mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "min": int(x.min()), "max": int(x.max()), "sum": int(x.sum())}


def tokenizer_revision(tokenizer) -> str | None:
    """Commit hash of the cached tokenizer snapshot, if it can be found."""
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(tokenizer.name_or_path, local_files_only=True)).name
    except Exception:
        return None


def build_traces(source: Path, out_dir: Path, n: int, seed: int, rates: list[float], burst: bool,
                 tokenizer=None) -> list[Path]:
    """Sample n requests once and write one trace per rate (and a burst trace), each with a sidecar."""
    source, out_dir = Path(source), Path(out_dir)
    names = [f"sharegpt_n{n}_seed{seed}_rate{r:g}" for r in rates] + ([f"sharegpt_n{n}_seed{seed}_burst"] * burst)
    paths = [out_dir / f"{name}.jsonl" for name in names]
    existing = [p for p in paths if p.exists() or p.with_suffix(".meta.json").exists()]
    if existing:
        raise FileExistsError(f"traces are never overwritten: {existing}")

    tokenizer = tokenizer or load_tokenizer()
    with open(source) as f:
        conversations = json.load(f)
    sample, stats = sample_requests(conversations, tokenizer, n, seed)
    del conversations
    prompts, output_lens = [p for p, _ in sample], [o for _, o in sample]
    arrival_seed = seed + 1

    repo = str(Path(__file__).resolve().parent.parent)
    commit = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "-C", repo, "status", "--porcelain", "--", "bench"], capture_output=True, text=True)
    common = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit or None,
        "git_dirty_bench": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "source": {"repo": SOURCE_REPO, "file": source.name, "sha256": sha256_file(source)},
        "tokenizer": {"name": tokenizer.name_or_path, "revision": tokenizer_revision(tokenizer)},
        "prompt_format": "Qwen chat template, default system prompt, add_generation_prompt=True",
        "selection": "first two turns human -> gpt; prompt = human turn, output_len = tokens of gpt reply",
        "filters": FILTERS, "n": n, "sample_seed": seed, "sampling_stats": stats,
        "prompt_len": length_stats([len(p) for p in prompts]), "output_len": length_stats(output_lens),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for path, rate in zip(paths, list(rates) + [None] * burst):
        arrivals = np.zeros(n) if rate is None else poisson_arrivals(rate, n, arrival_seed)
        write_trace(path, arrivals, prompts, output_lens)
        meta = {**common, "rate_req_s": rate, "burst": rate is None,
                "arrival_seed": None if rate is None else arrival_seed,
                "arrival_span_s": float(arrivals[-1]), "trace_sha256": sha256_file(path)}
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return paths


def main() -> None:
    """CLI: build ShareGPT traces for the given rates and/or a burst trace."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out-dir", type=Path, default=Path("traces"))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rates", type=float, nargs="*", default=[])
    p.add_argument("--burst", action="store_true")
    args = p.parse_args()
    if not args.rates and not args.burst:
        p.error("give --rates and/or --burst")
    for path in build_traces(args.source, args.out_dir, args.n, args.seed, args.rates, args.burst):
        print(path)


if __name__ == "__main__":
    main()
