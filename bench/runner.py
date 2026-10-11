"""Open-loop trace replay against an Engine, in wall-clock time. Writes raw per-request and per-step records.

Single process, single thread. Requests are admitted between steps; each keeps its *scheduled* arrival time,
so time spent waiting for the current step to end counts as queueing delay. Token timestamps are taken after
step() returns, which by the Engine contract is after the sampled tokens reach the CPU.
"""

import argparse
import bisect
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from bench.engines.base import Engine
from bench.engines.fake import wait_until
from bench.env import gpu_snapshot, static_env

WARMUP_OUTPUT_LEN = 8
MAX_CONSECUTIVE_EMPTY_STEPS = 100_000


@dataclass
class TraceRequest:
    id: int
    arrival_s: float
    prompt_ids: list[int]
    output_len: int


@dataclass
class RequestRecord:
    id: int
    arrival_s: float  # scheduled, relative to t0
    prompt_len: int
    output_len: int
    admit_s: float | None = None  # when add_request was called, relative to t0
    token_s: list[float] = field(default_factory=list)  # one timestamp per output token
    output_ids: list[int] = field(default_factory=list)

    @property
    def done(self) -> bool:
        """True once all output_len tokens have been received."""
        return len(self.token_s) == self.output_len


@dataclass
class StepRecord:
    start_s: float
    end_s: float
    n_tokens: int
    outstanding: int  # arrived by end_s (scheduled time) and not finished
    waiting: int  # arrived by end_s and no first token yet


def load_trace(path: Path) -> list[TraceRequest]:
    """Read a JSONL trace, validate it, and return requests sorted by arrival time."""
    reqs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                reqs.append(TraceRequest(row["id"], row["arrival_s"], row["prompt_ids"], row["output_len"]))
    if len({r.id for r in reqs}) != len(reqs):
        raise ValueError(f"{path}: duplicate request ids")
    for r in reqs:
        if r.output_len < 2 or not r.prompt_ids or r.arrival_s < 0:
            raise ValueError(f"{path}: invalid request {r.id}")
    return sorted(reqs, key=lambda r: r.arrival_s)


def warm_up(engine: Engine, reqs: list[TraceRequest], n: int) -> None:
    """Run n short requests (negative ids) to completion so one-off startup costs land outside the timed run."""
    for k, r in enumerate(reqs[:n]):
        engine.add_request(-(k + 1), r.prompt_ids, min(r.output_len, WARMUP_OUTPUT_LEN))
    while engine.has_unfinished():
        engine.step()


def replay(engine: Engine, reqs: list[TraceRequest]) -> tuple[list[RequestRecord], list[StepRecord]]:
    """Replay reqs open-loop starting now. Returns raw records; raises on any engine contract violation."""
    records = [RequestRecord(r.id, r.arrival_s, len(r.prompt_ids), r.output_len) for r in reqs]
    by_id = {rec.id: rec for rec in records}
    arrivals = [r.arrival_s for r in reqs]
    steps: list[StepRecord] = []
    n, i, n_started, n_finished, empty_streak = len(reqs), 0, 0, 0, 0

    t0 = time.perf_counter()
    while i < n or engine.has_unfinished():
        now = time.perf_counter() - t0
        while i < n and arrivals[i] <= now:
            engine.add_request(reqs[i].id, reqs[i].prompt_ids, reqs[i].output_len)
            records[i].admit_s = time.perf_counter() - t0
            i += 1

        if engine.has_unfinished():
            start = time.perf_counter() - t0
            out = engine.step()
            end = time.perf_counter() - t0
            for req_id, token_id, finished in out:
                rec = by_id.get(req_id)
                if rec is None or rec.admit_s is None or rec.done:
                    raise RuntimeError(f"token for unknown, unadmitted, or finished request {req_id}")
                if not rec.token_s:
                    n_started += 1
                rec.token_s.append(end)
                rec.output_ids.append(token_id)
                if finished != rec.done:
                    raise RuntimeError(f"request {req_id}: finished={finished} at token {len(rec.token_s)}"
                                       f" of {rec.output_len}")
                n_finished += finished
            empty_streak = 0 if out else empty_streak + 1
            if empty_streak > MAX_CONSECUTIVE_EMPTY_STEPS:
                raise RuntimeError("engine reports unfinished work but keeps returning no tokens")
            arrived = bisect.bisect_right(arrivals, end)
            steps.append(StepRecord(start, end, len(out), arrived - n_finished, arrived - n_started))
        elif i < n:
            wait_until(t0 + arrivals[i])

    unfinished = [rec.id for rec in records if not rec.done]
    if unfinished:
        raise RuntimeError(f"engine went idle with {len(unfinished)} unfinished requests, e.g. {unfinished[:5]}")
    return records, steps


def write_jsonl(path: Path, rows: list) -> None:
    """Write dataclass rows as one JSON object per line."""
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(asdict(row)) + "\n")


def run(engine: Engine, trace_path: Path, engine_name: str, engine_config: dict | None = None,
        results_root: Path = Path("results"), warmup: int = 2) -> Path:
    """Warm up, replay the trace, and write requests.jsonl, steps.jsonl, env.json, run.json to a new folder."""
    trace_path = Path(trace_path)
    reqs = load_trace(trace_path)
    run_dir = Path(results_root) / f"{datetime.now():%Y%m%d-%H%M%S}_{engine_name}_{trace_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a results folder

    env = {"static": static_env(trace_path)}
    warm_up(engine, reqs, warmup)
    env["gpu_before"] = gpu_snapshot()
    wall_start = time.time()
    records, steps = replay(engine, reqs)
    wall_end = time.time()
    env["gpu_after"] = gpu_snapshot()

    write_jsonl(run_dir / "requests.jsonl", records)
    write_jsonl(run_dir / "steps.jsonl", steps)
    (run_dir / "env.json").write_text(json.dumps(env, indent=2))
    (run_dir / "run.json").write_text(json.dumps({
        "engine": engine_name,
        "engine_config": engine_config or {},
        "trace": str(trace_path),
        "n_requests": len(reqs),
        "warmup_requests": warmup,
        "wall_start_unix": wall_start,
        "wall_end_unix": wall_end,
    }, indent=2))
    return run_dir


def main() -> None:
    """CLI: replay a trace against the fake engine (real engines are added as they are built)."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--engine", choices=["fake"], default="fake")
    p.add_argument("--a", type=float, default=1e-4, help="fake: prefill seconds per prompt token")
    p.add_argument("--b", type=float, default=5e-3, help="fake: seconds per decode step")
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()

    from bench.engines.fake import FakeEngine
    engine = FakeEngine(a=args.a, b=args.b)
    run_dir = run(engine, args.trace, args.engine, {"a": args.a, "b": args.b}, args.results_root, args.warmup)
    print(run_dir)


if __name__ == "__main__":
    main()
