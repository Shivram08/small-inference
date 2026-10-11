"""M0.2: runner mechanics. Open-loop admission, scheduled arrivals, queue depth, contract checks, and run output."""

import json
from datetime import datetime

import pytest

import bench.runner as runner
from bench.engines.fake import FakeEngine
from bench.env import sha256_file
from bench.runner import TraceRequest, load_trace, replay


def make_trace(path, rows):
    """Write (arrival_s, prompt_len, output_len) rows as a JSONL trace and return its path."""
    with open(path, "w") as f:
        for i, (arrival, p, n) in enumerate(rows):
            f.write(json.dumps({"id": i, "arrival_s": arrival, "prompt_ids": [1] * p, "output_len": n}) + "\n")
    return path


def reqs(rows):
    """Build TraceRequests from (arrival_s, prompt_len, output_len) rows."""
    return [TraceRequest(i, a, [1] * p, n) for i, (a, p, n) in enumerate(rows)]


class EngineWrapper:
    """Wraps a FakeEngine and lets a test tamper with its outputs to simulate engine bugs."""

    def __init__(self, inner, tamper):
        """tamper(out) -> out is applied to every step's output."""
        self.inner, self.tamper = inner, tamper

    def add_request(self, req_id, prompt_ids, output_len):
        """Forward to the wrapped engine."""
        self.inner.add_request(req_id, prompt_ids, output_len)

    def step(self):
        """Forward to the wrapped engine, then tamper with the output."""
        return self.tamper(self.inner.step())

    def has_unfinished(self):
        """Forward to the wrapped engine."""
        return self.inner.has_unfinished()


def test_idle_engine_admits_on_time_and_records_every_token():
    """With gaps much longer than service time, each request is admitted within ~1 ms of its scheduled arrival."""
    # Service time 10 * 1e-4 + 4 * 1e-3 = 5 ms, gaps 50 ms.
    rows = [(0.05 * k, 10, 5) for k in range(6)]
    records, steps = replay(FakeEngine(a=1e-4, b=1e-3), reqs(rows))

    lags = [rec.admit_s - rec.arrival_s for rec in records]
    print(f"\nidle admission lag: max {max(lags) * 1e6:.0f} µs, mean {sum(lags) / len(lags) * 1e6:.0f} µs")
    assert all(0 <= lag < 1e-3 for lag in lags)
    for rec, (arrival, _, n) in zip(records, rows):
        assert rec.arrival_s == arrival
        assert len(rec.token_s) == n == len(rec.output_ids)
        assert rec.admit_s < rec.token_s[0] and rec.token_s == sorted(rec.token_s)
    assert sum(s.n_tokens for s in steps) == sum(n for _, _, n in rows)


def test_busy_engine_keeps_scheduled_arrival():
    """A request arriving mid-step is admitted at the step boundary, but its recorded arrival stays the scheduled one."""
    # Request 0 occupies the engine for 1 ms + 4 * 20 ms. Request 1 arrives at 10 ms, mid-way through a decode step.
    records, _ = replay(FakeEngine(a=1e-4, b=20e-3), reqs([(0.0, 10, 5), (0.010, 10, 3)]))
    r0, r1 = records
    assert r1.arrival_s == 0.010
    assert r1.admit_s - r1.arrival_s > 5e-3  # waited for the step in progress to end
    assert r1.token_s[0] > r0.token_s[-1]  # FIFO: served only after request 0 finishes


def test_queue_depth_counts_scheduled_arrivals():
    """outstanding and waiting count requests by scheduled arrival, not by admission."""
    records, steps = replay(FakeEngine(a=1e-4, b=1e-3), reqs([(0.0, 10, 3)] * 3))
    first = steps[0]  # request 0 prefilled; requests 1 and 2 arrived but have no token yet
    assert (first.outstanding, first.waiting) == (3, 2)
    assert (steps[-1].outstanding, steps[-1].waiting) == (0, 0)
    assert all(s.waiting <= s.outstanding for s in steps)


def test_rejects_extra_tokens():
    """A token for an already-finished request is an engine bug."""
    def duplicate_final(out):
        """Emit the finishing token twice."""
        return out + [(rid, tok, False) for rid, tok, fin in out if fin]

    with pytest.raises(RuntimeError, match="finished request"):
        replay(EngineWrapper(FakeEngine(a=0, b=0), duplicate_final), reqs([(0.0, 4, 3)]))


def test_rejects_early_finish():
    """A finished flag before output_len tokens is an engine bug."""
    with pytest.raises(RuntimeError, match="finished=True"):
        replay(EngineWrapper(FakeEngine(a=0, b=0), lambda out: [(r, t, True) for r, t, _ in out]),
               reqs([(0.0, 4, 3)]))


def test_rejects_engine_going_idle_with_unfinished_requests():
    """An engine that drops a request makes the run fail instead of silently losing it."""
    inner = FakeEngine(a=0, b=0)
    eng = EngineWrapper(inner, lambda out: out)
    eng.add_request = lambda rid, p, n: None if rid == 1 else inner.add_request(rid, p, n)
    with pytest.raises(RuntimeError, match="unfinished"):
        replay(eng, reqs([(0.0, 4, 3), (0.0, 4, 3)]))


def test_load_trace_sorts_and_validates(tmp_path):
    """Traces come back sorted by arrival; duplicate ids and output_len < 2 are rejected."""
    path = make_trace(tmp_path / "t.jsonl", [(0.2, 3, 4), (0.1, 3, 4)])
    assert [r.arrival_s for r in load_trace(path)] == [0.1, 0.2]
    with pytest.raises(ValueError):
        load_trace(make_trace(tmp_path / "bad.jsonl", [(0.0, 3, 1)]))
    dup = tmp_path / "dup.jsonl"
    dup.write_text(path.read_text().replace('"id": 1', '"id": 0'))
    with pytest.raises(ValueError, match="duplicate"):
        load_trace(dup)


def test_run_writes_folder_and_excludes_warmup(tmp_path):
    """run() writes all four files; warmup requests are not recorded; env.json has the trace hash and GPU snapshots."""
    trace = make_trace(tmp_path / "small.jsonl", [(0.01 * k, 8, 4) for k in range(5)])
    run_dir = runner.run(FakeEngine(a=1e-4, b=1e-3), trace, "fake", {"a": 1e-4, "b": 1e-3},
                         results_root=tmp_path / "results", warmup=2)

    assert run_dir.name.endswith("_fake_small")
    rows = [json.loads(line) for line in open(run_dir / "requests.jsonl")]
    assert sorted(r["id"] for r in rows) == list(range(5))  # no negative warmup ids
    assert all(len(r["token_s"]) == r["output_len"] for r in rows)
    assert (run_dir / "steps.jsonl").stat().st_size > 0
    env = json.loads((run_dir / "env.json").read_text())
    assert env["static"]["trace_sha256"] == sha256_file(trace)
    assert {"gpu_before", "gpu_after"} <= env.keys()
    assert json.loads((run_dir / "run.json").read_text())["engine_config"] == {"a": 1e-4, "b": 1e-3}


def test_run_never_overwrites(tmp_path, monkeypatch):
    """A second run that would land in an existing folder fails before doing anything."""
    class FrozenDatetime:
        @staticmethod
        def now():
            """Always the same instant, so both runs pick the same folder name."""
            return datetime(2026, 1, 1, 12, 0, 0)

    monkeypatch.setattr(runner, "datetime", FrozenDatetime)
    trace = make_trace(tmp_path / "small.jsonl", [(0.0, 4, 2)])
    runner.run(FakeEngine(a=0, b=0), trace, "fake", results_root=tmp_path, warmup=0)
    with pytest.raises(FileExistsError):
        runner.run(FakeEngine(a=0, b=0), trace, "fake", results_root=tmp_path, warmup=0)
