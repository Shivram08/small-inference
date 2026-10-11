"""M0.5: trace building on a small synthetic ShareGPT file with the real Qwen tokenizer."""

import json

import numpy as np
import pytest

from bench.runner import load_trace
from bench.trace import build_traces, load_tokenizer, sample_requests


@pytest.fixture(scope="module")
def tok():
    """The cached Qwen tokenizer (skips if the model was never downloaded)."""
    try:
        return load_tokenizer()
    except OSError:
        pytest.skip("Qwen tokenizer not in the local HF cache")


def conv(*turns):
    """A ShareGPT conversation from (role, text) pairs."""
    return {"id": "x", "conversations": [{"from": r, "value": v} for r, v in turns]}


GOOD = [conv(("human", f"Question number {i}?"), ("gpt", f"This is answer number {i}, short.")) for i in range(30)]
BAD = {
    "not_human_gpt": conv(("gpt", "a continuation chunk"), ("human", "more")),
    "empty_text": conv(("human", "   "), ("gpt", "an answer to nothing at all")),
    "prompt_too_long": conv(("human", "word " * 1100), ("gpt", "a reply to a very long prompt")),
    "output_too_short": conv(("human", "hi"), ("gpt", "ok")),
    "output_too_long": conv(("human", "write a lot"), ("gpt", "word " * 600)),
}


def write_source(tmp_path, conversations):
    """Write a ShareGPT-format JSON file."""
    path = tmp_path / "sharegpt.json"
    path.write_text(json.dumps(conversations))
    return path


@pytest.mark.parametrize("reason", BAD)
def test_filters_reject_each_bad_case(tok, reason):
    """Each bad conversation is skipped, and counted under its own reason."""
    with pytest.raises(ValueError, match=f"'{reason}': 1"):
        sample_requests([BAD[reason]], tok, n=1, seed=0)


def test_good_conversations_all_kept(tok):
    """With only good conversations, every one scanned is kept."""
    sample, stats = sample_requests(GOOD, tok, n=len(GOOD), seed=0)
    assert len(sample) == stats["scanned"] == stats["kept"] == len(GOOD)


def test_prompt_is_chat_templated_and_output_len_is_reply_tokens(tok):
    """prompt_ids decode to the templated human turn; output_len is the raw token count of the reply."""
    (prompt_ids, output_len), = sample_requests(GOOD[:1], tok, n=1, seed=0)[0]
    text = tok.decode(prompt_ids)
    assert "<|im_start|>user\nQuestion number 0?<|im_end|>" in text and text.endswith("<|im_start|>assistant\n")
    assert output_len == len(tok("This is answer number 0, short.", add_special_tokens=False).input_ids)


def test_too_few_requests_raises(tok):
    """Asking for more requests than pass the filters is an error, not a short trace."""
    with pytest.raises(ValueError, match="pass the filters"):
        sample_requests(GOOD[:3] + list(BAD.values()), tok, n=5, seed=0)


def test_rates_share_sample_and_rescale_arrivals(tmp_path, tok):
    """All traces from one (n, seed) have identical requests; arrivals at rate 2r are exactly half those at rate r."""
    src = write_source(tmp_path, GOOD + list(BAD.values()))
    paths = build_traces(src, tmp_path / "traces", n=20, seed=3, rates=[0.5, 1.0], burst=True, tokenizer=tok)
    slow, fast, burst = (load_trace(p) for p in paths)
    assert [(r.prompt_ids, r.output_len) for r in slow] == [(r.prompt_ids, r.output_len) for r in fast] \
        == [(r.prompt_ids, r.output_len) for r in burst]
    np.testing.assert_allclose([r.arrival_s for r in fast], [r.arrival_s / 2 for r in slow], rtol=1e-12)
    assert all(r.arrival_s == 0.0 for r in burst)

    meta = json.loads(paths[0].with_suffix(".meta.json").read_text())
    assert meta["rate_req_s"] == 0.5 and meta["sample_seed"] == 3 and meta["arrival_seed"] == 4
    assert meta["tokenizer"]["name"] == "Qwen/Qwen2.5-0.5B-Instruct" and meta["tokenizer"]["revision"]
    assert meta["sampling_stats"]["kept"] == 20 and meta["filters"]["max_prompt"] == 1024


def test_same_seed_reproduces_byte_identical_trace(tmp_path, tok):
    """Rebuilding with the same source, n and seed gives the same bytes."""
    src = write_source(tmp_path, GOOD + list(BAD.values()))
    a, = build_traces(src, tmp_path / "a", n=10, seed=7, rates=[1.0], burst=False, tokenizer=tok)
    b, = build_traces(src, tmp_path / "b", n=10, seed=7, rates=[1.0], burst=False, tokenizer=tok)
    assert a.read_bytes() == b.read_bytes()


def test_never_overwrites(tmp_path, tok):
    """Building a trace that already exists fails before writing anything."""
    src = write_source(tmp_path, GOOD)
    build_traces(src, tmp_path, n=5, seed=0, rates=[], burst=True, tokenizer=tok)
    with pytest.raises(FileExistsError):
        build_traces(src, tmp_path, n=5, seed=0, rates=[], burst=True, tokenizer=tok)
