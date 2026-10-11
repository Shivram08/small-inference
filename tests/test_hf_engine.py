"""M0.7: the HF baseline adapter is greedy, ignores EOS, follows the Engine contract, and matches HF generate()."""

import pytest
import torch

from bench.engines.base import Engine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


@pytest.fixture(scope="module")
def engine():
    """One loaded HFEngine shared by the tests in this module."""
    from bench.engines.hf import HFEngine
    try:
        return HFEngine()
    except OSError:
        pytest.skip("Qwen2.5-0.5B-Instruct not in the local HF cache")


@pytest.fixture(scope="module")
def prompts():
    """Two chat-templated prompts of different lengths."""
    from bench.trace import load_tokenizer
    tok = load_tokenizer()
    texts = ["What is the capital of France?", "Write a haiku about GPUs, then explain each line in detail."]
    return [tok(tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False,
                                        add_generation_prompt=True), add_special_tokens=False).input_ids
            for t in texts]


def drain(engine):
    """Step until done; return the (req_id, token_id, finished) triples."""
    out = []
    while engine.has_unfinished():
        out.extend(engine.step())
    return out


NEVER_SAMPLED = 151935  # padded vocab row (tokenizer has 151,665 ids, embedding 151,936): untrained, never argmax


def reference_greedy(model, prompt_ids, n):
    """HF generate() with sampling and repetition penalty off and EOS stopping disabled: exactly n argmax tokens.

    eos_token_id=None does not work (generate falls back to the model's EOS ids and stops early), and
    min_new_tokens would mask EOS logits and change the argmax. So EOS is pointed at a token that is never sampled.
    """
    from transformers import GenerationConfig
    cfg = GenerationConfig(do_sample=False, max_new_tokens=n, repetition_penalty=1.0,
                           eos_token_id=NEVER_SAMPLED, pad_token_id=NEVER_SAMPLED)
    ids = torch.tensor([prompt_ids], device="cuda")
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=cfg)
    ref = out[0, len(prompt_ids):].tolist()
    assert len(ref) == n and NEVER_SAMPLED not in ref
    return ref


def test_conforms_to_protocol(engine):
    """HFEngine satisfies the Engine protocol."""
    assert isinstance(engine, Engine)


def test_matches_hf_generate_and_fifo(engine, prompts):
    """Tokens equal greedy generate() for each prompt; requests run FIFO with one finished flag each."""
    n = 40
    for i, p in enumerate(prompts):
        engine.add_request(i, p, n)
    out = drain(engine)
    assert [rid for rid, _, _ in out] == [0] * n + [1] * n
    for i, p in enumerate(prompts):
        mine = [tok for rid, tok, _ in out if rid == i]
        ref = reference_greedy(engine.model, p, n)
        assert mine == ref, f"prompt {i} diverges at token {next(k for k, (a, b) in enumerate(zip(mine, ref)) if a != b)}"
        assert [fin for rid, _, fin in out if rid == i] == [False] * (n - 1) + [True]


def test_ignores_eos(engine, prompts):
    """A short answer that would hit EOS keeps generating to exactly output_len tokens."""
    eos = {151645, 151643}
    engine.add_request(0, prompts[0], 120)  # "Paris..." ends well before 120 tokens
    tokens = [tok for _, tok, _ in drain(engine)]
    assert len(tokens) == 120
    assert eos & set(tokens), "expected the model to emit EOS at least once within 120 tokens"


def test_frees_cache_and_idles(engine, prompts):
    """After a request finishes, the engine holds no cache and an idle step returns []."""
    engine.add_request(0, prompts[0], 3)
    drain(engine)
    assert engine.running is None and not engine.has_unfinished() and engine.step() == []
