"""Phase 5 quantization reference: round-trip bounds, scale exactness, edge cases, and why per-channel K helps."""

import torch

from engine.quant import dequantize, quantize_per_channel, quantize_per_token, relative_error


def test_per_token_roundtrip_bound():
    """Each element is within half a quantization step of the original: |x - q*s| <= s/2 (plus fp16 scale slack)."""
    torch.manual_seed(0)
    x = torch.randn(37, 2, 64) * torch.rand(37, 2, 1) * 5
    q, s = quantize_per_token(x)
    assert q.dtype == torch.int8 and s.dtype == torch.float16 and s.shape == (37, 2)
    err = (x - dequantize(q, s[..., None])).abs()
    assert (err <= s.float()[..., None] * 0.5 * (1 + 1e-3) + 1e-7).all()
    assert q.abs().max() <= 127 and (q.abs().amax(-1) == 127).float().mean() > 0.9  # absmax hits the grid edge


def test_per_channel_roundtrip_bound():
    """One scale per (head, channel) over a block's 16 tokens; the same half-step bound per channel."""
    torch.manual_seed(1)
    x = torch.randn(16, 2, 64) * torch.rand(1, 2, 64) * 3  # [tokens, heads, channels]
    q, s = quantize_per_channel(x, dim=0)
    assert s.shape == (2, 64)
    err = (x - dequantize(q, s[None])).abs()
    assert (err <= s.float()[None] * 0.5 * (1 + 1e-3) + 1e-7).all()


def test_zeros_and_bf16_input():
    """An all-zero vector gets scale 0 and dequantizes to exact zeros; bf16 input works."""
    x = torch.zeros(3, 2, 64, dtype=torch.bfloat16)
    x[1, 0] = torch.linspace(-2, 2, 64)
    q, s = quantize_per_token(x)
    assert s[0, 0] == 0 and s[2, 1] == 0 and (q[0] == 0).all()
    assert torch.equal(dequantize(q, s[..., None])[0], torch.zeros(2, 64))
    assert relative_error(x, dequantize(q, s[..., None])) < 0.01


def test_per_channel_beats_per_token_with_outlier_channels():
    """Keys with a few large-magnitude channels shared by every token (the KIVI observation): per-token scales are
    set by the outliers and crush the other channels; per-channel scales isolate them."""
    torch.manual_seed(2)
    k = torch.randn(16, 2, 64)
    k[:, :, [3, 17, 40]] *= 30  # outlier channels
    qt, st = quantize_per_token(k)
    qc, sc = quantize_per_channel(k, dim=0)
    normal = [c for c in range(64) if c not in (3, 17, 40)]
    err_tok = relative_error(k[..., normal], dequantize(qt, st[..., None])[..., normal])
    err_ch = relative_error(k[..., normal], dequantize(qc, sc[None])[..., normal])
    print(f"\nnon-outlier channels: per-token error {err_tok:.3f}, per-channel {err_ch:.4f}")
    assert err_ch < err_tok / 5
