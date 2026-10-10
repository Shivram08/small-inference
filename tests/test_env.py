"""Environment sanity checks. Rerun after any driver, CUDA, torch, or triton upgrade."""

import pytest
import torch
import triton
import triton.language as tl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")


def test_cuda_device():
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"\n{props.name}: cc {props.major}.{props.minor}, {props.multi_processor_count} SMs, "
          f"{free / 2**30:.2f} / {total / 2**30:.2f} GiB free")
    assert (props.major, props.minor) == (8, 6)
    assert torch.cuda.is_bf16_supported()
    assert f"sm_{props.major}{props.minor}" in torch.cuda.get_arch_list()


def test_bf16_matmul():
    torch.manual_seed(0)
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.randn(4096, 4096, device="cuda")
    ref = a.double() @ b.double()
    out = (a.bfloat16() @ b.bfloat16()).double()
    rel = ((out - ref).norm() / ref.norm()).item()
    print(f"\nbf16 matmul 4096^3 relative error: {rel:.2e}")
    # Input rounding to bf16 (8-bit mantissa) gives about 2^-9 * sqrt(2) relative error per product,
    # and the output is rounded to bf16 again, so expect a few 1e-3.
    assert rel < 1e-2


@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@pytest.mark.parametrize("n", [2**24, 2**24 + 17, 1])  # exact multiple of BLOCK, ragged tail, single element
def test_triton_vector_add(n):
    x = torch.randn(n, device="cuda")
    y = torch.randn(n, device="cuda")
    out = torch.empty_like(x)
    BLOCK = 1024
    _add_kernel[(triton.cdiv(n, BLOCK),)](x, y, out, n, BLOCK=BLOCK)
    torch.testing.assert_close(out, x + y, rtol=0, atol=0)
