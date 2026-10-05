# Copyright (C) 2026 Kwanhee Lee and Dan Alistarh. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""group_mm_swiglu_quant (GEMM1 with SwiGLU + GEMM2 input quantization in the epilogue) against
group_mm -> silu_mul_quant_act, byte for byte."""

import pytest
import torch

import paired_nvfp4_kernels as pnk
from paired_nvfp4_kernels import reference as ref
from conftest import requires_gpu

ops = torch.ops.paired_nvfp4
requires_fused = pytest.mark.skipif(
    not pnk.fused_swiglu_tactics(), reason="no fused SwiGLU epilogue on this arch")

# (E, N intermediate, K hidden, cap)
# (3, 320, ...): 2N = 640 is a multiple of 128 but not 256, so a 256-row tile covers M partially.
SHAPES = [(4, 256, 512, 64), (8, 768, 1024, 37), (2, 128, 2048, 300), (3, 320, 512, 200)]


def operands(E, N, K, cap, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = ref.prune_paired_4of8(torch.randn(E, 2 * N, K, device="cuda", generator=g))
    w_packed, w_bs, _ = ref.quantize_weight(w)
    counts = torch.randint(0, cap + 1, (E,), device="cuda", dtype=torch.int32, generator=g)
    counts[0] = cap          # a full expert
    if E > 2:
        counts[1] = 0        # an empty one
    x = (torch.randn(E, cap, K, device="cuda", generator=g) * 2).to(torch.bfloat16)
    g1 = torch.rand(E, device="cuda", generator=g) + 0.5
    b_act, sfb = ops.quant_act(x, g1, counts, 2 * N)
    alphas = torch.rand(E, device="cuda", generator=g) * 0.02 + 0.005
    g2 = torch.rand(E, device="cuda", generator=g) * 8 + 1
    return w_packed, w_bs, counts, b_act, sfb, alphas, g2


@requires_gpu
@requires_fused
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_fused_swiglu_matches_unfused(shape):
    E, N, K, cap = shape
    w_packed, w_bs, counts, b_act, sfb, alphas, g2 = operands(E, N, K, cap, seed=sum(shape))

    a, em, sfa = ops.compress(w_packed, w_bs, K)
    c1 = torch.empty(E, cap, 2 * N, device="cuda", dtype=torch.bfloat16)
    ops.group_mm(c1, a, em, sfa, b_act, sfb, alphas, counts, 2 * N, K)
    b_ref, sf_ref = ops.silu_mul_quant_act(c1, g2, counts, K)

    perm = pnk.interleave_gate_up_rows(N, device="cuda")
    a_i, em_i, sfa_i = ops.compress(w_packed[:, perm].contiguous(), w_bs[:, perm].contiguous(), K)
    ran = 0
    for tactic in pnk.fused_swiglu_tactics():
        try:
            b_f, sf_f = ops.group_mm_swiglu_quant(a_i, em_i, sfa_i, b_act, sfb, alphas, counts, g2,
                                                  2 * N, K, *tactic)
        except RuntimeError as exc:
            if "can_implement" in str(exc):
                continue
            raise
        ran += 1
        torch.testing.assert_close(sf_f, sf_ref, rtol=0, atol=0, msg=f"sfb, tactic {tactic}")
        for e in range(E):
            n = int(counts[e])
            torch.testing.assert_close(b_f[e, :n], b_ref[e, :n], rtol=0, atol=0,
                                       msg=f"b_act expert {e}, tactic {tactic}")
    assert ran > 0


def test_interleave_gate_up_rows():
    perm = pnk.interleave_gate_up_rows(128)
    assert perm[:32].tolist() == list(range(32))
    assert perm[32:64].tolist() == list(range(128, 160))
    assert perm[64:96].tolist() == list(range(32, 64))
    assert sorted(perm.tolist()) == list(range(256))


@requires_gpu
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_silu_mul_quant_interleaved(shape):
    """silu_mul_quant_act on the interleaved GEMM1 output (the fused-epilogue weight order) gives
    the same bytes as on the plain [gate | up] one, so one compressed W13 serves both paths."""
    E, N, K, cap = shape
    g = torch.Generator(device="cuda").manual_seed(2)
    c1 = (torch.randn(E, cap, 2 * N, device="cuda", generator=g) * 3).to(torch.bfloat16)
    counts = torch.randint(0, cap + 1, (E,), device="cuda", dtype=torch.int32, generator=g)
    g2 = torch.rand(E, device="cuda", generator=g) + 0.5
    perm = pnk.interleave_gate_up_rows(N, device="cuda")
    b0, s0 = ops.silu_mul_quant_act(c1, g2, counts, K)
    b1, s1 = ops.silu_mul_quant_act(c1[..., perm].contiguous(), g2, counts, K, True)
    torch.testing.assert_close(s1, s0, rtol=0, atol=0)
    for e in range(E):
        n = int(counts[e])
        torch.testing.assert_close(b1[e, :n], b0[e, :n], rtol=0, atol=0)
