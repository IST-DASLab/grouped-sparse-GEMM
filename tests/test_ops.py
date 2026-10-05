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
"""Tests that drive the public ops from Python, with weights packed by paired_nvfp4_kernels.reference."""

import pytest
import torch

import paired_nvfp4_kernels as pnk
from paired_nvfp4_kernels import reference as ref
from conftest import GEMM_SHAPES, requires_gpu, requires_selftest, shape_id

pytestmark = requires_gpu

ops = torch.ops.paired_nvfp4


def probe(b_act, sfb, features, max_n, K, E):
    """fp4(b_act) * ue4m3(sfb) as the GEMM reads them, [E, max_n, K] float32 on the CPU."""
    return torch.ops._paired_nvfp4_test.probe_act_layout(
        b_act.contiguous(), sfb.contiguous().view(torch.uint8), features, max_n, K, E)


def make_counts(max_n, E, device):
    step = max(1, max_n // 4)
    return torch.tensor([max(1, max_n - (e % 4) * step) for e in range(E)],
                        device=device, dtype=torch.int32)


def make_alphas(E, device):
    return torch.tensor([0.5 + 1.5 * (e / (E - 1) if E > 1 else 0.0) for e in range(E)],
                        device=device, dtype=torch.float32)


def block_constant_activation(E, max_n, K, counts, device, fill=1.0):
    """bf16 [E, max_n, K], constant per 32-block with |value| in [0.25, 4]: quantization is exact up
    to the UE4M3 scale rounding, and relative errors stay meaningful (no values near zero)."""
    x = torch.full((E, max_n, K), fill, device=device, dtype=torch.float32)
    nb = K // ref.SF_BLOCK
    for e in range(E):
        c = int(counts[e])
        mag = torch.pow(2.0, torch.rand(c, nb, device=device) * 4.0 - 2.0)
        sgn = torch.where(torch.rand(c, nb, device=device) < 0.5, -1.0, 1.0)
        x[e, :c] = (sgn * mag).repeat_interleave(ref.SF_BLOCK, dim=1)
    return x.to(torch.bfloat16)


def make_operands(M, max_n, K, E, seed, device="cuda", counts=None):
    torch.manual_seed(seed)
    if counts is None:
        counts = make_counts(max_n, E, device)
    else:
        counts = torch.tensor(counts, device=device, dtype=torch.int32)
    alphas = make_alphas(E, device)
    w = ref.prune_paired_4of8(torch.randn(E, M, K, device=device)).to(torch.bfloat16)
    w_packed, w_sf, w_dq = ref.quantize_weight(w)
    a_comp, e_meta, sfa = ops.compress(w_packed, w_sf, K)
    x = block_constant_activation(E, max_n, K, counts, device)
    gscale = torch.ones(E, device=device, dtype=torch.float32)
    b_act, sfb = ops.quant_act(x, gscale, counts, M)
    return dict(counts=counts, alphas=alphas, w_dq=w_dq, a_comp=a_comp, e_meta=e_meta, sfa=sfa,
                b_act=b_act, sfb=sfb)


def run_group_mm(o, M, max_n, K, E, tactic=None):
    out = torch.zeros(E, max_n, M, device="cuda", dtype=torch.bfloat16)
    args = (out, o["a_comp"], o["e_meta"], o["sfa"], o["b_act"], o["sfb"], o["alphas"], o["counts"],
            M, K)
    ops.group_mm(*args, *(tactic or ()))
    return out


def test_compress_shapes():
    E, M, K = 2, 256, 512
    w_packed = torch.zeros(E, M, K // 2, dtype=torch.uint8, device="cuda")
    w_sf = torch.zeros(E, M, K // ref.SF_BLOCK, dtype=torch.uint8, device="cuda")
    a_comp, e_meta, sfa = ops.compress(w_packed, w_sf, K)
    torch.cuda.synchronize()
    assert a_comp.shape[0] == E and e_meta.shape[0] == E and sfa.dim() == 1
    # At an aligned shape the 4:8 operand stores exactly half the dense bytes (still 2 fp4/byte).
    assert a_comp.numel() * 2 == w_packed.numel()


def test_compress_rejects_per16_scales():
    E, M, K = 1, 256, 256
    w_packed = torch.zeros(E, M, K // 2, dtype=torch.uint8, device="cuda")
    w_sf16 = torch.zeros(E, M, K // 16, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="ceil"):
        ops.compress(w_packed, w_sf16, K)


@requires_selftest
@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shape", GEMM_SHAPES, ids=shape_id)
def test_reference_packed_weight_through_group_mm(shape, seed):
    """Weights packed by reference.quantize_weight (the checkpoint convention) flow through
    compress + group_mm: output matches alpha * (x_dq @ w_dq^T) with the linearly dequantized
    weight, so a wrong SFA placement or a per-16 / per-32 mixup fails loudly."""
    M, max_n, K, E = shape
    o = make_operands(M, max_n, K, E, seed)
    out = run_group_mm(o, M, max_n, K, E)
    x_dq = probe(o["b_act"], o["sfb"], M, max_n, K, E).to("cuda")
    num_bad = 0
    for e in range(E):
        c = int(o["counts"][e])
        want = o["alphas"][e] * (x_dq[e, :c] @ o["w_dq"][e].float().t())
        got = out[e, :c].float()
        num_bad += int(((got - want).abs() > 1e-1 + 1e-1 * want.abs()).sum())
    assert num_bad == 0


@requires_selftest
@pytest.mark.parametrize("config_id", range(12))
def test_group_mm_zero_count_experts_leave_padding_untouched(config_id):
    """Experts with no tokens (common under MoE routing) are skipped, and no tile is scheduled
    past an expert's token count: rows at or beyond the count rounded up to 256 (the largest tile
    N of any variant) keep the caller's bytes, and the valid rows match the reference."""
    configs = pnk.group_mm_configs()
    if config_id >= len(configs) or configs[config_id].endswith("_batched"):
        pytest.skip("no such grouped variant on this arch")
    M, max_n, K, E = 512, 512, 512, 8
    if "_sk" in configs[config_id] and K // 256 < int(configs[config_id].rsplit("_sk", 1)[1]):
        pytest.skip("split-K factor exceeds the K tiles of this shape")
    counts = [0, 1, 255, 256, 257, 512, 0, 64]
    o = make_operands(M, max_n, K, E, seed=0, counts=counts)
    sentinel = -12345.0
    out = torch.full((E, max_n, M), sentinel, device="cuda", dtype=torch.bfloat16)
    ops.group_mm(out, o["a_comp"], o["e_meta"], o["sfa"], o["b_act"], o["sfb"], o["alphas"],
                 o["counts"], M, K, config_id, *pnk.valid_clusters(configs[config_id])[0])
    x_dq = probe(o["b_act"], o["sfb"], M, max_n, K, E).to("cuda")
    for e, c in enumerate(counts):
        end = min(max_n, -(-c // 256) * 256)
        assert bool((out[e, end:] == sentinel).all()), f"expert {e}: rows >= {end} were written"
        want = o["alphas"][e] * (x_dq[e, :c] @ o["w_dq"][e].float().t())
        got = out[e, :c].float()
        assert int(((got - want).abs() > 1e-1 + 1e-1 * want.abs()).sum()) == 0, f"expert {e}"


SPLIT_K_SHAPES = [
    # (M, max_n, K, counts): many tiles per slot (the reduction slots are reused over several
    # rounds), zero-token and 1-token experts, K with a remainder split, many experts.
    (4096, 256, 1024, [256, 0, 1, 130, 255, 0, 64, 200]),
    (1536, 32, 2048, [0, 3, 0, 0, 7, 0, 1, 0] * 16),
    (512, 128, 768, [128, 5, 0, 77]),
]


@requires_selftest
@pytest.mark.parametrize("shape", SPLIT_K_SHAPES, ids=lambda s: f"{s[0]}x{s[1]}x{s[2]}xE{len(s[3])}")
def test_split_k_matches_unsplit(shape):
    """Split-K variants give the unsplit product (up to fp32 summation order), leave rows past each
    expert's last tile untouched, and are deterministic across launches."""
    M, max_n, K, counts = shape
    E = len(counts)
    configs = pnk.group_mm_configs()
    split_ids = [i for i, n in enumerate(configs) if "_sk" in n]
    if not split_ids:
        pytest.skip("no split-K variants on this arch")
    o = make_operands(M, max_n, K, E, seed=1, counts=counts)
    base = run_group_mm(o, M, max_n, K, E)
    ran = 0
    for cid in split_ids:
        splits = int(configs[cid].rsplit("_sk", 1)[1])
        if -(-K // 256) < splits:
            continue
        outs = []
        for _ in range(2):
            out = torch.full((E, max_n, M), -12345.0, device="cuda", dtype=torch.bfloat16)
            ops.group_mm(out, o["a_comp"], o["e_meta"], o["sfa"], o["b_act"], o["sfb"], o["alphas"],
                         o["counts"], M, K, cid, 1, 1)
            outs.append(out)
        torch.cuda.synchronize()
        assert torch.equal(outs[0], outs[1]), f"{configs[cid]} is not deterministic"
        for e, c in enumerate(counts):
            torch.testing.assert_close(outs[0][e, :c], base[e, :c], atol=1e-2, rtol=1e-2,
                                       msg=lambda m: f"{configs[cid]}, expert {e}: {m}")
            end = min(max_n, -(-c // 256) * 256)
            assert bool((outs[0][e, end:] == -12345.0).all()), f"{configs[cid]} wrote past expert {e}"
        ran += 1
    assert ran > 0


BENIGN_SF = 0x38   # UE4M3 1.0, the fill of SFB slots no quantizer writes


@requires_selftest
@pytest.mark.parametrize("op", ["quant_act", "silu_mul_quant_act"])
@pytest.mark.parametrize("shape", [(8, 64, 1024), (3, 200, 544), (5, 1, 7168)],
                         ids=lambda s: "E{}_n{}_K{}".format(*s))
def test_quantizer_writes_benign_fill(op, shape):
    """The quantizers write the benign fill of every SFB slot they do not quantize (padded rows,
    zero-count experts, atom slop past max_n and K) themselves, with no separate fill kernel. The
    probe reads the buffer through the GEMM's layout: padded rows must hold exactly 1.0 and valid
    rows their own (never-1.0) scales; the byte count covers the slop the probe cannot address."""
    E, max_n, K = shape
    g = torch.Generator(device="cuda").manual_seed(E * max_n + K)
    counts_list = [(e * 37 + 1) % (max_n + 1) for e in range(E)]
    counts_list[0] = 0
    counts = torch.tensor(counts_list, device="cuda", dtype=torch.int32)
    # A small global scale keeps every quantized scale far below 1.0 (distinct from the fill).
    gscale = torch.full((E,), 1e-2, device="cuda")
    if op == "quant_act":
        x = (torch.randn(E, max_n, K, device="cuda", generator=g) * 2).to(torch.bfloat16)
        _, sfb = ops.quant_act(x, gscale, counts, 256)
    else:
        x2 = (torch.randn(E, max_n, 2 * K, device="cuda", generator=g) * 2).to(torch.bfloat16)
        _, sfb = ops.silu_mul_quant_act(x2, gscale, counts, 256)
    KB = K // ref.SF_BLOCK
    sfb_u8 = sfb.view(torch.uint8)
    assert int((sfb_u8 == BENIGN_SF).sum()) == sfb_u8.numel() - sum(counts_list) * KB
    ones = torch.full((E, max_n, K // 2), 0x22, device="cuda", dtype=torch.uint8)  # e2m1 1.0 x 2
    scales = probe(ones, sfb, 256, max_n, K, E)
    rows = torch.arange(max_n)[None, :, None]
    valid = rows < torch.tensor(counts_list)[:, None, None]
    assert bool((scales[~valid.expand_as(scales)] == 1.0).all()), "padded slot not benign"
    assert bool((scales[valid.expand_as(scales)] != 1.0).all()), "valid scale overwritten"


@requires_selftest
def test_quant_act_layout_isolation():
    """Block-constant input away from zero: every element reconstructs within UE4M3 rounding."""
    E, max_n, K, M = 8, 64, 1024, 1024
    counts_list = [14, 8, 11, 6, 4, 6, 4, 11]
    torch.manual_seed(0)
    counts = torch.tensor(counts_list, dtype=torch.int32, device="cuda")
    x = block_constant_activation(E, max_n, K, counts, "cuda")
    gscale = torch.ones(E, dtype=torch.float32, device="cuda")
    b_act, sfb = ops.quant_act(x, gscale, counts, M)
    recon = probe(b_act, sfb, M, max_n, K, E)
    want = x.float().cpu()
    for e, c in enumerate(counts_list):
        rel = (recon[e, :c] - want[e, :c]).abs() / want[e, :c].abs()
        assert float(rel.max()) <= 0.15, f"expert {e}"


@requires_selftest
def test_quant_act_noise_floor():
    """Random input: RMS relative error is in the healthy 4-bit range."""
    E, max_n, K, M = 8, 64, 1024, 1024
    counts_list = [14, 8, 11, 6, 4, 6, 4, 11]
    torch.manual_seed(0)
    counts = torch.tensor(counts_list, dtype=torch.int32, device="cuda")
    x = torch.randn(E, max_n, K, dtype=torch.bfloat16, device="cuda")
    b_act, sfb = ops.quant_act(x, torch.ones(E, device="cuda"), counts, M)
    recon = probe(b_act, sfb, M, max_n, K, E)
    want = x.float().cpu()
    num = sum(float(((recon[e, :c] - want[e, :c]) ** 2).sum()) for e, c in enumerate(counts_list))
    den = sum(float((want[e, :c] ** 2).sum()) for e, c in enumerate(counts_list))
    assert 0.02 < (num / den) ** 0.5 < 0.3


@pytest.mark.parametrize("shape", [(512, 256, 512, 4), (768, 192, 1024, 4)], ids=shape_id)
def test_every_tactic_matches_default(shape):
    """All tile variants and clusters compute the same product as the default tactic."""
    M, max_n, K, E = shape
    o = make_operands(M, max_n, K, E, seed=0)
    base = run_group_mm(o, M, max_n, K, E)
    configs = pnk.group_mm_configs()
    ran, ran_configs = 0, set()
    for tactic in pnk.all_tactics():
        try:
            out = run_group_mm(o, M, max_n, K, E, tactic)
        except RuntimeError as err:
            if "can_implement" in str(err) or "initialize failed" in str(err):
                continue
            raise
        torch.cuda.synchronize()
        ran += 1
        ran_configs.add(configs[tactic[0]])
        for e in range(E):
            c = int(o["counts"][e])
            torch.testing.assert_close(out[e, :c], base[e, :c], atol=1e-2, rtol=1e-2,
                                       msg=lambda m: f"tactic {tactic}, expert {e}: {m}")
    # Every tile variant runs; only split-K variants may skip, when K has fewer tiles than splits.
    for name in configs:
        splits = int(name.rsplit("_sk", 1)[1]) if "_sk" in name else 1
        assert name in ran_configs or K // 256 < splits, f"{name} never ran"
    assert ran >= len(configs) - sum(1 for n in configs if n not in ran_configs)


def test_group_mm_rejects_wrong_features():
    M, max_n, K, E = 256, 128, 256, 2
    o = make_operands(M, max_n, K, E, seed=0)
    out = torch.zeros(E, max_n, 2 * M, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="compress"):
        ops.group_mm(out, o["a_comp"], o["e_meta"], o["sfa"], o["b_act"], o["sfb"], o["alphas"],
                     o["counts"], 2 * M, K)


def test_group_mm_rejects_bad_cluster():
    M, max_n, K, E = 256, 128, 256, 2
    o = make_operands(M, max_n, K, E, seed=0)
    # SM100: config 0 is a 2SM kernel and needs an even cluster_m. SM120: only 1x1 clusters.
    bad, msg = {100: ((0, 1, 1), "even cluster_m"), 120: ((0, 2, 1), "1x1 cluster")}[pnk.loaded_arch()]
    with pytest.raises(RuntimeError, match=msg):
        run_group_mm(o, M, max_n, K, E, tactic=bad)
