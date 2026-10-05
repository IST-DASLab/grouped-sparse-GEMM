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

"""dispatch_plan / scatter_quant_act_planned / moe_finalize against the torch dispatch they
replace (vLLM's capture-safe batched prepare/finalize), which the reference functions below
reproduce operation for operation."""

import pytest
import torch

from conftest import requires_gpu

ops = torch.ops.paired_nvfp4

# (T, topk, global experts, EP size, EP rank, cap or None = min(T, 1 << 30))
PLAN_CASES = [
    (1, 8, 128, 1, 0, None),        # decode, single GPU
    (5, 8, 384, 8, 3, None),        # decode, one EP rank: most routings are dropped
    (127, 8, 128, 1, 0, None),      # exactly one plan chunk (1016 routings)
    (129, 8, 128, 1, 0, None),      # just past one chunk
    (1000, 8, 384, 8, 0, None),     # multi-chunk EP rank
    (3000, 8, 384, 8, 7, 16),       # undersized cap: over-capacity routings drop
    (4096, 10, 512, 1, 0, None),    # 512 local experts (64 KiB plan smem)
    (2048, 6, 64, 2, 1, 300),
]


def routing(T, topk, E_global, seed, dtype=torch.int32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.rand(T, E_global, device="cuda", generator=g).argsort(dim=1)[:, :topk]
    w = torch.rand(T, topk, device="cuda", generator=g).softmax(dim=1)
    return ids.to(dtype).contiguous(), w.float().contiguous()


def ref_plan(topk_ids, first_expert, E, cap):
    """vLLM CaptureSafeBatchedPrepareAndFinalize._dispatch_plan."""
    T, topk = topk_ids.shape
    N = T * topk
    device = topk_ids.device
    flat_e = topk_ids.reshape(N).long() - first_expert
    flat_tok = torch.arange(T, device=device).unsqueeze(1).expand(T, topk).reshape(N)
    valid = (flat_e >= 0) & (flat_e < E)
    e_buckets = torch.where(valid, flat_e, torch.full_like(flat_e, E)).to(torch.int16)
    order = torch.argsort(e_buckets, stable=True)
    sorted_e = e_buckets.index_select(0, order)
    group_start = torch.searchsorted(sorted_e, sorted_e)
    rank_sorted = torch.arange(N, device=device) - group_start
    dest = torch.empty_like(rank_sorted)
    dest.scatter_(0, order, rank_sorted)
    edges = torch.searchsorted(sorted_e, torch.arange(E + 1, device=device, dtype=torch.int16))
    counts = edges[1:] - edges[:-1]
    in_range = valid & (dest < cap)
    trash = E * cap
    dest_global = torch.where(in_range, flat_e * cap + dest, torch.full_like(dest, trash))
    return flat_tok, dest_global, in_range, counts.clamp(max=cap).to(torch.int32)


def ref_finalize(fused, dest_global, in_range, topk_weights, apply_on_input):
    """vLLM CaptureSafeBatchedPrepareAndFinalize.finalize. The caller passes padded rows as zero,
    which is what the production path's row-0 pre-zero guarantees for the one row it reads."""
    E, cap, K = fused.shape
    T, topk = topk_weights.shape
    N = T * topk
    feo = fused.reshape(E * cap, K)
    dest_clamped = torch.where(in_range, dest_global, torch.zeros_like(dest_global))
    gathered = feo.index_select(0, dest_clamped)
    if apply_on_input:
        coeff = in_range.reshape(N, 1).to(gathered.dtype)
    else:
        tw = topk_weights.reshape(N)
        coeff = torch.where(in_range, tw, tw.new_zeros(())).reshape(N, 1).to(gathered.dtype)
    return (gathered * coeff).view(T, topk, K).sum(dim=1)


def case_args(case):
    T, topk, E_global, ep, rank, cap = case
    E = E_global // ep
    return T, topk, E_global, E, E * rank, (min(T, 1 << 30) if cap is None else cap)


@requires_gpu
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("case", PLAN_CASES, ids=str)
def test_dispatch_plan_matches_torch(case, id_dtype):
    T, topk, E_global, E, first, cap = case_args(case)
    ids, _ = routing(T, topk, E_global, seed=T, dtype=id_dtype)
    counts, route_dest, src_tok, src_route = ops.dispatch_plan(ids, first, E, cap)
    flat_tok, dest_global, in_range, ref_counts = ref_plan(ids, first, E, cap)

    torch.testing.assert_close(counts, ref_counts, rtol=0, atol=0)
    want = torch.where(in_range, dest_global, torch.full_like(dest_global, -1))
    torch.testing.assert_close(route_dest.reshape(-1).long(), want, rtol=0, atol=0)
    # Every kept row names its token and routing.
    kept = in_range.nonzero().squeeze(1)
    rows = dest_global[kept]
    torch.testing.assert_close(src_tok[rows].long(), flat_tok[kept], rtol=0, atol=0)
    torch.testing.assert_close(src_route[rows].long(), kept, rtol=0, atol=0)


@requires_gpu
def test_dispatch_plan_is_graph_safe():
    T, topk, E_global, E, first, cap = case_args((1000, 8, 384, 8, 2, None))
    ids, _ = routing(T, topk, E_global, seed=0)
    static = ids.clone()
    ops.dispatch_plan(static, first, E, cap)   # warm the allocator / attributes
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = ops.dispatch_plan(static, first, E, cap)
    for seed in (1, 2):
        new_ids, _ = routing(T, topk, E_global, seed=seed)
        static.copy_(new_ids)
        graph.replay()
        _, dest_global, in_range, ref_counts = ref_plan(new_ids, first, E, cap)
        torch.testing.assert_close(out[0], ref_counts, rtol=0, atol=0)
        want = torch.where(in_range, dest_global, torch.full_like(dest_global, -1))
        torch.testing.assert_close(out[1].reshape(-1).long(), want, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("with_weight", [False, True])
@pytest.mark.parametrize("case", [(5, 8, 384, 8, 3, None), (1000, 8, 384, 8, 0, None),
                                  (3000, 8, 384, 8, 7, 16), (129, 8, 128, 1, 0, None)], ids=str)
def test_scatter_quant_planned_matches_scatter_quant(case, with_weight):
    """Byte-identical to scatter_quant_act (itself gated against scatter -> quant_act) on every
    valid row of b_act, and on the whole sfb (padded slots keep the same benign fill)."""
    T, topk, E_global, E, first, cap = case_args(case)
    K = 512
    ids, w = routing(T, topk, E_global, seed=7)
    g = torch.Generator(device="cuda").manual_seed(3)
    a1 = (torch.randn(T, K, device="cuda", generator=g) * 3).to(torch.bfloat16)
    gscale = torch.rand(E, device="cuda", generator=g) + 0.5
    counts, _, src_tok, src_route = ops.dispatch_plan(ids, first, E, cap)
    flat_tok, dest_global, _, ref_counts = ref_plan(ids, first, E, cap)
    wf = w.reshape(-1) if with_weight else None

    b_new, s_new = ops.scatter_quant_act_planned(a1, src_tok, src_route, wf, gscale, counts, cap)
    b_old, s_old = ops.scatter_quant_act(a1, flat_tok, dest_global, wf, gscale, ref_counts, cap)
    torch.testing.assert_close(s_new, s_old, rtol=0, atol=0)
    for e in range(E):
        n = int(counts[e])
        torch.testing.assert_close(b_new[e, :n], b_old[e, :n], rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("apply_on_input", [False, True])
@pytest.mark.parametrize("case", PLAN_CASES, ids=str)
def test_moe_finalize_matches_torch(case, apply_on_input):
    """Bit-identical to the torch combine, including dropped routings and empty experts."""
    T, topk, E_global, E, first, cap = case_args(case)
    K = 1024
    ids, w = routing(T, topk, E_global, seed=11)
    counts, route_dest, _, _ = ops.dispatch_plan(ids, first, E, cap)
    _, dest_global, in_range, _ = ref_plan(ids, first, E, cap)
    g = torch.Generator(device="cuda").manual_seed(5)
    fused = torch.randn(E, cap, K, device="cuda", generator=g).to(torch.bfloat16)
    # Rows past each expert's count hold garbage the kernel must never read.
    pad = torch.arange(cap, device="cuda")[None, :] >= counts[:, None]
    fused[pad] = float("nan")

    out = torch.empty(T, K, device="cuda", dtype=torch.bfloat16)
    ops.moe_finalize(out, fused, route_dest, None if apply_on_input else w)
    fused_ref = fused.clone()
    fused_ref[pad] = 0.0   # the torch path multiplies dropped rows by 0 (NaN would poison it)
    want = ref_finalize(fused_ref, dest_global, in_range, w, apply_on_input)
    torch.testing.assert_close(out, want, rtol=0, atol=0)


@requires_gpu
@pytest.mark.parametrize("case", [(5, 8, 384, 8, 3, None), (1000, 8, 384, 8, 0, None),
                                  (129, 8, 128, 1, 0, None)], ids=str)
def test_prequantized_scatter_matches_scatter_quant(case):
    """quant_rows (before an all2all) + scatter_fp4_planned equals quantizing inside the scatter,
    byte for byte, when every expert shares the global scale."""
    T, topk, E_global, E, first, cap = case_args(case)
    K = 512
    ids, _ = routing(T, topk, E_global, seed=9)
    g = torch.Generator(device="cuda").manual_seed(4)
    a1 = (torch.randn(T, K, device="cuda", generator=g) * 3).to(torch.bfloat16)
    gscale = torch.full((E,), 1.7, device="cuda")
    counts, _, src_tok, src_route = ops.dispatch_plan(ids, first, E, cap)
    q, q_sf = ops.quant_rows(a1, gscale)
    b_new, s_new = ops.scatter_fp4_planned(q, q_sf, src_tok, counts, cap)
    b_ref, s_ref = ops.scatter_quant_act_planned(a1, src_tok, src_route, None, gscale, counts, cap)
    torch.testing.assert_close(s_new, s_ref, rtol=0, atol=0)
    for e in range(E):
        n = int(counts[e])
        torch.testing.assert_close(b_new[e, :n], b_ref[e, :n], rtol=0, atol=0)
