#!/usr/bin/env python3
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
"""Realistic-MoE benchmark of group_mm (every tactic) and quant_act, optionally against a dense
NVFP4 grouped GEMM.

    python tools/bench_moe.py                                  # all models, default token counts
    python tools/bench_moe.py --models mixtral-8x7b --tokens 1 512 --json out.json
    python tools/bench_moe.py --dense                          # + dense NVFP4 baseline (SM120)

Method (the numbers in docs/porting.md come from this):
- Expert shapes come from real models (w13 = fused gate/up, w2 = down projection). Token counts
  come from top-k routing of noisy logits with a mild per-expert popularity skew; max_n is the
  largest count rounded up to 32.
- Every launch is CUDA-graph replayed (no host launch overhead in the timing).
- Weights rotate through enough copies that the weights touched per cycle exceed 2x L2. Otherwise
  a decode-sized launch finds its few active experts in L2 from the previous launch, which does not
  happen in serving (the other layers run in between).
- Each configuration is warmed up and timed twice, keeping the better run. Runs vary by a few
  percent; compare builds interleaved and repeated before trusting a small difference.

--dense JIT-builds tools/dense_baseline/dense_nvfp4.cu (CUTLASS's SM120 pointer-array NVFP4
grouped GEMM with standard per-16 scales, arrays built on the device from the counts) and reports
the best of its tiles x raster orders per shape. It needs nvcc on PATH or CUDA_HOME, a host
compiler torch accepts, and ninja, the same toolchain the package builds with.
"""

import argparse
import json
import os
import sys

import torch

import paired_nvfp4_kernels as pnk
from paired_nvfp4_kernels import reference as ref

ops = torch.ops.paired_nvfp4

MODELS = {  # experts, top-k, hidden, moe intermediate
    "qwen3-30b-a3b": (128, 8, 2048, 768),
    "qwen3-235b":    (128, 8, 4096, 1536),
    "mixtral-8x7b":  (8, 2, 4096, 14336),
    "deepseek-v3":   (256, 8, 7168, 2048),
}
DEFAULT_TOKENS = [1, 16, 128, 512, 2048, 8192]


def routed_counts(T, E, topk, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bias = torch.randn(E, device="cuda", generator=g) * 0.5          # mild popularity skew
    logits = torch.randn(T, E, device="cuda", generator=g) + bias
    idx = logits.topk(topk, dim=1).indices.flatten()
    return torch.bincount(idx, minlength=E).to(torch.int32)


def l2_bytes():
    return torch.cuda.get_device_properties(0).L2_cache_size


def timeit(fns, reps=2, min_iters=30):
    """Time per call (us), graph-replayed, cycling through len(fns) weight-copy closures."""
    R = len(fns)
    it = max(2 * R, min_iters)
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(it):
            fns[i % R]()
    g.replay()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / it * 1e3)
    return best


def dram_bandwidth():
    x = torch.empty(1 << 30, dtype=torch.uint8, device="cuda")
    y = torch.empty_like(x)
    return 2 * x.numel() / (timeit([lambda: y.copy_(x)], min_iters=10) * 1e-6)


class SparseWeights:
    """compress() output for one (E, M, K), with the first-launch slowdown burned off."""

    def __init__(self, E, M, K):
        torch.manual_seed(0)
        wp, ws = [], []
        for e0 in range(0, E, 8):   # chunks keep the fp32 staging memory bounded
            w = ref.prune_paired_4of8(torch.randn(min(8, E - e0), M, K, device="cuda"))
            p, s, _ = ref.quantize_weight(w.to(torch.bfloat16))
            wp.append(p)
            ws.append(s)
            del w
        self.a_comp, self.e_meta, self.sfa = ops.compress(torch.cat(wp), torch.cat(ws), K)
        self.per_expert_bytes = (self.a_comp.numel() + self.e_meta.numel() + self.sfa.numel()) / E
        c = torch.full((E,), 64, dtype=torch.int32, device="cuda")
        xb, sb = ops.quant_act(torch.randn(E, 64, K, device="cuda", dtype=torch.bfloat16),
                               torch.ones(E, device="cuda"), c, M)
        o = torch.empty(E, 64, M, device="cuda", dtype=torch.bfloat16)
        for _ in range(30):
            ops.group_mm(o, self.a_comp, self.e_meta, self.sfa, xb, sb, torch.ones(E, device="cuda"),
                         c, M, K, *pnk.DEFAULT_TACTIC)
        torch.cuda.synchronize()

    def copies(self, n):
        return [(self.a_comp, self.e_meta, self.sfa)] + [
            (self.a_comp.clone(), self.e_meta.clone(), self.sfa.clone()) for _ in range(n - 1)]


def load_dense():
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    cutlass = os.environ.get("CUTLASS_DIR", os.path.join(here, "..", "third_party", "cutlass"))
    build = os.path.join(here, "dense_baseline", "build")
    os.makedirs(build, exist_ok=True)
    return load("paired_nvfp4_dense_baseline", [os.path.join(here, "dense_baseline", "dense_nvfp4.cu")],
                build_directory=build,
                extra_include_paths=[os.path.join(cutlass, "include"),
                                     os.path.join(cutlass, "tools", "util", "include")],
                extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a",
                                   "--expt-relaxed-constexpr", "--expt-extended-lambda"],
                extra_cflags=["-O3", "-std=c++17"])


def bench_case(E, M, K, counts, sw, tactics, dense, bw, l2):
    max_n = max(32, -(-int(counts.max()) // 32) * 32)
    active, ntok = int((counts > 0).sum()), int(counts.sum())
    x = torch.randn(E, max_n, K, device="cuda", dtype=torch.bfloat16)
    ones = torch.ones(E, device="cuda")
    b_act, sfb = ops.quant_act(x, ones, counts, M)
    out = torch.empty(E, max_n, M, device="cuda", dtype=torch.bfloat16)

    dense_per_expert = M * K // 2 + (dense.sf_sizes(M, 128, K)[0] if dense else M * K // 16)
    full = (sw.per_expert_bytes + (dense_per_expert if dense else 0)) * E
    n_copies = int(max(1, min(-(-2 * l2 // max(1, int(dense_per_expert * active))), (24 << 30) // full)))
    ws = sw.copies(n_copies)

    names = pnk.group_mm_configs()
    res = {"max_n": max_n, "active_experts": active, "tokens_routed": ntok, "weight_copies": n_copies}
    res["sparse"] = {}
    for cid, cm, cn in tactics:
        try:
            res["sparse"][f"{names[cid]}"] = timeit(
                [lambda w=w: ops.group_mm(out, *w, b_act, sfb, ones, counts, M, K, cid, cm, cn)
                 for w in ws])
        except RuntimeError:   # a tactic can_implement rejects for this shape
            pass
    res["sparse_bytes"] = sw.per_expert_bytes * active + ntok * (K // 2 + K // 32 + 2 * M)
    res["quant_act"] = timeit([lambda: ops.quant_act(x, ones, counts, M)])
    del ws

    if dense:
        da = torch.randint(0, 256, (E, M, K // 2), dtype=torch.uint8, device="cuda")
        dsfa = torch.full((E * dense.sf_sizes(M, 128, K)[0],), 0x38, dtype=torch.uint8, device="cuda")
        dsfb = torch.full((E * dense.sf_sizes(M, max_n, K)[1],), 0x38, dtype=torch.uint8, device="cuda")
        dws = [(da, dsfa)] + [(da.clone(), dsfa.clone()) for _ in range(n_copies - 1)]
        res["dense"] = {}
        for v, vname in enumerate(dense.variants()):
            for raster_n in (True, False):
                res["dense"][f"{vname}/{'N' if raster_n else 'M'}"] = timeit(
                    [lambda w=w, v=v, rn=raster_n:
                     dense.grouped_mm(out, w[0], w[1], b_act, dsfb, ones, counts, M, K, v, rn)
                     for w in dws])
        res["dense_bytes"] = dense_per_expert * active + ntok * (K // 2 + K // 16 + 2 * M)
        del dws, da, dsfa, dsfb
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--tokens", nargs="*", type=int, default=DEFAULT_TOKENS)
    ap.add_argument("--gemms", nargs="*", default=["w13", "w2"], choices=["w13", "w2"])
    ap.add_argument("--default-only", action="store_true", help="time only the default tactic")
    ap.add_argument("--dense", action="store_true", help="also time the dense NVFP4 baseline")
    ap.add_argument("--json", help="write every measurement to this file")
    args = ap.parse_args()

    if not pnk.is_available():
        sys.exit(f"no GPU matching the loaded arch module (SM{pnk.loaded_arch()})")
    dense = load_dense() if args.dense else None
    tactics = [pnk.DEFAULT_TACTIC] if args.default_only else pnk.all_tactics()
    names = pnk.group_mm_configs()
    default_name = names[pnk.DEFAULT_TACTIC[0]]
    bw, l2 = dram_bandwidth(), l2_bytes()
    print(f"# {torch.cuda.get_device_name()}  DRAM copy {bw / 1e9:.0f} GB/s  L2 {l2 >> 20} MB  "
          f"tactics {[names[t[0]] for t in tactics]}")
    hdr = f"{'case':30s} {'default':>8s} {'best':>8s} {'(tactic)':>28s} {'%bw':>6s} {'quant':>7s}"
    if dense:
        hdr += f" | {'dense':>8s} {'(config)':>26s} {'%bw':>6s} {'speedup':>7s}"
    print(hdr)
    rows = []
    for mname in args.models:
        E, topk, H, I = MODELS[mname]
        for gname in args.gemms:
            M, K = (2 * I, H) if gname == "w13" else (H, I)
            sw = SparseWeights(E, M, K)
            for T in args.tokens:
                r = bench_case(E, M, K, routed_counts(T, E, topk), sw, tactics, dense, bw, l2)
                sp = r["sparse"]
                bname, bt = min(sp.items(), key=lambda kv: kv[1])
                line = (f"{mname + ' ' + gname + ' T=' + str(T):30s} {sp.get(default_name, float('nan')):8.1f} "
                        f"{bt:8.1f} {bname:>28s} {r['sparse_bytes'] / bw * 1e6 / bt * 100:5.1f}% "
                        f"{r['quant_act']:7.1f}")
                if dense:
                    dname, dt = min(r["dense"].items(), key=lambda kv: kv[1])
                    line += (f" | {dt:8.1f} {dname.replace('grouped_', ''):>26s} "
                             f"{r['dense_bytes'] / bw * 1e6 / dt * 100:5.1f}% {dt / bt:6.2f}x")
                print(line, flush=True)
                rows.append(dict(model=mname, gemm=gname, T=T, E=E, M=M, K=K, **r))
            del sw
            torch.cuda.empty_cache()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=torch.cuda.get_device_name(), dram_bw=bw, rows=rows), f, indent=1)


if __name__ == "__main__":
    main()
