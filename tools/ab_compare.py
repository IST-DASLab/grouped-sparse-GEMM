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
"""Byte-level A/B of two builds of paired_nvfp4_kernels.

Two builds register the same torch library, so each runs in its own process:

    PYTHONPATH=<build A> python tools/ab_compare.py run a.pt
    PYTHONPATH=<build B> python tools/ab_compare.py run b.pt
    python tools/ab_compare.py compare a.pt b.pt

`run` feeds seeded inputs to every op (and to group_mm under every tactic) and saves the defined
part of each output: valid rows of b_act / out, full buffers of the weight operands and scales.
`compare` requires every saved tensor to be byte-identical.
"""

import importlib.util
import os
import sys

import torch

_REF_PATH = os.path.join(os.path.dirname(__file__), "..", "paired_nvfp4_kernels", "reference.py")
_spec = importlib.util.spec_from_file_location("pnk_reference", _REF_PATH)
ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref)

# (M, max_n, K, E): GEMM1-like and GEMM2-like shapes, one with an unaligned SFB layout.
SHAPES = [(1024, 256, 2048, 8), (2048, 192, 768, 16), (512, 128, 7168, 4)]
# (T, topk, K, E, cap) for scatter_quant_act
SCATTER_SHAPES = [(128, 8, 2048, 32, 64), (96, 4, 7168, 8, 96)]


def gen(seed):
    g = torch.Generator().manual_seed(seed)
    return g


def valid_rows(t, counts):
    return torch.cat([t[e, :int(c)].reshape(-1) for e, c in enumerate(counts.tolist())])


def run(path):
    import paired_nvfp4_kernels as pnk
    ops = torch.ops.paired_nvfp4
    dev = "cuda"
    out = {}
    tactics = pnk.all_tactics()
    for si, (M, max_n, K, E) in enumerate(SHAPES):
        g = gen(1000 + si)
        w = ref.prune_paired_4of8(torch.randn(E, M, K, generator=g)).to(torch.bfloat16)
        w_packed, w_sf, _ = ref.quantize_weight(w)
        a_comp, e_meta, sfa = ops.compress(w_packed.to(dev), w_sf.to(dev), K)
        out[f"s{si}/a_comp"], out[f"s{si}/e_meta"], out[f"s{si}/sfa"] = a_comp, e_meta, sfa

        counts = torch.randint(1, max_n + 1, (E,), generator=g, dtype=torch.int32)
        counts[0] = max_n
        gscale = (torch.rand(E, generator=g) * 4 + 0.25).float()
        x = (torch.randn(E, max_n, K, generator=g) * 3).to(torch.bfloat16)
        cd, gd = counts.to(dev), gscale.to(dev)
        b_act, sfb = ops.quant_act(x.to(dev), gd, cd, M)
        out[f"s{si}/quant/b_act"] = valid_rows(b_act, counts)
        out[f"s{si}/quant/sfb"] = sfb
        b_act1, sfb1 = ops.quant_act(x.to(dev), gd[:1].contiguous(), cd, M)
        out[f"s{si}/quant_shared_gscale/b_act"] = valid_rows(b_act1, counts)
        out[f"s{si}/quant_shared_gscale/sfb"] = sfb1

        x2 = (torch.randn(E, max_n, 2 * K, generator=g) * 2).to(torch.bfloat16)
        b2, s2 = ops.silu_mul_quant_act(x2.to(dev), gd, cd, M)
        out[f"s{si}/silu/b_act"] = valid_rows(b2, counts)
        out[f"s{si}/silu/sfb"] = s2

        alphas = (torch.rand(E, generator=g) + 0.5).float().to(dev)
        for tactic in tactics:
            d = torch.zeros(E, max_n, M, device=dev, dtype=torch.bfloat16)
            try:
                ops.group_mm(d, a_comp, e_meta, sfa, b_act, sfb, alphas, cd, M, K, *tactic)
                torch.cuda.synchronize()
            except RuntimeError as err:
                out[f"s{si}/group_mm/{tactic}/rejected"] = torch.tensor([1])
                continue
            out[f"s{si}/group_mm/{tactic}"] = valid_rows(d, counts)

    for si, (T, topk, K, E, cap) in enumerate(SCATTER_SHAPES):
        g = gen(2000 + si)
        a1 = (torch.randn(T, K, generator=g) * 2).to(torch.bfloat16)
        n = T * topk
        flat_tok = torch.arange(T).repeat_interleave(topk)
        expert = torch.randint(0, 2 * E, (n,), generator=g)          # half of them non-local
        dest = torch.full((n,), E * cap, dtype=torch.int64)
        fill = torch.zeros(E, dtype=torch.int64)
        for i in range(n):
            e = int(expert[i])
            if e < E and fill[e] < cap:
                dest[i] = e * cap + fill[e]
                fill[e] += 1
        counts = fill.to(torch.int32)
        gscale = (torch.rand(E, generator=g) * 4 + 0.25).float()
        topk_w = torch.rand(n, generator=g).float()
        for wname, w_arg in (("noweight", None), ("weight", topk_w.to(dev))):
            b, s = ops.scatter_quant_act(a1.to(dev), flat_tok.to(dev), dest.to(dev), w_arg,
                                         gscale.to(dev), counts.to(dev), cap)
            out[f"scatter{si}/{wname}/b_act"] = valid_rows(b, counts)
            out[f"scatter{si}/{wname}/sfb"] = s

    torch.save({k: v.detach().cpu() for k, v in out.items()}, path)
    print(f"saved {len(out)} tensors from {pnk.__file__} to {path}")


def compare(pa, pb):
    a, b = torch.load(pa), torch.load(pb)
    ok = True
    if set(a) != set(b):
        print("key sets differ:", sorted(set(a) ^ set(b)))
        ok = False
    for k in sorted(set(a) & set(b)):
        ta, tb = a[k], b[k]
        same = ta.shape == tb.shape and ta.dtype == tb.dtype and torch.equal(
            ta.view(torch.uint8) if ta.element_size() == 1 else ta,
            tb.view(torch.uint8) if tb.element_size() == 1 else tb)
        if not same:
            ok = False
            detail = ""
            if ta.shape == tb.shape and ta.is_floating_point():
                detail = f" max_abs_diff={float((ta.float() - tb.float()).abs().max()):.3e}"
            elif ta.shape == tb.shape:
                detail = f" mismatching={int((ta != tb).sum())}/{ta.numel()}"
            print(f"DIFF {k} shape {tuple(ta.shape)} vs {tuple(tb.shape)}{detail}")
    print(f"{'IDENTICAL' if ok else 'DIFFERENT'}: {len(set(a) & set(b))} common tensors")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "run":
        run(sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[1] == "compare":
        sys.exit(compare(sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
        sys.exit(2)
