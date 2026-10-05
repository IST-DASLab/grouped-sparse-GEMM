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
"""Paired-4:8 sparse NVFP4 (W4A4) grouped GEMM for MoE expert layers.

Importing the package registers the ops on the torch dispatcher:

    torch.ops.paired_nvfp4.compress(w_packed, w_blockscale, k) -> (a_comp, e_meta, sfa)
    torch.ops.paired_nvfp4.quant_act(x, gscale, expert_num_tokens, features) -> (b_act, sfb)
    torch.ops.paired_nvfp4.silu_mul_quant_act(x2, gscale, expert_num_tokens, features,
                                              interleaved=False, activation="silu", beta=1.0,
                                              linear_beta=-1.0) -> (b_act, sfb)
    torch.ops.paired_nvfp4.scatter_quant_act(a1, flat_tok, dest_global, topk_weights, gscale,
                                             expert_num_tokens, cap) -> (b_act, sfb)
    torch.ops.paired_nvfp4.dispatch_plan(topk_ids, first_expert, num_local_experts, cap)
        -> (expert_num_tokens, route_dest, src_tok, src_route)
    torch.ops.paired_nvfp4.scatter_quant_act_planned(a1, src_tok, src_route, topk_weights,
                                                     gscale, expert_num_tokens, cap)
        -> (b_act, sfb)
    torch.ops.paired_nvfp4.quant_rows(x, gscale) -> (q, q_sf)
    torch.ops.paired_nvfp4.scatter_fp4_planned(q, q_sf, src_tok, expert_num_tokens, cap)
        -> (b_act, sfb)
    torch.ops.paired_nvfp4.moe_finalize(out, fused, route_dest, topk_weights) -> ()
    torch.ops.paired_nvfp4.group_mm(out, a_comp, e_meta, sfa, b_act, sfb, alphas,
                                    expert_num_tokens, features, k,
                                    config_id=0, cluster_m=2, cluster_n=1) -> ()
    torch.ops.paired_nvfp4.group_mm_configs() -> list[str]

See README.md for the tensor contracts.
"""

import importlib
import importlib.util

import torch

__version__ = "0.13.0"

__all__ = [
    "DEFAULT_TACTIC",
    "all_tactics",
    "fused_swiglu_tactics",
    "interleave_gate_up_rows",
    "built_archs",
    "group_mm_configs",
    "is_available",
    "loaded_arch",
    "suggest_tactic",
    "valid_clusters",
]

# SM number -> extension module, in the order setup.py may build them.
_ARCH_MODULES = {100: "_C_sm100", 120: "_C_sm120"}


def _current_sm():
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _load():
    # Every arch module registers the same torch library, so exactly one may be loaded per process:
    # the one matching the current device, else the first one built (registration still works
    # without a GPU, which is enough for import-time checks).
    built = []
    for sm, name in _ARCH_MODULES.items():
        if importlib.util.find_spec(f"{__name__}.{name}") is not None:
            built.append(sm)
    if not built:
        raise ImportError("paired_nvfp4_kernels was installed without any compiled arch module")
    cur = _current_sm()
    sm = cur if cur in built else built[0]
    importlib.import_module(f"{__name__}.{_ARCH_MODULES[sm]}")
    return sm, built


_LOADED_SM, _BUILT_SMS = _load()


def loaded_arch():
    """SM number of the extension module loaded in this process (e.g. 100)."""
    return _LOADED_SM


def built_archs():
    """SM numbers with a compiled extension module in this installation."""
    return list(_BUILT_SMS)


def is_available():
    """True iff the current CUDA device matches the loaded arch module and the ops are registered."""
    if _current_sm() != _LOADED_SM:
        return False
    ns = getattr(torch.ops, "paired_nvfp4", None)
    return ns is not None and hasattr(ns, "group_mm") and hasattr(ns, "compress")


# ---------------------------------------------------------------------------------------------
# Autotuning surface. A group_mm tactic is (config_id, cluster_m, cluster_n): config_id indexes
# the compiled tile variants and the cluster is a runtime argument (no recompilation). The
# cluster rules below mirror each arch's tactics.cu; the op itself re-validates every tactic.
# ---------------------------------------------------------------------------------------------

# Per arch: the default tactic (also the op schema defaults) and the legal cluster dims.
#   SM100: dynamic clusters, each dim 1, 2 or 4 (2SM kernels pair CTAs along M).
#   SM120: no cluster multicast on this arch, so the cluster is always 1x1.
_DEFAULT_TACTICS = {100: (0, 2, 1), 120: (2, 1, 1)}
_CLUSTER_DIMS = {100: (1, 2, 4), 120: (1,)}

DEFAULT_TACTIC = _DEFAULT_TACTICS[_LOADED_SM]


def group_mm_configs():
    """Names of the compiled tile variants, indexed by config_id."""
    return list(torch.ops.paired_nvfp4.group_mm_configs())


def _is_2sm(config_name):
    return config_name.split("_")[1] == "2sm"   # sparse_<sm mode>_<TileM>x<TileN>


def valid_clusters(config_name, max_product=16):
    """Legal (cluster_m, cluster_n) pairs for one tile variant."""
    pows = _CLUSTER_DIMS[_LOADED_SM]
    out = []
    for cm in pows:
        if _is_2sm(config_name) and cm % 2 != 0:
            continue
        for cn in pows:
            if cm * cn <= max_product:
                out.append((cm, cn))
    return out


def all_tactics(max_cluster_product=16):
    """Every candidate (config_id, cluster_m, cluster_n).

    A candidate can still be rejected for a particular shape by the kernel's can_implement, which
    raises RuntimeError; sweeps should catch it and skip the candidate.
    """
    tactics = []
    for config_id, name in enumerate(group_mm_configs()):
        for cm, cn in valid_clusters(name, max_cluster_product):
            tactics.append((config_id, cm, cn))
    return tactics


def interleave_gate_up_rows(n, device=None):
    """Row order of W13 ([gate; up], 2n rows) for group_mm_swiglu_quant.

    Rows are interleaved in 64-row blocks: gate rows 32b .. 32b + 31, then up rows
    n + 32b .. n + 32b + 31. Apply to the dense packed weight and its block scales before
    compress(): ``w13[:, perm]``, ``w13_scale[:, perm]``. n must be a multiple of 64.
    """
    if n % 64:
        raise ValueError(f"intermediate size must be a multiple of 64, got {n}")
    b = torch.arange(n // 32, device=device).repeat_interleave(32) * 32
    r = torch.arange(32, device=device).repeat(n // 32)
    gate = b + r
    perm = torch.stack([gate.view(-1, 32), (gate + n).view(-1, 32)], dim=1)
    return perm.reshape(-1)


def fused_swiglu_tactics(max_cluster_product=16):
    """Candidate (config_id, cluster_m, cluster_n) for group_mm_swiglu_quant (empty if the loaded
    arch has no fused epilogue)."""
    ns = torch.ops.paired_nvfp4
    if not hasattr(ns, "group_mm_swiglu_configs"):
        return []
    tactics = []
    for config_id, name in enumerate(ns.group_mm_swiglu_configs()):
        for cm, cn in valid_clusters(name, max_cluster_product):
            tactics.append((config_id, cm, cn))
    return tactics


# SM120: max_n at or above which the 256x128 tile beats 128x64 (compute-bound prefill). Measured
# with tools/bench_moe.py on an RTX PRO 6000 over Qwen3-30B-A3B, Qwen3-235B-A22B, Mixtral-8x7B and
# DeepSeek-V3 shapes, 1-8192 tokens: this rule is within 0.5% of the per-shape best on the
# geometric mean and 7% at worst.
_SM120_LARGE_TILE_MIN_N = 768


def suggest_tactic(max_n):
    """A good group_mm tactic for a per-expert token capacity of max_n, without autotuning.

    On SM120 this picks sparse_1sm_256x128 for max_n >= 768 and sparse_1sm_128x64 (the default)
    otherwise. On other archs it returns DEFAULT_TACTIC. An autotuner over all_tactics() can still
    do better for a specific shape, e.g. with the split-K variants for decode on few, long-K experts.
    """
    if _LOADED_SM == 120:
        names = group_mm_configs()
        name = "sparse_1sm_256x128" if max_n >= _SM120_LARGE_TILE_MIN_N else "sparse_1sm_128x64"
        return (names.index(name), 1, 1)
    return DEFAULT_TACTIC
