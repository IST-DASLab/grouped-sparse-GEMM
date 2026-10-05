"""vLLM general plugin: run the paired48 backend's dispatch/combine on the kernel package's fused
ops (dispatch_plan, scatter_quant_act_planned, moe_finalize) instead of vLLM's torch glue.

For measuring the ops end to end before the vLLM patch adopts them. Put this directory on
PYTHONPATH; vLLM discovers the `vllm.general_plugins` entry point in the bundled dist-info and
calls register() in every engine and worker process. The override touches only
CaptureSafeBatchedPrepareAndFinalize, the P/F that the paired48 backend uses on a single GPU and
with TP+EP; the batched CuteDSL backend does not use the fused scatter path and is left alone.
"""

import torch

_saved = None


def install():
    global _saved
    if _saved is not None:
        return
    import paired_nvfp4_kernels  # noqa: F401  registers torch.ops.paired_nvfp4
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.prepare_finalize.batched_capture_safe import (
        CaptureSafeBatchedPrepareAndFinalize as PF,
    )

    ops = torch.ops.paired_nvfp4
    orig_prepare, orig_finalize = PF.prepare, PF.finalize

    def plan(pf, topk_ids):
        cached = pf._plan_cache
        if cached is not None and cached[0] is topk_ids:
            return cached[1]
        E, T = pf.num_local_experts, topk_ids.size(0)
        cap = min(pf.max_num_tokens, T)
        p = ops.dispatch_plan(topk_ids.contiguous(), E * pf.rank, E, cap) + (cap,)
        pf._plan_cache = (topk_ids, p)
        return p

    def prepare(pf, a1, topk_weights, topk_ids, num_experts, expert_map,
                apply_router_weight_on_input, quant_config, defer_input_quant=False):
        if not (pf.fuse_scatter_quant and quant_config.quant_dtype == "nvfp4"):
            return orig_prepare(pf, a1, topk_weights, topk_ids, num_experts, expert_map,
                                apply_router_weight_on_input, quant_config, defer_input_quant)
        counts, _, src_tok, src_route, cap = plan(pf, topk_ids)
        meta = mk.ExpertTokensMetadata(expert_num_tokens=counts, expert_num_tokens_cpu=None)
        w = (topk_weights.reshape(-1).to(torch.float32).contiguous()
             if apply_router_weight_on_input else None)
        b, sf = ops.scatter_quant_act_planned(
            a1, src_tok, src_route, w, quant_config.a1_gscale.reshape(-1).contiguous(),
            counts, cap)
        return b, sf, meta, None, None

    def finalize(pf, output, fused_expert_output, topk_weights, topk_ids,
                 apply_router_weight_on_input, weight_and_reduce_impl):
        if not pf.fuse_scatter_quant:
            return orig_finalize(pf, output, fused_expert_output, topk_weights, topk_ids,
                                 apply_router_weight_on_input, weight_and_reduce_impl)
        _, route_dest, _, _, _ = plan(pf, topk_ids)
        w = None if apply_router_weight_on_input else topk_weights.to(torch.float32).contiguous()
        ops.moe_finalize(output, fused_expert_output.contiguous(), route_dest, w)

    PF.prepare, PF.finalize = prepare, finalize
    _saved = (PF, orig_prepare, orig_finalize)
    import sys
    print(f"p48_fused_glue: fused dispatch/combine installed "
          f"(paired_nvfp4_kernels from {paired_nvfp4_kernels.__file__})", file=sys.stderr,
          flush=True)


def uninstall():
    global _saved
    if _saved is not None:
        PF, p, f = _saved
        PF.prepare, PF.finalize = p, f
        _saved = None


def register():
    install()
