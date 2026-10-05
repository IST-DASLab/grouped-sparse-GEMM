"""Layer-level MoE benchmark: paired48 sparse NVFP4 vs the FlashInfer dense NVFP4 backends.

Times one whole MoE layer as vLLM runs it -- router top-k, prepare (dispatch + activation quant),
both expert GEMMs, SwiGLU, finalize (unpermute + top-k weighted reduce) -- captured in a CUDA
graph, for every backend in the same process on the same GPU and token counts. Each backend is
built through vLLM's own NVFP4 oracle (select -> convert weights -> quant config -> modular
kernel), so the weight shuffles, prepare/finalize and glue are the production code paths.

Weights are synthetic: dense backends get group-16 NVFP4, paired48 gets paired-4:8 sparse
group-32 NVFP4 (the only format it runs). Kernel time does not depend on the values; they are
kept finite so no path hits NaN/Inf slow paths.

EP emulation: --ep-size S --ep-rank R builds one rank of pure EP (tokens replicated, the rank
owns experts [R*E/S, (R+1)*E/S)), which is the per-GPU MoE work of vLLM TP+EP serving. The
EP all-reduce of the partial output is not included (it is the same for every backend).

Run inside the vLLM venv on one SM100 GPU:
    python bench/moe_layer_bench.py --shape kimi --ep-size 8 \
        --backends paired48_nvfp4,flashinfer_trtllm,flashinfer_cutlass,flashinfer_cutedsl \
        --tokens 1,4,16,64,128,256,512,1024,2048,4096,8192 --csv out.csv
"""

import argparse
import csv
import os
import sys
import time

import torch

SHAPES = {
    # name: (num_experts, topk, hidden K, intermediate N)
    "kimi": (384, 8, 7168, 2048),
    "qwen3_30b": (128, 8, 2048, 768),
    "qwen35_397b": (512, 10, 4096, 1024),
}


def make_weights(E, N, K, group, sparse, device):
    """Random NVFP4 expert weights in compressed-tensors layout (w13 = [gate; up])."""
    g = torch.Generator(device=device).manual_seed(0)

    def packed(rows, cols):
        w = torch.randint(0, 256, (E, rows, cols // 2), dtype=torch.uint8, device=device,
                          generator=g)
        if sparse:
            # Paired 4:8: each byte holds one fp4 pair; in every 4-byte chunk keep 2 bytes.
            chunks = w.view(E, rows, cols // 8, 4)
            keep = torch.rand(chunks.shape, device=device, generator=g).argsort(-1) < 2
            chunks.mul_(keep.to(torch.uint8))
        return w

    def scales(rows, cols):
        s = torch.rand((E, rows, cols // group), device=device, generator=g) * 0.02 + 0.005
        return s.to(torch.float8_e4m3fn)

    return packed(2 * N, K), scales(2 * N, K), packed(K, N), scales(K, N)


class FakeLayer(torch.nn.Module):
    """The attributes the NVFP4 oracle and experts read from a RoutedExperts layer."""

    def __init__(self, moe_config, activation):
        super().__init__()
        self.moe_config = moe_config
        self.activation = activation


def build(backend, E, topk, K, N, ep_size, ep_rank, max_tokens, device):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import (
        convert_to_nvfp4_moe_kernel_format,
        make_nvfp4_moe_kernel,
        make_nvfp4_moe_quant_config,
        select_nvfp4_moe_backend,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
        kNvfp4Static,
    )
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk

    paired = backend == "paired48_nvfp4"
    group = 32 if paired else 16
    E_local = E // ep_size

    pcfg = FusedMoEParallelConfig.make_no_parallel()
    if ep_size > 1:
        pcfg.ep_size, pcfg.ep_rank, pcfg.use_ep = ep_size, ep_rank, True
    moe = FusedMoEConfig(
        num_experts=E,
        experts_per_token=topk,
        hidden_dim=K,
        intermediate_size=N,
        num_local_experts=E_local,
        num_logical_experts=E,
        moe_parallel_config=pcfg,
        activation=MoEActivation.SILU,
        in_dtype=torch.bfloat16,
        device=device,
        routing_method=RoutingMethodType.Renormalize,
        max_num_tokens=max_tokens,
        moe_backend=backend,
        max_capture_size=max_tokens,
        intermediate_size_per_partition=N,
    )
    nvfp4_backend, experts_cls = select_nvfp4_moe_backend(
        config=moe, weight_key=kNvfp4Static, activation_key=kNvfp4Dynamic,
        group_size=group)

    layer = FakeLayer(moe, MoEActivation.SILU)
    w13, w13_s, w2, w2_s = make_weights(E_local, N, K, group, paired, device)
    for name, t in (("w13_weight", w13), ("w13_weight_scale", w13_s),
                    ("w2_weight", w2), ("w2_weight_scale", w2_s)):
        layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    ones = torch.ones(E_local, dtype=torch.float32, device=device)
    (w13, w13_s, w13_s2, a13_s, w2, w2_s, w2_s2, a2_s) = convert_to_nvfp4_moe_kernel_format(
        nvfp4_backend=nvfp4_backend, layer=layer,
        w13=layer.w13_weight, w13_scale=layer.w13_weight_scale, w13_scale_2=ones.clone(),
        a13_scale=torch.ones(E_local, 2, dtype=torch.float32, device=device),
        w2=layer.w2_weight, w2_scale=layer.w2_weight_scale, w2_scale_2=ones.clone(),
        a2_scale=ones.clone(), is_act_and_mul=True)
    from vllm.model_executor.utils import replace_parameter
    for name, t in (("w13_weight", w13), ("w13_weight_scale", w13_s), ("w2_weight", w2),
                    ("w2_weight_scale", w2_s), ("w13_weight_scale_2", w13_s2),
                    ("w2_weight_scale_2", w2_s2)):
        if hasattr(layer, name):
            replace_parameter(layer, name, t)
        else:
            layer.register_parameter(name, torch.nn.Parameter(t, requires_grad=False))
    layer.w13_input_scale, layer.w2_input_scale = a13_s, a2_s
    qcfg = make_nvfp4_moe_quant_config(
        backend=nvfp4_backend, w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale, w13_scale_2=layer.w13_weight_scale_2,
        w2_scale_2=layer.w2_weight_scale_2, a13_scale=a13_s, a2_scale=a2_s, layer=layer)

    expert_map = None
    if ep_size > 1:
        expert_map = torch.full((E,), -1, dtype=torch.int32, device=device)
        expert_map[ep_rank * E_local:(ep_rank + 1) * E_local] = torch.arange(
            E_local, dtype=torch.int32, device=device)
    routing_tables = None
    kernel = make_nvfp4_moe_kernel(moe_quant_config=qcfg, moe_config=moe,
                                   experts_cls=experts_cls, backend=nvfp4_backend,
                                   routing_tables=routing_tables)
    kernel.fused_experts.process_weights_after_loading(layer)
    monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)
    return layer, kernel, expert_map, monolithic, nvfp4_backend.value, experts_cls.__name__



class FusedGlue:
    """The "paired48_fused" arm: vLLM's capture-safe P/F on the kernel package's fused
    dispatch_plan / scatter_quant_act_planned / moe_finalize, via the same override the vLLM
    plugin in bench/vllm_plugin installs for serving."""

    def __enter__(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vllm_plugin"))
        import p48_fused_glue
        self.mod = p48_fused_glue
        p48_fused_glue.install()
        return self

    def __exit__(self, *exc):
        self.mod.uninstall()


def make_forward(layer, kernel, expert_map, monolithic, E, topk):
    from vllm.model_executor.layers.fused_moe import fused_topk
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    def fwd(x, logits):
        if monolithic:
            return kernel.apply_monolithic(
                x, layer.w13_weight, layer.w2_weight, logits,
                activation=MoEActivation.SILU, global_num_experts=E,
                expert_map=expert_map, apply_router_weight_on_input=False)
        tw, ti, _ = fused_topk(x, logits, topk, renormalize=True)
        return kernel.apply(
            hidden_states=x, w1=layer.w13_weight, w2=layer.w2_weight,
            topk_weights=tw, topk_ids=ti,
            activation=MoEActivation.SILU, global_num_experts=E,
            expert_map=expert_map, apply_router_weight_on_input=False)

    return fwd


def time_graph(fwd, x, logits, iters, warmup, flush):
    for _ in range(3):  # eager warmup (JIT / autotune first-call paths)
        fwd(x, logits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fwd(x, logits)
        with torch.cuda.graph(graph, stream=stream):
            fwd(x, logits)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if flush is not None:
            flush.fill_(1)  # evict the weights from L2, as the model's other layers would
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        graph.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    times.sort()
    replays = int(os.environ.get("BENCH_PROFILE_REPLAYS", "0"))
    if replays:  # nsys --capture-range=cudaProfilerApi records exactly these replays
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for _ in range(replays):
            if flush is not None:
                flush.fill_(1)
            graph.replay()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    return times[len(times) // 2], times[len(times) // 10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="kimi", choices=sorted(SHAPES))
    ap.add_argument("--ep-size", type=int, default=1)
    ap.add_argument("--ep-rank", type=int, default=0)
    ap.add_argument("--backends", default="paired48_nvfp4,flashinfer_trtllm,"
                    "flashinfer_cutlass,flashinfer_cutedsl,flashinfer_cutedsl_batched")
    ap.add_argument("--tokens", default="1,4,16,64,128,256,512,1024,2048,4096,8192")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-autotune", action="store_true",
                    help="skip each backend's autotuner (vLLM kernel_warmup runs it by default)")
    ap.add_argument("--small-max-n", type=int, default=512,
                    help="paired48 decode tactic bucket = vLLM's max CUDA-graph capture size")
    ap.add_argument("--no-l2-flush", action="store_true",
                    help="keep L2 warm between replays (weights may stay resident)")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    init_workspace_manager(device)
    E, topk, K, N = SHAPES[args.shape]
    tokens = [int(t) for t in args.tokens.split(",")]
    max_tokens = max(tokens)
    flush = None if args.no_l2_flush else torch.empty(
        512 << 20, dtype=torch.uint8, device=device)
    rows = []
    print(f"# shape={args.shape} E={E} topk={topk} K={K} N={N} ep={args.ep_rank}/{args.ep_size}"
          f" gpu={torch.cuda.get_device_name()}", flush=True)
    import contextlib
    for arm in args.backends.split(","):
        # "paired48_fused": the paired48 backend with the fused plan/scatter/finalize glue.
        backend = "paired48_nvfp4" if arm == "paired48_fused" else arm
        glue = FusedGlue() if arm == "paired48_fused" else contextlib.nullcontext()
        with set_current_vllm_config(VllmConfig()), glue:
            try:
                layer, kernel, emap, mono, bname, cls = build(
                    backend, E, topk, K, N, args.ep_size, args.ep_rank, max_tokens, device)
            except Exception as exc:  # report and keep going with the other backends
                import traceback; traceback.print_exc()
                print(f"[{arm}] build failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            fwd = make_forward(layer, kernel, emap, mono, E, topk)
            if not args.no_autotune:
                t0 = time.time()
                autotune(backend, kernel, layer, fwd, E, K, max_tokens,
                         min(args.small_max_n, max_tokens), device)
                print(f"[{arm}] autotuned in {time.time() - t0:.0f}s", flush=True)
            print(f"[{arm}] -> {bname} / {cls}{' (monolithic)' if mono else ''}", flush=True)
            g = torch.Generator(device=device).manual_seed(1)
            for T in tokens:
                x = torch.randn(T, K, dtype=torch.bfloat16, device=device, generator=g) * 0.1
                logits = torch.randn(T, E, dtype=torch.float32, device=device, generator=g)
                try:
                    med, p10 = time_graph(fwd, x, logits, args.iters, args.warmup, flush)
                except Exception as exc:
                    print(f"[{arm}] T={T} failed: {type(exc).__name__}: {exc}", flush=True)
                    torch.cuda.synchronize()
                    continue
                rows.append(dict(shape=args.shape, ep_size=args.ep_size, backend=arm,
                                 resolved=bname, tokens=T, median_us=round(med, 2),
                                 p10_us=round(p10, 2)))
                print(f"  T={T:6d}  median {med:9.1f} us   p10 {p10:9.1f} us", flush=True)
            del layer, kernel, fwd
            torch.cuda.empty_cache()
    if args.csv and rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    by = {}
    for r in rows:
        by.setdefault(r["tokens"], {})[r["backend"]] = r["median_us"]
    names = [b for b in args.backends.split(",") if any(b in v for v in by.values())]
    print("\n# median us per MoE layer (CUDA graph)")
    print("tokens  " + "  ".join(f"{n:>24s}" for n in names))
    for T in sorted(by):
        print(f"{T:6d}  " + "  ".join(f"{by[T].get(n, float('nan')):24.1f}" for n in names))
    if "paired48_nvfp4" in names:
        print("\n# speedup of paired48 over each backend (>1 = paired48 faster)")
        for T in sorted(by):
            p = by[T].get("paired48_nvfp4")
            if p:
                print(f"{T:6d}  " + "  ".join(
                    f"{by[T].get(n, float('nan')) / p:24.2f}" for n in names))


def autotune(backend, kernel, layer, fwd, E, K, max_tokens, small_max_n, device):
    """Run each backend's own autotuner the way vLLM's kernel_warmup does."""
    if backend == "paired48_nvfp4":
        from types import SimpleNamespace
        from vllm.model_executor.warmup.paired_nvfp4_warmup import paired_nvfp4_autotune
        layer.quant_method = SimpleNamespace(moe_kernel=kernel)
        paired_nvfp4_autotune(layer, small_max_n=small_max_n)
        return
    import vllm.utils.flashinfer as fi_utils
    buckets = fi_utils.flashinfer_get_hybrid_num_tokens_buckets(max_tokens)
    x = torch.randn(max_tokens, K, dtype=torch.bfloat16, device=device) * 0.1
    logits = torch.randn(max_tokens, E, dtype=torch.float32, device=device)
    with fi_utils.autotune(tuning_buckets=buckets):
        fwd(x, logits)
    torch.cuda.synchronize()


if __name__ == "__main__":
    sys.exit(main())
