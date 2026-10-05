"""Bucket an nsys cuda_gpu_kern_sum CSV into MoE stages, per forward.

Profile with BENCH_PROFILE_REPLAYS=K and `nsys profile --capture-range=cudaProfilerApi`, so the
report holds exactly K graph replays (plus K L2-flush fills, dropped here), then:

    python bench/kernel_buckets.py run.kern.csv --forwards K
"""

import argparse
import csv
import re

BUCKETS = [
    ("gemm", r"GemmUniversal|group_mm|Bmm_|bmm_|gemm|Gemm|grouped_gemm|MoeFC|moe_gemm|Sm100.*Mma"),
    ("act_quant", r"scatter_quant|quant_act|fp4_quant|cvt_fp4|scaled_fp4|Quantize|quantize|expandInputRows"),
    ("swiglu", r"silu|swiglu|doActivation|activation"),
    ("routing", r"topkGating|topk|routing|Routing|moe_sort|argsort|computeStrides|buildExpertMaps"),
    ("finalize", r"finalize|Finalize|reduce_kernel|unpermute|index_add"),
    ("sort_plan", r"RadixSort|radix_sort|bitonicSort|warpMergeSort|searchsorted|scatter_gather|indexSelect|index_select|gather"),
    ("elementwise", r"elementwise|FillFunctor|arange|where|compare|copy|BinaryFunctor|mul|add|memset"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--forwards", type=int, default=20, help="BENCH_PROFILE_REPLAYS")
    args = ap.parse_args()
    rows = list(csv.reader(open(args.csv)))
    h = next(i for i, r in enumerate(rows) if r and r[0].startswith("Time"))
    hdr = rows[h]
    kernels = [dict(zip(hdr, r)) for r in rows[h + 1:] if len(r) == len(hdr)]
    per_bucket, per_kernel, launches = {}, [], 0
    for k in kernels:
        n = int(k["Instances"])
        calls = n / args.forwards
        us = float(k["Total Time (ns)"]) / args.forwards / 1e3
        name = k["Name"]
        if "FillFunctor<unsigned char>" in name and us > 20:
            continue  # the harness's 512 MiB L2 flush, not part of the layer
        bucket = next((b for b, pat in BUCKETS if re.search(pat, name)), "other")
        per_bucket[bucket] = per_bucket.get(bucket, 0.0) + us
        per_kernel.append((us, calls, bucket, name))
        launches += calls
    total = sum(per_bucket.values())
    print(f"{args.csv}: {total:.1f} us GPU time / forward, {launches:g} launches / forward")
    for b, us in sorted(per_bucket.items(), key=lambda kv: -kv[1]):
        print(f"  {b:12s} {us:8.1f} us  {100 * us / total:5.1f}%")
    print("  top kernels:")
    for us, calls, b, name in sorted(per_kernel, reverse=True)[:12]:
        print(f"    {us:8.1f} us  x{calls:g}  [{b}] {name[:110]}")


if __name__ == "__main__":
    main()
