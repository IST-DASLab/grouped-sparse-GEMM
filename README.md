# grouped-sparse-GEMM

[![arXiv](https://img.shields.io/badge/arXiv-2610.02241-b31b1b.svg)](https://arxiv.org/abs/2610.02241)

A grouped GEMM for Mixture-of-Experts layers whose expert weights are **paired-4:8 sparse and
NVFP4 quantized**, with NVFP4 activations (W4A4). It runs the expert projections on the sparse
tensor cores of NVIDIA Blackwell (SM100 and SM120): each expert's weight takes half the bytes of dense
NVFP4 (plus 4:8 metadata) and is multiplied at twice the dense NVFP4 math throughput.

The package ships as a PyTorch extension, `paired_nvfp4_kernels`, that registers
`torch.ops.paired_nvfp4.*`: a load-time weight compressor, three fused activation quantizers,
and the grouped GEMM itself. All per-forward ops are CUDA-graph-capture safe.

| | |
|---|---|
| Arch | SM100 (B200, GB200) and SM120 (RTX 5090, RTX PRO 6000); one extension module per arch, see [docs/porting.md](docs/porting.md) for the SM120 port |
| Toolkit | CUDA >= 12.8 (validated with 13.0), PyTorch with CUDA (validated with 2.11), a host compiler PyTorch accepts (GCC >= 9) |
| Weights | paired-4:8 sparse, e2m1 values, one UE4M3 scale per 32 K elements, per-expert fp32 alpha |
| Activations | bf16 in, quantized on the fly to e2m1 + UE4M3 (per 32), optional per-expert global scale |
| Output | bf16 |
| Built on | [CUTLASS](https://github.com/NVIDIA/cutlass) v4.8.0, plus two patched kernel headers adding grouped (MoE) support to the SM100 and SM120 sparse GEMMs |

## Install

```bash
git clone --recurse-submodules --shallow-submodules https://github.com/IST-DASLab/grouped-sparse-GEMM
cd grouped-sparse-GEMM
# inside the environment whose torch you will run with:
MAX_JOBS=8 pip wheel . --no-deps --no-build-isolation -w dist
pip install --no-deps dist/paired_nvfp4_kernels-*.whl
python -c "import paired_nvfp4_kernels as p; print(p.is_available(), p.group_mm_configs())"
```

`--no-build-isolation` is required because `setup.py` imports torch to build the extension, and
the extension must link against the torch you run with. Install with `--no-deps`: the wheel
declares an unpinned `torch` dependency, and letting pip resolve it can replace a pinned
`torch==X+cuYYY` (and then the extension fails to load with an undefined `torch::Library` symbol).

Build options (environment variables): `PAIRED_NVFP4_ARCHS` (default `100a`; `120a` for SM120,
`"100a;120a"` for one wheel carrying both, which loads the module matching the device),
`PAIRED_NVFP4_BUILD_TESTS=1` (compile the self-test ops the test suite uses), `CUTLASS_DIR`
(default `third_party/cutlass`), `MAX_JOBS` (the tile variants compile in parallel).

## Quick start

```python
import torch
import paired_nvfp4_kernels
from paired_nvfp4_kernels import reference

ops = torch.ops.paired_nvfp4
E, M, K, max_n = 8, 2048, 1024, 256            # experts, out_features, in_features, token capacity

# Load time: prune + quantize (normally done offline), then compress once.
w = reference.prune_paired_4of8(torch.randn(E, M, K, device="cuda"))
w_packed, w_blockscale, _ = reference.quantize_weight(w)       # uint8 [E,M,K/2], uint8 [E,M,K/32]
a_comp, e_meta, sfa = ops.compress(w_packed, w_blockscale, K)

# Every forward: per-expert padded activations + real token counts.
x = torch.randn(E, max_n, K, device="cuda", dtype=torch.bfloat16)
counts = torch.randint(1, max_n + 1, (E,), device="cuda", dtype=torch.int32)
gscale = torch.ones(E, device="cuda")                           # activation global scale
b_act, sfb = ops.quant_act(x, gscale, counts, M)

alphas = torch.ones(E, device="cuda")                           # per-expert dequant scalar (global scales)
out = torch.empty(E, max_n, M, device="cuda", dtype=torch.bfloat16)
ops.group_mm(out, a_comp, e_meta, sfa, b_act, sfb, alphas, counts, M, K)
# out[e, :counts[e]] == alphas[e] * dequant(b_act[e]) @ dequant(w[e]).T ; rows past counts[e] untouched
```

## Weight format

`compress` takes an ordinary dense NVFP4 tensor that already satisfies the sparsity pattern; the
4:8 compression and metadata are computed at load time. For each expert weight `W[M, K]`:

1. **Paired 4:8 sparsity along K.** In every group of 8 consecutive K elements, i.e. 4 adjacent
   pairs, exactly 2 pairs are zero. The pruning unit is a *pair* of fp4 values, which is what
   the SM100 sparse MMA's metadata encodes; an element-wise 4-of-8 pattern is not representable.
2. **NVFP4 values with 32-element scales.** `s = ue4m3(amax(|W| over 32 K elements) / 6)`,
   `q = e2m1(W / s)`. The sparse MMA reads one scale per 32 dense K elements (16 of which survive
   pruning), not one per 16 as in dense NVFP4, so standard NVFP4 checkpoints (group size 16) are
   not compatible without requantization.
3. **Packing.** Two e2m1 codes per byte, lower K index in the low nibble: `w_packed` is
   `uint8 [E, M, K/2]`; `w_blockscale` holds raw UE4M3 bytes in natural order, `[E, M, K/32]`.

Per-tensor global scales (weight and activation) are folded into the per-expert `alphas`.
[`paired_nvfp4_kernels/reference.py`](paired_nvfp4_kernels/reference.py) is the executable spec
(prune, quantize, pack, dequantize) and is what the tests pack weights with.

## Ops

Shapes use `E` experts, `M` = out_features, `K` = in_features, `max_n` = per-expert token
capacity. All tensors are CUDA and contiguous.

| op | when | inputs | outputs |
|---|---|---|---|
| `compress(w_packed, w_blockscale, k)` | load | `uint8 [E,M,K/2]`, `uint8 [E,M,K/32]` | `a_comp`, `e_meta` (uint8, kernel layouts), `sfa` (uint8, swizzled) |
| `quant_act(x, gscale, expert_num_tokens, features)` | forward | `bf16 [E,max_n,K]`, `f32 [E]` or `[1]`, `i32 [E]` | `b_act uint8 [E,max_n,K/2]`, `sfb uint8` (swizzled) |
| `silu_mul_quant_act(x2, gscale, expert_num_tokens, features, interleaved=False, activation="silu", beta=1.0, linear_beta=-1.0)` | forward | `bf16 [E,max_n,2N]` = `[gate \| up]` | `b_act uint8 [E,max_n,N/2]`, `sfb` |
| `scatter_quant_act(a1, flat_tok, dest_global, topk_weights, gscale, expert_num_tokens, cap)` | forward | token-order `bf16 [T,K]` + dispatch plan (`i64 [T*topk]` each), optional `f32 [T*topk]` | `b_act uint8 [E,cap,K/2]`, `sfb` |
| `dispatch_plan(topk_ids, first_expert, num_local_experts, cap)` | forward | `i32`/`i64 [T,topk]` global expert ids; this rank owns `[first_expert, first_expert+E)` | `expert_num_tokens i32 [E]`, `route_dest i32 [T,topk]`, `src_tok i32 [E*cap]`, `src_route i32 [E*cap]` |
| `scatter_quant_act_planned(a1, src_tok, src_route, topk_weights, gscale, expert_num_tokens, cap)` | forward | token-order `bf16 [T,K]` + the plan, optional `f32 [T*topk]` | `b_act uint8 [E,cap,K/2]`, `sfb` |
| `quant_rows(x, gscale)` | forward | token-order `bf16 [T,K]`, shared `f32` global scale | `q uint8 [T,K/2]`, linear `sf [T,K/32]` |
| `scatter_fp4_planned(q, q_sf, src_tok, expert_num_tokens, cap)` | forward | `quant_rows` output (e.g. after an all2all) + the plan | `b_act uint8 [E,cap,K/2]`, `sfb` |
| `moe_finalize(out, fused, route_dest, topk_weights)` | forward | `bf16 [E,cap,K]` expert output, the plan's `route_dest`, `f32 [T,topk]` or None | writes `out bf16 [T,K]` in place |
| `group_mm(out, a_comp, e_meta, sfa, b_act, sfb, alphas, expert_num_tokens, features, k, config_id=0, cluster_m=2, cluster_n=1)` | forward | the above, `f32 [E]`, `i32 [E]` | writes `out bf16 [E,max_n,M]` in place |
| `group_mm_configs()` | any | | tile-variant names, indexed by `config_id` |

Contracts worth knowing:

- **Per-expert padded layout.** Activations and outputs are `[E, max_n, ...]`, one fixed-size
  slot per expert, and `expert_num_tokens[e] <= max_n` gives the real count. The grouped kernel
  addresses experts at a fixed stride (CUTLASS `MoEProblemShape`); it cannot consume a
  summed-ragged `[sum tokens, K]` buffer.
- **Padding is never written.** Rows at or past `expert_num_tokens[e]` of `b_act` and `out` are
  left untouched (their bytes are undefined), and their scale slots hold a benign nonzero fill.
- **Quantization convention** (FlashInfer-compatible): `sfb = ue4m3(amax/6 * gscale[e])`,
  `b_act = e2m1(x * gscale[e] / sfb)`. Quantizing against the rounded stored scale leaves pure
  e2m1 rounding error.
- **`silu_mul_quant_act` is bit-exact** with `silu_and_mul` (bf16 rounding chain as in vLLM)
  followed by `quant_act`; with `activation="situ"` (Kimi's SiTU-GLU,
  `beta·tanh(g/beta)·sigmoid(g) · linear_beta·tanh(up/linear_beta)`, the up softcap off for
  `linear_beta <= 0`) it is bit-exact with vLLM's `situ_and_mul` instead. `group_mm_swiglu_quant`
  takes the same `activation, beta, linear_beta` arguments. **`scatter_quant_act`** fuses the dispatch gather/scatter with
  quantization: `dest_global[i]` is `e*cap + r`, or `E*cap` to drop a routing (non-local expert
  or over capacity); `topk_weights` applies the router weight on the input.
- **Dispatch and combine.** `dispatch_plan` gives each kept routing the row
  `e*cap + r` (`r` = its stable rank among this rank's routings to `e`) and marks the rest
  `route_dest = -1` (another rank's expert, or `r >= cap`); `src_tok` / `src_route` map each kept
  row back to its token and routing. `scatter_quant_act_planned` is byte-identical to
  `scatter_quant_act` on valid rows but only does work for kept rows. `moe_finalize` sums
  `bf16(row * bf16(w))` over a token's kept routings in fp32, in top-k order (bit-identical to the
  torch gather / multiply / sum it replaces); pass `topk_weights=None` when the router weight was
  applied on the input. Both are deterministic and never read padded rows.
- **`features`** is the consuming GEMM's M. The quantizers accept it for symmetry; `group_mm`
  checks it (and `k`) against the shapes `compress` produced.
- **CUDA graphs.** Forward ops only enqueue on the current stream; token counts are read on the
  device, never copied to the host.

## Tactics

A `group_mm` tactic is `(config_id, cluster_m, cluster_n)`.

**SM100** compiles eight tile variants
(`sparse_2sm_256x128`, `sparse_2sm_256x256`, `sparse_1sm_128x128`, `sparse_1sm_128x256`, and the
small-token tiles `sparse_2sm_256x64`, `sparse_1sm_128x64`, `sparse_2sm_256x64_k512`,
`sparse_1sm_128x64_k512`; `_k512` stages 512 K per pipeline step instead of 256) and takes the
thread-block cluster at runtime: each dimension 1, 2 or 4, at most 16 CTAs, and 2SM variants
need an even `cluster_m`. `paired_nvfp4_kernels.all_tactics()` enumerates the candidates; a
candidate the kernel cannot run for a given shape raises `RuntimeError` from `can_implement`, so
autotuners should catch and skip. The default, `(0, 2, 1)`, is a good general choice.

**SM120** has no cluster multicast, so the cluster is always 1x1 and any other cluster raises
`RuntimeError`. Its tile variants, by `config_id`:

| id | name | use |
|---|---|---|
| 0 | `sparse_1sm_128x128` | general |
| 1 | `sparse_1sm_128x128_batched` | reference path: computes (and writes) all `max_n` rows of every expert, same bytes on valid rows |
| 2 | `sparse_1sm_128x64` | **default** `(2, 1, 1)`: best single tile, especially at small token counts |
| 3 | `sparse_1sm_256x128` | compute-bound prefill (large `max_n`) |
| 4-7 | `sparse_1sm_128x{64,128}_sk{2,4}` | split-K: few tiles and long K, e.g. expert-parallel decode |

`paired_nvfp4_kernels.suggest_tactic(max_n)` gives a good choice without autotuning (256x128 for
`max_n >= 768`, else 128x64). A split-K variant needs at least as many 256-wide K tiles as splits
and otherwise raises the `can_implement` `RuntimeError`.

`paired_nvfp4_kernels.DEFAULT_TACTIC`, `valid_clusters`, `all_tactics()` and `suggest_tactic()`
follow the loaded arch.

## Tests

```bash
PAIRED_NVFP4_BUILD_TESTS=1 pip wheel . --no-deps --no-build-isolation -w dist
pip install --no-deps --force-reinstall dist/paired_nvfp4_kernels-*.whl
cd /tmp && python -m pytest /path/to/grouped-sparse-GEMM/tests -v
```

Run pytest from outside the source tree so `import paired_nvfp4_kernels` resolves to the installed
build. `test_registration.py` and `test_reference.py` run anywhere; the rest need a GPU matching
the loaded arch module (SM100 or SM120), and most need the self-test ops. `tools/ab_compare.py`
does a byte-level comparison of two builds, and `tools/validate_sm100.sh` runs it plus the tests
on a B200 against v0.10.0. `tools/bench_moe.py` benchmarks every tactic on real MoE shapes, and with
`--dense` against a dense NVFP4 grouped GEMM (SM120); docs/porting.md has its results.

## Repository layout

```
csrc/
  arch.cuh                  selects the arch config (PAIRED_NVFP4_SM)
  sm100/config.cuh          all SM100 types: elements, tiles, schedules, layouts, compressor
  sm100/group_mm_*.cu       one CUTLASS instantiation per tile variant
  sm100/tactics.cu          tile table and cluster legality
  sm120/                    the same three pieces for SM120
  layout.cuh                operand shapes, SF layouts, SF scatter        (arch-independent)
  quant.cuh                 the three activation quantizers                (arch-independent)
  group_mm.cuh              tile-variant runner                            (arch-independent)
  ops.cu, ops.h             validation, allocation, torch registration
  testing/                  self-test ops (PAIRED_NVFP4_BUILD_TESTS=1)
  cutlass_overrides/        the patched CUTLASS headers, first on the include path
paired_nvfp4_kernels/       Python package: arch-module loader, tactics, reference.py
patches/                    the same CUTLASS changes as patches against v4.8.0
third_party/cutlass         CUTLASS v4.8.0 (submodule)
docs/                       design notes and the porting guide
```

## CUTLASS dependency

The only non-stock code the kernels need from CUTLASS is two headers. Stock CUTLASS (v4.8.0 is
the newest release) implements the block-scaled sparse GEMMs for single and batched problems only;
the patched headers add the grouped path:

- `cutlass/gemm/kernel/sm100_sparse_gemm_tma_warpspecialized.hpp` (SM100): detects
  `MoEProblemShape`, selects the group tile scheduler, computes each work tile's shape from the
  per-expert token count, and gives the non-dynamic scheduler its own pipeline instead of the CLC
  fetch pipeline.
- `cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp` (SM120):
  detects `MoEProblemShape`, sets every collective up with the max shape `(M, max_n, K, E)`
  (the batched addressing), selects the SM120 group tile scheduler, and adds the producer warp
  and pipeline that the static scheduler needs (the stock kernel only drives the dynamic one).

Both files are carried in `csrc/cutlass_overrides/`, which setup.py puts ahead of the CUTLASS
include path; the same changes are in `patches/` as diffs against v4.8.0, and each file carries a
note naming the change above NVIDIA's unchanged license header. A compile-time check fails the
build if a stock header is picked up instead.

## Limitations

- On SM120 the grouped kernel pays a fixed ~1-3 µs to find each CTA's first tile, and split-K only
  helps with few, long-K tiles; see [docs/porting.md](docs/porting.md).
- The per-expert padded layout sizes buffers by `E * max_n`, not by routed tokens.
- The fused activation is SwiGLU (`silu(gate) * up`) only.
- No fake (meta) implementations yet, so `torch.compile` cannot trace through the ops; wrap them
  in a custom-op boundary (as vLLM's MoE layer does).

## License

Apache-2.0, see [LICENSE](LICENSE). Copyright (C) 2026 Kwanhee Lee and Dan Alistarh.

The C++/CUDA sources derived from CUTLASS keep NVIDIA's copyright and BSD-3-Clause license in
their headers, marked "Modified by Kwanhee Lee and Dan Alistarh". [NOTICE](NOTICE) reproduces the
CUTLASS and pybind11 licenses; both files ship inside the wheel.

## Citation

This kernel accompanies the paper below; the compression framework is
[MoESQ](https://github.com/IST-DASLab/MoESQ). If you use it, please cite
([DOI: 10.48550/arXiv.2610.02241](https://doi.org/10.48550/arXiv.2610.02241)):

```bibtex
@misc{lee2026hardwarenativejointsparsequantizationtrillionscale,
      title={Hardware-Native Joint Sparse-Quantization for Trillion-Scale Mixture-of-Experts},
      author={Kwanhee Lee and Namhoon Lee and Dan Alistarh},
      year={2026},
      eprint={2610.02241},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      doi={10.48550/arXiv.2610.02241},
      url={https://arxiv.org/abs/2610.02241},
}
```
