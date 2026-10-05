# Changelog

## 0.13.0

SM120 fused SwiGLU + FP4 GEMM1 epilogue, programmatic dependent launch (PDL) across the MoE chain,
and no separate SFB fill kernels. The CUTLASS patches in `patches/` are regenerated to match the
kernel overrides again (they had not picked up the PDL wait and the prologue fill).

Programmatic dependent launch (PDL) across the MoE chain, and no separate SFB fill kernels.
- `dispatch_plan`, the compacted quantizers (`quant_act`, `silu_mul_quant_act`,
  `scatter_quant_act_planned`, `scatter_fp4_planned`), `moe_finalize` and, on SM100, both GEMMs
  launch with the programmatic-stream-serialization attribute (`csrc/pdl.cuh`), so each kernel's
  launch and prologue overlap its predecessor's tail. Every such kernel waits
  (`griddepcontrol.wait`) before its first dependent global read; the SM100 GEMM kernel override
  waits before constructing the grouped scheduler (which reads the per-expert counts). SM100
  builds define `CUTLASS_ENABLE_GDC_FOR_SM100`; the SM120 GEMM still launches without PDL.
  `PAIRED_NVFP4_PDL=0` turns it off.
- The benign SFB pre-fill (`torch::full` per call) is written by the producers themselves:
  those quantizers and the fused SwiGLU epilogue (in every CTA, before its first tile) fill every
  slot they do not quantize (`fill_benign_sfb`). The output bytes are unchanged.
- Whole MoE layer on B200 (`bench/moe_layer_bench.py`, CUDA graph): Kimi-K2.5 EP8 rank
  1.11/1.08/1.05/1.02/1.04x at T=1/4/16/256/1024 (41.0 -> 36.9 us at T=1), Qwen3-30B
  1.06/1.14/1.10/1.06/1.05x.

Fused SwiGLU + FP4 GEMM1 epilogue on SM120 (`csrc/sm120/swiglu_epilogue.cuh`).
- `group_mm_swiglu_quant` on SM120, tactics `fused_1sm_128x64` and `fused_1sm_256x128`;
  byte-identical to `group_mm` -> `silu_mul_quant_act(interleaved=True)`, as on SM100. vLLM's
  `paired_nvfp4_supports_fused_swiglu()` turns on with no vLLM change.
- The store reads the mma.sync accumulator fragments of both MMA warp groups directly (each
  MMA_N index covers 16 CTA columns, so subtiles are static), uses 32-token subtiles on 128-row
  tiles and 16-token subtiles on 256-row tiles so the exchange buffers fit in the base
  epilogue's shared storage, and quantizes one block per 4-lane quad (`quantize_block_quad`).
  The SM120 kernel override calls the epilogue's `prologue_fill` (benign SFB fill) in every CTA.
- GEMM1 + SwiGLU on an RTX PRO 6000, best fused vs best unfused tactic + `silu_mul_quant_act`
  (Qwen3-30B-A3B / Qwen3-235B / Mixtral-8x7B / DeepSeek-V3, 1-4096 tokens): 1.06-1.30x
  everywhere except Mixtral at 1024 tokens (0.99x); Qwen3-30B 1.30/1.19/1.08x at T=1/4/16.
- Whole MoE layer through vLLM on an RTX PRO 6000 (`bench/moe_layer_bench.py`), fused vs
  unfused: Qwen3-30B 1.00-1.12x, Kimi-K2.5 EP8 rank 1.00-1.05x, Qwen3.5-397B EP8 rank
  1.02-1.06x over T=1-4096.

## 0.12.0

SM120 (RTX 5090, RTX PRO 6000) integration completed.
- SM120 default tactic is `sparse_1sm_128x64` (config_id 2). `suggest_tactic(max_n)` picks a tile
  without autotuning: `sparse_1sm_256x128` for max_n >= 768, `sparse_1sm_128x64` otherwise
  (within 0.5% of the per-shape best on the geometric mean, 7% at worst). On SM100 it returns
  `DEFAULT_TACTIC`.
- SM120 split-K reduces in one step: the final split sums the other splits' partials in split
  order (deterministic), instead of a chained split-order reduction.
- Against the best dense NVFP4 grouped GEMM on SM120, 1.07-1.56x faster (median 1.43x) over 48
  realistic MoE cases (docs/porting.md).
- `tools/bench_moe.py` (realistic-MoE benchmark, optional dense baseline in `tools/dense_baseline/`)
  and `tools/validate_sm100.sh` (B200 re-validation against v0.10.0).

## 0.11.0

SM100 performance release: fused MoE dispatch/combine ops, 64-token tiles and a GEMM1 epilogue
with fused SwiGLU + FP4 quantization. It also carries the first SM120 backend (below); the
SM120 port is completed in 0.12.0.

Fused SwiGLU + FP4 GEMM1 epilogue (SM100): `group_mm_swiglu_quant` computes GEMM1 and writes
GEMM2's FP4 activations and block scales directly, skipping the bf16 GEMM1 output and the
separate `silu_mul_quant_act` launch.
- Custom epilogue collective (`csrc/sm100/swiglu_epilogue.cuh`) over the stock TMA epilogue:
  TMEM accumulators to shared memory, then the exact vLLM chain bf16(bf16(silu(g)) * up) and
  group-32 UE4M3 quantization. Byte-identical to GEMM1 + `silu_mul_quant_act`.
- W13 rows must be interleaved in 64-row blocks (32 gate rows, then the matching 32 up rows):
  `interleave_gate_up_rows(n)`; `silu_mul_quant_act(..., interleaved=True)` reads that order so
  the unfused path can share the weights.
- Four tile variants (`group_mm_swiglu_configs()`, `fused_swiglu_tactics()`): 2SM 256x64
  (K 256/512), 1SM 128x64, 2SM 256x128. Work is clipped at each expert's token count and at M.
- GEMM1 + SwiGLU on a Kimi-K2.5 EP8 rank: 1.59/1.22/1.09/1.10x at T=1/16/256/4096. With the
  vLLM autotuner choosing fused or unfused per capacity bucket, Kimi-K2.5 TP8+EP serving on B200
  gains 0.9-3.0% decode and 1.3% prefill throughput, greedy outputs identical.

SM100 small-token tiles: `sparse_2sm_256x64`, `sparse_1sm_128x64` and their 512-deep-K
versions `_k512` (config ids 4-7), through an overridden block-scaled sparse collective that
allows Cta N = 64 (`patches/cutlass-v4.8.0-sm100-sparse-ctan64.patch`). Every new tactic is
byte-identical to the default. The vLLM autotuner picks `sparse_2sm_256x64_k512` 2x1 for the decode
bucket: whole MoE layer 1.02-1.07x faster on a Kimi-K2.5 EP8 rank and 1.08-1.17x on Qwen3-30B for
T <= 512 (B200, `bench/moe_layer_bench.py`, fused dispatch arm); larger T is unchanged.

SM120 (RTX 5090, RTX PRO 6000) backend.
- New `_C_sm120` module (`PAIRED_NVFP4_ARCHS=120a`, or `"100a;120a"` for one wheel with both):
  `csrc/sm120/` config, the grouped `sparse_1sm_128x128` tile (default tactic `(0, 1, 1)`) and a
  batched reference variant `sparse_1sm_128x128_batched`.
- Grouped (MoEProblemShape) support for the SM120 sparse GEMM, vendored as
  `csrc/cutlass_overrides/.../sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp` and
  `patches/cutlass-v4.8.0-sm120-sparse-grouped.patch`. See docs/porting.md.
- SM120 clusters are always 1x1; `group_mm` rejects any other cluster.
- `group_mm.cuh` picks the GEMM mode (grouped or batched) from each variant's problem shape.
- Python: `DEFAULT_TACTIC`, `valid_clusters` and `all_tactics()` follow the loaded arch.
- Tests: arch-aware cluster rejection test; new test for zero-count experts and untouched rows
  past each expert's last tile.
- SM120 tiles: `sparse_1sm_128x64` and `sparse_1sm_256x128`; split-K variants
  `sparse_1sm_128x{64,128}_sk{2,4}` (`PersistentTileSchedulerSm120GroupSplitK`, a new header next
  to the SM120 kernel override; deterministic split-order reduction through per-slot buffers).
- SM120 tuning (docs/porting.md): the group scheduler rasters AlongN (Mixtral-8x7B w13 9-12%
  faster at 512-2048 tokens, other shapes within about 1%); `quant_launch` caps its row grid by
  the SM count (`kQuantRowBlocksPerSm`, 8 on SM120; SM100 keeps its ~2048), 11-21% faster
  quantizers at small and mid token counts for 128/256-expert models.

New ops, arch-independent (`csrc/moe_routing.cuh`), so every arch module gets them:
- `dispatch_plan`: routing ids to per-expert rows in one launch at decode sizes (three above 1024
  routings), replacing the torch argsort / searchsorted plan (~20 launches). Same rows as before.
- `scatter_quant_act_planned`: the scatter quantizer over kept rows only. Byte-identical to
  `scatter_quant_act` on valid rows.
- `moe_finalize`: the top-k weighted combine back to token order, reading only kept rows.
  Bit-identical to the torch gather / multiply / sum.
- `quant_rows` + `scatter_fp4_planned`: quantize token rows once (for example before an all2all,
  so FP4 travels instead of bf16) and place the received rows into GEMM1's batched operands. With
  one global scale, byte-identical to `scatter_quant_act_planned` on the bf16 rows.

On one Kimi-K2.5 EP8 rank (48 of 384 experts, B200, whole MoE layer under a CUDA graph, measured
with `bench/moe_layer_bench.py`), swapping vLLM's torch dispatch/combine for these ops takes the
layer from 102 to 49 us at T=1, 293 to 191 us at T=256 and 2114 to 404 us at T=8192.

- The three quantizers rebuild the per-block counts prefix with a warp scan instead of one thread
  looping over all experts. Every block runs that prologue and the grid is sized for E * max_n
  rows, so it dominated when most blocks have few rows: 3-7% off the whole Qwen3-30B MoE layer
  (128 experts), neutral on a Kimi EP8 rank (48 experts). Bytes unchanged.

Fixes
- `scatter_quant_act` no longer stores a block to the trash row for every dropped routing. Under
  EP that was most routings writing the same row; nothing ever read it.

Tools
- `bench/moe_layer_bench.py`, `bench/e2e_tp_serving.sh`, `bench/kernel_buckets.py`: layer-level
  and serving comparisons against the FlashInfer NVFP4 MoE backends.

## 0.10.0

(Tagged `v0.10.0`; `setup.py` at that commit still reads `0.1.0`.)

First standalone release. Extracted from the CUTLASS fork where the kernel was developed
(`paired_nvfp4_kernels` 0.0.10, under
`examples/92_blackwell_moe_gemm/python_bench/paired_nvfp4_kernels`). The op names, schemas and the
Python import name are unchanged, so existing callers (the vLLM `paired48_nvfp4` MoE backend)
need no changes. The GEMM and quantizer device code is restructured (shared helpers, no duplicated
scan) but computes the same bytes: on B200, `tools/ab_compare.py` found all 125 outputs
byte-identical to 0.0.10 (compress, the three quantizers, and group_mm under all 30 tactics on
three shapes).

Provenance (paths relative to the old package directory):

| 0.0.10 | 0.10.0 |
|---|---|
| `csrc/kernel.cuh` | split: `csrc/sm100/config.cuh` (types), `csrc/layout.cuh` (shapes, SF scatter), `csrc/quant.cuh` (quantizers) |
| `csrc/group_mm_impl.cuh` | `csrc/group_mm.cuh` |
| `csrc/group_mm_*.cu` | `csrc/sm100/group_mm_*.cu` |
| `csrc/ops.cu`, `csrc/ops.h` | same names; tactic table moved to `csrc/sm100/tactics.cu` |
| `csrc/testutil.cu` | `csrc/testing/selftest.cu` (+ `legacy_quant.cuh`) |
| `tests/test_*.py`, `probe_act_layout.py` | `tests/test_selftests.py`, `tests/test_ops.py` (pytest) |
| `tools/pack_paired48_checkpoint.py` | dropped; packing math in `reference.py` |
| CUTLASS fork `include/.../sm100_sparse_gemm_tma_warpspecialized.hpp` | `csrc/cutlass_overrides/...` + `patches/` |
| fork `examples/92_blackwell_moe_gemm/*.cu`, `python_bench/*` (prototype ops, benchmarks) | not carried over |

Packaging
- Licensed under Apache-2.0 (Kwanhee Lee and Dan Alistarh). C++/CUDA files derived from CUTLASS
  keep NVIDIA's BSD-3-Clause notice with a "Modified by" note; `NOTICE` carries the CUTLASS and
  pybind11 licenses.
- CUTLASS is a pinned submodule (v4.8.0, the newest release) instead of the whole repository
  being a CUTLASS fork (which was based on v4.5.1). Rebasing the modified header onto v4.8.0 picked
  up only upstream's own edits to that file (SMEM / TMEM capacities now read from `ArchTag`).
  The one modified CUTLASS header is vendored in `csrc/cutlass_overrides/` and also shipped as a
  patch in `patches/`.
- Sources split into an arch config (`csrc/sm100/`) and arch-independent layers, and the build
  produces one extension module per target arch (`_C_sm100`) to prepare other ports.
- Dropped the vestigial `wheel` entry from `build-system.requires`, which broke
  `python -m build --no-isolation`.

Behavior
- `group_mm` validates operand dtypes, contiguity and sizes against what `compress` produced for
  the given `features` and `k`, so a wrong `features` / `k` raises instead of computing garbage.
- The arch check and SM-count query are cached per device (previously every op call queried the
  device properties).
- `scatter_quant_act` with zero routed entries is a no-op instead of launching an empty grid.

Removed
- The non-grouped GEMM self-tests (`selftest_nongrouped*`), diagnostics for an investigation that
  is closed.
- The prototype extension and benchmarks that preceded the package; they remain in the fork.
- `tools/pack_paired48_checkpoint.py` (vLLM-specific checkpoint minting); its packing functions
  live on in `paired_nvfp4_kernels/reference.py`.

Tests
- The standalone gate scripts are now a pytest suite. Added: CPU tests of the reference format,
  a check that every tactic matches the default, the per-32 packing test in the default run, and
  tests of the new argument validation.
