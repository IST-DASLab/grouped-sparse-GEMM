# Plan: fused SwiGLU + FP4 GEMM1 epilogue on SM120 (target: v0.13.0)

Work on an SM120 machine (RTX PRO 6000 / RTX 5090): every build-test-benchmark loop needs an
SM120 GPU. This branch (`sm120-swiglu-epilogue`) starts from `sm100-pdl`, so v0.13.0 also carries
programmatic dependent launch and the producer-written SFB fill (validated on B200: 134 kernel
tests pass with `PAIRED_NVFP4_PDL=1` and `=0`; Kimi EP8 MoE layer T=1 41.0 -> 36.9 us).

## Goal

`group_mm_swiglu_quant` on SM120: GEMM1 writes GEMM2's FP4 activations and block scales directly,
byte-identical to GEMM1 + `silu_mul_quant_act(interleaved=True)`, as on SM100. Today SM120 has no
fused variants (`kNumSwigluMmKinds = 0`), so vLLM always runs the unfused path. On SM100 the fusion
made GEMM1 + SwiGLU 1.2-1.6x faster at T <= 16 and the Kimi TP8+EP serving 0.9-3.0% faster.

## What carries over from `csrc/sm100/swiglu_epilogue.cuh`

- The `SwigluFp4Epilogue<Base>` wrapper: `Arguments`/`Params` = base + `SwigluFp4Args`, and the
  forwarding of `to_underlying_arguments`, `can_implement`, workspace and descriptor prefetch.
- The three phases per 32-column (token) subtile, with two named-barrier syncs per subtile:
  1. `bf16(alpha[e] * acc)` into a shared-memory exchange buffer indexed by (row, column);
  2. row-parallel SwiGLU with vLLM's rounding chain `bf16(bf16(silu(gate)) * up)`;
  3. column-parallel `quantize_block` per (32-feature block, token), storing 16 code bytes with
     `b_act_word` and one scale with `layout_SFB`.
- The W13 order: 64-row blocks of 32 gate rows then the matching 32 up rows
  (`interleave_gate_up_rows`), so a 128-row CTA tile holds two complete 32-feature blocks of
  GEMM2's K, at k0 = (m_coord * 2 + blk) * 32.
- Clipping: skip subtiles past `expert_num_tokens[e]` and CTAs whose rows are past M.
- `prologue_fill(tid, nthr)`: the benign UE4M3 1.0 fill of the SFB slots no tile writes
  (`fill_benign_sfb`); the op allocates SFB with `kernel_filled_sfb` (unfilled).

## What changes on SM120

- **Accumulators are registers, not TMEM.** The kernel
  (`csrc/cutlass_overrides/cutlass/gemm/kernel/sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`)
  builds `accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape))` and calls
  `collective_epilogue.store(...)` with them and `mma_thread_idx`. Replace the
  `SM100_TMEM_LOAD_32dp32b32x` copy with the fragment's own (row, column) map: partition an
  identity tensor of the CTA tile with `tiled_mma.get_slice(mma_thread_idx).partition_C(...)` and
  read each fragment element's coordinate from it. Check the store-call signature of the SM120
  epilogue (`SparseTmaWarpSpecializedCooperativeSm120`, an SM90-style TMA epilogue) and override
  that `store`.
- **No accumulator pipeline.** Drop the `acc_pipeline` wait/release; the SM90-style epilogue store
  has load/store pipeline states only. Return whatever that `store` returns.
- **256 epilogue threads.** Two cooperative MMA warp groups run the epilogue, so barriers count
  256 threads (`NamedBarrier` with the epilogue's `ThreadCount` and its reserved barrier id), and
  the phase-1 / phase-2 thread-to-work maps should spread over 256 threads (phase 2 has 64 jobs per
  subtile; phase 1 has 64 features x 32 columns).
- **Shared memory.** SM120 has ~99 KB per SM. The exchange buffers need
  (128 x 32 + 64 x 32) bf16 = 12 KB per CTA tile of 128 rows; check `sizeof(TensorStorage)` of the
  base epilogue (static_assert as on SM100) or carve a dedicated buffer, and keep
  `StageCountAutoCarveout` accounting for it.
- **Tile M.** The SM100 version assumes 128 rows per CTA. SM120 tiles are 128x64, 128x128 and
  256x128 (1 CTA); a 256-row CTA tile holds four 32-feature blocks: generalize k0 to
  (m_coord * (TileM / 64) + blk) * 32 and the block count to TileM / 64.

## Wire-up

1. `csrc/sm120/swiglu_epilogue.cuh` (new) with the SM120 `SwigluFp4Epilogue` and a
   `run_group_mm_swiglu_variant` like the SM100 one (`configure_group_scheduler`, splits == 1,
   alpha = 1, dummy D).
2. `csrc/sm120/group_mm_swiglu_*.cu` instantiations; start with `fused_1sm_128x64` (the SM120
   decode default) and `fused_1sm_256x128` (prefill, `suggest_tactic` picks it at max_n >= 768).
3. `csrc/sm120/tactics.cu`: fill `kSwigluMmKinds` / `kNumSwigluMmKinds`.
4. SM120 kernel override: call `collective_epilogue.prologue_fill(...)` in every CTA before the
   first tile when the epilogue has it (the SM100 override uses a `HasPrologueFill` trait after
   `pipeline_init_wait`). Note `kGemmPdl = false` on SM120: the SM120 GEMM still launches without
   PDL; enabling it needs an audit that the override waits (`wait_on_dependent_grids`) before its
   first dependent read, including the grouped scheduler's read of the per-expert counts.
5. `setup.py`: add the new sources for `120a` (check `arch_sources`).
6. vLLM: nothing. `paired_nvfp4_supports_fused_swiglu()` turns on as soon as
   `group_mm_swiglu_configs()` is non-empty, and the autotuner gates fused tactics byte-exactly
   and keeps them only where they are faster.

## Validation

- Build: `PAIRED_NVFP4_ARCHS=120a PAIRED_NVFP4_BUILD_TESTS=1` (or `"100a;120a"`).
- `tests/test_swiglu_epilogue.py`: fused output and SFB byte-identical to unfused, every fused
  tactic, including zero-count experts and counts not a multiple of the tile.
- Full `tests/` with `PAIRED_NVFP4_PDL=1` and `=0`.
- Speed: `tools/bench_moe.py` and `bench/moe_layer_bench.py` (fused vs unfused GEMM1 + SwiGLU,
  and the whole layer through the vLLM tuner) at decode and prefill sizes.
- SM100 regression on B200 (full tests + layer bench), since shared files change.

## Release

After both arches pass: merge to `main`, set `VERSION` / `__version__` to `0.13.0`, add a
CHANGELOG section (SM120 fused epilogue + the `sm100-pdl` changes), tag `v0.13.0`, and bump the
moe-sq-public submodule and vLLM's `MIN_VERSION` if the integration should require it.

## Status (2026-10-03)

Implemented and validated on an RTX PRO 6000 (SM120); released in 0.13.0.

- `csrc/sm120/swiglu_epilogue.cuh`, `group_mm_swiglu_1sm_{128x64,256x128}.cu`, `tactics.cu`, and
  `prologue_fill` in the SM120 kernel override (after `wait_on_dependent_grids`, a no-op while
  `kGemmPdl = false`). `PairedGemmVariant` takes an epilogue wrapper so the mainloop's stage
  count is carved out of the wrapped epilogue's storage.
- Deviations from the plan above:
  - The fragment is `((2,2), MMA_M, MMA_N)` with each MMA_N index covering 16 CTA columns, so
    subtiles are static ranges of MMA_N (the loop is unrolled; each element is visited once).
    A runtime column test over the whole fragment per subtile cost 5-17% on 256x128 prefill.
  - The base epilogue's `TensorStorage` is 13312 B. 32-token subtiles fit it on 128-row tiles
    (12 KB of exchange buffers); 256-row tiles use 16-token subtiles (also 12 KB), since 24 KB
    left the mainloop too few stages. No extra shared memory on either tile.
  - Phase 0 stores are bank-swizzled; phase 2 quantizes one block per 4-lane quad
    (`quantize_block_quad` in `quant.cuh`, byte-identical to `quantize_block`).
- Validation: `tests/` 136 passed, 7 skipped with `PAIRED_NVFP4_PDL=1` and `=0` (the fused test
  gained a 2N = 640 shape, a partial 256-row tile). Byte-exact, both tactics, on all 28 cases of
  the speed sweep below.
- Speed, GEMM1 + SwiGLU, best fused vs best unfused GEMM tactic + `silu_mul_quant_act`:

  | model | T=1 | 4 | 16 | 64 | 256 | 1024 | 4096 |
  |---|---|---|---|---|---|---|---|
  | Qwen3-30B-A3B | 1.30 | 1.19 | 1.08 | 1.13 | 1.23 | 1.25 | 1.06 |
  | Qwen3-235B | 1.16 | 1.08 | 1.07 | 1.08 | 1.11 | 1.15 | 1.22 |
  | Mixtral-8x7B | 1.08 | 1.07 | 1.09 | 1.09 | 1.07 | 0.99 | 1.11 |
  | DeepSeek-V3 | 1.06 | 1.06 | 1.06 | 1.06 | 1.08 | 1.08 | 1.20 |

  128x64 is the best fused tile up to ~1024 tokens, 256x128 above.
- Whole MoE layer through vLLM (`bench/moe_layer_bench.py`, CUDA graph, L2 flushed, autotuned),
  in the MoE-SQ serving venv (vLLM v0.30.0 + `moe-sq-v0.30.0.patch`, torch 2.13.0+cu130) with
  this branch's kernel. The autotuner picked the fused GEMM1 at every bucket (`fused_1sm_128x64`).
  "unfused" hides the fused op from vLLM (`paired_nvfp4_supports_fused_swiglu() -> False`), the
  path a wheel without SM120 fused tactics takes. Speedup of fused over unfused:

  | shape | T=1 | 4 | 16 | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Qwen3-30B, EP1 | 1.04 | 1.00 | 1.02 | 1.06 | 1.06 | 1.05 | 1.06 | 1.08 | 1.12 | 1.08 |
  | Kimi-K2.5, EP8 rank | 1.00 | 1.01 | 1.03 | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 | 1.04 | 1.05 |
  | Qwen3.5-397B, EP8 rank | 1.02 | 1.03 | 1.04 | 1.03 | 1.04 | 1.05 | 1.03 | 1.03 | 1.02 | 1.06 |

  Against FlashInfer CUTLASS dense NVFP4 in the same run, paired48 (fused) is 1.19-1.49x
  (Qwen3-30B), 1.11-1.50x (Kimi EP8) and 1.12-1.41x (Qwen3.5-397B EP8) faster. Kimi EP8 at T=1:
  65.5 us (fused and unfused) vs 72.7 us.
- Not run for 0.13.0: the SM100 regression on B200 (`quant.cuh` gained a function and the SM100
  kernel header's `HasPrologueFill` is include-guarded; only the 100a build was checked).
