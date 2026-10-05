# The SM120 port (RTX 5090, RTX PRO 6000 Blackwell)

Status: grouped and batched SM120 backends are built and pass the full test suite on an RTX PRO
6000 (188 SMs). Tile coverage and scheduler tuning are still open (see the end). File and line
references are to CUTLASS **v4.8.0**, the version the repository pins.

## What carried over unchanged

- **The checkpoint format.** The SM120 sparse NVFP4 MMA also reads one UE4M3 scale per 32 K
  elements (`sm1xx_common.inl:472-477`, `:173`); `sm120/config.cuh` static_asserts it. Weights
  keep the same paired-4:8, per-32 packing, and `reference.py` stays the spec. `compress` runs
  CUTLASS's SM120 `StructuredSparseCompressor` specialization, so the in-memory A / E layouts are
  SM120's own, built from the same checkpoint bytes.
- **The operand layouts.** RowMajor sparse A (alignment 64), ColMajor B (alignment 32) and
  ColMajor C / D are exactly what the upstream SM120 example
  (`examples/80_blackwell_geforce_sparse_gemm/80b_...`) uses.
- **The quantizers, the layout helpers, `ops.cu`, the op schemas and the tests.** They only
  consume the config aliases, which `sm120/config.cuh` defines under the same names.

## What differs on SM120

- The block-scaled sparse GEMM is a different kernel family: warp-level `mma.sync` with register
  operands, one producer warp group (A/E/SF loads, B loads, scheduler, C loads) and two
  cooperative MMA warp groups (`sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`).
- The cluster is a static 1x1x1 (`sm120_blockscaled_sparse_mma_builder.inl:179`: no programmatic
  multicast), so every tactic is `(config_id, 1, 1)` and `check_cluster` rejects anything else.
- Shared memory is ~99 KB per SM. The builder sizes stages automatically; when two full stages do
  not fit, it falls back to 1.5 stages of B with the metadata read from L2
  (`sm120_blockscaled_sparse_mma_builder.inl:103-120`).
- The epilogue schedule is `SparseTmaWarpSpecializedCooperativeSm120`. Its default
  `LinearCombination` fusion supports the strided `alpha_ptr` with `dAlpha = (0, 0, 1)`, so the
  per-expert alpha works unchanged.

## How grouped mode works on SM120

B, D and SFB are addressed at a fixed per-expert stride, so the grouped kernel sets up every
collective (TMA descriptors, mainloop, epilogue) with the max shape `(M, max_n, K, E)`, exactly
the batched addressing with L = E. Only the scheduler changes: `PersistentTileSchedulerSm90Group`
(selected for SM120 at `tile_scheduler.hpp:405-417`) walks each expert's device-side token count
and never schedules a tile past it. Unlike the SM100 patch, no per-tile effective shape is needed:
M and K are shared, and the last partial N tile writes into padded rows, whose bytes are undefined
by contract. Rows past the last scheduled tile are left untouched.

The changes in `csrc/cutlass_overrides/.../sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp`
(also `patches/cutlass-v4.8.0-sm120-sparse-grouped.patch`):

- detect `MoEProblemShape`; `ProblemShapeGemm` is the max shape, built from `max_m/max_n/max_k`
  (not `get_host_problem_shape(0)`, which returns group 0's count when host counts are present);
- select the group scheduler and hand it the MoE shape in the setup, grid and workspace paths;
  accept `kGrouped` in `can_implement`;
- drive the static scheduler. The stock kernel only configures the scheduler pipeline and runs the
  Warp1 producer loop for the dynamic (CLC) scheduler, but every consumer waits on that pipeline.
  Warp1 now publishes tiles through a `PipelineAsync` whose consumers are the two mainloop load
  warps, the MMA warps and (when C is read) the epilogue load warp. The loop is `while (valid)`, so
  a CTA with no valid first tile publishes nothing (a published response nobody consumes would
  hang `producer_tail`);
- give that pipeline its own shared storage: the group scheduler's `SharedStorage::pipeline()`
  returns the barriers by value;
- advance the consumers' scheduler pipeline state for the grouped scheduler too (stock code only
  advances it for the dynamic one);
- skip the single-argument `fetch_next_work(work)` fallback, which the group scheduler lacks and
  never needs (`valid_warpgroup_in_work_tile` is always true).

## Validation

On the RTX PRO 6000: the full `tests/` suite, including a zero-count / untouched-padding test; and
a stress run comparing grouped against batched byte for byte on valid rows (zero-count experts,
all experts empty, E up to 128 so the scheduler's 32-group warp walk wraps, many tiles per CTA,
K up to 2048, a CUDA graph replayed with new counts, 50 repeated launches).

## Tuning (RTX PRO 6000)

Measured with graph-replayed launches on realistic MoE shapes (Qwen3-30B-A3B, Qwen3-235B-A22B,
Mixtral-8x7B, DeepSeek-V3; top-k routed counts, 1 to 8192 tokens), interleaving builds and
repeating the runs.

- **Raster order: AlongN** (`configure_group_scheduler` in `sm120/config.cuh`). The group
  scheduler's default, AlongM, sweeps an expert's M (weight) tiles for one N tile, so an expert
  with several N tiles streams its weights from DRAM once per N tile. AlongN reuses each weight
  tile from L2 across its N tiles: Mixtral-8x7B w13 is 12% faster at 512 tokens and 9.5% at 2048,
  and the other shapes change by under 1%, except DeepSeek-V3 at 8192 tokens (+1-2%, 23 N tiles
  per expert). A swizzle above 1 was slower everywhere.
- **Scheduler pipeline depth: unchanged (2).** 4 and 8 stages measured the same.
- **Persistent grid: unchanged (SM count).** Smaller grids win only in scattered wave-quantization
  cases, and lose at mid and large sizes.
- **Quantizer row grid: 8 blocks per SM** (`kQuantRowBlocksPerSm`; SM100 keeps its ~2048 as
  14 x 148). The grid is sized from the padded `E * max_n` rows, and every block rebuilds the
  counts prefix before finding it has no rows, so the former fixed 2048 cost 11-21% at small and
  mid token counts for 128- and 256-expert models; large token counts are unchanged.

### Tiles and split-K

- **Tiles.** `sparse_1sm_128x64` (the default) and `sparse_1sm_256x128` joined 128x128.
  With 64 or fewer tokens per expert, 128x64 wastes half as much MMA work on the unused N range
  as 128x128, and its smaller stages leave room for a deeper pipeline in 99 KB: Qwen3-30B-A3B w13
  at 1 token drops from 17.6 to 12.8 µs, Mixtral-8x7B w13 from 104 to 64 µs. 256x128 wins compute-bound prefill (Mixtral-8x7B w2 at 8192 tokens: 1072 -> 965 µs).
  128x256 does not build (the SM120 sparse builder cannot tile a 256-wide N block).
  `suggest_tactic(max_n)` picks 256x128 for max_n >= 768 and 128x64 otherwise: within 0.5% of the
  per-shape best on the geometric mean, 7% at worst.
- **Split-K** (`sparse_1sm_128x{64,128}_sk{2,4}`, `PersistentTileSchedulerSm120GroupSplitK` in
  `csrc/cutlass_overrides/.../sm120_tile_scheduler_group_splitk.hpp`). Each output tile's K range
  becomes 2 or 4 work units; the splits of a tile run in the same round on the CTAs of one slot,
  and reduce in one step through per-slot fp32 partial buffers (deterministic; the last split runs
  the epilogue). It pays only when the tiles are far fewer than the SMs and K is long, e.g. under
  expert parallelism at decode: DeepSeek-V3 w13 with one active expert 21.1 -> 18.9 µs (sk4),
  Mixtral-8x7B w2 at 1 token 42.9 -> 39.1 µs (sk2). Ordinary top-k decode on one GPU already has
  about 100-450 tiles, and there split-K only adds overhead; it stays an autotuning option.

### Against dense NVFP4

`tools/bench_moe.py --dense` compares with the best of 8 configurations (4 tiles and schedules x 2
raster orders) of CUTLASS's SM120 dense NVFP4 pointer-array grouped GEMM
(`tools/dense_baseline/`, from example 79d; per-16 scales, per-expert arrays built on the device,
padded rows skipped). Across the 48 cases, the best sparse tactic is 1.07-1.56x faster (median
1.43x):

| model | w13 speedup (1 -> 8192 tokens) | w2 speedup (1 -> 8192 tokens) |
|---|---|---|
| Qwen3-30B-A3B | 1.39 1.54 1.33 1.35 1.27 1.21 | 1.32 1.39 1.46 1.49 1.20 1.07 |
| Qwen3-235B-A22B | 1.47 1.49 1.45 1.50 1.34 1.36 | 1.45 1.39 1.32 1.36 1.22 1.17 |
| Mixtral-8x7B | 1.54 1.54 1.53 1.41 1.40 1.43 | 1.43 1.55 1.56 1.44 1.35 1.40 |
| DeepSeek-V3 | 1.52 1.53 1.51 1.53 1.50 1.50 | 1.49 1.44 1.38 1.42 1.38 1.28 |

(token counts 1, 16, 128, 512, 2048, 8192). The sparse weight moves about 0.6x the dense bytes,
so ~1.6x is the memory-bound ceiling; dense reaches 93-100% of DRAM bandwidth and sparse 80-95%
on most memory-bound shapes. The weakest cases are large-token w2 GEMMs with short K
(Qwen3-30B-A3B w2, K = 768), where neither kernel is bandwidth-bound.

Remaining costs:

- **A fixed ~1-3 µs grouped overhead:** every warp walks the device-side counts to find its first
  tile (about 0.5 µs per 32 experts). Batched mode takes its first tile straight from `blockIdx`.
- **Short-K split-K overhead.** Per-unit fixed costs (pipeline fill, epilogue) dominate when each
  split has only 1-2 K tiles; stream-K style balancing would be the next step if expert-parallel
  decode needs it.

## Open work

- **SM100 re-validation.** The shared layers changed (`group_mm.cuh` picks grouped or batched from
  the variant's problem shape and passes a split factor; `quant_launch` takes the SM count). SM100
  compiles, but `tests/` and `tools/ab_compare.py` against a v0.10.0 build should be re-run on a
  B200: `tools/validate_sm100.sh` does both.
- **CUTLASS version.** When bumping, re-apply both patches in `patches/` to the new stock headers,
  diff against the overrides (only upstream's own edits should differ), and re-run the tests on
  both archs.
