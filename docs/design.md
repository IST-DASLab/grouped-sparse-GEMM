# Design notes

## The MoE forward this package serves

One MoE layer with SwiGLU experts (`w13` = fused gate and up projection, `w2` = down projection)
runs as six kernel launches at decode sizes (eight above one plan chunk), and no bf16 activation
tensor sized by expert capacity is ever written:

```
topk_ids [T, topk]
   │  dispatch_plan                                         counts, route_dest, row -> token
   ▼
a1 [T, K] bf16, token order
   │  scatter_quant_act_planned                             gather + scatter + quantize
   ▼
b1, sfb1   [E, cap, K/2] e2m1 + UE4M3                        per-expert padded
   │  group_mm(w13)
   ▼
c1 [E, cap, 2N] bf16
   │  silu_mul_quant_act                                    SwiGLU + quantize
   ▼
b2, sfb2   [E, cap, N/2]
   │  group_mm(w2)
   ▼
out [E, cap, K] bf16
   │  moe_finalize                                          top-k weighted sum, token order
   ▼
y [T, K] bf16
```

`quant_act` is the unfused alternative to the scatter step, for callers that already hold batched
bf16 activations (for example after an all-to-all dispatch, which also does the combine).
`scatter_quant_act` is the older scatter op that takes a torch-built plan (`flat_tok`,
`dest_global`); it produces the same bytes.

## Dispatch and combine (moe_routing.cuh)

The plan and the combine are plain CUDA over routing indices and bf16 rows, so they are
arch-independent. Both are deterministic (no atomics) and capture safe.

- **Rank = stable-sort rank.** A routing's row within its expert is its rank among this rank's
  routings to that expert in flat (token-major) order: the same row the torch plan's stable argsort
  gave, so expert rows, GEMM inputs and outputs are unchanged. Within a 1024-routing chunk the rank
  is `__match_any_sync` within the warp plus a per-warp histogram in shared memory. A single-chunk
  plan (decode) is one launch; larger ones add a per-chunk histogram and a one-block scan.
- **Only local rows do work.** Under EP most routings go to other ranks' experts. The plan marks
  them `route_dest = -1`, `scatter_quant_act_planned` walks only the kept rows (the compacted-row
  grid of the other quantizers), and `moe_finalize` reads only kept rows. The torch combine this
  replaces gathered, weighted and summed all `T * topk` rows, which at T = 4096 on one Kimi EP8
  rank was ~0.8 ms of a 1.3 ms layer.
- **Same rounding as the torch combine.** `moe_finalize` computes `bf16(row * bf16(w))` per
  routing and sums in fp32 in top-k order with one final rounding, which is bit-identical to
  index_select, a bf16 multiply and an fp32 `sum(dim=topk)`.

## Operand roles

CUTLASS's structured-sparse MMA takes the sparse operand as A. The weight is the sparse operand,
so the GEMM computes `D[M, N] = W[M, K] · X[K, N]` with M = out_features and N = tokens, and D is
laid out column-major. Column-major `[features, tokens]` is byte-identical to row-major
`[tokens, features]`, which is what callers want, and it also avoids the row-major output's
N >= 8 alignment floor, which matters at decode sizes.

## Per-expert padded layout

The grouped kernel is driven by CUTLASS's `MoEProblemShape`: every expert shares M and K, and
each has its own token count N_e. B, D and SFB are addressed at a fixed per-expert stride of
`max_n`, and there is no pointer-array variant of the sparse kernel. The consequences:

- Activation and output buffers are `[E, max_n, ...]`, and memory scales with `E * max_n`
  rather than with routed tokens.
- `expert_num_tokens[E]` is read on the device. The host sizes the persistent grid from the SM
  count and each work tile takes its N from the count, so no device-to-host copy is needed.
- Rows past an expert's count are never written by any op in this package. Nothing reads them,
  since every consumer clips at the count, so leaving them undefined costs nothing.

## Scale-factor layout

The sparse NVFP4 MMA reads one UE4M3 scale per 32 dense K elements; the dense NVFP4 MMA reads one
per 16. The kernel config takes this constant (`SFVecSize`) from the instantiated MMA, and every
op derives its K granularity from it.

SFA and SFB live in CUTLASS's swizzled `tile_atom_to_shape_SF{A,B}` layouts. Nothing in this
package computes swizzled offsets by hand: every producer writes a scale through the same CuTe
layout functor the kernel reads with (`sf(row, k, expert) = s`, evaluated once per block), so the
layouts match by construction. Two properties of these layouts are used:

- A, E and SFA do not depend on N, and SFB does not depend on M. Load-time `compress` and the
  scatter quantizer build their layouts with a placeholder for the extent they do not know.
- The layouts round up to the scale atom (N to 128, K / 32 to 4), so a buffer can hold slots no
  row maps to. Every op pre-fills SFB with UE4M3 1.0. Padded and valid rows share SF tiles, and a
  zero scale there would produce 0/0.

## Weight compression

`compress` runs CUTLASS's `StructuredSparseCompressor` once at load. It produces the compressed
weight (half the K extent, still two fp4 per byte) and the 4:8 metadata, then scatters the
natural-order scales into the SFA layout. Checkpoints therefore store plain dense NVFP4 with the
zeros in place; the kernel-specific layouts never reach disk, and a future arch with a different
compressor reads the same checkpoints.

## Activation quantizers

All three producers share one tail: block amax, then the UE4M3 scale, then the hardware
`cvt.rn.satfinite.e2m1x2.f32` pack into one 16-byte store per 32 elements. They also share one
convention: the divisor is the *rounded* stored scale, so the only error is e2m1 rounding.

Three implementation decisions, each backed by a measurement on B200:

- **Compacted rows.** `quant_act` and `silu_mul_quant_act` iterate over `sum(counts)` real rows,
  not `E * max_n`. At MoE fill ratios (top-k / E) padded rows are the vast majority of the
  buffer, so the kernel only has work proportional to real tokens. The row count lives on the
  device, so the host sizes the grid for occupancy and the kernel grid-strides; this is what
  keeps it capture-safe.
- **No integer division in the index math.** K blocks go on x and rows on y, so each thread
  recovers its coordinates without divides. In an earlier flat-index version, three 64-bit
  divides per thread were most of the padded-row cost: removing the padded-row stores alone cut
  bytes by ~90% but time by under 10%, because the kernel had stopped being bandwidth-bound.
- **Per-block prefix scan instead of a scan kernel.** Each block rebuilds the exclusive prefix of
  `counts` in shared memory. For tens to a few hundred experts this is far cheaper than a second
  launch, which at decode sizes would cost more than the whole op.

`silu_mul_quant_act` replicates vLLM's `silu_and_mul` rounding chain exactly
(`bf16(bf16(silu(gate)) * up)`), so swapping the fused op for the unfused pair changes no bits.

## group_mm

Each tile variant is compiled in its own translation unit so nvcc builds them in parallel.
SM100 has 128- and 256-token tiles plus 64-token ones for decode, where an expert sees a handful
of tokens and a 128-row B tile is mostly padding that still moves through shared memory. Stock
CUTLASS v4.8.0 rejects Cta N = 64 for the block-scaled sparse collective although the collective
carries the N = 64 scale-factor-B paths, so an overridden copy of that header allows it
(`patches/cutlass-v4.8.0-sm100-sparse-ctan64.patch`); N < 64 is rejected by the builders even for
dense block-scaled GEMMs. The thread-block cluster is a runtime argument (Blackwell dynamic clusters), so a
tactic is `(config_id, cluster_m, cluster_n)` with no recompilation. (SM120 has no cluster multicast: its
kernels are built for a static 1x1 cluster, and its tactics are `(config_id, 1, 1)`.) A compile-time check
asserts that every variant shares the default variant's operand layouts, so any tactic can
consume the same `compress` / quantizer output. The per-expert `alpha` is applied in the
epilogue through CUTLASS's strided `alpha_ptr` form (`alpha[e]` via `dAlpha = (0, 0, 1)`).

## Known costs and open items

- Output and activation buffers scale with `E * max_n` (see the padded layout above).
- `scatter_quant_act` (the torch-plan variant) still uses flat indexing with 64-bit divides and a
  thread per routing, dropped ones included. `scatter_quant_act_planned` replaces it.
- `quant_launch` caps the row grid at `kQuantRowBlocksPerSm` blocks per SM (arch config). Blocks
  past the real rows still rebuild the counts prefix (a warp scan), so the cap matters at small
  token counts.
- No fake (meta) kernels are registered, so the ops are opaque to `torch.compile` tracing.
- **GEMM1 writes the bf16 intermediate.** At large token counts the MoE layer is bound by bf16
  round trips, not by the sparse MMA: on one Kimi-K2.5 EP8 rank at T = 8192 the two GEMMs take
  ~220 us, `silu_mul_quant_act` ~30 us and `moe_finalize` ~50 us; on Qwen3-30B (one GPU) the same
  three are ~230 / 82 / 62 us. `silu_mul_quant_act` is arithmetic-bound, not memory-bound (60-70%
  issue slots, ~10-20% DRAM), because it keeps vLLM's exact `silu_and_mul` rounding chain (libdevice
  `expf`, IEEE divide). Fusing SwiGLU and the group-32 FP4 quantization into GEMM1's epilogue would
  hide that work behind the mainloop and replace the `[E, cap, 2N]` bf16 write with the
  `[E, cap, N/2]` FP4 operand. CUTLASS v4.8 already covers the quantization half:
  `Sm100BlockScaleFactorColStore` (epilogue/fusion/sm100_visitor_store_tma_warpspecialized.hpp)
  quantizes a column-major D along M with SFVecSize 32 as one warp `redux` abs-max, because after
  the 32dp32b TMEM load each epilogue thread owns one M row (a feature) across its N columns; and a
  column-major FP4 D is byte-for-byte GEMM2's K-major B. The custom half is the gate: interleave
  W13 rows at `compress` in 32-row groups (gate rows of a group in one warp, the matching up rows in
  the next), exchange through shared memory as ColStore's SFVecSize-64 path does, apply the exact
  SwiGLU chain, and store a D tile with half the accumulator's M rows (so a stock TMA-store
  epilogue does not fit as is).
- **Decode with very few active experts.** At T = 1 a Kimi EP8 rank has ~1 active expert, whose
  weights are streamed by only the 16-32 CTAs of its M tiles (~0.6 TB/s: GEMM1 ~16 us). Every
  compiled tactic is within noise of the chosen one there, so closing this needs split-K across
  CTAs (partial sums over K chunks plus a reduction), which the grouped sparse kernel does not
  support.
- **PDL.** Launching `group_mm` with programmatic dependent launch alone measured no change
  (0.96-1.00x, within timer resolution): the producers never trigger early, and CUDA graphs already
  hide most launch gaps. A useful chain needs every op (quantizers, plan, finalize and the SFB
  pre-fill) to wait and trigger.
