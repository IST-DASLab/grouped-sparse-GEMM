/*
 * Modified by Kwanhee Lee and Dan Alistarh.
 */

/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Activation quantizers that emit the B operand and its SFB scales in exactly the layout
           group_mm reads.

    Three producers share one quantization tail:
      quant_act       batched bf16 x[E, max_n, K]            -> (b_act, sfb)   generic
      silu_mul_quant  batched bf16 [gate | up][E, max_n, 2N]  -> (b_act, sfb)   GEMM2 input
      scatter_quant   token-order bf16 a1[T, K] + dispatch plan -> (b_act, sfb) GEMM1 input

    Quantization (per SFVecSize = 32 element block along K; matches FlashInfer's NVFP4 convention,
    so the caller's per-expert alpha is unchanged):
        block_scale = amax(|x| over the block) / 6          (6 = largest e2m1 magnitude)
        sfb         = ue4m3(block_scale * gscale[e])
        b_act       = e2m1(x * gscale[e] / float(sfb))       RN, saturating to +-6
    The divisor is the rounded stored scale, not block_scale: the GEMM multiplies fp4 * sfb, so
    quantizing against the dequantized scale leaves only e2m1 rounding, with no per-block
    ue4m3(s)/s bias. Hence b_act * sfb ~= x * gscale[e], and alpha undoes gscale.

    Padding contract (all three producers): rows t >= expert_num_tokens[e] are never written.
    group_mm clips each expert to its count, so their b_act bytes are never read, and their sfb
    slots keep the benign nonzero pre-fill the op allocates (see benign_sf_byte).
*/

#pragma once

#include <cuda_runtime.h>

#include "layout.cuh"
#include "pdl.cuh"

namespace paired_nvfp4 {

// Scale 32 bf16 values by `inv` and pack them to one 16-byte word of e2m1 nibbles (element 2j in
// the low nibble of byte j, CuTe's sub-byte order). NumericArrayConverter<ElementB, float, 8>
// lowers to the hardware cvt.rn.satfinite.e2m1x2.f32 when CUDA_PTX_FP4FP6_CVT_ENABLED (sm_100a,
// sm_120a, ...), one instruction per converted pair; otherwise it falls back to a scalar loop that
// produces the same bytes. Converting 8 floats at a time keeps register pressure low.
CUTLASS_DEVICE uint4 quantize_block_e2m1(cutlass::bfloat16_t const (&vals)[SFVecSize], float inv) {
  static_assert(SFVecSize == 32, "quantize_block_e2m1 packs exactly 32 nibbles into a uint4");
  using Cvt8 = cutlass::NumericArrayConverter<ElementB, float, 8>;
  uint4 packed;
  uint32_t* words = reinterpret_cast<uint32_t*>(&packed);
  CUTLASS_PRAGMA_UNROLL
  for (int w = 0; w < 4; ++w) {
    cutlass::Array<float, 8> in;
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < 8; ++j) in[j] = float(vals[w * 8 + j]) * inv;
    cutlass::Array<ElementB, 8> out8 = Cvt8::convert(in);
    words[w] = reinterpret_cast<uint32_t const&>(out8);
  }
  return packed;
}

// Block scale and the fp4 packing of one 32-element block, shared by every producer.
CUTLASS_DEVICE uint4 quantize_block(cutlass::bfloat16_t const (&vals)[SFVecSize], float gscale,
                                    ElementSF& scale_out) {
  float amax = 0.f;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < SFVecSize; ++j) amax = fmaxf(amax, fabsf(float(vals[j])));
  float bsf = amax > 0.f ? amax / 6.f : 1.f;
  scale_out = ElementSF(bsf * gscale);
  float inv = float(scale_out) > 0.f ? gscale / float(scale_out) : 0.f;
  return quantize_block_e2m1(vals, inv);
}

// quantize_block split over the 4 lanes of a quad (lanes 4i .. 4i + 3 of a warp, every lane of
// which must call it): lane q holds elements [8q, 8q + 8) of the block and gets word q of the
// packed codes. Byte-identical to quantize_block: the max is exact in any order, and each word is
// the same 8-element conversion.
CUTLASS_DEVICE uint32_t quantize_block_quad(cutlass::bfloat16_t const (&vals)[8], float gscale,
                                            ElementSF& scale_out) {
  static_assert(SFVecSize == 32, "quantize_block_quad splits a 32-element block over 4 lanes");
  using Cvt8 = cutlass::NumericArrayConverter<ElementB, float, 8>;
  float amax = 0.f;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(float(vals[j])));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  float bsf = amax > 0.f ? amax / 6.f : 1.f;
  scale_out = ElementSF(bsf * gscale);
  float inv = float(scale_out) > 0.f ? gscale / float(scale_out) : 0.f;
  cutlass::Array<float, 8> in;
  CUTLASS_PRAGMA_UNROLL
  for (int j = 0; j < 8; ++j) in[j] = float(vals[j]) * inv;
  cutlass::Array<ElementB, 8> out8 = Cvt8::convert(in);
  return reinterpret_cast<uint32_t const&>(out8);
}

// 64-byte vectorized load of the 32 bf16 values at `src` (16-byte aligned: K % 32 == 0).
CUTLASS_DEVICE void load_block(cutlass::bfloat16_t (&dst)[SFVecSize],
                               cutlass::bfloat16_t const* src) {
  CUTLASS_PRAGMA_UNROLL
  for (int v = 0; v < int(SFVecSize * sizeof(cutlass::bfloat16_t) / sizeof(uint4)); ++v)
    reinterpret_cast<uint4*>(dst)[v] = reinterpret_cast<uint4 const*>(src)[v];
}

// Byte address of the 16-byte b_act word holding elements [k0, k0 + 32) of row t of expert e.
// Exact because every stride_B component is a multiple of 32 elements.
CUTLASS_DEVICE uint4* b_act_word(ElementB* b_act, StrideB stride_B, int e, int t, int k0) {
  long long elem_off = (long long)e * cute::get<2>(stride_B) +
                       (long long)t * cute::get<0>(stride_B) + k0;
  return reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(b_act) + (elem_off >> 1));
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Launch geometry for the compacted (row, K block) quantizers.
//
// K blocks go on x, so a warp walks consecutive blocks of one row and its 16-byte b_act stores
// coalesce; rows go on y, with several rows per block when K is short, targeting ~256 threads.
// The kernels grid-stride over the compacted rows (real tokens only, sum(counts) of them), whose
// count lives on the device: the host sizes y for occupancy against the worst case E * max_n and
// never reads counts, so there is no device-to-host sync and CUDA-graph capture is unaffected.
// The row grid is capped at kQuantRowBlocksPerSm (arch config) blocks per SM: blocks past the
// real rows still rebuild the counts prefix before finding no work, so over-provisioning the
// padded worst case costs time at small and mid token counts.
/////////////////////////////////////////////////////////////////////////////////////////////////

struct QuantLaunch { dim3 grid; dim3 block; };

inline QuantLaunch quant_launch(int KB, int max_n, int E, int sm_count) {
  constexpr int kTargetThreads = 256;
  constexpr long long kMaxGridY = 65535;   // CUDA gridDim.y limit
  long long max_row_blocks = (long long)kQuantRowBlocksPerSm * (sm_count > 0 ? sm_count : 1);
  if (max_row_blocks > kMaxGridY) max_row_blocks = kMaxGridY;
  int bx = KB < kTargetThreads ? KB : kTargetThreads;
  if (bx < 1) bx = 1;
  int by = kTargetThreads / bx;
  if (by < 1) by = 1;
  unsigned gx = (unsigned)((KB + bx - 1) / bx);
  long long worst = (long long)E * max_n;
  long long rowblocks = (worst + by - 1) / by;
  if (rowblocks > max_row_blocks) rowblocks = max_row_blocks;
  if (rowblocks < 1) rowblocks = 1;
  return QuantLaunch{dim3(gx, (unsigned)rowblocks, 1u), dim3((unsigned)bx, (unsigned)by, 1u)};
}

// Dynamic shared memory for the per-block counts scan: s_cnt[E] + s_off[E + 1].
inline size_t quant_smem(int E) { return sizeof(int) * (size_t)(2 * E + 1); }

// Every block rebuilds the exclusive prefix sum of counts in shared memory. E is small (tens to a
// few hundred local experts), so this costs far less than a separate scan kernel's launch at
// decode sizes. The scan runs in the first warp: each lane sums a contiguous run of experts, a
// warp shuffle scan offsets the runs, and each lane writes its run's prefixes. A serial scan by
// one thread made this prologue the dominant cost of the quantizers whenever most blocks have
// few rows to process (every block runs it, and the grid is sized for E * max_n rows). Must be
// reached by every thread of the block. Returns sum(counts).
CUTLASS_DEVICE int block_count_prefix(int const* __restrict__ counts, int E, int* s_mem) {
  int* s_cnt = s_mem;
  int* s_off = s_mem + E;
  int tid  = int(threadIdx.y) * int(blockDim.x) + int(threadIdx.x);
  int nthr = int(blockDim.x) * int(blockDim.y);
  for (int i = tid; i < E; i += nthr) s_cnt[i] = counts[i];
  __syncthreads();
  if (tid < 32) {
    int per = (E + 31) / 32;
    int lo = tid * per, hi = min(lo + per, E);
    int run = 0;
    for (int i = lo; i < hi; ++i) run += s_cnt[i];
    int incl = run;
    CUTLASS_PRAGMA_UNROLL
    for (int d = 1; d < 32; d <<= 1) {
      int v = __shfl_up_sync(0xffffffffu, incl, d);
      if (tid >= d) incl += v;
    }
    int acc = incl - run;   // exclusive prefix of this lane's run
    for (int i = lo; i < hi; ++i) { s_off[i] = acc; acc += s_cnt[i]; }
    if (tid == 31) s_off[E] = incl;
  }
  __syncthreads();
  return s_off[E];
}

// Prologue of the compacted quantizers: wait for the producers of counts / inputs (PDL), let the
// consumer GEMM launch, rebuild the counts prefix, and write the benign fill of every SFB slot
// this launch does not quantize (so the op needs no separate fill kernel). Must be reached by
// every thread of the block. Returns sum(counts).
CUTLASS_DEVICE int quant_prologue(int const* __restrict__ counts, int E, int* s_mem,
                                  ElementSF* sfb, int rows, int KB) {
  pdl::wait();
  pdl::launch_dependents();
  int total = block_count_prefix(counts, E, s_mem);
  int const bthr = int(blockDim.x * blockDim.y);
  int const tid = int((blockIdx.y * gridDim.x + blockIdx.x) * unsigned(bthr) +
                      threadIdx.y * blockDim.x + threadIdx.x);
  int const nthr = int(gridDim.x * gridDim.y) * bthr;
  fill_benign_sfb(sfb, rows, KB, E, [s_mem](int e) { return s_mem[e]; }, tid, nthr);
  return total;
}

// Compacted row r -> (expert e, row t within e): the largest e with s_off[e] <= r.
CUTLASS_DEVICE void compact_row_to_expert(int const* s_off, int E, int r, int& e, int& t) {
  int lo = 0, hi = E;
  while (lo + 1 < hi) { int mid = (lo + hi) >> 1; if (s_off[mid] <= r) lo = mid; else hi = mid; }
  e = lo;
  t = r - s_off[e];
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// quant_act: batched bf16 x[E, max_n, K] -> b_act [E, max_n, K/2] (stride_B = {max_n, K, E}) and
// swizzled sfb. One thread per (compacted row, 32-wide K block).
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class LayoutSF>
__global__ void quant_act_kernel(cutlass::bfloat16_t const* __restrict__ x,
                                 ElementB* __restrict__ b_act,
                                 ElementSF* __restrict__ sfb, LayoutSF layout_SFB,
                                 StrideB stride_B, float const* __restrict__ gscale,
                                 int const* __restrict__ counts,
                                 int max_n, int K, int E, int gscale_per_expert) {
  int KB = K / SFVecSize;
  int kb = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int k0 = kb * SFVecSize;
  extern __shared__ int s_mem[];
  int total_rows = quant_prologue(counts, E, s_mem, sfb, max_n, KB);
  int const* s_off = s_mem + E;
  if (kb >= KB) return;   // only after the block-wide barriers above

  auto sf = make_tensor(cute::recast_ptr<ElementSF>(sfb), layout_SFB);
  int r_stride = int(gridDim.y) * int(blockDim.y);
  for (int r = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y); r < total_rows;
       r += r_stride) {
    int e, t;
    compact_row_to_expert(s_off, E, r, e, t);
    float g = gscale_per_expert ? gscale[e] : gscale[0];

    alignas(16) cutlass::bfloat16_t vals[SFVecSize];
    load_block(vals, x + ((long long)e * max_n + t) * K + k0);
    ElementSF s;
    uint4 packed = quantize_block(vals, g, s);
    sf(t, k0, e) = s;
    *b_act_word(b_act, stride_B, e, t, k0) = packed;
  }
}

template <class LayoutSFB>
inline void quant_act(cutlass::bfloat16_t const* x, ElementB* b_act, ElementSF* sfb,
                      LayoutSFB layout_SFB, StrideB stride_B, float const* gscale, int const* counts,
                      int max_n, int K, int E, int gscale_per_expert, int sm_count,
                      cudaStream_t stream) {
  QuantLaunch lc = quant_launch(K / SFVecSize, max_n, E, sm_count);
  pdl::launch(quant_act_kernel<LayoutSFB>, lc.grid, lc.block, quant_smem(E), stream,
              x, b_act, sfb, layout_SFB, stride_B, gscale, counts, max_n, K, E, gscale_per_expert);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// silu_mul_quant: GEMM1 output x2[E, max_n, 2N] ([gate | up], vLLM silu_and_mul order) ->
// b_act [E, max_n, N/2] and swizzled sfb for GEMM2, without materializing silu(gate) * up.
//
// Bit-exactness contract: the SwiGLU replicates vLLM's silu_and_mul rounding chain,
// v = bf16(bf16(silu_f32(gate)) * up), so the fused output equals silu_and_mul followed by
// quant_act byte for byte (tests/test_silu_mul_quant.py). Keeping silu in fp32 would be more
// accurate, but it would silently change model numerics relative to the unfused path.
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class LayoutSF>
__global__ void silu_mul_quant_kernel(cutlass::bfloat16_t const* __restrict__ x2,
                                      ElementB* __restrict__ b_act,
                                      ElementSF* __restrict__ sfb, LayoutSF layout_SFB,
                                      StrideB stride_B, float const* __restrict__ gscale,
                                      int const* __restrict__ counts,
                                      int max_n, int N, int E, int gscale_per_expert,
                                      int interleaved) {
  int KB = N / SFVecSize;
  int kb = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int k0 = kb * SFVecSize;
  extern __shared__ int s_mem[];
  int total_rows = quant_prologue(counts, E, s_mem, sfb, max_n, KB);
  int const* s_off = s_mem + E;
  if (kb >= KB) return;

  auto sf = make_tensor(cute::recast_ptr<ElementSF>(sfb), layout_SFB);
  int r_stride = int(gridDim.y) * int(blockDim.y);
  for (int r = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y); r < total_rows;
       r += r_stride) {
    int e, t;
    compact_row_to_expert(s_off, E, r, e, t);
    float g = gscale_per_expert ? gscale[e] : gscale[0];

    alignas(16) cutlass::bfloat16_t gate[SFVecSize], up[SFVecSize];
    cutlass::bfloat16_t const* row = x2 + ((long long)e * max_n + t) * (2LL * N);
    // Plain: [gate | up] halves. Interleaved (the fused-epilogue weight order): 64-column blocks,
    // 32 gate columns then the matching 32 up columns.
    int const gate_off = interleaved ? (k0 / SFVecSize) * 2 * SFVecSize : k0;
    int const up_off = interleaved ? gate_off + SFVecSize : N + k0;
    load_block(gate, row + gate_off);
    load_block(up, row + up_off);

    cutlass::bfloat16_t vals[SFVecSize];
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < SFVecSize; ++j) {
      float gf = float(gate[j]);
      cutlass::bfloat16_t s_bf(gf / (1.0f + expf(-gf)));
      vals[j] = cutlass::bfloat16_t(float(s_bf) * float(up[j]));
    }
    ElementSF s;
    uint4 packed = quantize_block(vals, g, s);
    sf(t, k0, e) = s;
    *b_act_word(b_act, stride_B, e, t, k0) = packed;
  }
}

template <class LayoutSFB>
inline void silu_mul_quant(cutlass::bfloat16_t const* x2, ElementB* b_act, ElementSF* sfb,
                           LayoutSFB layout_SFB, StrideB stride_B, float const* gscale,
                           int const* counts, int max_n, int N, int E, int gscale_per_expert,
                           int sm_count, cudaStream_t stream, int interleaved = 0) {
  QuantLaunch lc = quant_launch(N / SFVecSize, max_n, E, sm_count);
  pdl::launch(silu_mul_quant_kernel<LayoutSFB>, lc.grid, lc.block, quant_smem(E), stream,
              x2, b_act, sfb, layout_SFB, stride_B, gscale, counts, max_n, N, E, gscale_per_expert,
              interleaved);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_quant: token-order a1[T, K] + a dispatch plan -> batched b_act [E, cap, K/2] and
// swizzled sfb for GEMM1, fusing the gather, the scatter into per-expert rows and the quantizer.
// Work scales with routed entries (T * topk), not with capacity E * cap.
//
// Plan (vLLM's capture-safe batched prepare/finalize):
//   flat_tok[i]     source token of routed entry i
//   dest_global[i]  destination row e * cap + r, or the trash row E * cap for routings to a
//                   non-local expert (EP) or past capacity
// Trash entries do no work: the trash row is never read, so nothing is stored there.
//
// topk_w (nullable) applies the router weight on the input with torch's rounding chain,
// bf16(float(x) * float(bf16(w))), byte-identical to scaling then scattering in torch.
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class LayoutSF>
__global__ void scatter_quant_kernel(cutlass::bfloat16_t const* __restrict__ a1,
                                     int64_t const* __restrict__ flat_tok,
                                     int64_t const* __restrict__ dest_global,
                                     float const* __restrict__ topk_w,
                                     ElementB* __restrict__ b_act,   // (E * cap + 1) rows
                                     ElementSF* __restrict__ sfb, LayoutSF layout_SFB,
                                     float const* __restrict__ gscale,
                                     long long N, int K, int cap,
                                     long long trash_row, int gscale_per_expert) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int KB = K / SFVecSize;
  long long total = N * KB;
  if (idx >= total) return;
  int kb = int(idx % KB);
  long long i = idx / KB;
  int k0 = kb * SFVecSize;

  long long dest = dest_global[i];
  // Dropped routings (non-local expert, or past capacity) do no work. Storing a block for them
  // would make every dropped routing hit the same trash row: under EP that is most of the
  // routings, all contending for one row's cache lines.
  if (dest == trash_row) return;
  // Row-flat addressing, identical to ColMajor B with stride_B = (K, 1, cap * K).
  uint4* dst = reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(b_act) +
                                        (((long long)dest * K + k0) >> 1));
  int e = int(dest / cap);
  int r = int(dest % cap);
  float g = gscale_per_expert ? gscale[e] : gscale[0];

  alignas(16) cutlass::bfloat16_t vals[SFVecSize];
  load_block(vals, a1 + (long long)flat_tok[i] * K + k0);
  if (topk_w != nullptr) {
    cutlass::bfloat16_t w_bf(topk_w[i]);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < SFVecSize; ++j)
      vals[j] = cutlass::bfloat16_t(float(vals[j]) * float(w_bf));
  }

  ElementSF s;
  uint4 packed = quantize_block(vals, g, s);
  auto sf = make_tensor(cute::recast_ptr<ElementSF>(sfb), layout_SFB);
  sf(r, k0, e) = s;
  *dst = packed;
}

template <class LayoutSFB>
inline void scatter_quant(cutlass::bfloat16_t const* a1, int64_t const* flat_tok,
                          int64_t const* dest_global, float const* topk_w,
                          ElementB* b_act, ElementSF* sfb, LayoutSFB layout_SFB,
                          float const* gscale, long long N, int K, int cap,
                          long long trash_row, int gscale_per_expert, cudaStream_t stream) {
  long long total = N * (K / SFVecSize);
  int threads = 256;
  long long blocks = (total + threads - 1) / threads;
  if (blocks == 0) return;
  scatter_quant_kernel<LayoutSFB><<<dim3((unsigned)blocks), dim3(threads), 0, stream>>>(
      a1, flat_tok, dest_global, topk_w, b_act, sfb, layout_SFB, gscale, N, K, cap,
      trash_row, gscale_per_expert);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_quant_planned: token-order a1[T, K] + the dispatch_plan output (moe_routing.cuh) ->
// batched b_act [E, cap, K/2] and swizzled sfb for GEMM1. Same outputs as scatter_quant on every
// valid row, but the grid walks only the kept rows -- sum(counts) of them, recovered to (e, r)
// through the per-block counts prefix like quant_act -- so dropped routings (most of them under
// EP) cost nothing, and there are no 64-bit divides in the index math.
//   src_tok[e * cap + r]    source token of row (e, r)
//   src_route[e * cap + r]  its routing index, which selects the router weight when topk_w is set
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class LayoutSF>
__global__ void scatter_quant_planned_kernel(cutlass::bfloat16_t const* __restrict__ a1,
                                             int const* __restrict__ src_tok,
                                             int const* __restrict__ src_route,
                                             float const* __restrict__ topk_w,
                                             ElementB* __restrict__ b_act,
                                             ElementSF* __restrict__ sfb, LayoutSF layout_SFB,
                                             StrideB stride_B, float const* __restrict__ gscale,
                                             int const* __restrict__ counts,
                                             int cap, int K, int E, int gscale_per_expert) {
  int KB = K / SFVecSize;
  int kb = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int k0 = kb * SFVecSize;
  extern __shared__ int s_mem[];
  int total_rows = quant_prologue(counts, E, s_mem, sfb, cap, KB);
  int const* s_off = s_mem + E;
  if (kb >= KB) return;   // only after the block-wide barriers above

  auto sf = make_tensor(cute::recast_ptr<ElementSF>(sfb), layout_SFB);
  int r_stride = int(gridDim.y) * int(blockDim.y);
  for (int row = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y); row < total_rows;
       row += r_stride) {
    int e, r;
    compact_row_to_expert(s_off, E, row, e, r);
    float g = gscale_per_expert ? gscale[e] : gscale[0];
    long long slot = (long long)e * cap + r;

    alignas(16) cutlass::bfloat16_t vals[SFVecSize];
    load_block(vals, a1 + (long long)src_tok[slot] * K + k0);
    if (topk_w != nullptr) {
      cutlass::bfloat16_t w_bf(topk_w[src_route[slot]]);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < SFVecSize; ++j)
        vals[j] = cutlass::bfloat16_t(float(vals[j]) * float(w_bf));
    }
    ElementSF s;
    uint4 packed = quantize_block(vals, g, s);
    sf(r, k0, e) = s;
    *b_act_word(b_act, stride_B, e, r, k0) = packed;
  }
}

template <class LayoutSFB>
inline void scatter_quant_planned(cutlass::bfloat16_t const* a1, int const* src_tok,
                                  int const* src_route, float const* topk_w, ElementB* b_act,
                                  ElementSF* sfb, LayoutSFB layout_SFB, StrideB stride_B,
                                  float const* gscale, int const* counts, int cap, int K, int E,
                                  int gscale_per_expert, int sm_count, cudaStream_t stream) {
  QuantLaunch lc = quant_launch(K / SFVecSize, cap, E, sm_count);
  pdl::launch(scatter_quant_planned_kernel<LayoutSFB>, lc.grid, lc.block, quant_smem(E), stream,
              a1, src_tok, src_route, topk_w, b_act, sfb, layout_SFB, stride_B, gscale, counts,
              cap, K, E, gscale_per_expert);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// quant_rows: token-order bf16 x[T, K] -> row-major e2m1 [T, K/2] and linear UE4M3 scales [T, K/32],
// with one shared global scale. For quantizing before an all2all: the rows travel as FP4, and
// scatter_fp4_planned places them into the batched operands. With the same global scale the result
// is byte-identical to quantizing inside the scatter (quantize_block on the same 32 values).
/////////////////////////////////////////////////////////////////////////////////////////////////

static __global__ void quant_rows_kernel(cutlass::bfloat16_t const* __restrict__ x,
                                  uint8_t* __restrict__ q, uint8_t* __restrict__ sf,
                                  float const* __restrict__ gscale, long long T, int K) {
  int KB = K / SFVecSize;
  int kb = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (kb >= KB) return;
  float g = gscale[0];
  for (long long t = (long long)blockIdx.y * blockDim.y + threadIdx.y; t < T;
       t += (long long)gridDim.y * blockDim.y) {
    alignas(16) cutlass::bfloat16_t vals[SFVecSize];
    load_block(vals, x + t * K + (long long)kb * SFVecSize);
    ElementSF s;
    uint4 packed = quantize_block(vals, g, s);
    reinterpret_cast<uint4*>(q + t * (K / 2))[kb] = packed;
    sf[t * KB + kb] = reinterpret_cast<uint8_t const&>(s);
  }
}

inline void quant_rows(cutlass::bfloat16_t const* x, uint8_t* q, uint8_t* sf, float const* gscale,
                       long long T, int K, int sm_count, cudaStream_t stream) {
  if (T == 0) return;
  QuantLaunch lc = quant_launch(K / SFVecSize, int(T < (1LL << 30) ? T : (1LL << 30)), 1,
                                sm_count);
  quant_rows_kernel<<<lc.grid, lc.block, 0, stream>>>(x, q, sf, gscale, T, K);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_fp4_planned: pre-quantized token rows (quant_rows output, e.g. after an all2all) + the
// dispatch_plan output -> batched b_act [E, cap, K/2] and swizzled sfb for GEMM1. A byte copy over
// the kept rows; with the same global scale, byte-identical to scatter_quant_planned on the bf16
// rows.
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class LayoutSF>
__global__ void scatter_fp4_planned_kernel(uint8_t const* __restrict__ q,
                                           uint8_t const* __restrict__ q_sf,
                                           int const* __restrict__ src_tok,
                                           ElementB* __restrict__ b_act,
                                           ElementSF* __restrict__ sfb, LayoutSF layout_SFB,
                                           StrideB stride_B, int const* __restrict__ counts,
                                           int cap, int K, int E) {
  int KB = K / SFVecSize;
  int kb = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  int k0 = kb * SFVecSize;
  extern __shared__ int s_mem[];
  int total_rows = quant_prologue(counts, E, s_mem, sfb, cap, KB);
  int const* s_off = s_mem + E;
  if (kb >= KB) return;   // only after the block-wide barriers above

  auto sf = make_tensor(cute::recast_ptr<ElementSF>(sfb), layout_SFB);
  int r_stride = int(gridDim.y) * int(blockDim.y);
  for (int row = int(blockIdx.y) * int(blockDim.y) + int(threadIdx.y); row < total_rows;
       row += r_stride) {
    int e, r;
    compact_row_to_expert(s_off, E, row, e, r);
    long long t = src_tok[(long long)e * cap + r];
    *b_act_word(b_act, stride_B, e, r, k0) =
        reinterpret_cast<uint4 const*>(q + t * (K / 2))[kb];
    sf(r, k0, e) = reinterpret_cast<ElementSF const&>(q_sf[t * KB + kb]);
  }
}

template <class LayoutSFB>
inline void scatter_fp4_planned(uint8_t const* q, uint8_t const* q_sf, int const* src_tok,
                                ElementB* b_act, ElementSF* sfb, LayoutSFB layout_SFB,
                                StrideB stride_B, int const* counts, int cap, int K, int E,
                                int sm_count, cudaStream_t stream) {
  QuantLaunch lc = quant_launch(K / SFVecSize, cap, E, sm_count);
  pdl::launch(scatter_fp4_planned_kernel<LayoutSFB>, lc.grid, lc.block, quant_smem(E), stream,
              q, q_sf, src_tok, b_act, sfb, layout_SFB, stride_B, counts, cap, K, E);
}

} // namespace paired_nvfp4
