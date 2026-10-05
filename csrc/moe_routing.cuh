/***************************************************************************************************
 * Copyright (C) 2026 Kwanhee Lee and Dan Alistarh. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 **************************************************************************************************/

/*! \file
    \brief MoE dispatch plan and combine for the per-expert padded layout (arch-independent).

    Plain CUDA over routing indices and bf16 rows: no tensor cores and no scale-factor layouts, so
    this header does not depend on the arch config and every arch module compiles it unchanged.

    dispatch_plan: topk_ids[T, topk] (global expert ids) -> for this rank's contiguous expert shard
        [first_expert, first_expert + E):
          counts[E]            tokens per local expert, clamped to cap
          route_dest[T * topk] e * cap + r for a kept routing, -1 for a dropped one (non-local
                               expert, or rank r >= cap)
          src_tok[E * cap]     source token of row (e, r); rows r >= counts[e] are not written
          src_route[E * cap]   routing index t * topk + k of row (e, r), same contract
        r is the routing's 0-based rank among this rank's routings to e in flat (token-major)
        order, i.e. exactly the stable-argsort rank the torch plan computed, so every expert row
        holds the same token as before.

    finalize: out[t] = sum over k of bf16(fused[route_dest[t, k]] * bf16(w[t, k])) for kept
        routings, accumulated in fp32 in k order and rounded once -- the rounding chain of the
        torch combine (index_select, bf16 multiply, fp32 sum over topk). Dropped routings read
        nothing.

    Both are deterministic (no atomics) and only enqueue work on the stream, so they are
    CUDA-graph-capture safe.

    Plan work layout: routings are processed in chunks of kPlanChunk (one thread each). Within a
    chunk a routing's rank is its warp-local rank (__match_any_sync) plus the counts of the same
    expert in earlier warps (a per-warp histogram in shared memory). A plan with one chunk (every
    decode batch up to kPlanChunk / topk tokens) is a single launch; a larger one adds a
    per-chunk histogram pass and a one-block exclusive scan over chunks.
*/

#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "pdl.cuh"

namespace paired_nvfp4 {

constexpr int kPlanChunk = 1024;            // routings per plan block (one per thread)
constexpr int kPlanWarps = kPlanChunk / 32;

// Shared memory of one plan block: a uint16 histogram per warp and expert.
inline size_t plan_smem(int E) { return sizeof(uint16_t) * (size_t)kPlanWarps * (size_t)E; }

template <class IdT>
__device__ __forceinline__ int local_expert(IdT const* topk_ids, long long i, long long N,
                                            int first_expert, int E) {
  if (i >= N) return E;                    // past the end: the "invalid" bucket
  long long e = (long long)topk_ids[i] - first_expert;
  return (e >= 0 && e < E) ? int(e) : E;
}

// Per-warp histogram of the chunk into s_hist[warp][E] (uint16; a warp adds at most 32).
// Returns this thread's local expert (E = invalid) and its rank among earlier lanes of its warp
// that route to the same expert.
template <class IdT>
__device__ __forceinline__ void chunk_warp_hist(IdT const* topk_ids, long long N, int first_expert,
                                                int E, uint16_t* s_hist, int& e_out,
                                                int& warp_rank) {
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  for (int j = tid; j < kPlanWarps * E; j += blockDim.x) s_hist[j] = 0;
  __syncthreads();
  long long i = (long long)blockIdx.x * kPlanChunk + tid;
  int e = local_expert(topk_ids, i, N, first_expert, E);
  unsigned peers = __match_any_sync(0xffffffffu, e);
  warp_rank = __popc(peers & ((1u << lane) - 1u));
  bool leader = warp_rank == 0;
  if (e < E && leader) s_hist[warp * E + e] = uint16_t(__popc(peers));
  __syncthreads();
  e_out = e;
}

// Pass 1 (multi-chunk plans only): per-chunk expert totals into hist[chunk][E].
template <class IdT>
__global__ void plan_hist_kernel(IdT const* __restrict__ topk_ids, long long N, int first_expert,
                                 int E, int* __restrict__ hist) {
  extern __shared__ uint16_t s_hist[];
  pdl::wait();   // topk_ids come from the router kernel
  pdl::launch_dependents();
  int e, warp_rank;
  chunk_warp_hist(topk_ids, N, first_expert, E, s_hist, e, warp_rank);
  for (int x = threadIdx.x; x < E; x += blockDim.x) {
    int acc = 0;
    for (int w = 0; w < kPlanWarps; ++w) acc += s_hist[w * E + x];
    hist[(long long)blockIdx.x * E + x] = acc;
  }
}

// Pass 2 (multi-chunk plans only): in-place exclusive scan of hist over chunks, per expert, and
// the clamped totals. One block; thread x owns expert x, reads are coalesced across experts.
static __global__ void plan_scan_kernel(int* __restrict__ hist, int num_chunks, int E, int cap,
                                 int* __restrict__ counts) {
  pdl::wait();
  pdl::launch_dependents();
  for (int x = threadIdx.x; x < E; x += blockDim.x) {
    int run = 0;
    for (int c = 0; c < num_chunks; ++c) {
      int v = hist[(long long)c * E + x];
      hist[(long long)c * E + x] = run;
      run += v;
    }
    counts[x] = run < cap ? run : cap;
  }
}

// Pass 3 (every plan): ranks and destinations. base (nullable) is the scanned hist; with a
// single chunk it is null and this block also writes the counts.
template <class IdT>
__global__ void plan_assign_kernel(IdT const* __restrict__ topk_ids, long long N, int topk,
                                   int first_expert, int E, int cap,
                                   int const* __restrict__ base, int* __restrict__ counts,
                                   int* __restrict__ route_dest, int* __restrict__ src_tok,
                                   int* __restrict__ src_route) {
  extern __shared__ uint16_t s_hist[];
  pdl::wait();   // topk_ids come from the router kernel
  pdl::launch_dependents();
  int e, warp_rank;
  chunk_warp_hist(topk_ids, N, first_expert, E, s_hist, e, warp_rank);
  int warp = threadIdx.x >> 5;
  // Exclusive prefix over warps per expert, in place (earlier warps of this chunk).
  for (int x = threadIdx.x; x < E; x += blockDim.x) {
    int acc = 0;
    for (int w = 0; w < kPlanWarps; ++w) {
      int v = s_hist[w * E + x];
      s_hist[w * E + x] = uint16_t(acc);
      acc += v;
    }
    if (base == nullptr) counts[x] = acc < cap ? acc : cap;   // single chunk: acc is the total
  }
  __syncthreads();
  long long i = (long long)blockIdx.x * kPlanChunk + threadIdx.x;
  if (i >= N) return;
  int dest = -1;
  if (e < E) {
    int r = warp_rank + int(s_hist[warp * E + e]) +
            (base != nullptr ? base[(long long)blockIdx.x * E + e] : 0);
    if (r < cap) {
      dest = e * cap + r;
      src_tok[dest] = int(i / topk);
      src_route[dest] = int(i);
    }
  }
  route_dest[i] = dest;
}

// hist_ws: int [num_chunks * E] device scratch, only used when num_chunks > 1.
template <class IdT>
inline void dispatch_plan(IdT const* topk_ids, long long N, int topk, int first_expert, int E,
                          int cap, int* counts, int* route_dest, int* src_tok, int* src_route,
                          int* hist_ws, cudaStream_t stream) {
  int num_chunks = int((N + kPlanChunk - 1) / kPlanChunk);
  if (num_chunks == 0) {   // no tokens: counts are zero
    cudaMemsetAsync(counts, 0, sizeof(int) * (size_t)E, stream);
    return;
  }
  size_t smem = plan_smem(E);
  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(plan_assign_kernel<IdT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(smem));
    cudaFuncSetAttribute(plan_hist_kernel<IdT>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         int(smem));
  }
  int const* base = nullptr;
  if (num_chunks > 1) {
    pdl::launch(plan_hist_kernel<IdT>, dim3(num_chunks), dim3(kPlanChunk), smem, stream,
                topk_ids, N, first_expert, E, hist_ws);
    pdl::launch(plan_scan_kernel, dim3(1), dim3(E < 1024 ? ((E + 31) / 32) * 32 : 1024), 0, stream,
                hist_ws, num_chunks, E, cap, counts);
    base = hist_ws;
  }
  pdl::launch(plan_assign_kernel<IdT>, dim3(num_chunks), dim3(kPlanChunk), smem, stream,
              topk_ids, N, topk, first_expert, E, cap, base, counts, route_dest, src_tok,
              src_route);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// finalize: one block per (token, 1024-column slab); each thread owns 8 consecutive columns
// (one 16-byte load per kept routing).
/////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int kFinalizeThreads = 128;
constexpr int kFinalizeVec = 8;             // bf16 per thread per load

static __global__ void finalize_kernel(__nv_bfloat16 const* __restrict__ fused,   // [E * cap, K]
                                int const* __restrict__ route_dest,         // [T * topk]
                                float const* __restrict__ topk_w,           // [T * topk] or null
                                __nv_bfloat16* __restrict__ out,            // [T, K]
                                int topk, int K) {
  pdl::wait();   // fused rows come from GEMM2
  pdl::launch_dependents();
  long long t = blockIdx.x;
  int col = (int(blockIdx.y) * kFinalizeThreads + int(threadIdx.x)) * kFinalizeVec;
  if (col >= K) return;
  float acc[kFinalizeVec];
#pragma unroll
  for (int j = 0; j < kFinalizeVec; ++j) acc[j] = 0.f;
  for (int k = 0; k < topk; ++k) {
    int d = route_dest[t * topk + k];
    if (d < 0) continue;
    uint4 raw = *reinterpret_cast<uint4 const*>(fused + (long long)d * K + col);
    __nv_bfloat16 const* v = reinterpret_cast<__nv_bfloat16 const*>(&raw);
    if (topk_w != nullptr) {
      float w = __bfloat162float(__float2bfloat16(topk_w[t * topk + k]));
#pragma unroll
      for (int j = 0; j < kFinalizeVec; ++j)
        acc[j] += __bfloat162float(__float2bfloat16(__bfloat162float(v[j]) * w));
    } else {
#pragma unroll
      for (int j = 0; j < kFinalizeVec; ++j) acc[j] += __bfloat162float(v[j]);
    }
  }
  uint4 packed;
  __nv_bfloat16* o = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
  for (int j = 0; j < kFinalizeVec; ++j) o[j] = __float2bfloat16(acc[j]);
  *reinterpret_cast<uint4*>(out + t * K + col) = packed;
}

inline void finalize(__nv_bfloat16 const* fused, int const* route_dest, float const* topk_w,
                     __nv_bfloat16* out, long long T, int topk, int K, cudaStream_t stream) {
  if (T == 0) return;
  int slabs = (K + kFinalizeThreads * kFinalizeVec - 1) / (kFinalizeThreads * kFinalizeVec);
  pdl::launch(finalize_kernel, dim3((unsigned)T, (unsigned)slabs), dim3(kFinalizeThreads), 0,
              stream, fused, route_dest, topk_w, out, topk, K);
}

} // namespace paired_nvfp4
