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
    \brief Operand shape / layout helpers shared by the compress, quantize and group_mm ops.

    Every op builds its layouts through these helpers, so the producer of a buffer and the GEMM
    that consumes it agree by construction.
*/

#pragma once

#include <cuda_runtime.h>

#include "arch.cuh"

namespace paired_nvfp4 {

inline SparseProblemShape make_sparse_shape(int features, int max_n, int k, int groups) {
  return make_tuple(features, max_n, k, groups);
}

// A tile variant's GEMM problem: grouped (MoEProblemShape, kGrouped) clips expert e to
// expert_num_tokens[e] on the device; batched (<M,N,K,L>, kBatched) runs all max_n rows of every
// expert. Both address B / D / SFB at the same fixed per-expert stride.
template <class ProblemShape_>
inline constexpr bool kIsGroupedProblem =
    cutlass::gemm::detail::is_moe_problem_shape<ProblemShape_>::value;

template <class ProblemShape_>
inline ProblemShape_ make_problem_shape(int features, int max_n, int k, int groups,
                                        int32_t* expert_num_tokens) {
  if constexpr (kIsGroupedProblem<ProblemShape_>) {
    return {features, max_n, k, groups, expert_num_tokens, /*host=*/nullptr};
  } else {
    return make_tuple(features, max_n, k, groups);
  }
}

template <class ProblemShape_>
constexpr cutlass::gemm::GemmUniversalMode gemm_mode() {
  return kIsGroupedProblem<ProblemShape_> ? cutlass::gemm::GemmUniversalMode::kGrouped
                                          : cutlass::gemm::GemmUniversalMode::kBatched;
}

// The A / E / SFA layouts do not depend on N, and SFB does not depend on M. Ops that do not know
// the other extent use this placeholder (the sparse kernel's minimum tile extent).
static constexpr int kPlaceholderExtent = 128;

// Physical (post-alignment) extents of the compressed weight A and its metadata E.
struct PhysicalDims { int MAlignedAC, KAlignedAC, MAlignedE, KAlignedE; };

inline PhysicalDims physical_dims(int features, int k, int groups) {
  SparseProblemShape sp = make_sparse_shape(features, kPlaceholderExtent, k, groups);
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {features, k, groups});
  CompressorUtility cu(sp, stride_A);
  return {cu.get_tensorA_m_physical(), cu.get_tensorA_k_physical(),
          cu.get_metadata_m_physical(), cu.get_metadata_k_physical()};
}

// Element counts of the SFA / SFB buffers, i.e. size(filter_zeros(layout)). Both round their
// extents up to the scale-factor atom, so they can exceed outer * ceil(K / SFVecSize) * groups.
inline int sfa_numel(int features, int max_n, int k, int groups) {
  return int(cute::size(cute::filter_zeros(
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_sparse_shape(features, max_n, k, groups)))));
}
inline int sfb_numel(int features, int max_n, int k, int groups) {
  return int(cute::size(cute::filter_zeros(
      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(features, max_n, k, groups)))));
}

inline int kblocks(int k) { return (k + SFVecSize - 1) / SFVecSize; }

// Scale-factor scatter: a natural-order [groups, outer, ceil(K / SFVecSize)] array into the
// kernel's swizzled tile_atom_to_shape_SF{A,B} layout (outer = M for SFA, N for SFB). The store
// goes through the same CuTe layout functor the GEMM reads with, so the swizzle is correct by
// construction. One thread per (outer, K block, group): all SFVecSize elements of a block map to
// one physical slot, so the functor is evaluated once, at the block's first element. The
// destination must be pre-zeroed; slots outside the logical extent keep that value.
template <class LayoutSF>
__global__ void scatter_sf_kernel(ElementSF const* __restrict__ src,
                                  ElementSF* __restrict__ dst,
                                  LayoutSF layout, int outer, int K, int groups) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int kblk = (K + SFVecSize - 1) / SFVecSize;
  long long total = (long long)outer * kblk * groups;
  if (idx >= total) return;
  int kb = int(idx % kblk);
  int o  = int((idx / kblk) % outer);
  int g  = int(idx / ((long long)kblk * outer));
  auto sf = make_tensor(cute::recast_ptr<ElementSF>(dst), layout);
  sf(o, kb * SFVecSize, g) = src[((long long)g * outer + o) * kblk + kb];
}

template <class LayoutSF>
inline void scatter_sf(ElementSF const* src, ElementSF* dst, LayoutSF layout,
                       int outer, int K, int groups, cudaStream_t stream) {
  long long total = (long long)outer * kblocks(K) * groups;
  int threads = 256;
  long long blocks = (total + threads - 1) / threads;
  scatter_sf_kernel<LayoutSF><<<dim3((unsigned)blocks), dim3(threads), 0, stream>>>(
      src, dst, layout, outer, K, groups);
}

// A UE4M3 byte for 1.0: the benign fill for SFB slots no quantizer writes (padded rows and atom
// slop). It must be nonzero: padded and valid rows share SF tiles, and a zero scale there
// produces 0/0.
CUTLASS_HOST_DEVICE uint8_t benign_sf_byte() {
  ElementSF one(1.0f);
  return reinterpret_cast<uint8_t const&>(one);
}

// The producer-side form of that pre-fill: writes the benign byte to every byte of a K-major SFB
// buffer (tile_atom_to_shape_SFB over [rows, K, E]) except the slots (e, r < count_of(e),
// kb < KB) its producer quantizes, so the buffer ends up exactly as a torch::full pre-fill
// followed by the producer's stores, without the separate fill kernel. Every thread of the grid
// calls it (thread tid of nthr), in any order relative to the producer's stores: they touch
// disjoint bytes.
//
// Layout (SfKMajorAtom tiled with Step<_2,_1,_3>): 512-byte atoms of 128 rows x 4 K blocks, K
// atoms innermost, then row atoms, then experts. Within an atom, row r and K block kb sit at byte
// (r % 32) * 16 + (r / 32 % 4) * 4 + kb % 4, so each 16-byte chunk holds the 4 K blocks of rows
// rr, rr + 32, rr + 64 and rr + 96 of one atom. tests/test_ops.py checks it against the layout.
template <class CountOf>
CUTLASS_DEVICE void fill_benign_sfb(ElementSF* sfb, int rows, int KB, int E, CountOf count_of,
                                    int tid, int nthr) {
  int const n_rt = (rows + 127) / 128;
  int const n_kt = (KB + 3) / 4;
  int const chunks = E * n_rt * n_kt * 32;   // sfb_numel / 16, which is an int
  uint32_t const b = benign_sf_byte();
  uint32_t const w = b * 0x01010101u;
  uint8_t* base = reinterpret_cast<uint8_t*>(sfb);
  for (int c = tid; c < chunks; c += nthr) {
    int const rr = c & 31;
    int a = c >> 5;
    int const kt = a % n_kt;
    a /= n_kt;
    int const r0 = (a % n_rt) * 128 + rr;
    int const cnt = count_of(a / n_rt);
    uint8_t* p = base + 16LL * c;
    if (r0 >= cnt) {   // no quantized row in this chunk
      *reinterpret_cast<uint4*>(p) = make_uint4(w, w, w, w);
      continue;
    }
    int const k_have = KB - kt * 4;   // K blocks of this atom column that exist, >= 1
    for (int g = 0; g < 4; ++g) {
      uint8_t* pg = p + 4 * g;
      if (r0 + 32 * g >= cnt) {
        *reinterpret_cast<uint32_t*>(pg) = w;
      } else {
        for (int j = k_have; j < 4; ++j) pg[j] = uint8_t(b);   // K slop of a quantized row
      }
    }
  }
}

// Outputs and scales of the GEMM1 epilogue that fuses SwiGLU and GEMM2's input quantization
// (sm100/swiglu_epilogue.cuh). Arch-aliased types only, so the shared op code can fill it.
struct SwigluFp4Args {
  ElementB* b_act = nullptr;           // GEMM2 B operand [E, cap, N/2] (e2m1 codes)
  ElementSF* sfb = nullptr;            // GEMM2 SFB, swizzled (layout_SFB); the kernel writes
                                       // the benign fill of unquantized slots (prologue_fill)
  LayoutSFB layout_SFB{};
  StrideB stride_B{};                  // GEMM2 B stride, packed {cap, N, E}
  float const* alphas = nullptr;       // GEMM1 per-expert alpha [E]
  float const* gscale = nullptr;       // GEMM2 activation global scale, [E] or [1]
  int gscale_per_expert = 0;
  int const* expert_num_tokens = nullptr;   // [E]; token rows at or past the count are skipped
  int sfb_rows = 0;                    // SFB extents for the benign fill: cap, N / 32, E
  int sfb_kblocks = 0;
  int experts = 0;
};

} // namespace paired_nvfp4
