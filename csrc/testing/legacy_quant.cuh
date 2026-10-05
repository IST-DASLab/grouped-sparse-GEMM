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

// Test-only reference: the original two-stage activation quantizer. It writes b_act plus a
// natural-order sf_lin[E, max_n, K / SFVecSize] that scatter_sf then swizzles into SFB. The
// sf_debug self-test uses the un-swizzled intermediate to tell scale-value errors from swizzle
// errors, and gates the fused production quant_act against it byte for byte on valid rows.
// Unlike the production kernel it visits padded rows: zero b_act and sf_lin = ue4m3(gscale[e]).

#pragma once

#include "../quant.cuh"

namespace paired_nvfp4 {

// static: a non-template __global__ in a header needs internal linkage.
static __global__ void legacy_quant_act_kernel(cutlass::bfloat16_t const* __restrict__ x,
                                               ElementB* __restrict__ b_act,
                                               ElementSF* __restrict__ sf_lin,
                                               StrideB stride_B, float const* __restrict__ gscale,
                                               int const* __restrict__ counts,
                                               int max_n, int K, int E, int gscale_per_expert) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int KB = K / SFVecSize;
  long long total = (long long)E * max_n * KB;
  if (idx >= total) return;
  int kb = int(idx % KB);
  int t  = int((idx / KB) % max_n);
  int e  = int(idx / ((long long)KB * max_n));
  int k0 = kb * SFVecSize;
  float g = gscale_per_expert ? gscale[e] : gscale[0];

  uint4* dst = b_act_word(b_act, stride_B, e, t, k0);
  long long sf_idx = ((long long)e * max_n + t) * KB + kb;

  uint4 packed = make_uint4(0u, 0u, 0u, 0u);
  if (t < counts[e]) {
    alignas(16) cutlass::bfloat16_t vals[SFVecSize];
    load_block(vals, x + ((long long)e * max_n + t) * K + k0);
    float amax = 0.f;
    #pragma unroll
    for (int j = 0; j < SFVecSize; ++j) amax = fmaxf(amax, fabsf(float(vals[j])));
    float bsf = amax > 0.f ? amax / 6.f : 1.f;
    ElementSF s = ElementSF(bsf * g);
    sf_lin[sf_idx] = s;
    float inv = float(s) > 0.f ? g / float(s) : 0.f;
    packed = quantize_block_e2m1(vals, inv);
  } else {
    sf_lin[sf_idx] = ElementSF(g);
  }
  *dst = packed;
}

} // namespace paired_nvfp4
