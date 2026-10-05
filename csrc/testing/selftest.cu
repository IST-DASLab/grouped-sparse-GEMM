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
    \brief Test-only self-tests, compiled in when PAIRED_NVFP4_BUILD_TESTS=1.

    Registers torch.ops._paired_nvfp4_test.*. Each entry point builds its inputs on the host,
    writes them through the kernel's own CuTe layouts (so FP4 sub-byte packing and the SF swizzle
    are correct by construction), drives the production ops, and returns a small float tensor of
    error counters that tests/ turns into pass / fail:

      selftest               prune + quantize -> compress -> group_mm vs a host dequant reference
      selftest_quant_act     quant_act -> group_mm vs a host reference
      selftest_quant_act_diff  quant_act bytes vs a host replica of the quantizer
      selftest_sf_debug      legacy two-stage quantizer vs host, and the fused op vs legacy
      selftest_silu_mul_quant  fused SwiGLU+quant vs silu_and_mul -> quant_act, byte for byte
      selftest_scatter_quant   fused scatter+quant vs scatter -> quant_act, byte for byte
      probe_act_layout       read (b_act, sfb) back through the kernel layouts (host)

    References use the quantized operands, so the only residual in the GEMM checks is the
    fp32 -> bf16 output rounding; atol = rtol = 1e-1 is comfortable.
*/

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "cutlass/util/host_tensor.h"

#include "../quant.cuh"
#include "../ops.h"
#include "legacy_quant.cuh"

namespace paired_nvfp4 {

#if PAIRED_NVFP4_ENABLED

namespace {

inline int current_sm() {
  int device = 0, major = 0, minor = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  return major * 10 + minor;
}

inline uint8_t sf_byte(float scale) {           // round a positive scale to UE4M3, return its byte
  ElementSF sf = ElementSF(scale);
  uint8_t b;
  std::memcpy(&b, &sf, 1);
  return b;
}
inline float sf_value(float scale) {            // value the kernel will actually see for that scale
  return float(ElementSF(scale));
}

// Quantize one length-K row to NVFP4 (one scale per SFVecSize block): writes fp4 through `dst(coord_outer, k, g)`,
// fills block scales (UE4M3 bytes) into bs[outer_local*Kb + kb] and the float dequant into deq.
template <class Element, class DstTensor>
void quantize_row(std::vector<float>& row, DstTensor& dst, int outer_idx, int g, int K, int Kb,
                  uint8_t* bs_row, float* deq_row) {
  for (int kb = 0; kb < Kb; ++kb) {
    int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
    float bmax = 0.f;
    for (int k = k0; k < k1; ++k) bmax = std::max(bmax, std::fabs(row[k]));
    float scale = bmax > 0.f ? bmax / 6.f : 1.f;   // FP4 (e2m1) max magnitude = 6
    bs_row[kb] = sf_byte(scale);
    float sval = sf_value(scale);
    for (int k = k0; k < k1; ++k) {
      Element q = Element(row[k] / scale);
      dst(outer_idx, k, g) = q;
      deq_row[k] = float(q) * sval;
    }
  }
}

} // namespace

// Returns a CPU float32 tensor [max_abs_err, max_rel_err, ref_rms, num_bad].
torch::Tensor paired_nvfp4_selftest(int64_t M_, int64_t maxn_, int64_t K_, int64_t E_, int64_t seed_) {
  TORCH_CHECK(current_sm() == kArchSm, "selftest requires SM", kArchSm);
  const int M = int(M_), max_n = int(maxn_), K = int(K_), E = int(E_);
  const int Kb = kblocks(K);
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of 16 for this test");
  TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8 for 4:8 pruning");

  const c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, 0));
  auto dev = torch::Device(torch::kCUDA, 0);
  auto u8_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);

  std::mt19937 gen{static_cast<uint32_t>(seed_)};  // brace-init avoids the most-vexing-parse
  std::normal_distribution<float> nd(0.f, 1.f);

  // Per-expert token counts (exercise padding: {max_n, 3/4, 1/2, 1/4, ...}) and distinct alphas.
  std::vector<int32_t> counts(E);
  std::vector<float>   alphas(E);
  for (int e = 0; e < E; ++e) {
    int step = std::max(1, max_n / 4);
    counts[e] = std::max(1, max_n - (e % 4) * step);
    alphas[e] = 0.5f + 1.5f * (E > 1 ? float(e) / float(E - 1) : 0.f);  // span [0.5, 2.0]
  }

  // ---- Weight: dense FP4 (RowMajor, groups along M) + UE4M3 block scales + host dequant ----
  cutlass::HostTensor<ElementA, LayoutTagA> tA(cutlass::make_Coord(M * E, K));
  std::memset(tA.host_data(), 0, size_t(M) * E * K / 2);
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, E});
  auto Aten = make_tensor(cute::recast_ptr<ElementA>(tA.host_data()),
                          make_layout(make_shape(M, K, E), stride_A));
  CompressorUtility comp_util(make_sparse_shape(M, kPlaceholderExtent, K, E), stride_A);
  std::vector<uint8_t> Wbs(size_t(E) * M * Kb, 0);
  std::vector<float>   svals(size_t(E) * M * Kb, 0.f);
  std::vector<float>   Wdq(size_t(E) * M * K, 0.f);

  // Quantize the full (dense) weight first; the paired-4:8 mask is applied afterwards by the
  // compressor's OWN structure_sparse_zero_mask_fill so the sparsity granularity (whole nv_float4_t
  // pairs, half zeroed per LogicalElemsAMmaRawPerChunk chunk) matches what the compressor encodes.
  for (int e = 0; e < E; ++e) {
    for (int m = 0; m < M; ++m) {
      std::vector<float> row(K);
      for (int k = 0; k < K; ++k) row[k] = nd(gen);
      for (int kb = 0; kb < Kb; ++kb) {
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        float bmax = 0.f;
        for (int k = k0; k < k1; ++k) bmax = std::max(bmax, std::fabs(row[k]));
        float scale = bmax > 0.f ? bmax / 6.f : 1.f;
        Wbs[(size_t(e) * M + m) * Kb + kb]   = sf_byte(scale);
        svals[(size_t(e) * M + m) * Kb + kb] = sf_value(scale);
        for (int k = k0; k < k1; ++k) Aten(m, k, e) = ElementA(row[k] / scale);
      }
    }
  }
  comp_util.structure_sparse_zero_mask_fill(tA.host_data(), static_cast<uint64_t>(seed_) + 777);
  // Host dequant from the MASKED fp4 values (zeroed pairs -> 0); kernel applies the same scales.
  for (int e = 0; e < E; ++e)
    for (int m = 0; m < M; ++m)
      for (int k = 0; k < K; ++k)
        Wdq[(size_t(e) * M + m) * K + k] =
            float(ElementA(Aten(m, k, e))) * svals[(size_t(e) * M + m) * Kb + k / SFVecSize];

  torch::Tensor W_packed =
      torch::from_blob(tA.host_data(), {E, M, K / 2}, u8_cpu).clone().to(dev);
  torch::Tensor W_blockscale =
      torch::from_blob(Wbs.data(), {E, M, Kb}, u8_cpu).clone().to(dev);

  // ---- Activations: dense FP4 in kernel ColMajor B-layout + scattered SFB + host dequant ----
  cutlass::HostTensor<ElementB, LayoutTagB> tB(cutlass::make_Coord(K, max_n * E));
  std::memset(tB.host_data(), 0, size_t(K) * max_n * E / 2);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E});
  auto Bten = make_tensor(cute::recast_ptr<ElementB>(tB.host_data()),
                          make_layout(make_shape(max_n, K, E), stride_B));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(M, max_n, K, E));
  std::vector<uint8_t> SFB_host(size_t(sfb_numel(M, max_n, K, E)), 0);
  auto sfb_t = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(SFB_host.data())),
                           layout_SFB);
  std::vector<float>   Xdq(size_t(E) * max_n * K, 0.f);
  std::vector<uint8_t> sfb_tmp(Kb);

  for (int e = 0; e < E; ++e) {
    for (int t = 0; t < counts[e]; ++t) {
      std::vector<float> row(K);
      for (int k = 0; k < K; ++k) row[k] = nd(gen);  // DIAGNOSTIC: random activations -> SFB (direct fill) varies
      quantize_row<ElementB>(row, Bten, t, e, K, Kb, sfb_tmp.data(), &Xdq[(size_t(e) * max_n + t) * K]);
      // Write every (t, k, e) slot through the same cute tensor the kernel reads (handles swizzle).
      for (int k = 0; k < K; ++k) {
        ElementSF v; std::memcpy(&v, &sfb_tmp[k / SFVecSize], 1);
        sfb_t(t, k, e) = v;
      }
    }
  }

  torch::Tensor B_act = torch::from_blob(tB.host_data(), {E, max_n, K / 2}, u8_cpu).clone().to(dev);
  torch::Tensor SFB =
      torch::from_blob(SFB_host.data(), {int64_t(SFB_host.size())}, u8_cpu).clone().to(dev);

  torch::Tensor alphas_t =
      torch::from_blob(alphas.data(), {E}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev);
  torch::Tensor counts_t =
      torch::from_blob(counts.data(), {E}, torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev);

  // ---- Drive the REAL ops ----
  auto compressed = paired_nvfp4_compress(W_packed, W_blockscale, K);
  torch::Tensor A_comp = std::get<0>(compressed);
  torch::Tensor E_meta = std::get<1>(compressed);
  torch::Tensor SFA    = std::get<2>(compressed);

  torch::Tensor D =
      torch::zeros({E, max_n, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  paired_nvfp4_group_mm(D, A_comp, E_meta, SFA, B_act, SFB, alphas_t, counts_t, M, K,
                        PAIRED_NVFP4_DEFAULT_CONFIG_ID, PAIRED_NVFP4_DEFAULT_CLUSTER_M,
                        PAIRED_NVFP4_DEFAULT_CLUSTER_N);

  // ---- Host reference + comparison (read SFA/SFB THROUGH the cute layouts the kernel uses) ----
  (void)Wdq; (void)Xdq;
  torch::Tensor Dh = D.to(torch::kFloat32).cpu().contiguous();
  const float* d = Dh.data_ptr<float>();
  torch::Tensor sfa_cpu = SFA.cpu().contiguous();
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_sparse_shape(M, max_n, K, E));
  auto sfa_view = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(sfa_cpu.data_ptr<uint8_t>())), layout_SFA);
  auto sfb_view = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(SFB_host.data())), layout_SFB);
  double max_abs = 0.0, max_rel = 0.0, ref_sq = 0.0;
  long num_bad = 0, num_cmp = 0;
  const double atol = 1e-1, rtol = 1e-1;
  for (int e = 0; e < E; ++e) {
    for (int t = 0; t < counts[e]; ++t) {
      for (int m = 0; m < M; ++m) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          double a = double(float(ElementA(Aten(m, k, e)))) * double(float(ElementSF(sfa_view(m, k, e))));
          double b = double(float(ElementB(Bten(t, k, e)))) * double(float(ElementSF(sfb_view(t, k, e))));
          acc += a * b;
        }
        double ref = double(alphas[e]) * acc;
        double got = double(d[(size_t(e) * max_n + t) * M + m]);
        double ae = std::fabs(got - ref);
        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, ae / (std::fabs(ref) + 1e-6));
        ref_sq += ref * ref;
        if (ae > atol + rtol * std::fabs(ref)) ++num_bad;
        ++num_cmp;
      }
    }
  }
  double ref_rms = num_cmp ? std::sqrt(ref_sq / num_cmp) : 0.0;

  auto out = torch::empty({4}, torch::TensorOptions().dtype(torch::kFloat32));
  out[0] = float(max_abs);
  out[1] = float(max_rel);
  out[2] = float(ref_rms);
  out[3] = float(num_bad);
  return out;
}

// LAYOUT PROBE (test-only): read an EXTERNALLY-produced (b_act, sfb) pair through the EXACT cute
// layouts group_mm consumes and return the dequantized activations [E, max_n, K] float32 =
//   ElementB(b_act read as Bten(t,k,e)) * ElementSF(sfb read as sfb_view(t,k,e)).
// Feed your FlashInfer/reordered tensors for (features, max_n, K, E); if the result reconstructs the
// pre-quant activation (within fp4 rounding), the byte layout matches what the kernel reads. A
// mismatch in L-nesting (expert plane stride) or the in-plane 128x4 SF swizzle shows up as garbage
// or mispaired scales here, isolated from the GEMM. Reads exactly as the kernel: B is ColMajor with
// stride_B={max_n,K,E}; sfb through tile_atom_to_shape_SFB(features,max_n,K,E). No GPU compute (host
// read), so it works on CPU or CUDA tensors.
torch::Tensor paired_nvfp4_probe_act_layout(torch::Tensor b_act, torch::Tensor sfb,
                                            int64_t features_, int64_t maxn_, int64_t K_, int64_t E_) {
  const int features = int(features_), max_n = int(maxn_), K = int(K_), E = int(E_);
  TORCH_CHECK(b_act.dim() == 3, "b_act must be [E, max_n, K/2]");
  TORCH_CHECK(int(b_act.size(0)) == E && int(b_act.size(1)) == max_n && int(b_act.size(2)) == K / 2,
              "b_act must be [E, max_n, K/2]; got [", b_act.size(0), ",", b_act.size(1), ",", b_act.size(2), "]");
  TORCH_CHECK(b_act.scalar_type() == torch::kUInt8 && sfb.scalar_type() == torch::kUInt8,
              "b_act and sfb must be uint8");
  const int64_t need = sfb_numel(features, max_n, K, E);
  TORCH_CHECK(int64_t(sfb.numel()) >= need,
              "sfb too small: have ", int64_t(sfb.numel()), " elems, kernel reads ", need,
              " (= ceil(max_n/128)*128 * ceil(ceil(K/16)/4)*4 * E)");

  auto b_cpu   = b_act.to(torch::kCPU).contiguous();
  auto sfb_cpu = sfb.to(torch::kCPU).contiguous();

  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E});
  auto Bten = make_tensor(
      cute::recast_ptr<ElementB>(reinterpret_cast<ElementB*>(b_cpu.data_ptr<uint8_t>())),
      make_layout(make_shape(max_n, K, E), stride_B));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(features, max_n, K, E));
  auto sfb_view = make_tensor(
      cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(sfb_cpu.data_ptr<uint8_t>())),
      layout_SFB);

  auto out = torch::empty({E, max_n, K}, torch::TensorOptions().dtype(torch::kFloat32));
  float* o = out.data_ptr<float>();
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < max_n; ++t)
      for (int k = 0; k < K; ++k)
        o[(size_t(e) * max_n + t) * K + k] =
            float(ElementB(Bten(t, k, e))) * float(ElementSF(sfb_view(t, k, e)));
  return out;
}

// END-TO-END validation of quant_act through the REAL group_mm, with an INDEPENDENT reference
// (host weight x host activation) — bypasses the probe entirely. Weight is random FP4 (4:8 pruned)
// like the main self-test; activation x is CONSTANT per SFVecSize block, |val| in [0.25,4], so fp4 is exact
// and the reference can use the original host x directly. If group_mm(compress(W), quant_act(x))
// matches alpha * (Wdq @ x), then quant_act emits kernel-correct bytes. Returns [max_abs, max_rel,
// ref_rms, num_bad].
torch::Tensor paired_nvfp4_selftest_quant_act(int64_t M_, int64_t maxn_, int64_t K_, int64_t E_, int64_t seed_) {
  TORCH_CHECK(current_sm() == kArchSm, "selftest_quant_act requires SM", kArchSm);
  const int M = int(M_), max_n = int(maxn_), K = int(K_), E = int(E_), Kb = kblocks(K);
  TORCH_CHECK(K % SFVecSize == 0 && K % 8 == 0, "K must be a multiple of 16");

  const c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, 0));
  auto dev = torch::Device(torch::kCUDA, 0);
  auto u8_cpu = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
  std::mt19937 gen{static_cast<uint32_t>(seed_)};
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ur(0.f, 1.f);
  std::uniform_int_distribution<int> ub(0, 1);

  std::vector<int32_t> counts(E);
  std::vector<float>   alphas(E);
  for (int e = 0; e < E; ++e) {
    int step = std::max(1, max_n / 4);
    counts[e] = std::max(1, max_n - (e % 4) * step);
    alphas[e] = 0.5f + 1.5f * (E > 1 ? float(e) / float(E - 1) : 0.f);
  }

  // ---- Weight: random FP4 (RowMajor) + UE4M3 block scales + 4:8 mask (same as the main self-test).
  cutlass::HostTensor<ElementA, LayoutTagA> tA(cutlass::make_Coord(M * E, K));
  std::memset(tA.host_data(), 0, size_t(M) * E * K / 2);
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, E});
  auto Aten = make_tensor(cute::recast_ptr<ElementA>(tA.host_data()),
                          make_layout(make_shape(M, K, E), stride_A));
  CompressorUtility comp_util(make_sparse_shape(M, kPlaceholderExtent, K, E), stride_A);
  std::vector<uint8_t> Wbs(size_t(E) * M * Kb, 0);
  for (int e = 0; e < E; ++e)
    for (int m = 0; m < M; ++m) {
      std::vector<float> row(K);
      for (int k = 0; k < K; ++k) row[k] = nd(gen);
      for (int kb = 0; kb < Kb; ++kb) {
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        float bmax = 0.f;
        for (int k = k0; k < k1; ++k) bmax = std::max(bmax, std::fabs(row[k]));
        float s = bmax > 0.f ? bmax / 6.f : 1.f;
        Wbs[(size_t(e) * M + m) * Kb + kb] = sf_byte(s);
        for (int k = k0; k < k1; ++k) Aten(m, k, e) = ElementA(row[k] / s);
      }
    }
  comp_util.structure_sparse_zero_mask_fill(tA.host_data(), static_cast<uint64_t>(seed_) + 777);
  torch::Tensor W_packed     = torch::from_blob(tA.host_data(), {E, M, K / 2}, u8_cpu).clone().to(dev);
  torch::Tensor W_blockscale = torch::from_blob(Wbs.data(), {E, M, Kb}, u8_cpu).clone().to(dev);
  auto compressed = paired_nvfp4_compress(W_packed, W_blockscale, K);
  torch::Tensor A_comp = std::get<0>(compressed), E_meta = std::get<1>(compressed), SFA = std::get<2>(compressed);

  // Host weight dequant read through SFA (validated path): Wdq(m,k,e) = fp4 * SFA.
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_sparse_shape(M, max_n, K, E));
  torch::Tensor sfa_cpu = SFA.cpu().contiguous();
  auto sfa_view = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(sfa_cpu.data_ptr<uint8_t>())), layout_SFA);

  // ---- Activation x [E,max_n,K]: CONSTANT per SFVecSize block, |val| in [0.25,4] (fp4-exact).
  std::vector<float> xh(size_t(E) * max_n * K, 0.f);
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < counts[e]; ++t)
      for (int kb = 0; kb < Kb; ++kb) {
        float c = (ub(gen) ? 1.f : -1.f) * std::pow(2.f, ur(gen) * 4.f - 2.f);   // [0.25, 4]
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        for (int k = k0; k < k1; ++k) xh[(size_t(e) * max_n + t) * K + k] = c;
      }
  std::vector<cutlass::bfloat16_t> xbf(size_t(E) * max_n * K);
  for (size_t i = 0; i < xbf.size(); ++i) xbf[i] = cutlass::bfloat16_t(xh[i]);
  torch::Tensor x = torch::from_blob(xbf.data(), {E, max_n, K},
                                     torch::TensorOptions().dtype(torch::kBFloat16)).clone().to(dev);
  torch::Tensor gscale   = torch::ones({E}, torch::TensorOptions().dtype(torch::kFloat32)).to(dev);
  torch::Tensor counts_t = torch::from_blob(counts.data(), {E}, torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev);
  torch::Tensor alphas_t = torch::from_blob(alphas.data(), {E}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev);

  // ---- The ops under test: quant_act (B side) then the real grouped GEMM.
  auto qa = paired_nvfp4_quant_act(x, gscale, counts_t, M);
  torch::Tensor b_act = std::get<0>(qa), sfb = std::get<1>(qa);
  torch::Tensor D = torch::zeros({E, max_n, M}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  paired_nvfp4_group_mm(D, A_comp, E_meta, SFA, b_act, sfb, alphas_t, counts_t, M, K,
                        PAIRED_NVFP4_DEFAULT_CONFIG_ID, PAIRED_NVFP4_DEFAULT_CLUSTER_M,
                        PAIRED_NVFP4_DEFAULT_CLUSTER_N);

  // Reference activation = the quantized b_act * sfb the GEMM actually multiplies, read back through
  // the same CuTe B / SFB layouts. Comparing against the ideal fp32 input instead would fold in the
  // UE4M3 scale rounding (~6% per block), and with random-signed 4:8 weights the length-K dot product
  // cancels to |ref| ~ 0 often enough that relative error explodes on correct output. Reading back
  // the stored product leaves only the fp32 -> bf16 output rounding.
  torch::Tensor b_cpu   = b_act.to(torch::kCPU).contiguous();
  torch::Tensor sfb_cpu = sfb.to(torch::kCPU).contiguous();
  StrideB stride_Bq = cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E});
  auto Bq = make_tensor(cute::recast_ptr<ElementB>(reinterpret_cast<ElementB*>(b_cpu.data_ptr<uint8_t>())),
                        make_layout(make_shape(max_n, K, E), stride_Bq));
  auto layout_SFBq = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(M, max_n, K, E));
  auto sfbq = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(sfb_cpu.data_ptr<uint8_t>())), layout_SFBq);

  // ---- End-to-end reference: D_ref[e,t,m] = alpha[e] * sum_k Wdq(m,k,e) * (b_act*sfb)(t,k,e).
  torch::Tensor Dh = D.to(torch::kFloat32).cpu().contiguous();
  const float* d = Dh.data_ptr<float>();
  double max_abs = 0, max_rel = 0, ref_sq = 0;
  long num_bad = 0, num_cmp = 0;
  const double atol = 1e-1, rtol = 1e-1;
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < counts[e]; ++t)
      for (int m = 0; m < M; ++m) {
        double acc = 0;
        for (int k = 0; k < K; ++k) {
          double w = double(float(ElementA(Aten(m, k, e)))) * double(float(ElementSF(sfa_view(m, k, e))));
          double bq = double(float(ElementB(Bq(t, k, e)))) * double(float(ElementSF(sfbq(t, k, e))));
          acc += w * bq;
        }
        double ref = double(alphas[e]) * acc;
        double got = double(d[(size_t(e) * max_n + t) * M + m]);
        double ae = std::fabs(got - ref);
        max_abs = std::max(max_abs, ae);
        max_rel = std::max(max_rel, ae / (std::fabs(ref) + 1e-6));
        ref_sq += ref * ref;
        if (ae > atol + rtol * std::fabs(ref)) ++num_bad;
        ++num_cmp;
      }
  double ref_rms = num_cmp ? std::sqrt(ref_sq / num_cmp) : 0.0;
  auto out = torch::empty({4}, torch::TensorOptions().dtype(torch::kFloat32));
  out[0] = float(max_abs); out[1] = float(max_rel); out[2] = float(ref_rms); out[3] = float(num_bad);
  return out;
}

// BYTE-LEVEL diagnostic: produce (b_act, sfb) BOTH via the host reference quant (the SAME formula
// quant_act_kernel runs: amax/6 -> e2m1, scale -> ue4m3, written through the kernel's cute B / SFB
// layouts) AND via the quant_act op, for the IDENTICAL bf16 activation, then count raw-byte
// mismatches. For a constant-per-SFVecSize-block activation, host and device quant are bit-deterministic and
// MUST agree byte-for-byte if quant_act is correct, so this localizes a quant_act bug to b_act vs sfb
// vs NEITHER (the latter would indict the GEMM-test reference, not quant_act). The host replicate
// reads the SAME bf16 values the device reads (float(xbf), not the fp32 xh) so there is no spurious
// bf16-rounding mismatch. Returns
//   [b_act_mismatch, sfb_mismatch, b_act_first_idx, sfb_first_idx,
//    probe_max_rel, n_active_elem, b_total_bytes, sfb_total_bytes].
torch::Tensor paired_nvfp4_selftest_quant_act_diff(int64_t M_, int64_t maxn_, int64_t K_, int64_t E_, int64_t seed_) {
  TORCH_CHECK(current_sm() == kArchSm, "selftest_quant_act_diff requires SM", kArchSm);
  const int M = int(M_), max_n = int(maxn_), K = int(K_), E = int(E_), Kb = kblocks(K);
  TORCH_CHECK(K % SFVecSize == 0 && K % 8 == 0, "K must be a multiple of 16");

  const c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, 0));
  auto dev = torch::Device(torch::kCUDA, 0);
  std::mt19937 gen{static_cast<uint32_t>(seed_)};
  std::uniform_real_distribution<float> ur(0.f, 1.f);
  std::uniform_int_distribution<int> ub(0, 1);

  std::vector<int32_t> counts(E);
  for (int e = 0; e < E; ++e) {
    int step = std::max(1, max_n / 4);
    counts[e] = std::max(1, max_n - (e % 4) * step);
  }
  const float g = 1.0f;  // gscale = ones (matches the device call below)

  // Activation x [E,max_n,K]: CONSTANT per SFVecSize block, |val| in [0.25,4].
  std::vector<float> xh(size_t(E) * max_n * K, 0.f);
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < counts[e]; ++t)
      for (int kb = 0; kb < Kb; ++kb) {
        float c = (ub(gen) ? 1.f : -1.f) * std::pow(2.f, ur(gen) * 4.f - 2.f);
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        for (int k = k0; k < k1; ++k) xh[(size_t(e) * max_n + t) * K + k] = c;
      }
  std::vector<cutlass::bfloat16_t> xbf(xh.size());
  for (size_t i = 0; i < xbf.size(); ++i) xbf[i] = cutlass::bfloat16_t(xh[i]);
  auto xbf_at = [&](int e, int t, int k) -> float {
    return float(xbf[(size_t(e) * max_n + t) * K + k]);   // the EXACT value the device kernel reads
  };

  torch::Tensor x = torch::from_blob(xbf.data(), {E, max_n, K},
                                     torch::TensorOptions().dtype(torch::kBFloat16)).clone().to(dev);
  torch::Tensor gscale   = torch::ones({E}, torch::TensorOptions().dtype(torch::kFloat32)).to(dev);
  torch::Tensor counts_t = torch::from_blob(counts.data(), {E}, torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev);

  // ---- HOST reference quant: mirror quant_act_kernel EXACTLY (incl. padding), through the kernel's
  //      cute B (ColMajor, stride_B={max_n,K,E}) and SFB (tile_atom_to_shape_SFB) layouts. ----
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E});
  cutlass::HostTensor<ElementB, LayoutTagB> tBh(cutlass::make_Coord(K, max_n * E));
  std::memset(tBh.host_data(), 0, size_t(K) * max_n * E / 2);
  auto Bh = make_tensor(cute::recast_ptr<ElementB>(tBh.host_data()),
                        make_layout(make_shape(max_n, K, E), stride_B));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(M, max_n, K, E));
  // SFB slots outside the written rows (padded rows, and the atom slop from rounding max_n up to 128
  // and K / SFVecSize up to 4) hold the op's benign nonzero pre-fill, so the reference starts from
  // the same byte, not from zero; otherwise unaligned shapes differ on exactly those slots.
  uint8_t benign_sf;
  { ElementSF one(1.0f); std::memcpy(&benign_sf, &one, 1); }
  std::vector<uint8_t> SFBh(size_t(sfb_numel(M, max_n, K, E)), benign_sf);
  auto sfbh = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(SFBh.data())), layout_SFB);
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < max_n; ++t)
      for (int kb = 0; kb < Kb; ++kb) {
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        if (t < counts[e]) {
          float amax = 0.f;
          for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(xbf_at(e, t, k)));
          float bsf = amax > 0.f ? amax / 6.f : 1.f;
          ElementSF s = ElementSF(bsf * g);
          // Mirror quant_act_kernel: quantize against the DEQUANTIZED stored scale.
          float inv = float(s) > 0.f ? g / float(s) : 0.f;
          for (int k = k0; k < k1; ++k) { sfbh(t, k, e) = s; Bh(t, k, e) = ElementB(xbf_at(e, t, k) * inv); }
        } else {
          ElementSF s = ElementSF(g);
          for (int k = k0; k < k1; ++k) { sfbh(t, k, e) = s; Bh(t, k, e) = ElementB(0.f); }
        }
      }

  // ---- DEVICE quant via the op under test (gscale = ones, so g = 1 as in the host replicate). ----
  auto qa = paired_nvfp4_quant_act(x, gscale, counts_t, M);
  torch::Tensor b_act = std::get<0>(qa).to(torch::kCPU).contiguous();
  torch::Tensor sfb   = std::get<1>(qa).to(torch::kCPU).contiguous();

  // ---- Raw byte comparison (both buffers use the identical physical layout). ----
  // b_act is compared on valid rows only: the op leaves padded rows unwritten, while this host
  // replica mirrors the legacy quantizer and zero-fills them. sfb is compared over the whole buffer:
  // the replica writes ue4m3(gscale) into padded slots, which with gscale = 1 is exactly the op's
  // benign pre-fill byte.
  const uint8_t* bq = b_act.data_ptr<uint8_t>();
  const uint8_t* bh = reinterpret_cast<const uint8_t*>(tBh.host_data());
  const long b_row_bytes = long(K) / 2;
  long b_total = 0, b_mis = 0, b_first = -1;
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < counts[e]; ++t) {
      long off = (long(e) * max_n + t) * b_row_bytes;
      b_total += b_row_bytes;
      for (long b = 0; b < b_row_bytes; ++b)
        if (bq[off + b] != bh[off + b]) { ++b_mis; if (b_first < 0) b_first = off + b; }
    }

  const uint8_t* sq = sfb.data_ptr<uint8_t>();
  long s_total = long(SFBh.size()), s_mis = 0, s_first = -1;
  for (long i = 0; i < s_total; ++i) if (sq[i] != SFBh[i]) { ++s_mis; if (s_first < 0) s_first = i; }

  // ---- Cross-check: dequant the DEVICE (b_act, sfb) through the cute layouts vs the bf16 input. ----
  auto Bq = make_tensor(cute::recast_ptr<ElementB>(reinterpret_cast<ElementB*>(b_act.data_ptr<uint8_t>())),
                        make_layout(make_shape(max_n, K, E), stride_B));
  auto sfbq = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>())), layout_SFB);
  double probe_max_rel = 0.0; long n_active = 0;
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < counts[e]; ++t)
      for (int k = 0; k < K; ++k) {
        double got = double(float(ElementB(Bq(t, k, e)))) * double(float(ElementSF(sfbq(t, k, e))));
        double ref = double(xbf_at(e, t, k));
        probe_max_rel = std::max(probe_max_rel, std::fabs(got - ref) / (std::fabs(ref) + 1e-6));
        ++n_active;
      }

  auto out = torch::empty({8}, torch::TensorOptions().dtype(torch::kFloat32));
  out[0] = float(b_mis);   out[1] = float(s_mis);
  out[2] = float(b_first); out[3] = float(s_first);
  out[4] = float(probe_max_rel); out[5] = float(n_active);
  out[6] = float(b_total); out[7] = float(s_total);
  return out;
}

// Device probe: compute layout_SFB(o, kb*16, g) ON DEVICE (layout passed by value, exactly as
// scatter_sf_kernel receives it) into out[(g*outer+o)*KB+kb]. Comparing to the host evaluation of the
// SAME layout tells us whether the cute layout indexes identically on host and device.
template <class LayoutSF>
static __global__ void sf_offset_kernel(LayoutSF layout, int outer, int K, int groups, int* out) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  int KB = K / SFVecSize;
  long long total = (long long)outer * KB * groups;
  if (idx >= total) return;
  int kb = int(idx % KB);
  int o  = int((idx / KB) % outer);
  int g  = int(idx / ((long long)KB * outer));
  out[idx] = int(layout(cute::make_coord(o, kb * SFVecSize, g)));
}

// DECOMPOSING debug probe for the quant_act SFB path. Replicates the LEGACY two-stage internals
// (quant_act_kernel -> scatter_sf) so it can expose the intermediate linear sf_lin, then tests the
// stages SEPARATELY:
//   [0] lin_mis : device sf_lin (scale VALUES) vs an independent host computation  -> a value/scale bug
//   [1] swz_mis : device scatter_sf(sfb) vs a HOST functor-write of the SAME DEVICE sf_lin -> a SWIZZLE
//                 (offset/layout) bug, with the scale values held identical so only placement differs
//   [2] off_mis : device-evaluated layout_SFB(coord) vs host-evaluated -> host/device layout mismatch
//   [9] fused_mis : the PRODUCTION op (fused single-kernel quant_act) vs the legacy two-stage path,
//                 raw-byte equality over BOTH outputs (b_act + sfb). Same quant math + same
//                 functor-addressed SF slots => MUST be 0; nonzero isolates a fused-path regression.
// Exactly one of [0]/[1] (and possibly [2]) should be nonzero, naming the broken stage unambiguously.
// Returns [lin_mis, swz_mis, off_mis, lin_first, swz_first, off_first, lin_total, sfb_total, n_blocks,
//          fused_mis].
torch::Tensor paired_nvfp4_selftest_sf_debug(int64_t M_, int64_t maxn_, int64_t K_, int64_t E_, int64_t seed_) {
  TORCH_CHECK(current_sm() == kArchSm, "selftest_sf_debug requires SM", kArchSm);
  const int M = int(M_), max_n = int(maxn_), K = int(K_), E = int(E_), KB = K / SFVecSize;
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of 16");

  const c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, 0));
  auto dev = torch::Device(torch::kCUDA, 0);
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(dev);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(0).stream();
  std::mt19937 gen{static_cast<uint32_t>(seed_)};
  std::uniform_real_distribution<float> ur(0.f, 1.f);
  std::uniform_int_distribution<int> ub(0, 1);

  std::vector<int32_t> counts(E);
  for (int e = 0; e < E; ++e) { int step = std::max(1, max_n / 4); counts[e] = std::max(1, max_n - (e % 4) * step); }
  const float g = 1.0f;

  std::vector<float> xh(size_t(E) * max_n * K, 0.f);
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < counts[e]; ++t)
      for (int kb = 0; kb < KB; ++kb) {
        float c = (ub(gen) ? 1.f : -1.f) * std::pow(2.f, ur(gen) * 4.f - 2.f);
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        for (int k = k0; k < k1; ++k) xh[(size_t(e) * max_n + t) * K + k] = c;
      }
  std::vector<cutlass::bfloat16_t> xbf(xh.size());
  for (size_t i = 0; i < xbf.size(); ++i) xbf[i] = cutlass::bfloat16_t(xh[i]);
  auto xbf_at = [&](int e, int t, int k) -> float { return float(xbf[(size_t(e) * max_n + t) * K + k]); };
  torch::Tensor x = torch::from_blob(xbf.data(), {E, max_n, K}, torch::TensorOptions().dtype(torch::kBFloat16)).clone().to(dev);
  torch::Tensor gscale   = torch::ones({E}, torch::TensorOptions().dtype(torch::kFloat32)).to(dev);
  torch::Tensor counts_t = torch::from_blob(counts.data(), {E}, torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev);

  // --- Replicate quant_act internals so we can read sf_lin out BEFORE the scatter. ---
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E});
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(M, max_n, K, E));
  torch::Tensor b_act  = torch::empty({E, max_n, K / 2}, u8);
  // Same benign pre-fill as the production op: scatter_sf writes only in-range slots, so slop keeps
  // the initial value.
  uint8_t benign_legacy;
  { ElementSF one(1.0f); std::memcpy(&benign_legacy, &one, 1); }
  torch::Tensor sfb    = torch::full({sfb_numel(M, max_n, K, E)}, int64_t(benign_legacy), u8);
  torch::Tensor sf_lin = torch::empty({int64_t(E) * max_n * KB}, u8);

  long long total = (long long)E * max_n * KB;
  int threads = 256;
  long long blocks = (total + threads - 1) / threads;
  legacy_quant_act_kernel<<<dim3((unsigned)blocks), dim3(threads), 0, stream>>>(
      reinterpret_cast<cutlass::bfloat16_t const*>(x.data_ptr()),
      reinterpret_cast<ElementB*>(b_act.data_ptr<uint8_t>()),
      reinterpret_cast<ElementSF*>(sf_lin.data_ptr<uint8_t>()),
      stride_B, gscale.data_ptr<float>(), counts_t.data_ptr<int32_t>(), max_n, K, E, /*gpe=*/1);
  torch::Tensor sf_lin_dev = sf_lin.to(torch::kCPU).contiguous();   // device-computed linear scales
  scatter_sf(reinterpret_cast<ElementSF const*>(sf_lin.data_ptr<uint8_t>()),
             reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>()), layout_SFB, /*outer=*/max_n, K, E, stream);
  torch::Tensor sfb_dev = sfb.to(torch::kCPU).contiguous();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));

  // --- [0] scale-value check: independent host sf_lin vs device sf_lin (linear, no swizzle). ---
  const uint8_t* sld = sf_lin_dev.data_ptr<uint8_t>();
  long lin_total = long(size_t(E) * max_n * KB), lin_mis = 0, lin_first = -1;
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < max_n; ++t)
      for (int kb = 0; kb < KB; ++kb) {
        long li = (long(e) * max_n + t) * KB + kb;
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        float val;
        if (t < counts[e]) {
          float amax = 0.f;
          for (int k = k0; k < k1; ++k) amax = std::max(amax, std::fabs(xbf_at(e, t, k)));
          val = (amax > 0.f ? amax / 6.f : 1.f) * g;
        } else val = g;
        if (sld[li] != sf_byte(val)) { ++lin_mis; if (lin_first < 0) lin_first = li; }
      }

  // --- [1] swizzle check: host functor-write of the DEVICE sf_lin vs device scatter sfb (same values). ---
  // SFB slots outside the written rows (padded rows, and the atom slop from rounding max_n up to 128
  // and K / SFVecSize up to 4) hold the op's benign nonzero pre-fill, so the reference starts from
  // the same byte, not from zero; otherwise unaligned shapes differ on exactly those slots.
  uint8_t benign_sf;
  { ElementSF one(1.0f); std::memcpy(&benign_sf, &one, 1); }
  std::vector<uint8_t> SFBh(size_t(sfb_numel(M, max_n, K, E)), benign_sf);
  auto sfbh = make_tensor(cute::recast_ptr<ElementSF>(reinterpret_cast<ElementSF*>(SFBh.data())), layout_SFB);
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < max_n; ++t)
      for (int kb = 0; kb < KB; ++kb) {
        long li = (long(e) * max_n + t) * KB + kb;
        ElementSF v; std::memcpy(&v, &sld[li], 1);
        int k0 = kb * SFVecSize, k1 = std::min(k0 + SFVecSize, K);
        for (int k = k0; k < k1; ++k) sfbh(t, k, e) = v;
      }
  const uint8_t* sd = sfb_dev.data_ptr<uint8_t>();
  long sfb_total = long(SFBh.size()), swz_mis = 0, swz_first = -1;
  for (long i = 0; i < sfb_total; ++i) if (sd[i] != SFBh[i]) { ++swz_mis; if (swz_first < 0) swz_first = i; }

  // --- [2] layout-offset check: device-evaluated layout_SFB(coord) vs host-evaluated. ---
  torch::Tensor d_off = torch::empty({int64_t(E) * max_n * KB}, torch::TensorOptions().dtype(torch::kInt32).device(dev));
  sf_offset_kernel<decltype(layout_SFB)><<<dim3((unsigned)blocks), dim3(threads), 0, stream>>>(
      layout_SFB, max_n, K, E, d_off.data_ptr<int32_t>());
  torch::Tensor d_off_cpu = d_off.to(torch::kCPU).contiguous();
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  const int32_t* doff = d_off_cpu.data_ptr<int32_t>();
  long off_mis = 0, off_first = -1;
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < max_n; ++t)
      for (int kb = 0; kb < KB; ++kb) {
        long li = (long(e) * max_n + t) * KB + kb;
        int ho = int(layout_SFB(cute::make_coord(t, kb * SFVecSize, e)));
        if (doff[li] != ho) { ++off_mis; if (off_first < 0) off_first = li; }
      }

  // --- [9] fused-vs-legacy gate: the production op (fused single-kernel) must reproduce the legacy
  //     two-stage outputs byte-for-byte (b_act from the direct quant_act_kernel launch above, sfb
  //     from scatter_sf). M is the `features` arg, so the op builds the identical SFB layout.
  auto qa_fused = paired_nvfp4_quant_act(x, gscale, counts_t, M);
  torch::Tensor b_fused   = std::get<0>(qa_fused).to(torch::kCPU).contiguous();
  torch::Tensor sfb_fused = std::get<1>(qa_fused).to(torch::kCPU).contiguous();
  torch::Tensor b_legacy  = b_act.to(torch::kCPU).contiguous();
  long fused_mis = 0;
  {
    TORCH_CHECK(b_fused.numel() == b_legacy.numel() && long(sfb_fused.numel()) == sfb_total,
                "fused/legacy quant_act output size mismatch");
    // b_act: valid rows only. The LEGACY kernel still zero-fills padded rows; the fused
    // production kernel skips their store (never read -- group_mm clips to counts[e]), so the two
    // agree byte-for-byte exactly where it matters. The sfb half of the gate stays full-buffer.
    const uint8_t* pf = b_fused.data_ptr<uint8_t>();
    const uint8_t* pl = b_legacy.data_ptr<uint8_t>();
    const long b_row_bytes = long(K) / 2;
    for (int e = 0; e < E; ++e)
      for (int t = 0; t < counts[e]; ++t) {
        long off = (long(e) * max_n + t) * b_row_bytes;
        for (long b = 0; b < b_row_bytes; ++b) if (pf[off + b] != pl[off + b]) ++fused_mis;
      }
    const uint8_t* qf = sfb_fused.data_ptr<uint8_t>();
    for (long i = 0; i < sfb_total; ++i) if (qf[i] != sd[i]) ++fused_mis;
  }

  auto out = torch::empty({10}, torch::TensorOptions().dtype(torch::kFloat32));
  out[0] = float(lin_mis);   out[1] = float(swz_mis);   out[2] = float(off_mis);
  out[3] = float(lin_first); out[4] = float(swz_first); out[5] = float(off_first);
  out[6] = float(lin_total); out[7] = float(sfb_total); out[8] = float(E * max_n * KB);
  out[9] = float(fused_mis);
  return out;
}

// vLLM-EXACT silu_and_mul replica (reference half of the silu_mul_quant gate): one thread per
// output element, a2 = bf16(bf16(silu_f(gate)) * up) — silu in fp32 rounded to bf16, then a bf16
// multiply — the same rounding chain as vLLM's scalar AND packed (__hmul2) silu_and_mul paths,
// and the same chain silu_mul_quant_kernel replicates. Same expf (libdevice) on both sides.
static __global__ void ref_silu_mul_kernel(cutlass::bfloat16_t const* __restrict__ x2,
                                           cutlass::bfloat16_t* __restrict__ a2,
                                           long long total /* E*max_n*N */, int N) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  long long row = i / N;
  int j = int(i % N);
  float gf = float(x2[row * (2LL * N) + j]);
  cutlass::bfloat16_t s_bf(gf / (1.0f + expf(-gf)));
  a2[i] = cutlass::bfloat16_t(float(s_bf) * float(x2[row * (2LL * N) + N + j]));
}

// BYTE-LEVEL gate for the fused SwiGLU+quant op: run BOTH
//   reference: ref_silu_mul_kernel (vLLM-exact silu_and_mul) -> paired_nvfp4_quant_act
//   fused    : paired_nvfp4_silu_mul_quant_act
// on the same random bf16 x2 [E, max_n, 2N] (per-expert gscale, staggered counts) and raw-byte
// compare both outputs. The fused kernel replicates the reference rounding chain exactly, so the
// only acceptable result is 0 mismatches in both buffers.
// Returns [b_mis, sfb_mis, b_first, sfb_first, b_total, sfb_total].
torch::Tensor paired_nvfp4_selftest_silu_mul_quant(int64_t M_, int64_t maxn_, int64_t N_,
                                                   int64_t E_, int64_t seed_) {
  TORCH_CHECK(current_sm() == kArchSm, "selftest_silu_mul_quant requires SM", kArchSm);
  const int M = int(M_), max_n = int(maxn_), N = int(N_), E = int(E_);
  TORCH_CHECK(N % SFVecSize == 0, "N must be a multiple of SFVecSize=", SFVecSize);

  const c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, 0));
  auto dev = torch::Device(torch::kCUDA, 0);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(0).stream();
  std::mt19937 gen{static_cast<uint32_t>(seed_)};
  std::normal_distribution<float> nd(0.f, 1.f);

  // Staggered counts (exercise padding) + per-expert gscale (exercise the gpe path).
  std::vector<int32_t> counts(E);
  std::vector<float>   gs(E);
  for (int e = 0; e < E; ++e) {
    int step = std::max(1, max_n / 4);
    counts[e] = std::max(1, max_n - (e % 4) * step);
    gs[e] = 0.5f + 0.25f * float(e % 8);
  }

  std::vector<cutlass::bfloat16_t> x2h(size_t(E) * max_n * 2 * N);
  for (auto& v : x2h) v = cutlass::bfloat16_t(nd(gen));
  torch::Tensor x2 = torch::from_blob(x2h.data(), {E, max_n, 2 * N},
                                      torch::TensorOptions().dtype(torch::kBFloat16)).clone().to(dev);
  torch::Tensor gscale   = torch::from_blob(gs.data(), {E}, torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev);
  torch::Tensor counts_t = torch::from_blob(counts.data(), {E}, torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev);

  // Reference chain: device silu_and_mul replica -> the (already validated) quant_act op.
  torch::Tensor a2 = torch::empty({E, max_n, N}, torch::TensorOptions().dtype(torch::kBFloat16).device(dev));
  long long total = (long long)E * max_n * N;
  int threads = 256;
  long long blocks = (total + threads - 1) / threads;
  ref_silu_mul_kernel<<<dim3((unsigned)blocks), dim3(threads), 0, stream>>>(
      reinterpret_cast<cutlass::bfloat16_t const*>(x2.data_ptr()),
      reinterpret_cast<cutlass::bfloat16_t*>(a2.data_ptr()), total, N);
  auto qa_ref = paired_nvfp4_quant_act(a2, gscale, counts_t, M);

  // Fused op under test.
  auto qa_fused = paired_nvfp4_silu_mul_quant_act(x2, gscale, counts_t, M);

  torch::Tensor b_ref    = std::get<0>(qa_ref).to(torch::kCPU).contiguous();
  torch::Tensor sfb_ref  = std::get<1>(qa_ref).to(torch::kCPU).contiguous();
  torch::Tensor b_fus    = std::get<0>(qa_fused).to(torch::kCPU).contiguous();
  torch::Tensor sfb_fus  = std::get<1>(qa_fused).to(torch::kCPU).contiguous();
  TORCH_CHECK(b_ref.numel() == b_fus.numel() && sfb_ref.numel() == sfb_fus.numel(),
              "fused/reference output size mismatch");

  // b_act is compared on valid rows only: both sides leave padded rows unwritten (the reference
  // chain ends in quant_act). sfb is compared over the whole buffer: both ops pre-fill the same
  // benign byte and write the same valid-row slots.
  const uint8_t* br = b_ref.data_ptr<uint8_t>();
  const uint8_t* bf = b_fus.data_ptr<uint8_t>();
  const long b_row_bytes = long(N) / 2;
  long b_total = 0, b_mis = 0, b_first = -1;
  for (int e = 0; e < E; ++e)
    for (int t = 0; t < counts[e]; ++t) {
      long off = (long(e) * max_n + t) * b_row_bytes;
      b_total += b_row_bytes;
      for (long b = 0; b < b_row_bytes; ++b)
        if (br[off + b] != bf[off + b]) { ++b_mis; if (b_first < 0) b_first = off + b; }
    }
  const uint8_t* sr = sfb_ref.data_ptr<uint8_t>();
  const uint8_t* sf = sfb_fus.data_ptr<uint8_t>();
  long s_total = long(sfb_ref.numel()), s_mis = 0, s_first = -1;
  for (long i = 0; i < s_total; ++i) if (sr[i] != sf[i]) { ++s_mis; if (s_first < 0) s_first = i; }

  auto out = torch::empty({6}, torch::TensorOptions().dtype(torch::kFloat32));
  out[0] = float(b_mis);   out[1] = float(s_mis);
  out[2] = float(b_first); out[3] = float(s_first);
  out[4] = float(b_total); out[5] = float(s_total);
  return out;
}

// BYTE-LEVEL gate for the fused dispatch-scatter+quant op: emulate the CaptureSafe dispatch plan
// on the host (within-expert rank in flat order — the stable-argsort semantics — including
// NON-LOCAL experts (ids >= E, EP shard misses) and capacity-OVERFLOW routings, both -> trash),
// then run BOTH
//   reference: host-scatter the bf16 rows into [E, cap, K] -> paired_nvfp4_quant_act
//              (with a REAL features=M_ref — byte equality vs the fused op's placeholder-M SFB
//              layout simultaneously proves tile_atom_to_shape_SFB's M-independence)
//   fused    : paired_nvfp4_scatter_quant_act on the token-order a1 + the plan
// and compare VALID rows only (rows < counts[e]): b_act bytes flat, sfb slots through the cute
// layout functor. Padded rows deviate by design (fused: garbage b_act / benign-fill SF — never
// read by the GEMM); instead assert every fused SFB byte is NONZERO (the 0/0 guard).
// Runs twice: without router weights, and with apply_router_weight_on_input (host replicate of
// torch's bf16-cast-then-bf16-multiply chain).
// Returns [b_mis, sfb_mis, b_mis_w, sfb_mis_w, sfb_zero, valid_rows].
torch::Tensor paired_nvfp4_selftest_scatter_quant(int64_t T_, int64_t topk_, int64_t K_,
                                                  int64_t E_, int64_t cap_, int64_t seed_) {
  TORCH_CHECK(current_sm() == kArchSm, "selftest_scatter_quant requires SM", kArchSm);
  const int T = int(T_), topk = int(topk_), K = int(K_), E = int(E_), cap = int(cap_);
  const int KB = K / SFVecSize;
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of SFVecSize=", SFVecSize);

  const c10::cuda::CUDAGuard guard(torch::Device(torch::kCUDA, 0));
  auto dev = torch::Device(torch::kCUDA, 0);
  std::mt19937 gen{static_cast<uint32_t>(seed_)};
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_int_distribution<int> ue(0, E + 1);   // E, E+1 = non-local (EP) -> trash
  std::uniform_real_distribution<float> uw(0.1f, 2.f);

  // Token-order activations + per-expert gscale.
  std::vector<cutlass::bfloat16_t> a1h(size_t(T) * K);
  for (auto& v : a1h) v = cutlass::bfloat16_t(nd(gen));
  std::vector<float> gs(E);
  for (int e = 0; e < E; ++e) gs[e] = 0.5f + 0.25f * float(e % 8);
  torch::Tensor a1 = torch::from_blob(a1h.data(), {T, K},
                                      torch::TensorOptions().dtype(torch::kBFloat16)).clone().to(dev);
  torch::Tensor gscale = torch::from_blob(gs.data(), {E},
                                          torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev);

  // Host plan: flat order i = t*topk + j; rank = running per-expert counter (== stable argsort).
  const long long N = (long long)T * topk;
  const long long trash = (long long)E * cap;
  std::vector<int64_t> flat_tok(N), dest(N);
  std::vector<float> wts(N);
  std::vector<int> counter(E, 0);
  for (long long i = 0; i < N; ++i) {
    int t = int(i / topk);
    int e = ue(gen);
    flat_tok[i] = t;
    wts[i] = uw(gen);
    if (e >= E) { dest[i] = trash; continue; }            // non-local expert
    int r = counter[e]++;
    dest[i] = (r < cap) ? (long long)e * cap + r : trash; // capacity overflow -> trash
  }
  std::vector<int32_t> counts(E);
  long long valid_rows = 0;
  for (int e = 0; e < E; ++e) { counts[e] = std::min(counter[e], cap); valid_rows += counts[e]; }

  torch::Tensor flat_tok_t = torch::from_blob(flat_tok.data(), {int64_t(N)},
                                              torch::TensorOptions().dtype(torch::kInt64)).clone().to(dev);
  torch::Tensor dest_t = torch::from_blob(dest.data(), {int64_t(N)},
                                          torch::TensorOptions().dtype(torch::kInt64)).clone().to(dev);
  torch::Tensor wts_t = torch::from_blob(wts.data(), {int64_t(N)},
                                         torch::TensorOptions().dtype(torch::kFloat32)).clone().to(dev);
  torch::Tensor counts_t = torch::from_blob(counts.data(), {E},
                                            torch::TensorOptions().dtype(torch::kInt32)).clone().to(dev);

  const int M_ref = 2048;   // a REAL features for the reference SFB layout (fused uses placeholder)
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(M_ref, cap, K, E));

  // One pass = (host-scatter -> quant_act reference) vs fused, compare valid rows.
  auto run_pass = [&](bool weighted, long& b_mis, long& s_mis) {
    // Reference scatter on host (bf16; weighted = torch's bf16(w) cast then bf16 multiply).
    std::vector<cutlass::bfloat16_t> bref(size_t(E) * cap * K, cutlass::bfloat16_t(0.f));
    for (long long i = 0; i < N; ++i) {
      if (dest[i] == trash) continue;
      cutlass::bfloat16_t const* srow = &a1h[size_t(flat_tok[i]) * K];
      cutlass::bfloat16_t* drow = &bref[size_t(dest[i]) * K];
      if (weighted) {
        cutlass::bfloat16_t w_bf(wts[i]);
        for (int k = 0; k < K; ++k) drow[k] = cutlass::bfloat16_t(float(srow[k]) * float(w_bf));
      } else {
        std::memcpy(drow, srow, size_t(K) * sizeof(cutlass::bfloat16_t));
      }
    }
    torch::Tensor bref_t = torch::from_blob(bref.data(), {E, cap, K},
                                            torch::TensorOptions().dtype(torch::kBFloat16)).clone().to(dev);
    auto qa_ref = paired_nvfp4_quant_act(bref_t, gscale, counts_t, M_ref);
    auto qa_fus = paired_nvfp4_scatter_quant_act(
        a1, flat_tok_t, dest_t,
        weighted ? std::optional<torch::Tensor>(wts_t) : std::nullopt,
        gscale, counts_t, cap);

    torch::Tensor b_r = std::get<0>(qa_ref).to(torch::kCPU).contiguous();
    torch::Tensor s_r = std::get<1>(qa_ref).to(torch::kCPU).contiguous();
    torch::Tensor b_f = std::get<0>(qa_fus).to(torch::kCPU).contiguous();
    torch::Tensor s_f = std::get<1>(qa_fus).to(torch::kCPU).contiguous();
    TORCH_CHECK(b_r.sizes() == b_f.sizes() && s_r.numel() == s_f.numel(),
                "scatter_quant output shape mismatch (b ", b_f.sizes(), " sfb ", s_f.numel(),
                " vs ref ", b_r.sizes(), " / ", s_r.numel(), ")");

    // b_act: valid rows only, flat row-bytes.
    const uint8_t* pr = b_r.data_ptr<uint8_t>();
    const uint8_t* pf = b_f.data_ptr<uint8_t>();
    for (int e = 0; e < E; ++e)
      for (int r = 0; r < counts[e]; ++r) {
        size_t off = (size_t(e) * cap + r) * (K / 2);
        for (int b = 0; b < K / 2; ++b)
          if (pr[off + b] != pf[off + b]) ++b_mis;
      }
    // sfb: valid-row slots through the layout functor (one slot per K-block).
    auto sr = make_tensor(cute::recast_ptr<ElementSF>(
                  reinterpret_cast<ElementSF*>(s_r.data_ptr<uint8_t>())), layout_SFB);
    auto sf = make_tensor(cute::recast_ptr<ElementSF>(
                  reinterpret_cast<ElementSF*>(s_f.data_ptr<uint8_t>())), layout_SFB);
    for (int e = 0; e < E; ++e)
      for (int r = 0; r < counts[e]; ++r)
        for (int kb = 0; kb < KB; ++kb) {
          uint8_t vr, vf;
          ElementSF er = sr(r, kb * SFVecSize, e), ef = sf(r, kb * SFVecSize, e);
          std::memcpy(&vr, &er, 1); std::memcpy(&vf, &ef, 1);
          if (vr != vf) ++s_mis;
        }
    return s_f;   // for the nonzero scan below (last pass's fused sfb)
  };

  long b_mis = 0, s_mis = 0, b_mis_w = 0, s_mis_w = 0;
  run_pass(false, b_mis, s_mis);
  torch::Tensor s_f_last = run_pass(true, b_mis_w, s_mis_w);

  // 0/0 guard: EVERY fused SFB byte (valid, padded, and atom slop) must be nonzero.
  long sfb_zero = 0;
  const uint8_t* sz = s_f_last.data_ptr<uint8_t>();
  for (long i = 0; i < long(s_f_last.numel()); ++i) if (sz[i] == 0) ++sfb_zero;

  auto out = torch::empty({6}, torch::TensorOptions().dtype(torch::kFloat32));
  out[0] = float(b_mis);   out[1] = float(s_mis);
  out[2] = float(b_mis_w); out[3] = float(s_mis_w);
  out[4] = float(sfb_zero); out[5] = float(valid_rows);
  return out;
}

#else  // arch not supported by this toolkit

torch::Tensor paired_nvfp4_selftest(int64_t, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "paired_nvfp4 selftest was not built with arch support");
}
torch::Tensor paired_nvfp4_selftest_silu_mul_quant(int64_t, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "paired_nvfp4 selftest_silu_mul_quant was not built with arch support");
}
torch::Tensor paired_nvfp4_selftest_scatter_quant(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "paired_nvfp4 selftest_scatter_quant was not built with arch support");
}
torch::Tensor paired_nvfp4_selftest_quant_act(int64_t, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "paired_nvfp4 selftest_quant_act was not built with arch support");
}
torch::Tensor paired_nvfp4_selftest_quant_act_diff(int64_t, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "paired_nvfp4 selftest_quant_act_diff was not built with arch support");
}
torch::Tensor paired_nvfp4_selftest_sf_debug(int64_t, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "paired_nvfp4 selftest_sf_debug was not built with arch support");
}
torch::Tensor paired_nvfp4_probe_act_layout(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t) {
  TORCH_CHECK(false, "paired_nvfp4 probe_act_layout was not built with arch support");
}

#endif

} // namespace paired_nvfp4

// selftest takes no Tensor arguments, so it must be registered as a backend-agnostic catch-all
// (CompositeImplicitAutograd via the def-with-function overload) rather than per-backend CPU/CUDA
// kernels — otherwise the dispatcher has no key to select and raises NotImplementedError.
TORCH_LIBRARY_FRAGMENT(_paired_nvfp4_test, m) {
  m.def("selftest(int M, int max_n, int K, int E, int seed) -> Tensor",
        &paired_nvfp4::paired_nvfp4_selftest);
  m.def("probe_act_layout(Tensor b_act, Tensor sfb, int features, int max_n, int K, int E) -> Tensor",
        &paired_nvfp4::paired_nvfp4_probe_act_layout);
  m.def("selftest_quant_act(int M, int max_n, int K, int E, int seed) -> Tensor",
        &paired_nvfp4::paired_nvfp4_selftest_quant_act);
  m.def("selftest_quant_act_diff(int M, int max_n, int K, int E, int seed) -> Tensor",
        &paired_nvfp4::paired_nvfp4_selftest_quant_act_diff);
  m.def("selftest_sf_debug(int M, int max_n, int K, int E, int seed) -> Tensor",
        &paired_nvfp4::paired_nvfp4_selftest_sf_debug);
  m.def("selftest_silu_mul_quant(int M, int max_n, int N, int E, int seed) -> Tensor",
        &paired_nvfp4::paired_nvfp4_selftest_silu_mul_quant);
  m.def("selftest_scatter_quant(int T, int topk, int K, int E, int cap, int seed) -> Tensor",
        &paired_nvfp4::paired_nvfp4_selftest_scatter_quant);
}
