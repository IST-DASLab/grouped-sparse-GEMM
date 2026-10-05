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
    \brief torch.ops.paired_nvfp4.* : argument validation, allocation and registration.

    Load time
      compress(w_packed, w_blockscale, k) -> (a_comp, e_meta, sfa)
    Every forward
      quant_act(x, gscale, expert_num_tokens, features) -> (b_act, sfb)
      silu_mul_quant_act(x2, gscale, expert_num_tokens, features) -> (b_act, sfb)
      scatter_quant_act(a1, flat_tok, dest_global, topk_weights, gscale, expert_num_tokens, cap)
          -> (b_act, sfb)
      dispatch_plan(topk_ids, first_expert, num_local_experts, cap)
          -> (expert_num_tokens, route_dest, src_tok, src_route)
      scatter_quant_act_planned(a1, src_tok, src_route, topk_weights, gscale, expert_num_tokens,
                                cap) -> (b_act, sfb)
      quant_rows(x, gscale) -> (q, q_sf)
      scatter_fp4_planned(q, q_sf, src_tok, expert_num_tokens, cap) -> (b_act, sfb)
      moe_finalize(out, fused, route_dest, topk_weights)
          out[t] = sum_k w[t, k] * fused[route_dest[t, k]] over kept routings
      group_mm(out, a_comp, e_meta, sfa, b_act, sfb, alphas, expert_num_tokens, features, k,
               config_id, cluster_m, cluster_n)
          out[e, :n_e] = alphas[e] * (a_comp[e] @ b_act[e, :n_e]),  n_e = expert_num_tokens[e]
    Introspection
      group_mm_configs() -> str[]   tile-variant names, indexed by config_id

    Every forward op only enqueues work on the current CUDA stream and never synchronizes or reads
    device memory on the host, so all of them are CUDA-graph-capture safe. See README.md for the
    tensor contracts.
*/

#include <array>
#include <mutex>
#include <optional>
#include <vector>

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "moe_routing.cuh"
#include "quant.cuh"
#include "ops.h"

#define PAIRED_NVFP4_STR_(x) #x
#define PAIRED_NVFP4_STR(x) PAIRED_NVFP4_STR_(x)

namespace paired_nvfp4 {

namespace {

constexpr int kMaxDevices = 64;

struct DeviceInfo { int sm_version; int sm_count; };

// Queried once per device, so the per-forward path never calls cudaGetDeviceProperties.
DeviceInfo const& device_info(int device) {
  static std::array<DeviceInfo, kMaxDevices> cache{};
  static std::array<std::once_flag, kMaxDevices> once;
  TORCH_CHECK(device >= 0 && device < kMaxDevices, "unexpected CUDA device index ", device);
  std::call_once(once[device], [device] {
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cache[device].sm_version = major * 10 + minor;
    cache[device].sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
  });
  return cache[device];
}

void check_arch(torch::Tensor const& t) {
  int v = device_info(t.device().index()).sm_version;
  TORCH_CHECK(v == kArchSm,
              "paired_nvfp4 ops in this module are built for SM", kArchSm,
              " but the tensor is on a compute capability ", v / 10, ".", v % 10, " device");
}

void check_gscale_counts(torch::Tensor const& gscale, torch::Tensor const& counts, int E) {
  TORCH_CHECK(gscale.scalar_type() == torch::kFloat32, "gscale must be float32");
  TORCH_CHECK(int(gscale.numel()) == E || gscale.numel() == 1,
              "gscale must be [E] (per expert) or [1] (shared)");
  TORCH_CHECK(counts.scalar_type() == torch::kInt32 && int(counts.numel()) == E,
              "expert_num_tokens must be [E] int32");
  TORCH_CHECK(gscale.is_contiguous() && counts.is_contiguous(),
              "gscale and expert_num_tokens must be contiguous");
}

torch::Tensor benign_sfb(int64_t numel, torch::Device device) {
  return torch::full({numel}, int64_t(benign_sf_byte()),
                     torch::TensorOptions().dtype(torch::kUInt8).device(device));
}

// The gated activation named by an op's (activation, beta, linear_beta) arguments.
GatedAct make_gated_act(std::string const& activation, double beta, double linear_beta) {
  GatedAct a;
  if (activation == "silu") {
    a.kind = GatedAct::kSilu;
  } else if (activation == "situ") {
    TORCH_CHECK(beta > 0, "situ: beta must be positive, got ", beta);
    a.kind = GatedAct::kSitu;
    a.beta = float(beta);
    a.linear_beta = float(linear_beta);
  } else {
    TORCH_CHECK(false, "activation must be 'silu' or 'situ', got '", activation, "'");
  }
  return a;
}

// SFB output of a producer that writes the benign fill itself (fill_benign_sfb): no fill kernel.
torch::Tensor kernel_filled_sfb(int64_t numel, torch::Device device) {
  return torch::empty({numel}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
}

} // namespace

/////////////////////////////////////////////////////////////////////////////////////////////////
// compress: dense packed NVFP4 weight -> sparse operand A + metadata E + swizzled SFA.
//   W_packed      [E, out_features, K/2]           uint8   e2m1, paired-4:8 zeros in place
//   W_blockscale  [E, out_features, K/SFVecSize]   1-byte  UE4M3, natural order, one per 32 K
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
paired_nvfp4_compress(torch::Tensor W_packed, torch::Tensor W_blockscale, int64_t k) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(W_packed.is_cuda() && W_blockscale.is_cuda(), "inputs must be CUDA tensors");
  check_arch(W_packed);
  TORCH_CHECK(W_packed.dim() == 3, "w_packed must be [E, out_features, K/2]");
  TORCH_CHECK(W_blockscale.dim() == 3, "w_blockscale must be [E, out_features, ceil(K/SFVecSize)]");
  TORCH_CHECK(W_packed.is_contiguous() && W_blockscale.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(W_packed.scalar_type() == torch::kUInt8, "w_packed must be uint8 (2 e2m1 per byte)");

  const int E = int(W_packed.size(0));
  const int M = int(W_packed.size(1));
  const int K = int(k);
  TORCH_CHECK(K % 2 == 0 && int(W_packed.size(2)) == K / 2, "w_packed last dim must equal K/2");
  TORCH_CHECK(int(W_blockscale.size(0)) == E && int(W_blockscale.size(1)) == M,
              "w_blockscale must match w_packed in [E, out_features]");
  TORCH_CHECK(int(W_blockscale.size(2)) == kblocks(K),
              "w_blockscale last dim must be ceil(K/", SFVecSize, ") = ", kblocks(K), ", got ",
              int(W_blockscale.size(2)), ". The sparse NVFP4 MMA uses one scale per ", SFVecSize,
              " dense K elements; checkpoints quantized with 16-element groups are incompatible");
  TORCH_CHECK(W_blockscale.element_size() == 1, "w_blockscale must be a 1-byte (e4m3) dtype");

  const c10::cuda::CUDAGuard device_guard(W_packed.device());
  const int device = W_packed.device().index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();

  PhysicalDims pd = physical_dims(M, K, E);
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(W_packed.device());
  // KAlignedAC counts fp4 elements, two per byte.
  TORCH_CHECK(pd.KAlignedAC % 2 == 0, "KAlignedAC must be even for fp4 packing, got ", pd.KAlignedAC);
  torch::Tensor A_comp = torch::empty({E, pd.MAlignedAC, pd.KAlignedAC / 2}, u8);
  torch::Tensor E_meta = torch::empty({E, pd.MAlignedE, pd.KAlignedE}, u8);
  torch::Tensor SFA    = torch::zeros({sfa_numel(M, kPlaceholderExtent, K, E)}, u8);

  SparseProblemShape sp = make_sparse_shape(M, kPlaceholderExtent, K, E);
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, E});

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count  = device_info(device).sm_count;

  typename Compressor::Arguments comp_args{
      sp,
      {reinterpret_cast<ElementA*>(W_packed.data_ptr<uint8_t>()), stride_A,
       reinterpret_cast<ElementA*>(A_comp.data_ptr<uint8_t>()),
       reinterpret_cast<ElementE*>(E_meta.data_ptr<uint8_t>())},
      {hw_info}};

  Compressor compressor;
  size_t comp_ws_bytes = Compressor::get_workspace_size(comp_args);
  torch::Tensor comp_ws = torch::empty({int64_t(comp_ws_bytes)}, u8);
  TORCH_CHECK(compressor.can_implement(comp_args) == cutlass::Status::kSuccess,
              "paired_nvfp4.compress: compressor can_implement failed");
  TORCH_CHECK(compressor.initialize(comp_args, comp_ws.data_ptr(), stream) == cutlass::Status::kSuccess,
              "paired_nvfp4.compress: compressor initialize failed");
  TORCH_CHECK(compressor.run(stream) == cutlass::Status::kSuccess,
              "paired_nvfp4.compress: compressor run failed");

  scatter_sf(reinterpret_cast<ElementSF const*>(W_blockscale.data_ptr<uint8_t>()),
             reinterpret_cast<ElementSF*>(SFA.data_ptr<uint8_t>()),
             Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(sp), /*outer=*/M, K, /*groups=*/E,
             stream);

  return {A_comp, E_meta, SFA};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// quant_act: batched bf16 x[E, max_n, K] -> (b_act [E, max_n, K/2], sfb). `features` is the
// consuming GEMM's M; it is accepted for API symmetry (the SFB layout does not depend on M).
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_quant_act(torch::Tensor x, torch::Tensor gscale, torch::Tensor expert_num_tokens,
                       int64_t features) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(x.is_cuda() && gscale.is_cuda() && expert_num_tokens.is_cuda(), "inputs must be CUDA");
  check_arch(x);
  TORCH_CHECK(x.dim() == 3, "x must be [E, max_n, K]");
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && x.is_contiguous(), "x must be contiguous bf16");
  const int E = int(x.size(0)), max_n = int(x.size(1)), K = int(x.size(2));
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of ", SFVecSize, ", got ", K);
  check_gscale_counts(gscale, expert_num_tokens, E);
  const int gpe = int(gscale.numel()) == E ? 1 : 0;

  const c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();

  const int M = int(features);
  torch::Tensor b_act = torch::empty({E, max_n, K / 2},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(x.device()));
  torch::Tensor sfb = kernel_filled_sfb(sfb_numel(M, max_n, K, E), x.device());

  quant_act(reinterpret_cast<cutlass::bfloat16_t const*>(x.data_ptr()),
            reinterpret_cast<ElementB*>(b_act.data_ptr<uint8_t>()),
            reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>()),
            Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(M, max_n, K, E)),
            cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E}),
            gscale.data_ptr<float>(), expert_num_tokens.data_ptr<int32_t>(),
            max_n, K, E, gpe, device_info(x.device().index()).sm_count, stream);
  return {b_act, sfb};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// silu_mul_quant_act: GEMM1 output x2[E, max_n, 2N] ([gate | up]) -> (b_act [E, max_n, N/2], sfb)
// for GEMM2. Byte-identical to silu_and_mul followed by quant_act (see quant.cuh).
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_silu_mul_quant_act(torch::Tensor x2, torch::Tensor gscale,
                                torch::Tensor expert_num_tokens, int64_t features, bool interleaved,
                                std::string activation, double beta, double linear_beta) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(x2.is_cuda() && gscale.is_cuda() && expert_num_tokens.is_cuda(), "inputs must be CUDA");
  check_arch(x2);
  TORCH_CHECK(x2.dim() == 3, "x2 must be [E, max_n, 2N] ([gate | up] halves)");
  TORCH_CHECK(x2.scalar_type() == torch::kBFloat16 && x2.is_contiguous(), "x2 must be contiguous bf16");
  const int E = int(x2.size(0)), max_n = int(x2.size(1));
  TORCH_CHECK(x2.size(2) % 2 == 0, "x2 last dim must be even ([gate | up]), got ", x2.size(2));
  const int N = int(x2.size(2)) / 2;
  TORCH_CHECK(N % SFVecSize == 0, "intermediate width N must be a multiple of ", SFVecSize,
              ", got ", N);
  check_gscale_counts(gscale, expert_num_tokens, E);
  const int gpe = int(gscale.numel()) == E ? 1 : 0;

  const c10::cuda::CUDAGuard device_guard(x2.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x2.device().index()).stream();

  const int M = int(features);
  torch::Tensor b_act = torch::empty({E, max_n, N / 2},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(x2.device()));
  torch::Tensor sfb = kernel_filled_sfb(sfb_numel(M, max_n, N, E), x2.device());

  silu_mul_quant(reinterpret_cast<cutlass::bfloat16_t const*>(x2.data_ptr()),
                 reinterpret_cast<ElementB*>(b_act.data_ptr<uint8_t>()),
                 reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>()),
                 Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_sparse_shape(M, max_n, N, E)),
                 cutlass::make_cute_packed_stride(StrideB{}, {max_n, N, E}),
                 gscale.data_ptr<float>(), expert_num_tokens.data_ptr<int32_t>(),
                 max_n, N, E, gpe, device_info(x2.device().index()).sm_count, stream, interleaved ? 1 : 0,
                 make_gated_act(activation, beta, linear_beta));
  return {b_act, sfb};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_quant_act: token-order a1[T, K] + dispatch plan -> (b_act [E, cap, K/2], sfb) for
// GEMM1. expert_num_tokens only supplies E here. On valid rows the output is byte-identical to
// gathering, scattering and then calling quant_act.
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_scatter_quant_act(torch::Tensor a1, torch::Tensor flat_tok,
                               torch::Tensor dest_global,
                               std::optional<torch::Tensor> topk_weights,
                               torch::Tensor gscale, torch::Tensor expert_num_tokens,
                               int64_t cap) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(a1.is_cuda() && flat_tok.is_cuda() && dest_global.is_cuda() && gscale.is_cuda() &&
              expert_num_tokens.is_cuda(), "inputs must be CUDA");
  check_arch(a1);
  TORCH_CHECK(a1.dim() == 2, "a1 must be [T, K] (token-order hidden states)");
  TORCH_CHECK(a1.scalar_type() == torch::kBFloat16 && a1.is_contiguous(), "a1 must be contiguous bf16");
  const int K = int(a1.size(1));
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of ", SFVecSize, ", got ", K);
  TORCH_CHECK(flat_tok.scalar_type() == torch::kInt64 && dest_global.scalar_type() == torch::kInt64,
              "flat_tok and dest_global must be int64");
  TORCH_CHECK(flat_tok.dim() == 1 && dest_global.dim() == 1 &&
              flat_tok.numel() == dest_global.numel(), "flat_tok and dest_global must be same-length 1D");
  TORCH_CHECK(flat_tok.is_contiguous() && dest_global.is_contiguous(), "plan tensors must be contiguous");
  TORCH_CHECK(cap > 0, "cap must be positive");
  const long long N = (long long)flat_tok.numel();
  const int E = int(expert_num_tokens.numel());
  check_gscale_counts(gscale, expert_num_tokens, E);
  const int gpe = int(gscale.numel()) == E ? 1 : 0;
  float const* topk_w_ptr = nullptr;
  if (topk_weights.has_value()) {
    torch::Tensor const& w = *topk_weights;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 && w.is_contiguous() &&
                w.numel() == N, "topk_weights must be contiguous float32 [N]");
    topk_w_ptr = w.data_ptr<float>();
  }

  const c10::cuda::CUDAGuard device_guard(a1.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a1.device().index()).stream();

  // One extra trailing row absorbs non-local and over-capacity routings; it is never read.
  torch::Tensor b_ext = torch::empty({int64_t(E) * cap + 1, int64_t(K) / 2},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(a1.device()));
  torch::Tensor b_act = b_ext.narrow(0, 0, int64_t(E) * cap).view({E, cap, K / 2});
  torch::Tensor sfb = benign_sfb(sfb_numel(kPlaceholderExtent, int(cap), K, E), a1.device());

  scatter_quant(reinterpret_cast<cutlass::bfloat16_t const*>(a1.data_ptr()),
                flat_tok.data_ptr<int64_t>(), dest_global.data_ptr<int64_t>(), topk_w_ptr,
                reinterpret_cast<ElementB*>(b_ext.data_ptr<uint8_t>()),
                reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>()),
                Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                    make_sparse_shape(kPlaceholderExtent, int(cap), K, E)),
                gscale.data_ptr<float>(), N, K, int(cap),
                /*trash_row=*/int64_t(E) * cap, gpe, stream);
  return {b_act, sfb};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// dispatch_plan: routing ids -> this rank's per-expert rows (moe_routing.cuh). Arch-independent.
// Output shapes depend only on host ints (T, topk, E, cap), so the op is capture safe.
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
paired_nvfp4_dispatch_plan(torch::Tensor topk_ids, int64_t first_expert,
                           int64_t num_local_experts, int64_t cap) {
  TORCH_CHECK(topk_ids.is_cuda(), "topk_ids must be CUDA");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.is_contiguous(),
              "topk_ids must be contiguous [T, topk]");
  TORCH_CHECK(topk_ids.scalar_type() == torch::kInt32 || topk_ids.scalar_type() == torch::kInt64,
              "topk_ids must be int32 or int64");
  const int E = int(num_local_experts);
  TORCH_CHECK(E >= 1 && E <= 4096, "num_local_experts must be in [1, 4096], got ", E);
  TORCH_CHECK(cap >= 1, "cap must be positive");
  TORCH_CHECK(int64_t(E) * cap < (int64_t(1) << 31), "E * cap must fit int32");
  const long long T = topk_ids.size(0), topk = topk_ids.size(1);
  const long long N = T * topk;
  TORCH_CHECK(N < (int64_t(1) << 31), "T * topk must fit int32");

  const c10::cuda::CUDAGuard device_guard(topk_ids.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(topk_ids.device().index()).stream();
  auto i32 = torch::TensorOptions().dtype(torch::kInt32).device(topk_ids.device());
  torch::Tensor counts = torch::empty({E}, i32);
  torch::Tensor route_dest = torch::empty({T, topk}, i32);
  torch::Tensor src_tok = torch::empty({int64_t(E) * cap}, i32);
  torch::Tensor src_route = torch::empty({int64_t(E) * cap}, i32);
  const long long chunks = (N + kPlanChunk - 1) / kPlanChunk;
  torch::Tensor hist = torch::empty({chunks > 1 ? chunks * E : 1}, i32);
  if (topk_ids.scalar_type() == torch::kInt32) {
    dispatch_plan(topk_ids.data_ptr<int32_t>(), N, int(topk), int(first_expert), E, int(cap),
                  counts.data_ptr<int32_t>(), route_dest.data_ptr<int32_t>(),
                  src_tok.data_ptr<int32_t>(), src_route.data_ptr<int32_t>(),
                  hist.data_ptr<int32_t>(), stream);
  } else {
    dispatch_plan(topk_ids.data_ptr<int64_t>(), N, int(topk), int(first_expert), E, int(cap),
                  counts.data_ptr<int32_t>(), route_dest.data_ptr<int32_t>(),
                  src_tok.data_ptr<int32_t>(), src_route.data_ptr<int32_t>(),
                  hist.data_ptr<int32_t>(), stream);
  }
  return {counts, route_dest, src_tok, src_route};
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// scatter_quant_act_planned: token-order a1[T, K] + dispatch_plan output -> (b_act [E, cap, K/2],
// sfb) for GEMM1. Byte-identical to scatter_quant_act on valid rows; work scales with kept rows.
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_scatter_quant_act_planned(torch::Tensor a1, torch::Tensor src_tok,
                                       torch::Tensor src_route,
                                       std::optional<torch::Tensor> topk_weights,
                                       torch::Tensor gscale, torch::Tensor expert_num_tokens,
                                       int64_t cap) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(a1.is_cuda() && src_tok.is_cuda() && src_route.is_cuda() && gscale.is_cuda() &&
              expert_num_tokens.is_cuda(), "inputs must be CUDA");
  check_arch(a1);
  TORCH_CHECK(a1.dim() == 2, "a1 must be [T, K] (token-order hidden states)");
  TORCH_CHECK(a1.scalar_type() == torch::kBFloat16 && a1.is_contiguous(), "a1 must be contiguous bf16");
  const int K = int(a1.size(1));
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of ", SFVecSize, ", got ", K);
  const int E = int(expert_num_tokens.numel());
  check_gscale_counts(gscale, expert_num_tokens, E);
  TORCH_CHECK(cap > 0, "cap must be positive");
  TORCH_CHECK(src_tok.scalar_type() == torch::kInt32 && src_route.scalar_type() == torch::kInt32 &&
              src_tok.is_contiguous() && src_route.is_contiguous() &&
              src_tok.numel() == int64_t(E) * cap && src_route.numel() == int64_t(E) * cap,
              "src_tok and src_route must be contiguous int32 [E * cap] (dispatch_plan output)");
  const int gpe = int(gscale.numel()) == E ? 1 : 0;
  float const* topk_w_ptr = nullptr;
  if (topk_weights.has_value()) {
    torch::Tensor const& w = *topk_weights;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 && w.is_contiguous(),
                "topk_weights must be contiguous float32 [T * topk]");
    topk_w_ptr = w.data_ptr<float>();
  }

  const c10::cuda::CUDAGuard device_guard(a1.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(a1.device().index()).stream();

  torch::Tensor b_act = torch::empty({E, cap, K / 2},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(a1.device()));
  torch::Tensor sfb = kernel_filled_sfb(sfb_numel(kPlaceholderExtent, int(cap), K, E), a1.device());

  scatter_quant_planned(reinterpret_cast<cutlass::bfloat16_t const*>(a1.data_ptr()),
                        src_tok.data_ptr<int32_t>(), src_route.data_ptr<int32_t>(), topk_w_ptr,
                        reinterpret_cast<ElementB*>(b_act.data_ptr<uint8_t>()),
                        reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>()),
                        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                            make_sparse_shape(kPlaceholderExtent, int(cap), K, E)),
                        cutlass::make_cute_packed_stride(StrideB{}, {int(cap), K, E}),
                        gscale.data_ptr<float>(), expert_num_tokens.data_ptr<int32_t>(),
                        int(cap), K, E, gpe, device_info(a1.device().index()).sm_count, stream);
  return {b_act, sfb};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// quant_rows / scatter_fp4_planned: quantize token rows once (before an all2all), then place the
// received FP4 rows into GEMM1's batched operands. With one global scale, the pair is
// byte-identical to scatter_quant_act_planned on the bf16 rows.
/////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_quant_rows(torch::Tensor x, torch::Tensor gscale) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(x.is_cuda() && gscale.is_cuda(), "inputs must be CUDA");
  check_arch(x);
  TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16 && x.is_contiguous(),
              "x must be contiguous bf16 [T, K]");
  const long long T = x.size(0);
  const int K = int(x.size(1));
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of ", SFVecSize, ", got ", K);
  TORCH_CHECK(gscale.scalar_type() == torch::kFloat32 && gscale.numel() >= 1,
              "gscale must be float32 with the shared global scale first");
  const c10::cuda::CUDAGuard device_guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(x.device().index()).stream();
  auto u8 = torch::TensorOptions().dtype(torch::kUInt8).device(x.device());
  torch::Tensor q = torch::empty({T, K / 2}, u8);
  torch::Tensor sf = torch::empty({T, K / SFVecSize}, u8);
  quant_rows(reinterpret_cast<cutlass::bfloat16_t const*>(x.data_ptr()), q.data_ptr<uint8_t>(),
             sf.data_ptr<uint8_t>(), gscale.data_ptr<float>(), T, K,
             device_info(x.device().index()).sm_count, stream);
  return {q, sf};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_scatter_fp4_planned(torch::Tensor q, torch::Tensor q_sf, torch::Tensor src_tok,
                                 torch::Tensor expert_num_tokens, int64_t cap) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(q.is_cuda() && q_sf.is_cuda() && src_tok.is_cuda() && expert_num_tokens.is_cuda(),
              "inputs must be CUDA");
  check_arch(q);
  TORCH_CHECK(q.dim() == 2 && q.scalar_type() == torch::kUInt8 && q.is_contiguous(),
              "q must be contiguous uint8 [T, K/2] (quant_rows output)");
  const int K = int(q.size(1)) * 2;
  TORCH_CHECK(K % SFVecSize == 0, "K must be a multiple of ", SFVecSize, ", got ", K);
  TORCH_CHECK(q_sf.dim() == 2 && q_sf.size(0) == q.size(0) && q_sf.size(1) == K / SFVecSize &&
              q_sf.element_size() == 1 && q_sf.is_contiguous(),
              "q_sf must be contiguous 1-byte [T, K/", SFVecSize, "] (quant_rows output)");
  const int E = int(expert_num_tokens.numel());
  TORCH_CHECK(expert_num_tokens.scalar_type() == torch::kInt32 && expert_num_tokens.is_contiguous(),
              "expert_num_tokens must be contiguous int32 [E]");
  TORCH_CHECK(cap > 0, "cap must be positive");
  TORCH_CHECK(src_tok.scalar_type() == torch::kInt32 && src_tok.is_contiguous() &&
              src_tok.numel() == int64_t(E) * cap,
              "src_tok must be contiguous int32 [E * cap] (dispatch_plan output)");
  const c10::cuda::CUDAGuard device_guard(q.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(q.device().index()).stream();
  torch::Tensor b_act = torch::empty({E, cap, K / 2},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
  torch::Tensor sfb = kernel_filled_sfb(sfb_numel(kPlaceholderExtent, int(cap), K, E), q.device());
  scatter_fp4_planned(q.data_ptr<uint8_t>(), reinterpret_cast<uint8_t const*>(q_sf.data_ptr()),
                      src_tok.data_ptr<int32_t>(),
                      reinterpret_cast<ElementB*>(b_act.data_ptr<uint8_t>()),
                      reinterpret_cast<ElementSF*>(sfb.data_ptr<uint8_t>()),
                      Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
                          make_sparse_shape(kPlaceholderExtent, int(cap), K, E)),
                      cutlass::make_cute_packed_stride(StrideB{}, {int(cap), K, E}),
                      expert_num_tokens.data_ptr<int32_t>(), int(cap), K, E,
                      device_info(q.device().index()).sm_count, stream);
  return {b_act, sfb};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// moe_finalize: combine the batched expert output back to token order (moe_routing.cuh).
// Arch-independent. topk_weights = None when the router weight was applied on the input.
/////////////////////////////////////////////////////////////////////////////////////////////////

void paired_nvfp4_moe_finalize(torch::Tensor& out, torch::Tensor fused, torch::Tensor route_dest,
                               std::optional<torch::Tensor> topk_weights) {
  TORCH_CHECK(out.is_cuda() && fused.is_cuda() && route_dest.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(out.dim() == 2 && out.scalar_type() == torch::kBFloat16 && out.is_contiguous(),
              "out must be contiguous bf16 [T, K]");
  TORCH_CHECK(fused.scalar_type() == torch::kBFloat16 && fused.is_contiguous() && fused.dim() >= 2,
              "fused must be contiguous bf16 [E, cap, K] (or [E * cap, K])");
  const long long T = out.size(0);
  const int K = int(out.size(1));
  TORCH_CHECK(fused.size(-1) == K, "fused and out must share K");
  TORCH_CHECK(K % kFinalizeVec == 0, "K must be a multiple of ", kFinalizeVec);
  TORCH_CHECK(route_dest.scalar_type() == torch::kInt32 && route_dest.is_contiguous() &&
              route_dest.dim() == 2 && route_dest.size(0) == T,
              "route_dest must be contiguous int32 [T, topk] (dispatch_plan output)");
  const int topk = int(route_dest.size(1));
  float const* w_ptr = nullptr;
  if (topk_weights.has_value()) {
    torch::Tensor const& w = *topk_weights;
    TORCH_CHECK(w.is_cuda() && w.scalar_type() == torch::kFloat32 && w.is_contiguous() &&
                w.numel() == T * topk, "topk_weights must be contiguous float32 [T, topk]");
    w_ptr = w.data_ptr<float>();
  }
  const c10::cuda::CUDAGuard device_guard(out.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(out.device().index()).stream();
  finalize(reinterpret_cast<__nv_bfloat16 const*>(fused.data_ptr()),
           route_dest.data_ptr<int32_t>(), w_ptr,
           reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), T, topk, K, stream);
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// group_mm_swiglu_quant: GEMM1 with SwiGLU and GEMM2's input quantization fused into the epilogue.
// a_comp / e_meta / sfa are compress() output of a W13 whose rows are interleaved in 64-row
// blocks (32 gate rows, then the 32 matching up rows; interleave_gate_up_rows in Python), so
// features = 2N. Returns GEMM2's operands (b_act [E, max_n, N/2], sfb), byte-identical to
// group_mm on the plain W13 followed by silu_mul_quant_act.
/////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> paired_nvfp4_group_mm_swiglu_configs() {
  std::vector<std::string> names;
  for (int i = 0; i < kNumSwigluMmKinds; ++i) names.emplace_back(kSwigluMmKinds[i].name);
  return names;
}

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_group_mm_swiglu_quant(torch::Tensor A_comp, torch::Tensor E_meta, torch::Tensor SFA,
                                   torch::Tensor B_act, torch::Tensor SFB, torch::Tensor alphas,
                                   torch::Tensor expert_num_tokens, torch::Tensor gscale,
                                   int64_t features, int64_t k, int64_t config_id,
                                   int64_t cluster_m, int64_t cluster_n, std::string activation,
                                   double beta, double linear_beta) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(kNumSwigluMmKinds > 0, "group_mm_swiglu_quant is not available on SM", kArchSm);
  TORCH_CHECK(A_comp.is_cuda() && E_meta.is_cuda() && SFA.is_cuda() && B_act.is_cuda() &&
              SFB.is_cuda() && alphas.is_cuda() && expert_num_tokens.is_cuda() && gscale.is_cuda(),
              "all tensors must be CUDA");
  check_arch(B_act);
  const int E = int(A_comp.size(0));
  const int M = int(features);
  const int K = int(k);
  const int max_n = int(B_act.size(1));
  const int N = M / 2;
  TORCH_CHECK(M % 128 == 0, "features (2N) must be a multiple of 128 (whole 64-row gate/up "
              "blocks per CTA), got ", M);
  TORCH_CHECK(B_act.dim() == 3 && int(B_act.size(0)) == E && int(B_act.size(2)) == K / 2,
              "b_act must be [E, max_n, K/2]");
  TORCH_CHECK(A_comp.is_contiguous() && E_meta.is_contiguous() && SFA.is_contiguous() &&
              B_act.is_contiguous() && SFB.is_contiguous() && alphas.is_contiguous(),
              "inputs must be contiguous");
  TORCH_CHECK(alphas.scalar_type() == torch::kFloat32 && int(alphas.numel()) == E,
              "alphas must be [E] float32");
  check_gscale_counts(gscale, expert_num_tokens, E);
  PhysicalDims pd = physical_dims(M, K, E);
  TORCH_CHECK(int(A_comp.size(1)) == pd.MAlignedAC && int(A_comp.size(2)) == pd.KAlignedAC / 2 &&
              int(E_meta.size(1)) == pd.MAlignedE && int(E_meta.size(2)) == pd.KAlignedE,
              "a_comp / e_meta shapes do not match compress() output for features=", M, ", k=", K);
  TORCH_CHECK(SFA.numel() >= sfa_numel(M, max_n, K, E), "sfa is smaller than the SFA layout");
  TORCH_CHECK(SFB.numel() >= sfb_numel(M, max_n, K, E), "sfb is smaller than the SFB layout");
  TORCH_CHECK(config_id >= 0 && config_id < kNumSwigluMmKinds,
              "config_id must be in [0, ", kNumSwigluMmKinds, "), got ", config_id);
  SwigluMmKind const& kind = kSwigluMmKinds[config_id];
  const int cm = int(cluster_m), cn = int(cluster_n);
  check_cluster(GroupMmKind{kind.name, kind.two_sm, nullptr}, cm, cn);

  const c10::cuda::CUDAGuard device_guard(B_act.device());
  torch::Tensor b_out = torch::empty({E, max_n, N / 2},
                                     torch::TensorOptions().dtype(torch::kUInt8).device(B_act.device()));
  torch::Tensor sfb_out = kernel_filled_sfb(sfb_numel(kPlaceholderExtent, max_n, N, E), B_act.device());

  SwigluFp4Args fused;
  fused.b_act = reinterpret_cast<ElementB*>(b_out.data_ptr<uint8_t>());
  fused.sfb = reinterpret_cast<ElementSF*>(sfb_out.data_ptr<uint8_t>());
  fused.layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
      make_sparse_shape(kPlaceholderExtent, max_n, N, E));
  fused.stride_B = cutlass::make_cute_packed_stride(StrideB{}, {max_n, N, E});
  fused.alphas = alphas.data_ptr<float>();
  fused.gscale = gscale.data_ptr<float>();
  fused.gscale_per_expert = int(gscale.numel()) == E ? 1 : 0;
  fused.expert_num_tokens = expert_num_tokens.data_ptr<int32_t>();
  fused.sfb_rows = max_n;
  fused.sfb_kblocks = N / SFVecSize;
  fused.experts = E;
  fused.act = make_gated_act(activation, beta, linear_beta);

  GroupMmParams params{b_out, A_comp, E_meta, SFA, B_act, SFB, alphas, expert_num_tokens,
                       M, K, cm, cn, device_info(B_act.device().index()).sm_count, /*splits=*/1};
  kind.run(params, fused);
  return {b_out, sfb_out};
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// group_mm: dispatch one tactic. config_id picks a compiled tile variant (sm*/tactics.cu),
// cluster_m / cluster_n the dynamic cluster (preferred == fallback).
/////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> paired_nvfp4_group_mm_configs() {
  std::vector<std::string> names;
  names.reserve(kNumGroupMmKinds);
  for (int i = 0; i < kNumGroupMmKinds; ++i) names.emplace_back(kGroupMmKinds[i].name);
  return names;
}

void paired_nvfp4_group_mm(torch::Tensor& D,
                           torch::Tensor A_comp,
                           torch::Tensor E_meta,
                           torch::Tensor SFA,
                           torch::Tensor B_act,
                           torch::Tensor SFB,
                           torch::Tensor alphas,
                           torch::Tensor expert_num_tokens,
                           int64_t features,
                           int64_t k,
                           int64_t config_id,
                           int64_t cluster_m,
                           int64_t cluster_n) {
#if PAIRED_NVFP4_ENABLED
  TORCH_CHECK(D.is_cuda() && A_comp.is_cuda() && E_meta.is_cuda() && SFA.is_cuda() &&
              B_act.is_cuda() && SFB.is_cuda() && alphas.is_cuda() && expert_num_tokens.is_cuda(),
              "all tensors must be CUDA");
  check_arch(D);
  TORCH_CHECK(D.dim() == 3 && B_act.dim() == 3 && A_comp.dim() == 3 && E_meta.dim() == 3,
              "out, b_act, a_comp and e_meta must be 3D");

  const int E     = int(A_comp.size(0));
  const int M     = int(features);
  const int K     = int(k);
  const int max_n = int(B_act.size(1));
  TORCH_CHECK(int(D.size(0)) == E && int(B_act.size(0)) == E && int(E_meta.size(0)) == E,
              "expert dim mismatch");
  TORCH_CHECK(int(D.size(1)) == max_n && int(D.size(2)) == M, "out must be [E, max_n, features]");
  TORCH_CHECK(int(B_act.size(2)) == K / 2, "b_act must be [E, max_n, K/2]");
  TORCH_CHECK(D.scalar_type() == torch::kBFloat16, "out must be bf16");
  TORCH_CHECK(A_comp.scalar_type() == torch::kUInt8 && E_meta.scalar_type() == torch::kUInt8 &&
              B_act.scalar_type() == torch::kUInt8 && SFA.element_size() == 1 &&
              SFB.element_size() == 1, "a_comp, e_meta and b_act must be uint8; sfa, sfb 1-byte");
  TORCH_CHECK(D.is_contiguous() && A_comp.is_contiguous() && E_meta.is_contiguous() &&
              SFA.is_contiguous() && B_act.is_contiguous() && SFB.is_contiguous() &&
              alphas.is_contiguous() && expert_num_tokens.is_contiguous(),
              "all group_mm tensors must be contiguous");
  TORCH_CHECK(alphas.scalar_type() == torch::kFloat32 && int(alphas.numel()) == E,
              "alphas must be [E] float32");
  TORCH_CHECK(expert_num_tokens.scalar_type() == torch::kInt32 && int(expert_num_tokens.numel()) == E,
              "expert_num_tokens must be [E] int32 (real per-expert token counts, <= max_n)");

  // The weight operands must be what compress() produced for this (features, k).
  PhysicalDims pd = physical_dims(M, K, E);
  TORCH_CHECK(int(A_comp.size(1)) == pd.MAlignedAC && int(A_comp.size(2)) == pd.KAlignedAC / 2 &&
              int(E_meta.size(1)) == pd.MAlignedE && int(E_meta.size(2)) == pd.KAlignedE,
              "a_comp / e_meta shapes do not match compress() output for features=", M, ", k=", K,
              " (check the features / k arguments)");
  TORCH_CHECK(SFA.numel() >= sfa_numel(M, max_n, K, E), "sfa is smaller than the SFA layout");
  TORCH_CHECK(SFB.numel() >= sfb_numel(M, max_n, K, E), "sfb is smaller than the SFB layout");

  TORCH_CHECK(config_id >= 0 && config_id < kNumGroupMmKinds,
              "config_id must be in [0, ", kNumGroupMmKinds, "), got ", config_id);
  GroupMmKind const& kind = kGroupMmKinds[config_id];
  const int cm = int(cluster_m), cn = int(cluster_n);
  check_cluster(kind, cm, cn);

  GroupMmParams params{D, A_comp, E_meta, SFA, B_act, SFB, alphas, expert_num_tokens,
                       M, K, cm, cn, device_info(D.device().index()).sm_count, kind.splits};
  kind.run(params);
#else
  TORCH_CHECK(false, "paired_nvfp4 was built without SM", kArchSm, " support");
#endif
}

} // namespace paired_nvfp4

TORCH_LIBRARY(paired_nvfp4, m) {
  m.def("compress(Tensor w_packed, Tensor w_blockscale, int k) -> (Tensor, Tensor, Tensor)");
  m.def("quant_act(Tensor x, Tensor gscale, Tensor expert_num_tokens, int features) "
        "-> (Tensor, Tensor)");
  m.def("silu_mul_quant_act(Tensor x2, Tensor gscale, Tensor expert_num_tokens, int features, "
        "bool interleaved=False, str activation=\"silu\", float beta=1.0, "
        "float linear_beta=-1.0) -> (Tensor, Tensor)");
  m.def("scatter_quant_act(Tensor a1, Tensor flat_tok, Tensor dest_global, "
        "Tensor? topk_weights, Tensor gscale, Tensor expert_num_tokens, int cap) "
        "-> (Tensor, Tensor)");
  m.def("dispatch_plan(Tensor topk_ids, int first_expert, int num_local_experts, int cap) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
  m.def("scatter_quant_act_planned(Tensor a1, Tensor src_tok, Tensor src_route, "
        "Tensor? topk_weights, Tensor gscale, Tensor expert_num_tokens, int cap) "
        "-> (Tensor, Tensor)");
  m.def("quant_rows(Tensor x, Tensor gscale) -> (Tensor, Tensor)");
  m.def("scatter_fp4_planned(Tensor q, Tensor q_sf, Tensor src_tok, "
        "Tensor expert_num_tokens, int cap) -> (Tensor, Tensor)");
  m.def("moe_finalize(Tensor(a!) out, Tensor fused, Tensor route_dest, Tensor? topk_weights) "
        "-> ()");
  m.def("group_mm(Tensor(a!) out, Tensor a_comp, Tensor e_meta, Tensor sfa, "
        "Tensor b_act, Tensor sfb, Tensor alphas, Tensor expert_num_tokens, "
        "int features, int k, "
        "int config_id=" PAIRED_NVFP4_STR(PAIRED_NVFP4_DEFAULT_CONFIG_ID) ", "
        "int cluster_m=" PAIRED_NVFP4_STR(PAIRED_NVFP4_DEFAULT_CLUSTER_M) ", "
        "int cluster_n=" PAIRED_NVFP4_STR(PAIRED_NVFP4_DEFAULT_CLUSTER_N) ") -> ()");
  m.def("group_mm_configs() -> str[]");
  m.def("group_mm_swiglu_quant(Tensor a_comp, Tensor e_meta, Tensor sfa, Tensor b_act, "
        "Tensor sfb, Tensor alphas, Tensor expert_num_tokens, Tensor gscale, int features, "
        "int k, int config_id=0, "
        "int cluster_m=" PAIRED_NVFP4_STR(PAIRED_NVFP4_DEFAULT_CLUSTER_M) ", "
        "int cluster_n=" PAIRED_NVFP4_STR(PAIRED_NVFP4_DEFAULT_CLUSTER_N) ", "
        "str activation=\"silu\", float beta=1.0, float linear_beta=-1.0) -> (Tensor, Tensor)");
  m.def("group_mm_swiglu_configs() -> str[]");
}

TORCH_LIBRARY_IMPL(paired_nvfp4, CUDA, m) {
  m.impl("compress", &paired_nvfp4::paired_nvfp4_compress);
  m.impl("quant_act", &paired_nvfp4::paired_nvfp4_quant_act);
  m.impl("silu_mul_quant_act", &paired_nvfp4::paired_nvfp4_silu_mul_quant_act);
  m.impl("scatter_quant_act", &paired_nvfp4::paired_nvfp4_scatter_quant_act);
  m.impl("dispatch_plan", &paired_nvfp4::paired_nvfp4_dispatch_plan);
  m.impl("scatter_quant_act_planned", &paired_nvfp4::paired_nvfp4_scatter_quant_act_planned);
  m.impl("quant_rows", &paired_nvfp4::paired_nvfp4_quant_rows);
  m.impl("scatter_fp4_planned", &paired_nvfp4::paired_nvfp4_scatter_fp4_planned);
  m.impl("moe_finalize", &paired_nvfp4::paired_nvfp4_moe_finalize);
  m.impl("group_mm", &paired_nvfp4::paired_nvfp4_group_mm);
  m.impl("group_mm_swiglu_quant", &paired_nvfp4::paired_nvfp4_group_mm_swiglu_quant);
}

TORCH_LIBRARY_IMPL(paired_nvfp4, CompositeExplicitAutograd, m) {
  m.impl("group_mm_configs", &paired_nvfp4::paired_nvfp4_group_mm_configs);
  m.impl("group_mm_swiglu_configs", &paired_nvfp4::paired_nvfp4_group_mm_swiglu_configs);
}

// torch.utils.cpp_extension imports the module as paired_nvfp4_kernels._C_sm<arch>, so CPython
// needs its PyInit symbol. The ops register through the static TORCH_LIBRARY initializers above.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "paired-4:8 NVFP4 W4A4 grouped sparse GEMM ops (torch.ops.paired_nvfp4)";
}
