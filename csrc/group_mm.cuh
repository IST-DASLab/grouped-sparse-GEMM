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
    \brief Templated body of one group_mm tile-variant runner.

    Each tile variant is instantiated in its own TU (sm100/group_mm_*.cu), so nvcc compiles the
    heavyweight CUTLASS instantiations in parallel. Arguments arrive validated by the dispatcher
    in ops.cu; this only marshals them into the variant's Gemm and launches on the current stream.
*/

#pragma once

#include <type_traits>

#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "layout.cuh"
#include "ops.h"
#include "pdl.cuh"

namespace paired_nvfp4 {

// compress() emits A / E / SFA and the quantizers emit B / SFB against the default variant's
// layouts. A variant with a different layout would silently misread those bytes, so any
// divergence is a compile error.
template <class Variant>
constexpr void assert_variant_layout_compatible() {
  // Only the patched kernel headers (csrc/cutlass_overrides) have grouped support; this fails if a
  // stock CUTLASS copy shadowed one on the include path.
  if constexpr (kIsGroupedProblem<typename Variant::ProblemShape>) {
    static_assert(Variant::Gemm::GemmKernel::IsGroupedGemmKernel,
                  "sparse kernel is not the grouped (MoEProblemShape) variant: csrc/cutlass_overrides "
                  "must precede third_party/cutlass/include on the include path");
  }
  using M = typename Variant::Gemm::GemmKernel::CollectiveMainloop;
  static_assert(std::is_same_v<typename M::SparseConfig, SparseConfig>,
                "tile variant changes SparseConfig: compress() output would be misread");
  static_assert(std::is_same_v<typename M::LayoutA, LayoutA>,
                "tile variant changes LayoutA (compressed weight)");
  static_assert(std::is_same_v<typename M::LayoutE, LayoutE>,
                "tile variant changes LayoutE (sparsity metadata)");
  static_assert(std::is_same_v<typename M::LayoutSFA, LayoutSFA>,
                "tile variant changes LayoutSFA (weight block scales)");
  static_assert(std::is_same_v<typename M::LayoutSFB, LayoutSFB>,
                "tile variant changes LayoutSFB (activation block scales)");
  static_assert(std::is_same_v<typename M::ArrayElementA, ArrayElementA> &&
                std::is_same_v<typename M::ArrayElementB, ArrayElementB>,
                "tile variant changes operand element views");
  static_assert(std::is_same_v<typename Variant::Gemm::GemmKernel::StrideB, StrideB> &&
                std::is_same_v<typename Variant::Gemm::GemmKernel::StrideC, StrideC> &&
                std::is_same_v<typename Variant::Gemm::GemmKernel::StrideD, StrideD>,
                "tile variant changes B/C/D strides");
  static_assert(int(M::Sm1xxBlkScaledConfig::SFVecSize) == SFVecSize,
                "tile variant changes SFVecSize");
}

template <class Variant>
void run_group_mm_variant(GroupMmParams const& p) {
  assert_variant_layout_compatible<Variant>();
  using VGemm = typename Variant::Gemm;

  const c10::cuda::CUDAGuard device_guard(p.D.device());
  const int device = p.D.device().index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();

  const int E     = int(p.A_comp.size(0));
  const int M     = p.features;
  const int K     = p.k;
  const int max_n = int(p.B_act.size(1));

  SparseProblemShape sp = make_sparse_shape(M, max_n, K, E);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, max_n, E});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, max_n, E});
  auto layout_A   = SparseConfig::fill_layoutA(sp);
  auto layout_E   = SparseConfig::fill_layoutE(sp);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(sp);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(sp);

  // Grouped variants get the per-expert counts only as a device pointer: the persistent grid is
  // sized from the SM count, and each work tile takes its effective N from expert_num_tokens[e]
  // on the device. No host copy of the counts, so the launch is CUDA-graph-capture safe. Batched
  // variants compute all max_n rows of every expert.
  using VProblemShape = typename Variant::ProblemShape;
  VProblemShape problem_shape = make_problem_shape<VProblemShape>(
      M, max_n, K, E, p.expert_num_tokens.data_ptr<int32_t>());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count  = p.sm_count;
  hw_info.cluster_shape          = dim3(p.cluster_m, p.cluster_n, 1);
  hw_info.cluster_shape_fallback = dim3(p.cluster_m, p.cluster_n, 1);

  typename VGemm::Arguments arguments{
      gemm_mode<VProblemShape>(),
      problem_shape,
      {reinterpret_cast<ArrayElementA*>(p.A_comp.data_ptr<uint8_t>()), layout_A,
       reinterpret_cast<ArrayElementB*>(p.B_act.data_ptr<uint8_t>()), stride_B,
       reinterpret_cast<ElementE*>(p.E_meta.data_ptr<uint8_t>()), layout_E,
       reinterpret_cast<ElementSF*>(p.SFA.data_ptr<uint8_t>()), layout_SFA,
       reinterpret_cast<ElementSF*>(p.SFB.data_ptr<uint8_t>()), layout_SFB},
      {{},
       reinterpret_cast<ElementC*>(p.D.data_ptr()), stride_C,   // C is unread (beta = 0)
       reinterpret_cast<ElementD*>(p.D.data_ptr()), stride_D},
      hw_info};

  if constexpr (kIsGroupedProblem<VProblemShape>) {
    configure_group_scheduler(arguments.scheduler, p.splits);
  }
  else {
    TORCH_CHECK(p.splits == 1, "paired_nvfp4.group_mm: split-K needs a grouped tile variant");
  }
  constexpr int kTileK = int(cute::size<2>(typename Variant::MmaTileShape{}));
  TORCH_CHECK(p.splits <= (K + kTileK - 1) / kTileK,
              "paired_nvfp4.group_mm: can_implement failed: split-K ", p.splits,
              " needs at least that many K tiles of ", kTileK, ", but K=", K);

  // Per-expert alpha: a flat [E] f32 device array read as alpha_ptr[group * dAlpha.z]. This is
  // the strided scalar-broadcast form, not alpha_ptr_array (which is an array of pointers).
  auto& fa = arguments.epilogue.thread;
  fa.alpha     = 0.f;
  fa.beta      = 0.f;
  fa.alpha_ptr = reinterpret_cast<ElementAccumulator const*>(p.alphas.data_ptr<float>());
  fa.beta_ptr  = nullptr;
  fa.dAlpha    = {cute::_0{}, cute::_0{}, 1};
  fa.dBeta     = {cute::_0{}, cute::_0{}, 0};

  size_t ws_bytes = VGemm::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(
      {int64_t(ws_bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(p.D.device()));

  VGemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "paired_nvfp4.group_mm: can_implement failed for tile ",
              Variant::TileM, "x", Variant::TileN, Variant::is_2sm ? " (2sm)" : " (1sm)",
              " cluster=", p.cluster_m, "x", p.cluster_n,
              " (M=", M, " max_n=", max_n, " K=", K, " E=", E, ")");
  TORCH_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
              "paired_nvfp4.group_mm: initialize failed (tile ",
              Variant::TileM, "x", Variant::TileN, ")");
  TORCH_CHECK(gemm.run(stream, nullptr, kGemmPdl && pdl::enabled()) == cutlass::Status::kSuccess,
              "paired_nvfp4.group_mm: run failed (tile ",
              Variant::TileM, "x", Variant::TileN, ")");
}

} // namespace paired_nvfp4
