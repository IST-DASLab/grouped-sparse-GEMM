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
    \brief SM100 (Blackwell datacenter, B200/GB200) type configuration for the paired-4:8 NVFP4
           W4A4 grouped sparse GEMM.

    Every arch-specific type lives here. The shared code (layout.cuh, quant.cuh, group_mm.cuh,
    ops.cu) only consumes the aliases this header defines, so a port to another arch provides a
    sibling header with the same names (see docs/porting.md).

    Operand roles (the structured-sparse operand of the MMA is A, so the weight is A):
        D[M,N] = alpha * ( A[M,K] * B[K,N] )
          A = weight       4:8 sparse, RowMajor FP4, compressed + metadata E    M = out_features
          B = activations  dense ColMajor FP4                                   N = tokens
          D = output       ColMajor [features, tokens] == RowMajor [tokens, features], bf16
          K = in_features

    The grouped kernel is driven by MoEProblemShape: each expert's B / D / SFB is addressed at a
    fixed per-expert stride of max_n, so those tensors are per-expert padded [E, max_n, ...], and
    the device array expert_num_tokens[E] (<= max_n) clips each expert to its real token count.
*/

#pragma once

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/transform/kernel/sparse_gemm_compressor.hpp"
#include "cutlass/transform/device/transform_universal_adapter.hpp"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#define PAIRED_NVFP4_ENABLED 1
#else
#define PAIRED_NVFP4_ENABLED 0
#endif

// Default group_mm tactic (config_id, cluster_m, cluster_n); also the torch schema defaults.
#define PAIRED_NVFP4_DEFAULT_CONFIG_ID 0
#define PAIRED_NVFP4_DEFAULT_CLUSTER_M 2
#define PAIRED_NVFP4_DEFAULT_CLUSTER_N 1

namespace paired_nvfp4 {

using namespace cute;

static constexpr int kArchSm = 100;

using ElementA     = cutlass::float_e2m1_t;
using ElementAPair = cutlass::nv_float4_t<ElementA>;
using LayoutTagA   = cutlass::layout::RowMajor;
static constexpr int AlignmentA = 64;   // sparse A is compressed along K, so 2x the dense alignment

using ElementE   = cute::uint8_t;       // sparsity metadata
using LayoutTagE = LayoutTagA;

using ElementB     = cutlass::float_e2m1_t;
using ElementBPair = cutlass::nv_float4_t<ElementB>;
using LayoutTagB   = cutlass::layout::ColumnMajor;
static constexpr int AlignmentB = 32;

using ElementSF = typename ElementAPair::ScaleFactorType;   // UE4M3 block scales

using ElementD   = cutlass::bfloat16_t;
using ElementC   = cutlass::bfloat16_t;
// ColMajor D[features, tokens] is exactly the RowMajor [tokens, features] output the caller wants,
// and avoids the RowMajor N >= 8 alignment floor.
using LayoutTagC = cutlass::layout::ColumnMajor;
using LayoutTagD = cutlass::layout::ColumnMajor;
static constexpr int AlignmentD = (16 * 8) / cutlass::sizeof_bits<ElementD>::value;
static constexpr int AlignmentC = (16 * 8) / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledSparseTensorOp;

using ClusterShape = Shape<int32_t, int32_t, _1>;   // dynamic: set per launch (the tactic)

using SparseProblemShape = Shape<int, int, int, int>;   // (M, N, K, groups), layout setup only
using ProblemShape       = cutlass::gemm::MoEProblemShape<Shape<int, int, int>>;

// One compiled GEMM tile. TileM is fixed by the SM mode (128 for 1SM, 256 for 2SM), TileK is 256,
// and the sparse NVFP4 MMA supports TileN in {128, 256}. The cluster shape is a runtime argument,
// so four instantiations cover the whole tactic space.
template <class KernelScheduleType,     // KernelSparseTmaWarpSpecialized{1,2}SmNvf4Sm100
          class EpilogueScheduleType,   // cutlass::epilogue::TmaWarpSpecialized{1,2}SmNvf4
          int TileN_,
          int TileK_ = 256>
struct PairedGemmVariant {
  static constexpr bool is_2sm =
      cute::is_base_of_v<cutlass::gemm::KernelSchedule2Sm, KernelScheduleType>;
  static constexpr int TileM = is_2sm ? 256 : 128;
  static constexpr int TileN = TileN_;
  using ProblemShape = paired_nvfp4::ProblemShape;
  static constexpr int TileK = TileK_;
  using MmaTileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutTagC, AlignmentC,
      ElementD, LayoutTagD, AlignmentD,
      EpilogueScheduleType>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementAPair, LayoutTagA, AlignmentA,
      ElementBPair, LayoutTagB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveoutEpi<CollectiveEpilogue>,
      KernelScheduleType>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Every shared data layout below derives from this variant; group_mm.cuh static_asserts that the
// other tile variants produce identical layouts, so compress() / quant_act() output is readable
// by every tactic.
using DefaultPairedGemm = PairedGemmVariant<
    cutlass::gemm::KernelSparseTmaWarpSpecialized2SmNvf4Sm100,
    cutlass::epilogue::TmaWarpSpecialized2SmNvf4, 128>;
using Gemm = typename DefaultPairedGemm::Gemm;
using Mainloop = typename Gemm::GemmKernel::CollectiveMainloop;

using Sm1xxBlkScaledConfig = typename Mainloop::Sm1xxBlkScaledConfig;

// SF block length along K, taken from the instantiated MMA. The sparse NVFP4 MMA reads one UE4M3
// scale per 32 dense K elements (16 survive the 4:8 prune); the dense NVFP4 MMA reads one per 16.
// Everything that sizes, writes or reads a scale derives its K granularity from this constant.
static constexpr int SFVecSize = Sm1xxBlkScaledConfig::SFVecSize;

using LayoutA   = typename Mainloop::LayoutA;
using LayoutE   = typename Mainloop::LayoutE;
using StrideA   = cutlass::gemm::TagToStrideA_t<LayoutTagA>;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutSFA = typename Mainloop::LayoutSFA;
using LayoutSFB = typename Mainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;

using ArrayElementA = typename Mainloop::ArrayElementA;
using ArrayElementB = typename Mainloop::ArrayElementB;

using SparseConfig = typename Mainloop::SparseConfig;
using CompressorUtility = cutlass::transform::kernel::StructuredSparseCompressorUtility<
    SparseProblemShape, ElementA, LayoutTagA, SparseConfig>;
using CompressorKernel = cutlass::transform::kernel::StructuredSparseCompressor<
    SparseProblemShape, ElementA, LayoutTagA, SparseConfig, ArchTag>;
using Compressor = cutlass::transform::device::TransformUniversalAdapter<CompressorKernel>;

// Quantizer row-grid cap, in blocks of 256 threads per SM (quant.cuh). 14 x 148 SMs keeps the
// ~2048 row blocks tuned on B200.
static constexpr int kQuantRowBlocksPerSm = 14;

// The GEMMs launch with programmatic dependent launch (csrc/pdl.cuh): built with
// CUTLASS_ENABLE_GDC_FOR_SM100, and the kernel override waits before its first dependent read.
static constexpr bool kGemmPdl = true;

// Grouped-scheduler arguments (raster order, swizzle). SM100 keeps CUTLASS's defaults and has no
// split-K variants (every SM100 tile kind has splits == 1).
template <class SchedulerArguments>
inline void configure_group_scheduler(SchedulerArguments&, int /*splits*/) {}

} // namespace paired_nvfp4
