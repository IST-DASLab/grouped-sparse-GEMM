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
    \brief SM120 (Blackwell GeForce / workstation: RTX 5090, RTX PRO 6000) type configuration for
           the paired-4:8 NVFP4 W4A4 grouped sparse GEMM.

    Same alias names and operand roles as sm100/config.cuh, which the shared layers consume:
        D[M,N] = alpha * ( A[M,K] * B[K,N] )
          A = weight       4:8 sparse, RowMajor FP4, compressed + metadata E    M = out_features
          B = activations  dense ColMajor FP4                                   N = tokens
          D = output       ColMajor [features, tokens] == RowMajor [tokens, features], bf16

    SM120 differences from SM100:
      - The sparse block-scaled MMA is warp-level mma.sync with register operands (no UMMA / TMEM);
        the kernel is sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp (one producer
        warp group, two cooperative MMA warp groups).
      - No cluster multicast: the cluster is a static 1x1x1, and a tactic's cluster is always 1x1.
      - ~99 KB of shared memory per SM, which bounds the tile / stage choices.
    The scale-factor block is still 32 dense K elements, so the checkpoint format is unchanged.
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

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
#define PAIRED_NVFP4_ENABLED 1
#else
#define PAIRED_NVFP4_ENABLED 0
#endif

// Default group_mm tactic (config_id, cluster_m, cluster_n); also the torch schema defaults.
// config_id 2 is sparse_1sm_128x64, the best single tile across the measured MoE shapes (see
// docs/porting.md); paired_nvfp4_kernels.suggest_tactic() also switches to 256x128 for large max_n.
#define PAIRED_NVFP4_DEFAULT_CONFIG_ID 2
#define PAIRED_NVFP4_DEFAULT_CLUSTER_M 1
#define PAIRED_NVFP4_DEFAULT_CLUSTER_N 1

namespace paired_nvfp4 {

using namespace cute;

static constexpr int kArchSm = 120;

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
using LayoutTagC = cutlass::layout::ColumnMajor;
using LayoutTagD = cutlass::layout::ColumnMajor;
static constexpr int AlignmentD = (16 * 8) / cutlass::sizeof_bits<ElementD>::value;
static constexpr int AlignmentC = (16 * 8) / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledSparseTensorOp;
using KernelSchedule     = cutlass::gemm::KernelSparseTmaWarpSpecializedNvf4Sm120;
using EpilogueSchedule   = cutlass::epilogue::SparseTmaWarpSpecializedCooperativeSm120;

using ClusterShape = Shape<_1, _1, _1>;   // SM120 has no cluster multicast

using SparseProblemShape = Shape<int, int, int, int>;   // (M, N, K, groups), layout setup only
using BatchedProblemShape = Shape<int, int, int, int>;  // kBatched with L = experts
using ProblemShape       = cutlass::gemm::MoEProblemShape<Shape<int, int, int>>;   // kGrouped

template <class Epilogue>
using UnwrappedEpilogue = Epilogue;

// One compiled GEMM tile. TileK is 256 (the metadata atom's K extent for nvf4); the cooperative
// kernel needs TileM >= 128. ProblemShape_ selects grouped (MoEProblemShape: tiles past an
// expert's token count are never scheduled) or batched (every expert computes all max_n rows).
// WrapEpilogue wraps the builder's epilogue (sm120/swiglu_epilogue.cuh); the mainloop's stage
// count is carved out of what the wrapped epilogue's shared storage leaves.
template <int TileM_, int TileN_, class ProblemShape_,
          template <class> class WrapEpilogue = UnwrappedEpilogue>
struct PairedGemmVariant {
  static constexpr bool is_2sm = false;
  static constexpr int TileM = TileM_;
  static constexpr int TileN = TileN_;
  using ProblemShape = ProblemShape_;
  using MmaTileShape = Shape<Int<TileM>, Int<TileN>, _256>;

  using CollectiveEpilogue = WrapEpilogue<typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementAccumulator,
      ElementC, LayoutTagC, AlignmentC,
      ElementD, LayoutTagD, AlignmentD,
      EpilogueSchedule>::CollectiveOp>;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementAPair, LayoutTagA, AlignmentA,
      ElementBPair, LayoutTagB, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Every shared data layout below derives from this variant; group_mm.cuh static_asserts that the
// other tile variants produce identical layouts, so compress() / quant_act() output is readable
// by every tactic.
using DefaultPairedGemm = PairedGemmVariant<128, 128, ProblemShape>;
using Gemm = typename DefaultPairedGemm::Gemm;
using Mainloop = typename Gemm::GemmKernel::CollectiveMainloop;

using Sm1xxBlkScaledConfig = typename Mainloop::Sm1xxBlkScaledConfig;

// SF block length along K, taken from the instantiated MMA (32 for the sparse NVFP4 MMA, as on
// SM100). Checkpoints are shared across archs, so a different value is a format break.
static constexpr int SFVecSize = Sm1xxBlkScaledConfig::SFVecSize;
static_assert(SFVecSize == 32, "SM120 sparse NVFP4 must read one scale per 32 K elements");

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

// Quantizer row-grid cap, in blocks of 256 threads per SM (quant.cuh). Measured on an RTX PRO
// 6000: within ~2% of the best cap at 8192 tokens, and 10-15% faster than the former fixed 2048
// at 128-2048 tokens.
static constexpr int kQuantRowBlocksPerSm = 8;

// The SM120 GEMM kernel override is not audited for reads before its grid-dependency wait, so
// it launches without PDL (the quantizers and routing kernels still use it).
static constexpr bool kGemmPdl = false;

// Grouped-scheduler arguments. Raster AlongN: consecutive work tiles walk an expert's N (token)
// tiles for one M (weight) tile, so each weight tile is reused from L2 across its N tiles instead
// of being re-read from DRAM once per N tile. Measured on an RTX PRO 6000: ~10% faster for
// experts with several N tiles (Mixtral-8x7B w13 at 512-2048 tokens), neutral elsewhere; a
// larger swizzle was slower everywhere, and a deeper scheduler pipeline made no difference.
// `splits` > 1 selects split-K (PersistentTileSchedulerSm120GroupSplitK), for decode-sized
// problems whose few tiles would otherwise leave most SMs idle.
template <class SchedulerArguments>
inline void configure_group_scheduler(SchedulerArguments& args, int splits) {
  args.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN;
  args.max_swizzle_size = 1;
  args.splits = splits;
}

} // namespace paired_nvfp4
