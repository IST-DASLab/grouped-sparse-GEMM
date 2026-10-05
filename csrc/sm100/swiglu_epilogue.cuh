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
    \brief SM100 GEMM1 epilogue with SwiGLU and the group-32 FP4 quantization fused in.

    GEMM1 computes the gate and up projections of every expert; this epilogue turns its fp32
    accumulators straight into GEMM2's B operand (e2m1 codes + UE4M3 scales), so the bf16
    [E, cap, 2N] intermediate is never written and silu_mul_quant_act is not launched.

    Row interleave. W13's rows are interleaved at load time in 64-row blocks: rows [64b, 64b + 32)
    are gate rows 32b .. 32b + 31 and rows [64b + 32, 64b + 64) the matching up rows
    (interleave_gate_up_rows in the Python package). Every CTA owns 128 accumulator rows, and a
    tcgen05.ld 32dp32b load gives warp w lanes 32w .. 32w + 31, one row per thread. So warps 0 and
    2 hold gate rows, warps 1 and 3 the matching up rows, and each warp pair covers 32 output
    features: exactly one group-32 scale block of GEMM2's K.

    Per 32-token epilogue subtile: every thread rounds its accumulators to bf16(alpha[e] * acc),
    the value the stock epilogue would have stored, into shared memory by (row, column). Then,
    in two data-parallel phases, each thread computes v = bf16(bf16(silu(gate)) * up) (vLLM's
    silu_and_mul chain) for one feature over 16 token columns, and one thread per (32-feature
    block, token) runs quantize_block, the silu_mul_quant_act tail, and stores 16 code bytes and
    the scale. Token rows past the expert's count, and CTAs past M, store nothing. The result is
    byte-identical to GEMM1 -> silu_mul_quant_act.

    The collective derives from the builder's TMA epilogue and replaces only store(): its
    pipelines, shared storage and load path are unused but keep the kernel's interface intact.
    Tiles with overlapping accumulators (N = 256) are not supported.
*/

#pragma once

#include "cute/atom/copy_traits_sm100.hpp"
#include "cutlass/arch/barrier.h"

#include "../quant.cuh"

namespace paired_nvfp4 {

template <class Base>
class SwigluFp4Epilogue : public Base {
 public:
  using typename Base::LoadPipeline;
  using typename Base::LoadPipelineState;
  using typename Base::StorePipeline;
  using typename Base::StorePipelineState;
  using typename Base::TensorStorage;
  using EpilogueTile = typename Base::EpilogueTile;
  static constexpr int ThreadCount = Base::ThreadCount;
  static_assert(ThreadCount == 128, "fused SwiGLU epilogue expects 4 epilogue warps");
  static_assert(cute::size<0>(EpilogueTile{}) == 128 && cute::size<1>(EpilogueTile{}) == 32,
                "fused SwiGLU epilogue expects a 128 x 32 epilogue subtile");
  static constexpr int kCols = 32;                        // token columns per subtile
  static constexpr int kBufElems = 128 * kCols;           // bf16 per exchange buffer
  static_assert(sizeof(TensorStorage) >= (kBufElems + kBufElems / 2) * sizeof(cutlass::bfloat16_t),
                "base epilogue smem too small for the exchange buffers");

  struct Arguments {
    typename Base::Arguments base;
    SwigluFp4Args fused;
  };
  struct Params {
    typename Base::Params base;
    SwigluFp4Args fused;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                  Arguments const& args, void* workspace) {
    return {Base::to_underlying_arguments(problem_shape, args.base, workspace), args.fused};
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return Base::can_implement(problem_shape, args.base);
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return Base::get_workspace_size(problem_shape, args.base);
  }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args, void* workspace,
                                              cudaStream_t stream,
                                              cutlass::CudaHostAdapter* cuda_adapter = nullptr) {
    return Base::initialize_workspace(problem_shape, args.base, workspace, stream, cuda_adapter);
  }
  CUTLASS_DEVICE static void prefetch_tma_descriptors(Params const& params) {
    Base::prefetch_tma_descriptors(params.base);
  }

  CUTLASS_DEVICE SwigluFp4Epilogue(Params const& params, TensorStorage& shared_tensors)
      : Base(params.base, shared_tensors), fused_(params.fused) {}

  // Benign fill of the SFB slots no tile quantizes (rows at or past each expert's count, atom
  // slop); the op allocates sfb unfilled. The kernel calls it in every CTA before its first tile
  // (HasPrologueFill); the stores in store() touch the other, disjoint bytes.
  CUTLASS_DEVICE void prologue_fill(int tid, int nthr) const {
    int const* counts = fused_.expert_num_tokens;
    fill_benign_sfb(fused_.sfb, fused_.sfb_rows, fused_.sfb_kblocks, fused_.experts,
                    [counts](int e) { return counts[e]; }, tid, nthr);
  }

  template <bool ReuseTmem = false, class AccumulatorPipeline, class AccumulatorPipelineState,
            class ProblemShapeMNKL, class CtaTileMNK, class CtaCoordMNKL, class MmaTileMNK,
            class TiledMma, class AccEngine, class AccLayout>
  CUTLASS_DEVICE auto store(LoadPipeline load_pipeline, LoadPipelineState load_pipe_consumer_state,
                            StorePipeline store_pipeline,
                            StorePipelineState store_pipe_producer_state,
                            AccumulatorPipeline acc_pipeline,
                            AccumulatorPipelineState acc_pipe_consumer_state,
                            ProblemShapeMNKL problem_shape_mnkl, CtaTileMNK cta_tile_mnk,
                            CtaCoordMNKL cta_coord_mnkl, MmaTileMNK, TiledMma,
                            cute::Tensor<AccEngine, AccLayout> accumulators,
                            TensorStorage& shared_tensors) {
    using namespace cute;
    static_assert(!ReuseTmem, "fused SwiGLU epilogue does not support overlapping accumulators");
    static_assert(size<0>(CtaTileMNK{}) == 128, "fused SwiGLU epilogue expects 128 rows per CTA");

    // The kernel passes the capacity-sized problem shape here (only the mainloop sees the
    // per-expert one), so the expert's real token count comes from the counts array.
    auto [m_coord, n_coord, k_coord, l_coord] = cta_coord_mnkl;
    int const thread_idx = int(threadIdx.x) % ThreadCount;
    int const e = int(l_coord);

    Tensor tAcc = accumulators(make_coord(_, _), _0{}, _0{});             // (CTA_M, CTA_N)
    Tensor tAcc_epi = flat_divide(tAcc, EpilogueTile{});                 // (128, 32, 1, EPI_N)
    TiledCopy t2r = make_tmem_copy(SM100_TMEM_LOAD_32dp32b32x{}, tAcc_epi(_, _, _0{}, _0{}));
    ThrCopy thr_t2r = t2r.get_slice(thread_idx);
    Tensor tTR_tAcc = thr_t2r.partition_S(tAcc_epi);                     // (T2R, T2R_M, T2R_N, 1, EPI_N)
    Tensor cD = make_identity_tensor(select<0, 1>(CtaTileMNK{}));        // (CTA_M, CTA_N) local
    Tensor tTR_cD = thr_t2r.partition_D(flat_divide(cD, EpilogueTile{}));
    Tensor tTR_rAcc = make_tensor<float>(shape(tTR_cD(_, _, _, _0{}, _0{})));
    static_assert(decltype(size(tTR_rAcc))::value == kCols, "one row x 32 columns per thread");

    auto* xbuf = reinterpret_cast<cutlass::bfloat16_t*>(&shared_tensors);   // [4][32][32]
    auto* vbuf = xbuf + kBufElems;                                          // [2][32][32]
    auto sync = []() CUTLASS_LAMBDA_FUNC_INLINE {
      cutlass::arch::NamedBarrier::sync(ThreadCount,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    };

    float const alpha = fused_.alphas[e];
    float const g = fused_.gscale_per_expert ? fused_.gscale[e] : fused_.gscale[0];
    // CTA rows [128 m, 128 m + 128) carry output features [64 m, 64 m + 64): two 32-feature
    // blocks of GEMM2's K, each one scale.
    int const n_tile0 = int(n_coord) * int(size<1>(CtaTileMNK{}));
    int const n_count = fused_.expert_num_tokens[e];
    // A cluster can extend past M (e.g. a 4x1 cluster on M = 256): those CTAs still drain their
    // accumulator stage below but store nothing.
    bool const rows_in_bounds = int(m_coord) * 128 < int(get<0>(problem_shape_mnkl));

    constexpr int NumEpiN = decltype(size<4>(tTR_tAcc))::value;
    auto acc_wait_token = acc_pipeline.consumer_try_wait(acc_pipe_consumer_state);
    CUTLASS_PRAGMA_NO_UNROLL
    for (int epi_n = 0; epi_n < NumEpiN; ++epi_n) {
      // Valid token columns of this subtile; uniform across the CTA. Empty subtiles (and CTAs
      // past M) only take part in the accumulator handshake.
      int const n_sub0 = n_tile0 + epi_n * kCols;
      int const valid = rows_in_bounds ? max(0, min(kCols, n_count - n_sub0)) : 0;
      if (epi_n == 0) acc_pipeline.consumer_wait(acc_pipe_consumer_state, acc_wait_token);
      if (valid > 0) copy(t2r, tTR_tAcc(_, _, _, _0{}, epi_n), tTR_rAcc);
      if (epi_n == NumEpiN - 1) {
        cutlass::arch::fence_view_async_tmem_load();
        acc_pipeline.consumer_release(acc_pipe_consumer_state);
        ++acc_pipe_consumer_state;
      }
      if (valid == 0) continue;

      // Phase 0: bf16(alpha * acc), the value the stock epilogue stores, placed by its (row,
      // column) coordinate: xbuf[row / 32][col][row % 32], conflict-free per feature below.
      // Two barriers per subtile suffice: phase 0 of the next subtile rewrites xbuf only after
      // every thread passed this subtile's second barrier (all phase-1 reads done), and phase 1
      // rewrites vbuf only after the next first barrier (every thread's phase 2 done).
      Tensor tTR_cD_n = tTR_cD(_, _, _, _0{}, epi_n);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kCols; ++j) {
        int const r = int(get<0>(tTR_cD_n(j)));
        int const c = int(get<1>(tTR_cD_n(j))) - epi_n * kCols;
        if (c < valid)
          xbuf[((r / 32) * kCols + c) * 32 + (r % 32)] = cutlass::bfloat16_t(alpha * tTR_rAcc(j));
      }
      sync();

      // Phase 1 (row-parallel): v = bf16(bf16(silu(gate)) * up), vLLM's silu_and_mul chain.
      // Thread t owns feature f = t % 64 (block f / 32) over 16 of the 32 token columns; results
      // go to vbuf[block][col][feature], one contiguous 32-feature row per (block, token).
      {
        int const f = thread_idx % 64, blk = f / 32, fl = f % 32;
        int const cbeg = (thread_idx / 64) * (kCols / 2);
        CUTLASS_PRAGMA_UNROLL
        for (int jj = 0; jj < kCols / 2; ++jj) {
          int const c = cbeg + jj;
          if (c >= valid) break;
          cutlass::bfloat16_t const gate = xbuf[((2 * blk) * kCols + c) * 32 + fl];
          cutlass::bfloat16_t const up = xbuf[((2 * blk + 1) * kCols + c) * 32 + fl];
          vbuf[(blk * kCols + c) * 32 + fl] = gated_act(gate, up, fused_.act);
        }
      }
      sync();

      // Phase 2 (column-parallel): one thread per (block, token) quantizes its 32 features with
      // quantize_block, the silu_mul_quant_act tail, and stores 16 code bytes and one scale.
      if (thread_idx < 2 * kCols) {
        int const blk = thread_idx / kCols, c = thread_idx % kCols;
        int const n = n_sub0 + c;
        if (c < valid) {
          int const k0 = (int(m_coord) * 2 + blk) * 32;
          alignas(16) cutlass::bfloat16_t vals[SFVecSize];
          load_block(vals, vbuf + (blk * kCols + c) * 32);
          ElementSF sf;
          uint4 packed = quantize_block(vals, g, sf);
          *b_act_word(fused_.b_act, fused_.stride_B, e, n, k0) = packed;
          fused_.sfb[fused_.layout_SFB(n, k0, e)] = sf;
        }
      }
    }
    sync();   // the next tile's phase 1 overwrites vbuf, still read by this tile's phase 2
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state,
                            acc_pipe_consumer_state);
  }

 private:
  SwigluFp4Args fused_;
};

}  // namespace paired_nvfp4

#include "../group_mm.cuh"

namespace paired_nvfp4 {

// A tile variant whose epilogue is SwigluFp4Epilogue: same mainloop (and so the same operand and
// scale layouts) as PairedGemmVariant, with the builder's epilogue wrapped.
template <class KernelScheduleType, class EpilogueScheduleType, int TileN_, int TileK_ = 256>
struct SwigluGemmVariant {
  using Plain = PairedGemmVariant<KernelScheduleType, EpilogueScheduleType, TileN_, TileK_>;
  static constexpr bool is_2sm = Plain::is_2sm;
  static constexpr int TileM = Plain::TileM;
  static constexpr int TileN = TileN_;
  using ProblemShape = typename Plain::ProblemShape;
  using CollectiveMainloop = typename Plain::CollectiveMainloop;
  using CollectiveEpilogue = SwigluFp4Epilogue<typename Plain::CollectiveEpilogue>;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, CollectiveMainloop, CollectiveEpilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// GEMM1 with the fused epilogue: p.D is GEMM2's B operand (uint8 [E, max_n, features / 4]);
// p.features is GEMM1's M (2N, rows interleaved), p.k its K.
template <class Variant>
void run_group_mm_swiglu_variant(GroupMmParams const& p, SwigluFp4Args fused) {
  assert_variant_layout_compatible<typename Variant::Plain>();
  using VGemm = typename Variant::Gemm;

  const c10::cuda::CUDAGuard device_guard(p.D.device());
  const int device = p.D.device().index();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();

  const int E = int(p.A_comp.size(0));
  const int M = p.features;
  const int K = p.k;
  const int max_n = int(p.B_act.size(1));

  SparseProblemShape sp = make_sparse_shape(M, max_n, K, E);
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {max_n, K, E});
  // The base epilogue's TMA descriptor for D is built but never used: it only needs a valid,
  // aligned address and the (M, max_n, E) shape.
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, max_n, E});
  auto layout_A = SparseConfig::fill_layoutA(sp);
  auto layout_E = SparseConfig::fill_layoutE(sp);
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(sp);
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(sp);

  using VProblemShape = typename Variant::ProblemShape;
  VProblemShape problem_shape = make_problem_shape<VProblemShape>(
      M, max_n, K, E, p.expert_num_tokens.data_ptr<int32_t>());

  cutlass::KernelHardwareInfo hw_info;
  hw_info.device_id = device;
  hw_info.sm_count = p.sm_count;
  hw_info.cluster_shape = dim3(p.cluster_m, p.cluster_n, 1);
  hw_info.cluster_shape_fallback = dim3(p.cluster_m, p.cluster_n, 1);

  typename VGemm::Arguments arguments{
      gemm_mode<VProblemShape>(),
      problem_shape,
      {reinterpret_cast<ArrayElementA*>(p.A_comp.data_ptr<uint8_t>()), layout_A,
       reinterpret_cast<ArrayElementB*>(p.B_act.data_ptr<uint8_t>()), stride_B,
       reinterpret_cast<ElementE*>(p.E_meta.data_ptr<uint8_t>()), layout_E,
       reinterpret_cast<ElementSF*>(p.SFA.data_ptr<uint8_t>()), layout_SFA,
       reinterpret_cast<ElementSF*>(p.SFB.data_ptr<uint8_t>()), layout_SFB},
      {{{},
        reinterpret_cast<ElementC*>(p.D.data_ptr()), StrideC{},
        reinterpret_cast<ElementD*>(p.D.data_ptr()), stride_D},
       fused},
      hw_info};
  if constexpr (kIsGroupedProblem<VProblemShape>) {
    configure_group_scheduler(arguments.scheduler, p.splits);
  }
  TORCH_CHECK(p.splits == 1, "paired_nvfp4.group_mm_swiglu_quant: split-K is not supported "
              "with the fused epilogue");
  auto& fa = arguments.epilogue.base.thread;   // alpha is applied by the fused store itself
  fa.alpha = 1.f;
  fa.beta = 0.f;

  size_t ws_bytes = VGemm::get_workspace_size(arguments);
  torch::Tensor workspace = torch::empty(
      {int64_t(ws_bytes)}, torch::TensorOptions().dtype(torch::kUInt8).device(p.D.device()));
  VGemm gemm;
  TORCH_CHECK(gemm.can_implement(arguments) == cutlass::Status::kSuccess,
              "paired_nvfp4.group_mm_swiglu_quant: can_implement failed for tile ",
              Variant::TileM, "x", Variant::TileN, " cluster=", p.cluster_m, "x", p.cluster_n,
              " (M=", M, " max_n=", max_n, " K=", K, " E=", E, ")");
  TORCH_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream) == cutlass::Status::kSuccess,
              "paired_nvfp4.group_mm_swiglu_quant: initialize failed");
  TORCH_CHECK(gemm.run(stream, nullptr, kGemmPdl && pdl::enabled()) == cutlass::Status::kSuccess,
              "paired_nvfp4.group_mm_swiglu_quant: run failed");
}

}  // namespace paired_nvfp4
