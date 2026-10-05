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
    \brief SM120 GEMM1 epilogue with SwiGLU and the group-32 FP4 quantization fused in.

    The SM120 counterpart of sm100/swiglu_epilogue.cuh (see there for the dataflow): GEMM1's fp32
    accumulators become GEMM2's B operand (e2m1 codes + UE4M3 scales) directly, byte-identical to
    GEMM1 -> silu_mul_quant_act(interleaved=True).

    Row interleave. W13's rows are interleaved in 64-row blocks of 32 gate rows then the matching
    32 up rows (interleave_gate_up_rows), so a CTA tile of TileM rows holds TileM / 64 complete
    32-feature blocks of GEMM2's K: local block b is GEMM2 K block m_coord * (TileM / 64) + b.

    SM120 differences from SM100:
      - The accumulators are mma.sync register fragments of the two cooperative MMA warp groups
        (256 threads), not TMEM: each fragment element's (row, column) comes from partitioning an
        identity tensor of the CTA tile with the thread's MMA slice.
      - The base is the SM90-style TMA epilogue: no accumulator pipeline, and store() returns the
        (load, store) pipeline states.
      - The exchange buffers may exceed the base epilogue's shared storage, so TensorStorage grows
        to fit them and the mainloop's stage count is carved out of what is left (config.cuh).

    Per token subtile (32 columns; 16 on 256-row tiles), as on SM100: every thread writes
    bf16(alpha[e] * acc) of its fragment elements in the subtile to shared memory by (row,
    column); then row-parallel SwiGLU
    v = bf16(bf16(silu(gate)) * up) over all 256 threads; then one thread per (32-feature block,
    token) runs quantize_block and stores 16 code bytes and one scale. Token columns past the
    expert's count, and 64-row blocks past M, store nothing.

    The collective derives from the builder's TMA epilogue and replaces store(); its pipelines and
    load path are unused (beta = 0, so the producer never loads C) but keep the kernel's interface.
*/

#pragma once

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
  using typename Base::PipelineStorage;
  using CtaTileMNK = typename Base::CtaTileMNK;

  static constexpr int kTileM = int(cute::size<0>(CtaTileMNK{}));
  static constexpr int kTileN = int(cute::size<1>(CtaTileMNK{}));
  static_assert(kTileM % 64 == 0, "fused SwiGLU epilogue needs whole 64-row gate/up blocks");
  static constexpr int kBlocks = kTileM / 64;             // 32-feature GEMM2 K blocks per tile
  static constexpr int kBaseBytes = int(sizeof(typename Base::TensorStorage));
  static constexpr int exchange_bytes(int cols) {
    return (kTileM + kBlocks * 32) * cols * int(sizeof(cutlass::bfloat16_t));
  }
  // Token columns per subtile: 32 when the exchange buffers fit in the base epilogue's storage
  // (128-row tiles), else 16, so a 256-row tile takes no shared memory from the mainloop stages.
  static constexpr int kCols = exchange_bytes(32) <= kBaseBytes ? 32 : 16;
  static_assert(kTileN % kCols == 0, "fused SwiGLU epilogue needs whole token subtiles");
  static constexpr int kXbufElems = kTileM * kCols;       // bf16 alpha * acc, [TileM/32][kCols][32]
  static constexpr int kVbufElems = kBlocks * 32 * kCols; // bf16 v, [kBlocks][kCols][32]
  static constexpr int kExchangeBytes = exchange_bytes(kCols);

  // The base storage (unused by the fused store), followed by more bytes if the exchange buffers,
  // laid from the start of the struct, need them.
  struct GrownTensorStorage : Base::TensorStorage {
    alignas(16) uint8_t extra[kExchangeBytes > kBaseBytes ? kExchangeBytes - kBaseBytes : 16];
  };
  using TensorStorage = cute::conditional_t<(kExchangeBytes > kBaseBytes), GrownTensorStorage,
                                            typename Base::TensorStorage>;
  static_assert(sizeof(TensorStorage) >= size_t(kExchangeBytes),
                "fused SwiGLU epilogue storage too small for the exchange buffers");
  struct SharedStorage {
    TensorStorage tensors;
    PipelineStorage pipeline;
  };

  struct Arguments {
    typename Base::Arguments base;
    SwigluFp4Args fused;
  };
  // Derives from the base params: the kernel reads members of it (tma_transaction_bytes).
  struct Params : Base::Params {
    SwigluFp4Args fused;
  };

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                  Arguments const& args, void* workspace) {
    Params params;
    static_cast<typename Base::Params&>(params) =
        Base::to_underlying_arguments(problem_shape, args.base, workspace);
    params.fused = args.fused;
    return params;
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

  CUTLASS_DEVICE SwigluFp4Epilogue(Params const& params, TensorStorage& shared_tensors)
      : Base(params, shared_tensors), fused_(params.fused) {}

  // Benign fill of the SFB slots no tile quantizes (rows at or past each expert's count, atom
  // slop); the op allocates sfb unfilled. The kernel calls it in every CTA before its first tile
  // (HasPrologueFill); the stores in store() touch the other, disjoint bytes.
  CUTLASS_DEVICE void prologue_fill(int tid, int nthr) const {
    int const* counts = fused_.expert_num_tokens;
    fill_benign_sfb(fused_.sfb, fused_.sfb_rows, fused_.sfb_kblocks, fused_.experts,
                    [counts](int e) { return counts[e]; }, tid, nthr);
  }

  // Never runs (beta = 0, so is_producer_load_needed() is false); forwards the base view of the
  // storage so the kernel's producer branch compiles.
  template <class... Args>
  CUTLASS_DEVICE auto load(LoadPipeline load_pipeline, LoadPipelineState load_pipe_producer_state,
                           Args&&... args) {
    return forward_load(load_pipeline, load_pipe_producer_state, static_cast<Args&&>(args)...);
  }

  template <class ProblemShapeMNKL, class TileShapeMNK, class TileCoordMNKL, class AccEngine,
            class AccLayout, class TiledMma>
  CUTLASS_DEVICE auto store(LoadPipeline load_pipeline, LoadPipelineState load_pipe_consumer_state,
                            StorePipeline store_pipeline,
                            StorePipelineState store_pipe_producer_state,
                            ProblemShapeMNKL problem_shape_mnkl, TileShapeMNK tile_shape_MNK,
                            TileCoordMNKL tile_coord_mnkl,
                            cute::Tensor<AccEngine, AccLayout> accumulators, TiledMma tiled_mma,
                            int thread_idx, TensorStorage& shared_tensors, int subtile_idx = -1) {
    using namespace cute;
    static_assert(is_rmem<AccEngine>::value, "accumulators must be register fragments");
    static_assert(size<0>(TileShapeMNK{}) == kTileM && size<1>(TileShapeMNK{}) == kTileN,
                  "kernel tile differs from the epilogue's CTA tile");
    constexpr int kThreads = int(size(TiledMma{}));
    static_assert(kThreads == 256, "fused SwiGLU epilogue expects two cooperative MMA warp groups");
    static_assert(kBlocks * kCols * 4 % kThreads == 0, "whole rounds of one quad per (block, token)");

    // The kernel passes the capacity-sized problem shape (only the mainloop sees the per-expert
    // one), so the expert's real token count comes from the counts array.
    auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;
    int const e = int(l_coord);

    // (row, column) within the CTA tile of every accumulator element of this thread. The
    // fragment is ((2, 2), MMA_M, MMA_N) with column = c0(thread) + v + 16 * MMA_N index and
    // c0 in [0, 16): MMA_N index ni covers exactly CTA columns [16 ni, 16 ni + 16), so a subtile
    // is a static range of ni and every element is visited once.
    Tensor cD = make_identity_tensor(take<0, 2>(TileShapeMNK{}));
    Tensor tCcD = tiled_mma.get_slice(thread_idx).partition_C(cD);   // (MMA, MMA_M, MMA_N)
    static_assert(decltype(size(tCcD))::value == decltype(size(accumulators))::value,
                  "accumulator / coordinate partition mismatch");
    static_assert(rank(AccLayout{}) == 3 && size<2>(AccLayout{}) * 16 == kTileN,
                  "expected 16 CTA columns per MMA_N index (SM120 m16n8 atoms, 2 warps along N)");
    constexpr int kNiPerSubtile = kCols / 16;

    // xbuf[row / 32][col][(row % 32) ^ swz(col)]: the swizzle spreads the 4 column pairs a warp
    // stores per fragment element over distinct banks; a 32-row run at one column stays a
    // permutation of 32 consecutive slots, so phase 1 reads it conflict-free.
    auto* xbuf = reinterpret_cast<cutlass::bfloat16_t*>(&shared_tensors);
    auto* vbuf = xbuf + kXbufElems;                                         // [kBlocks][kCols][32]
    auto xidx = [](int grp, int c, int rl) CUTLASS_LAMBDA_FUNC_INLINE {
      return (grp * kCols + c) * 32 + (rl ^ (((c >> 1) & 3) << 3));
    };
    auto sync = []() CUTLASS_LAMBDA_FUNC_INLINE {
      cutlass::arch::NamedBarrier::sync(kThreads,
                                        cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    };

    float const alpha = fused_.alphas[e];
    float const g = fused_.gscale_per_expert ? fused_.gscale[e] : fused_.gscale[0];
    int const n_tile0 = int(n_coord) * kTileN;
    int const n_count = fused_.expert_num_tokens[e];
    // 64-row blocks of this tile within M (a tail tile can cover M partially); uniform per CTA.
    int const m_rows = int(get<0>(problem_shape_mnkl)) - int(m_coord) * kTileM;
    int const blocks = max(0, min(kBlocks, m_rows / 64));

    // Unrolled so the accumulator indices stay static (register-resident).
    constexpr int kSubtiles = kTileN / kCols;
    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < kSubtiles; ++epi_n) {
      // Valid token columns of this subtile; uniform across the CTA.
      int const n_sub0 = n_tile0 + epi_n * kCols;
      int const valid = blocks > 0 ? max(0, min(kCols, n_count - n_sub0)) : 0;
      if (valid == 0) continue;

      // Phase 0: bf16(alpha * acc), the value the stock epilogue stores, by (row, column). Two
      // barriers per subtile suffice: phase 0 of the next subtile rewrites xbuf only after every
      // thread passed this subtile's second barrier (all phase-1 reads done), and phase 1
      // rewrites vbuf only after the next first barrier (every thread's phase 2 done).
      CUTLASS_PRAGMA_UNROLL
      for (int nl = 0; nl < kNiPerSubtile; ++nl) {
        int const ni = epi_n * kNiPerSubtile + nl;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size<1>(accumulators); ++mi) {
          CUTLASS_PRAGMA_UNROLL
          for (int v = 0; v < size<0>(accumulators); ++v) {
            int const r = int(get<0>(tCcD(v, mi, ni)));
            int const c = int(get<1>(tCcD(v, mi, ni))) - epi_n * kCols;
            if (c < valid)
              xbuf[xidx(r / 32, c, r % 32)] =
                  cutlass::bfloat16_t(alpha * float(accumulators(v, mi, ni)));
          }
        }
      }
      sync();

      // Phase 1 (row-parallel): v = bf16(bf16(silu(gate)) * up), vLLM's silu_and_mul chain.
      // Thread t owns feature f = t % (32 kBlocks) (block f / 32) over a contiguous share of the
      // subtile's token columns; results go to vbuf[block][col][feature].
      {
        constexpr int kFeatures = 32 * kBlocks;
        constexpr int kColShare = kCols * kFeatures / kThreads;
        int const f = thread_idx % kFeatures, blk = f / 32, fl = f % 32;
        int const cbeg = (thread_idx / kFeatures) * kColShare;
        if (blk < blocks) {
          CUTLASS_PRAGMA_UNROLL
          for (int jj = 0; jj < kColShare; ++jj) {
            int const c = cbeg + jj;
            if (c >= valid) break;
            float const gf = float(xbuf[xidx(2 * blk, c, fl)]);
            cutlass::bfloat16_t const up = xbuf[xidx(2 * blk + 1, c, fl)];
            cutlass::bfloat16_t const s_bf(gf / (1.0f + expf(-gf)));
            vbuf[(blk * kCols + c) * 32 + fl] = cutlass::bfloat16_t(float(s_bf) * float(up));
          }
        }
      }
      sync();

      // Phase 2 (column-parallel): one quad of threads per (block, token) quantizes its 32
      // features (quantize_block_quad, byte-identical to the silu_mul_quant_act tail); lane q
      // stores code word q of the 16-byte b_act word, lane 0 the scale. Every thread of a warp
      // runs the quad shuffles; a quad is active or idle as a whole.
      CUTLASS_PRAGMA_UNROLL
      for (int job0 = 0; job0 < kBlocks * kCols; job0 += kThreads / 4) {
        int const job = job0 + thread_idx / 4, q = thread_idx % 4;
        int const blk = job / kCols, c = job % kCols;
        alignas(16) cutlass::bfloat16_t vals[8];
        *reinterpret_cast<uint4*>(vals) =
            *reinterpret_cast<uint4 const*>(vbuf + (blk * kCols + c) * 32 + 8 * q);
        ElementSF sf;
        uint32_t const word = quantize_block_quad(vals, g, sf);
        if (blk < blocks && c < valid) {
          int const n = n_sub0 + c;
          int const k0 = (int(m_coord) * kBlocks + blk) * 32;
          reinterpret_cast<uint32_t*>(b_act_word(fused_.b_act, fused_.stride_B, e, n, k0))[q] = word;
          if (q == 0) fused_.sfb[fused_.layout_SFB(n, k0, e)] = sf;
        }
      }
    }
    sync();   // the next tile's phase 1 overwrites vbuf, still read by this tile's phase 2
    return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
  }

 private:
  template <class ProblemShapeMNKL, class TileShapeMNK, class TileCoordMNKL, class TiledMma>
  CUTLASS_DEVICE auto forward_load(LoadPipeline load_pipeline,
                                   LoadPipelineState load_pipe_producer_state,
                                   ProblemShapeMNKL problem_shape_mnkl, TileShapeMNK tile_shape_MNK,
                                   TileCoordMNKL tile_coord_mnkl, TiledMma tiled_mma,
                                   int thread_idx, TensorStorage& shared_tensors,
                                   int subtile_idx = -1) {
    return Base::load(load_pipeline, load_pipe_producer_state, problem_shape_mnkl, tile_shape_MNK,
                      tile_coord_mnkl, tiled_mma, thread_idx,
                      static_cast<typename Base::TensorStorage&>(shared_tensors), subtile_idx);
  }

  SwigluFp4Args fused_;
};

}  // namespace paired_nvfp4

#include "../group_mm.cuh"

namespace paired_nvfp4 {

// A grouped tile variant whose epilogue is SwigluFp4Epilogue: the same mainloop types (and so the
// same operand and scale layouts) as PairedGemmVariant, with its stage count re-carved for the
// fused epilogue's shared storage.
template <int TileM_, int TileN_>
using SwigluGemmVariant = PairedGemmVariant<TileM_, TileN_, ProblemShape, SwigluFp4Epilogue>;

// GEMM1 with the fused epilogue: p.D is GEMM2's B operand (uint8 [E, max_n, features / 4]);
// p.features is GEMM1's M (2N, rows interleaved), p.k its K.
template <class Variant>
void run_group_mm_swiglu_variant(GroupMmParams const& p, SwigluFp4Args fused) {
  assert_variant_layout_compatible<Variant>();
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
  // The base epilogue's TMA descriptors for C / D are built but never used: they only need a
  // valid, aligned address and the (M, max_n, E) shape.
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, max_n, E});
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
        reinterpret_cast<ElementC*>(p.D.data_ptr()), stride_C,
        reinterpret_cast<ElementD*>(p.D.data_ptr()), stride_D},
       fused},
      hw_info};
  configure_group_scheduler(arguments.scheduler, p.splits);
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
