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

// Dense NVFP4 grouped GEMM baselines for SM120, used by tools/bench_moe.py --dense. Not part of
// the package. Derived from CUTLASS examples/79_blackwell_geforce_gemm/79d (pointer-array grouped
// NVFP4 GEMM), with the per-expert arrays built on the device from the token counts.
//   dense_grouped_mm : CUTLASS SM120 pointer-array grouped GEMM (as in example 79d), per-expert
//                      problem sizes / pointers / SFB layouts built on the device from the
//                      token counts (graph-safe, padded rows are never computed).
//   dense_batched_mm : stock SM120 block-scaled GEMM in kBatched mode, L = experts.
// Operands mirror paired_nvfp4.group_mm: D[M, N] = alpha[e] * W[M, K] . X[K, N], W RowMajor,
// X ColMajor ([E, max_n, K/2] bytes), D ColMajor ([E, max_n, M] bf16), NVFP4 with UE4M3 per 16.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

using ElementPair = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;
using ElementD = cutlass::bfloat16_t;
constexpr int AlignA = 32, AlignB = 32, AlignD = 8;
using ArchTag = cutlass::arch::Sm120;
using OpClass = cutlass::arch::OpClassBlockScaledTensorOp;
using Cluster = Shape<_1, _1, _1>;

// ---------------------------------------------------------------------------------------------
// Grouped (pointer-array)
// ---------------------------------------------------------------------------------------------
using GroupShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;

template <class Tile, class MainSchedule>
struct Grouped {
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, void, LayoutD*, AlignD, ElementD, LayoutD*, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, ElementPair, LayoutA*, AlignA, ElementPair, LayoutB*, AlignB, float,
      Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      MainSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<GroupShape, Main, Epi>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using StrideA = typename Kernel::InternalStrideA;
  using StrideB = typename Kernel::InternalStrideB;
  using StrideD = typename Kernel::InternalStrideD;
  using LSFA = typename Kernel::CollectiveMainloop::InternalLayoutSFA;
  using LSFB = typename Kernel::CollectiveMainloop::InternalLayoutSFB;
  using Cfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using EA = typename Gemm::ElementA;
  using EB = typename Gemm::ElementB;
  using ESF = typename Kernel::CollectiveMainloop::ElementSF;
};

// One thread per expert: problem size, pointers and the N-dependent SFB layout from counts.
template <class G>
__global__ void setup_groups(int E, int M, int K, int max_n, int const* counts,
                             uint8_t const* a, uint8_t const* sfa, uint8_t const* b,
                             uint8_t const* sfb, ElementD* d, float* alphas,
                             long long sfa_per, long long sfb_per,
                             typename GroupShape::UnderlyingProblemShape* ps,
                             typename G::EA const** pA, typename G::EB const** pB,
                             typename G::ESF const** pSFA, typename G::ESF const** pSFB,
                             ElementD** pD, float** pAlpha,
                             typename G::StrideA* sA, typename G::StrideB* sB,
                             typename G::StrideD* sD, typename G::LSFA* lSFA,
                             typename G::LSFB* lSFB) {
  int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= E) return;
  int n = counts[e];
  ps[e] = make_shape(M, n, K);
  pA[e] = reinterpret_cast<typename G::EA const*>(a + (long long)e * M * (K / 2));
  pB[e] = reinterpret_cast<typename G::EB const*>(b + (long long)e * max_n * (K / 2));
  pSFA[e] = reinterpret_cast<typename G::ESF const*>(sfa + e * sfa_per);
  pSFB[e] = reinterpret_cast<typename G::ESF const*>(sfb + e * sfb_per);
  pD[e] = d + (long long)e * max_n * M;
  pAlpha[e] = alphas + e;
  sA[e] = cutlass::make_cute_packed_stride(typename G::StrideA{}, make_shape(M, K, 1));
  sB[e] = cutlass::make_cute_packed_stride(typename G::StrideB{}, make_shape(n > 0 ? n : 1, K, 1));
  sD[e] = cutlass::make_cute_packed_stride(typename G::StrideD{}, make_shape(M, n > 0 ? n : 1, 1));
  lSFA[e] = G::Cfg::tile_atom_to_shape_SFA(make_shape(M, n, K, 1));
  lSFB[e] = G::Cfg::tile_atom_to_shape_SFB(make_shape(M, n, K, 1));
}

template <class G>
void run_grouped(torch::Tensor out, torch::Tensor a, torch::Tensor sfa, torch::Tensor b,
                 torch::Tensor sfb, torch::Tensor alphas, torch::Tensor counts, int M, int K,
                 bool raster_n) {
  const int E = int(a.size(0)), max_n = int(b.size(1));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto opt = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
  auto buf = [&](size_t bytes) { return torch::empty({int64_t(bytes * E + 16)}, opt); };
  auto t_ps = buf(sizeof(typename GroupShape::UnderlyingProblemShape));
  auto t_pA = buf(8), t_pB = buf(8), t_pSFA = buf(8), t_pSFB = buf(8), t_pD = buf(8), t_pAl = buf(8);
  auto t_sA = buf(sizeof(typename G::StrideA)), t_sB = buf(sizeof(typename G::StrideB));
  auto t_sD = buf(sizeof(typename G::StrideD));
  auto t_lA = buf(sizeof(typename G::LSFA)), t_lB = buf(sizeof(typename G::LSFB));
  long long sfa_per = sfa.numel() / E, sfb_per = sfb.numel() / E;
  using PS = typename GroupShape::UnderlyingProblemShape;
  auto P = [](torch::Tensor& t) { return t.data_ptr<uint8_t>(); };
  setup_groups<G><<<(E + 127) / 128, 128, 0, stream>>>(
      E, M, K, max_n, counts.data_ptr<int>(), P(a), P(sfa), P(b), P(sfb),
      reinterpret_cast<ElementD*>(out.data_ptr()), alphas.data_ptr<float>(), sfa_per, sfb_per,
      reinterpret_cast<PS*>(P(t_ps)), reinterpret_cast<typename G::EA const**>(P(t_pA)),
      reinterpret_cast<typename G::EB const**>(P(t_pB)),
      reinterpret_cast<typename G::ESF const**>(P(t_pSFA)),
      reinterpret_cast<typename G::ESF const**>(P(t_pSFB)), reinterpret_cast<ElementD**>(P(t_pD)),
      reinterpret_cast<float**>(P(t_pAl)), reinterpret_cast<typename G::StrideA*>(P(t_sA)),
      reinterpret_cast<typename G::StrideB*>(P(t_sB)), reinterpret_cast<typename G::StrideD*>(P(t_sD)),
      reinterpret_cast<typename G::LSFA*>(P(t_lA)), reinterpret_cast<typename G::LSFB*>(P(t_lB)));

  cutlass::KernelHardwareInfo hw;
  hw.device_id = a.device().index();
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  typename G::Gemm::Arguments args;
  decltype(args.epilogue.thread) fa;
  fa.alpha = 0.f; fa.beta = 0.f; fa.alpha_ptr = nullptr; fa.beta_ptr = nullptr;
  fa.alpha_ptr_array = reinterpret_cast<float**>(P(t_pAl));
  fa.dAlpha = {_0{}, _0{}, 1};
  typename G::Kernel::TileSchedulerArguments sched;
  sched.raster_order = raster_n ? cutlass::gemm::kernel::detail::RasterOrderOptions::AlongN
                                : cutlass::gemm::kernel::detail::RasterOrderOptions::AlongM;
  args = typename G::Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {E, reinterpret_cast<PS*>(P(t_ps)), nullptr},
      {reinterpret_cast<typename G::EA const**>(P(t_pA)), reinterpret_cast<typename G::StrideA*>(P(t_sA)),
       reinterpret_cast<typename G::EB const**>(P(t_pB)), reinterpret_cast<typename G::StrideB*>(P(t_sB)),
       reinterpret_cast<typename G::ESF const**>(P(t_pSFA)), reinterpret_cast<typename G::LSFA*>(P(t_lA)),
       reinterpret_cast<typename G::ESF const**>(P(t_pSFB)), reinterpret_cast<typename G::LSFB*>(P(t_lB))},
      {fa, nullptr, reinterpret_cast<typename G::StrideD*>(P(t_sD)),
       reinterpret_cast<ElementD**>(P(t_pD)), reinterpret_cast<typename G::StrideD*>(P(t_sD))},
      hw, sched};
  typename G::Gemm gemm;
  auto ws = torch::empty({int64_t(G::Gemm::get_workspace_size(args)) + 16}, opt);
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "dense grouped: can_implement");
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "dense grouped: init");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "dense grouped: run");
}

using GCoop128  = Grouped<Shape<_128, _128, _128>, cutlass::gemm::collective::KernelScheduleAuto>;
using GPing128  = Grouped<Shape<_128, _128, _128>, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;
using GCoop256K = Grouped<Shape<_128, _128, _256>, cutlass::gemm::collective::KernelScheduleAuto>;
using GPing64N  = Grouped<Shape<_128, _64, _128>,  cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong>;

std::vector<std::string> variants() {
  return {"grouped_coop_128x128x128", "grouped_ping_128x128x128", "grouped_coop_128x128x256",
          "grouped_ping_128x64x128"};
}

void dense_grouped_mm(torch::Tensor out, torch::Tensor a, torch::Tensor sfa, torch::Tensor b,
                      torch::Tensor sfb, torch::Tensor alphas, torch::Tensor counts, int64_t M,
                      int64_t K, int64_t variant, bool raster_n) {
  const c10::cuda::CUDAGuard g(a.device());
  switch (variant) {
    case 0: return run_grouped<GCoop128>(out, a, sfa, b, sfb, alphas, counts, M, K, raster_n);
    case 1: return run_grouped<GPing128>(out, a, sfa, b, sfb, alphas, counts, M, K, raster_n);
    case 2: return run_grouped<GCoop256K>(out, a, sfa, b, sfb, alphas, counts, M, K, raster_n);
    case 3: return run_grouped<GPing64N>(out, a, sfa, b, sfb, alphas, counts, M, K, raster_n);
  }
  TORCH_CHECK(false, "bad variant");
}

// SF buffer sizes per expert (bytes) for an (M, max_n, K) expert, per-16 UE4M3.
std::vector<int64_t> sf_sizes(int64_t M, int64_t max_n, int64_t K) {
  using C = GCoop128::Cfg;
  return {int64_t(size(filter_zeros(C::tile_atom_to_shape_SFA(make_shape(int(M), int(max_n), int(K), 1))))),
          int64_t(size(filter_zeros(C::tile_atom_to_shape_SFB(make_shape(int(M), int(max_n), int(K), 1)))))};
}

// ---------------------------------------------------------------------------------------------
// Batched (stock SM120 block-scaled GEMM, L = experts)
// ---------------------------------------------------------------------------------------------
struct Batched {
  using Tile = Shape<_128, _128, _128>;
  using Epi = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OpClass, Tile, Cluster, cutlass::epilogue::collective::EpilogueTileAuto,
      float, float, void, LayoutD, AlignD, ElementD, LayoutD, AlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Main = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass, ElementPair, LayoutA, AlignA, ElementPair, LayoutB, AlignB, float, Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epi::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Main, Epi, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Cfg = typename Kernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
};

std::vector<int64_t> batched_sf_sizes(int64_t M, int64_t max_n, int64_t K, int64_t E) {
  using C = Batched::Cfg;
  auto s = make_shape(int(M), int(max_n), int(K), int(E));
  return {int64_t(size(filter_zeros(C::tile_atom_to_shape_SFA(s)))),
          int64_t(size(filter_zeros(C::tile_atom_to_shape_SFB(s))))};
}

void dense_batched_mm(torch::Tensor out, torch::Tensor a, torch::Tensor sfa, torch::Tensor b,
                      torch::Tensor sfb, torch::Tensor alphas, int64_t M, int64_t K) {
  const c10::cuda::CUDAGuard g(a.device());
  using B = Batched;
  const int E = int(a.size(0)), N = int(b.size(1));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  auto s = make_shape(int(M), N, int(K), E);
  typename B::Kernel::StrideA sA = cutlass::make_cute_packed_stride(typename B::Kernel::StrideA{}, {int(M), int(K), E});
  typename B::Kernel::StrideB sB = cutlass::make_cute_packed_stride(typename B::Kernel::StrideB{}, {N, int(K), E});
  typename B::Kernel::StrideD sD = cutlass::make_cute_packed_stride(typename B::Kernel::StrideD{}, {int(M), N, E});
  cutlass::KernelHardwareInfo hw;
  hw.device_id = a.device().index();
  hw.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
  typename B::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kBatched, s,
      {reinterpret_cast<typename B::Gemm::ElementA const*>(a.data_ptr()), sA,
       reinterpret_cast<typename B::Gemm::ElementB const*>(b.data_ptr()), sB,
       reinterpret_cast<cutlass::float_ue4m3_t const*>(sfa.data_ptr()), B::Cfg::tile_atom_to_shape_SFA(s),
       reinterpret_cast<cutlass::float_ue4m3_t const*>(sfb.data_ptr()), B::Cfg::tile_atom_to_shape_SFB(s)},
      {{}, nullptr, sD, reinterpret_cast<ElementD*>(out.data_ptr()), sD},
      hw};
  auto& fa = args.epilogue.thread;
  fa.alpha = 0.f; fa.beta = 0.f; fa.alpha_ptr = alphas.data_ptr<float>(); fa.dAlpha = {_0{}, _0{}, 1};
  typename B::Gemm gemm;
  auto ws = torch::empty({int64_t(B::Gemm::get_workspace_size(args)) + 16},
                         torch::TensorOptions().dtype(torch::kUInt8).device(a.device()));
  TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "dense batched: can_implement");
  TORCH_CHECK(gemm.initialize(args, ws.data_ptr(), stream) == cutlass::Status::kSuccess, "dense batched: init");
  TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "dense batched: run");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("variants", &variants);
  m.def("sf_sizes", &sf_sizes);
  m.def("batched_sf_sizes", &batched_sf_sizes);
  m.def("grouped_mm", &dense_grouped_mm);
  m.def("batched_mm", &dense_batched_mm);
}
