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
    \brief Grouped (MoE) tile scheduler with split-K for the SM120 sparse GEMM (grouped-sparse-GEMM;
           not part of CUTLASS).

    Extends PersistentTileSchedulerSm90Group: every output tile of the group walk is cut into
    `splits` K ranges. With splits == 1 it schedules exactly like the base scheduler.

    Work units u = t * S + s (tile t in group-walk order, split s) are dealt to the G persistent
    CTAs round-robin, u -> CTA u % G, with G a multiple of S. Hence the S splits of a tile run in
    the same round on the S consecutive CTAs of one "slot" (t % (G / S)), and a slot always sees
    split s on the same CTA. Each slot has S - 1 fp32 partial buffers and two counters:
      - splits 0..S-2 store their accumulators into partial buffer s, then add 1 to `arrived`;
      - split S - 1 waits until `arrived` reaches (round + 1) * (S - 1), adds the S - 1 partials
        in split order (deterministic), sets `consumed` to round + 1 and runs the epilogue; the
        other splits skip it;
      - in the next round, splits 0..S-2 first wait for `consumed` >= round, so they never
        overwrite a partial the previous round's reducer has not read yet.
    The reduction is one step, not a chain: the partials are written concurrently.

    Each thread stores and reloads only its own accumulator fragment, and every CTA has the same
    thread-to-fragment mapping, so the buffers need no coordinate math. Only the counters need a
    known start state; they are zeroed per launch by initialize_workspace (graph-safe), and the
    partials are fully written before they are read.

    The waits assume all G CTAs are co-resident (G <= SM count, one CTA per SM), as CUTLASS's
    stream-K scheduler does.
*/

#pragma once

#include "cutlass/arch/barrier.h"
#include "cutlass/workspace.h"
#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"

namespace cutlass::gemm::kernel::detail {

template <class GroupProblemShape, int SchedulerPipelineStageCount, class TileShape>
class PersistentTileSchedulerSm120GroupSplitK
    : public PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount> {
  using Base = PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount>;

public:
  static constexpr int TileM = int(cute::size<0>(TileShape{}));
  static constexpr int TileN = int(cute::size<1>(TileShape{}));
  static constexpr int TileK = int(cute::size<2>(TileShape{}));
  static constexpr bool IsDynamicPersistent = false;

  using typename Base::RasterOrder;
  using typename Base::RasterOrderOptions;
  using typename Base::Pipeline;
  using typename Base::PipelineStorage;
  using typename Base::PipelineState;
  using typename Base::ThrottlePipeline;
  using typename Base::ThrottlePipelineStorage;
  using typename Base::SharedStorage;
  using BaseParams = typename Base::Params;

  struct WorkTileInfo {
    int32_t M_idx = 0;
    int32_t N_idx = 0;
    int32_t L_idx = 0;
    int32_t is_valid_tile = 0;
    int32_t k_tile_start = 0;
    int32_t k_tile_count = 0;
    int32_t split = 0;
    int32_t splits = 1;
    int32_t slot = 0;
    int32_t round = 0;

    CUTLASS_HOST_DEVICE bool is_valid() const { return is_valid_tile != 0; }
    CUTLASS_HOST_DEVICE static WorkTileInfo invalid_work_tile() { return {-1, -1, -1, 0}; }
    CUTLASS_HOST_DEVICE bool is_final_split(uint32_t) const { return split == splits - 1; }
    CUTLASS_HOST_DEVICE int32_t reduction_subtile_idx() const { return -1; }
  };
  using SchedulerResponse = WorkTileInfo;

  struct Arguments : Base::Arguments {
    int splits = 1;
  };

  struct Params : BaseParams {
    int32_t splits = 1;
    int32_t k_tiles = 1;
    int32_t slots = 1;
    int32_t* flags = nullptr;
    float* partials = nullptr;
  };

  static constexpr size_t kFlagAlign = 128;

  static int grid_ctas(int sm_count, int splits) {
    return splits > 1 ? (sm_count / splits) * splits : sm_count;
  }

  // Two counters per slot (arrived, consumed).
  static size_t flag_bytes(int slots) {
    return (size_t(2 * slots) * sizeof(int32_t) + kFlagAlign - 1) / kFlagAlign * kFlagAlign;
  }

  static size_t partial_bytes(int slots, int splits) {
    return size_t(slots) * size_t(splits - 1) * TileM * TileN * sizeof(float);
  }

  static int resolved_sm_count(KernelHardwareInfo const& hw_info) {
    return hw_info.sm_count > 0 ? hw_info.sm_count
                                : KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
  }

  template <class TileShape_, class ClusterShape>
  static Params
  to_underlying_arguments(GroupProblemShape problem_shapes, TileShape_ tile_shape, ClusterShape cluster_shape,
                          KernelHardwareInfo const& hw_info, Arguments const& arguments,
                          void* workspace = nullptr, const uint32_t epilogue_subtile = 1,
                          uint32_t ktile_start_alignment_count = 1u) {
    int splits = arguments.splits > 1 ? arguments.splits : 1;
    KernelHardwareInfo hw = hw_info;
    hw.sm_count = grid_ctas(resolved_sm_count(hw_info), splits);
    Params params;
    static_cast<BaseParams&>(params) = Base::to_underlying_arguments(
        problem_shapes, tile_shape, cluster_shape, hw, arguments, workspace, epilogue_subtile,
        ktile_start_alignment_count);
    params.splits = splits;
    params.k_tiles = int32_t((cute::get<2>(problem_shapes.get_host_problem_shape(0)) + TileK - 1) / TileK);
    params.slots = hw.sm_count / splits;
    if (splits > 1) {
      params.flags = reinterpret_cast<int32_t*>(workspace);
      params.partials = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(workspace) + flag_bytes(params.slots));
    }
    return params;
  }

  template <class TileShape_, class ClusterShape>
  CUTLASS_HOST_DEVICE static dim3
  get_grid_shape(Params const& params, GroupProblemShape const& problem_shapes, TileShape_ tile_shape,
                 ClusterShape cluster_shape, KernelHardwareInfo hw_info, Arguments arguments,
                 bool truncate_by_problem_size = true) {
    hw_info.sm_count = params.slots * params.splits;
    return Base::get_grid_shape(params, problem_shapes, tile_shape, cluster_shape, hw_info, arguments,
                                truncate_by_problem_size);
  }

  static bool
  can_implement(Arguments const& args, KernelHardwareInfo const&) {
    return args.splits >= 1;
  }

  template <class ProblemShape, class ElementAccumulator>
  static size_t
  get_workspace_size(Arguments const& args, ProblemShape, KernelHardwareInfo const& hw_info,
                     uint32_t, const uint32_t = 1, uint32_t = 1) {
    if (args.splits <= 1) {
      return 0;
    }
    int slots = grid_ctas(resolved_sm_count(hw_info), args.splits) / args.splits;
    return flag_bytes(slots) + partial_bytes(slots, args.splits);
  }

  template <class ProblemShape, class ElementAccumulator>
  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace, cudaStream_t stream, ProblemShape,
                       KernelHardwareInfo const& hw_info, uint32_t, const uint32_t = 1, uint32_t = 1,
                       CudaHostAdapter* cuda_adapter = nullptr) {
    if (args.splits <= 1) {
      return Status::kSuccess;
    }
    int slots = grid_ctas(resolved_sm_count(hw_info), args.splits) / args.splits;
    return zero_workspace(workspace, flag_bytes(slots), stream, cuda_adapter);
  }

  PersistentTileSchedulerSm120GroupSplitK() = default;

  CUTLASS_DEVICE explicit
  PersistentTileSchedulerSm120GroupSplitK(Params const& params, SchedulerResponse* response_ptr)
      : Base(params, reinterpret_cast<typename Base::SchedulerResponse*>(response_ptr)),
        splits_(params.splits), k_tiles_(params.k_tiles), slots_(params.slots) {}

  // Warp-collective (the base group walk shuffles across the warp).
  CUTLASS_DEVICE WorkTileInfo
  work_for_unit(uint64_t unit) {
    uint64_t tile = splits_ > 1 ? unit / uint64_t(splits_) : unit;
    int32_t split = splits_ > 1 ? int32_t(unit - tile * uint64_t(splits_)) : 0;
    auto base = this->get_current_work_for_linear_idx(tile);
    if (!base.is_valid()) {
      return WorkTileInfo::invalid_work_tile();
    }
    WorkTileInfo w;
    w.M_idx = base.M_idx;
    w.N_idx = base.N_idx;
    w.L_idx = base.L_idx;
    w.is_valid_tile = 1;
    w.k_tile_start = int32_t((int64_t(split) * k_tiles_) / splits_);
    w.k_tile_count = int32_t((int64_t(split + 1) * k_tiles_) / splits_) - w.k_tile_start;
    w.split = split;
    w.splits = splits_;
    w.slot = int32_t(tile % uint64_t(slots_));
    w.round = int32_t(tile / uint64_t(slots_));
    return w;
  }

  template <class ClusterShape, typename CallbackBeforeCommit = WorkTileInfo(*)(WorkTileInfo)>
  CUTLASS_DEVICE WorkTileInfo
  initial_work_tile_info(ClusterShape, CallbackBeforeCommit = [] (WorkTileInfo r) { return r; }) {
    return work_for_unit(this->current_work_linear_idx_);
  }

  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState,
            typename CallbackBeforeCommit = WorkTileInfo(*)(WorkTileInfo)>
  CUTLASS_DEVICE auto
  advance_to_next_work(TileSchedulerPipeline& scheduler_pipeline,
                       TileSchedulerPipelineState scheduler_pipe_producer_state,
                       uint32_t advance_count = 1,
                       CallbackBeforeCommit = [] (WorkTileInfo r) { return r; }) {
    this->current_work_linear_idx_ += this->total_grid_size_ * uint64_t(advance_count);
    WorkTileInfo work_tile = work_for_unit(this->current_work_linear_idx_);
    scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
    if (cute::elect_one_sync()) {
      reinterpret_cast<WorkTileInfo*>(this->response_ptr_)[scheduler_pipe_producer_state.index()] = work_tile;
      cutlass::arch::fence_view_async_shared();
      scheduler_pipeline.producer_commit(scheduler_pipe_producer_state);
    }
    return cute::make_tuple(work_tile, true);
  }

  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState>
  CUTLASS_DEVICE auto
  fetch_next_work(WorkTileInfo, TileSchedulerPipeline& scheduler_pipeline,
                  TileSchedulerPipelineState scheduler_pipe_consumer_state) {
    scheduler_pipeline.consumer_wait(scheduler_pipe_consumer_state);
    WorkTileInfo work_tile = reinterpret_cast<WorkTileInfo*>(this->response_ptr_)[scheduler_pipe_consumer_state.index()];
    cutlass::arch::fence_view_async_shared();
    scheduler_pipeline.consumer_release(scheduler_pipe_consumer_state);
    return cute::make_tuple(work_tile, true);
  }

  template <class ProblemShape_MNKL, class TileShape_>
  CUTLASS_HOST_DEVICE static int
  get_work_k_tile_count(WorkTileInfo const& work_tile_info, ProblemShape_MNKL, TileShape_) {
    return work_tile_info.k_tile_count;
  }

  CUTLASS_HOST_DEVICE static uint32_t
  get_work_k_tile_start(WorkTileInfo const& work_tile_info) {
    return uint32_t(work_tile_info.k_tile_start);
  }

  CUTLASS_HOST_DEVICE static bool
  compute_epilogue(WorkTileInfo const& work_tile_info, Params const&) {
    return work_tile_info.split == work_tile_info.splits - 1;
  }

  CUTLASS_DEVICE static bool
  valid_warpgroup_in_work_tile(WorkTileInfo const&) {
    return true;
  }

  CUTLASS_DEVICE static bool
  requires_separate_reduction(Params const&) {
    return false;
  }

  // One-step split-K reduction through the slot's partial buffers (see the file comment). Called
  // by every MMA thread (num_barriers warp groups) after the mainloop of a work unit.
  template <class FrgTensorC>
  CUTLASS_DEVICE static void
  fixup(Params const& params, WorkTileInfo const& work, FrgTensorC& accumulators,
        uint32_t num_barriers, uint32_t /*barrier_idx*/) {
    if (work.splits <= 1) {
      return;
    }
    constexpr int FragSize = int(decltype(cute::size(cute::declval<FrgTensorC&>()))::value);
    static_assert(FragSize % 4 == 0, "accumulator fragment must be a multiple of 4 floats");
    static_assert(cute::is_same_v<typename FrgTensorC::value_type, float>, "split-K reduces fp32 accumulators");
    const int num_threads = int(num_barriers) * NumThreadsPerWarpGroup;
    const int tid = int(threadIdx.x) % num_threads;
    int32_t* arrived  = params.flags + 2 * work.slot;
    int32_t* consumed = arrived + 1;
    constexpr size_t kTileFloats = size_t(TileM) * TileN;
    float4* partials = reinterpret_cast<float4*>(
        params.partials + size_t(work.slot) * size_t(work.splits - 1) * kTileFloats);
    cutlass::arch::NamedBarrier barrier(num_threads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0);

    auto wait_at_least = [](int32_t* counter, int32_t value) {
      int32_t seen;
      while (true) {
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(seen) : "l"(counter) : "memory");
        if (seen >= value) break;
        __nanosleep(20);
      }
    };

    if (work.split < work.splits - 1) {
      if (work.round > 0) {
        if (tid == 0) wait_at_least(consumed, work.round);
        barrier.sync();
      }
      float4* mine = partials + size_t(work.split) * (kTileFloats / 4);
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < FragSize / 4; ++j) {
        __stcg(mine + size_t(j) * num_threads + tid,
               make_float4(accumulators(4 * j + 0), accumulators(4 * j + 1),
                           accumulators(4 * j + 2), accumulators(4 * j + 3)));
      }
      __threadfence();
      barrier.sync();
      if (tid == 0) {
        asm volatile("red.release.gpu.global.add.s32 [%0], 1;" :: "l"(arrived) : "memory");
      }
    }
    else {
      if (tid == 0) wait_at_least(arrived, (work.round + 1) * (work.splits - 1));
      barrier.sync();
      for (int p = 0; p < work.splits - 1; ++p) {
        float4 const* theirs = partials + size_t(p) * (kTileFloats / 4);
        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < FragSize / 4; ++j) {
          float4 v = __ldcg(theirs + size_t(j) * num_threads + tid);
          accumulators(4 * j + 0) += v.x;
          accumulators(4 * j + 1) += v.y;
          accumulators(4 * j + 2) += v.z;
          accumulators(4 * j + 3) += v.w;
        }
      }
      barrier.sync();
      if (tid == 0) {
        asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(consumed), "r"(work.round + 1) : "memory");
      }
    }
  }

private:
  int32_t splits_ = 1;
  int32_t k_tiles_ = 1;
  int32_t slots_ = 1;
};

} // namespace cutlass::gemm::kernel::detail
