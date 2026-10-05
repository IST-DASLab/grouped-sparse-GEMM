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

/*
    Programmatic dependent launch (PDL) for the MoE chain: dispatch_plan -> the quantizers ->
    group_mm (GEMM1, GEMM2) -> moe_finalize.

    A kernel launched with the programmatic-stream-serialization attribute may start while its
    predecessor on the stream is still running (once the predecessor triggers
    griddepcontrol.launch_dependents, or at the latest when it completes), so its launch and
    prologue overlap the predecessor's tail. Correctness rests on one rule, kept by every kernel
    launched through pdl::launch: it executes griddepcontrol.wait (pdl::wait) before its first
    global-memory access that could depend on an earlier kernel. The wait returns once all
    prerequisite grids have completed and their writes are visible, so the chain stays ordered
    transitively. Without the attribute, the wait is a no-op and the launch is an ordinary one.
    The CUTLASS GEMMs follow the same rule (CUTLASS_ENABLE_GDC_FOR_SM100, see setup.py).

    PAIRED_NVFP4_PDL=0 turns the attribute off for every op (read once per process).
*/

#pragma once

#include <cstdlib>
#include <utility>

#include <cuda_runtime.h>

namespace paired_nvfp4::pdl {

inline bool enabled() {
  static const bool on = [] {
    char const* v = std::getenv("PAIRED_NVFP4_PDL");
    return v == nullptr || v[0] != '0';
  }();
  return on;
}

// Wait for every prerequisite grid to complete (no-op without a programmatic launch).
__device__ __forceinline__ void wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Allow the next kernel on the stream to launch early. Only a scheduling hint: the dependent
// still waits for this grid's completion before it touches our outputs.
__device__ __forceinline__ void launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;");
#endif
}

// kernel<<<grid, block, smem, stream>>>(args...), with the PDL attribute when enabled().
template <class... KernelArgs, class... Args>
inline cudaError_t launch(void (*kernel)(KernelArgs...), dim3 grid, dim3 block, size_t smem,
                          cudaStream_t stream, Args&&... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = block;
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = enabled() ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kernel, std::forward<Args>(args)...);
}

} // namespace paired_nvfp4::pdl
