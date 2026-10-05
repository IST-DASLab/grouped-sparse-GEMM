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

// C++ entry points of the torch ops (ops.cu), the tile-variant runners (one TU each) and the
// arch's tactic table. The self-test (testing/selftest.cu) calls the ops through these directly.

#pragma once

#include <optional>
#include <string>
#include <tuple>
#include <vector>
#include <torch/torch.h>

namespace paired_nvfp4 {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
paired_nvfp4_compress(torch::Tensor W_packed, torch::Tensor W_blockscale, int64_t k);

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_quant_act(torch::Tensor x, torch::Tensor gscale, torch::Tensor expert_num_tokens,
                       int64_t features);

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_silu_mul_quant_act(torch::Tensor x2, torch::Tensor gscale,
                                torch::Tensor expert_num_tokens, int64_t features, bool interleaved = false);

std::tuple<torch::Tensor, torch::Tensor>
paired_nvfp4_scatter_quant_act(torch::Tensor a1, torch::Tensor flat_tok,
                               torch::Tensor dest_global,
                               std::optional<torch::Tensor> topk_weights,
                               torch::Tensor gscale, torch::Tensor expert_num_tokens,
                               int64_t cap);

// Validated arguments handed from the group_mm dispatcher to one tile-variant runner.
struct GroupMmParams {
  torch::Tensor& D;
  const torch::Tensor& A_comp;
  const torch::Tensor& E_meta;
  const torch::Tensor& SFA;
  const torch::Tensor& B_act;
  const torch::Tensor& SFB;
  const torch::Tensor& alphas;
  const torch::Tensor& expert_num_tokens;
  int features;
  int k;
  int cluster_m;
  int cluster_n;
  int sm_count;   // cached by the dispatcher, so the launch path never queries the device
  int splits;     // split-K factor of the tile variant (1 = no split)
};

// One compiled tile variant. The arch's tactics.cu defines the table, indexed by config_id.
struct GroupMmKind {
  const char* name;
  bool two_sm;
  void (*run)(GroupMmParams const&);
  int splits = 1;   // split-K: each output tile's K range is cut into this many work units
};
extern const GroupMmKind kGroupMmKinds[];
extern const int kNumGroupMmKinds;

// Tile variants of GEMM1 with the fused SwiGLU + FP4 epilogue (none on arches without one).
struct SwigluFp4Args;
struct SwigluMmKind {
  const char* name;
  bool two_sm;
  void (*run)(GroupMmParams const&, SwigluFp4Args const&);
};
extern const SwigluMmKind kSwigluMmKinds[];
extern const int kNumSwigluMmKinds;

// TORCH_CHECKs that (cluster_m, cluster_n) is a legal cluster for `kind` on this arch.
void check_cluster(GroupMmKind const& kind, int cluster_m, int cluster_n);

std::vector<std::string> paired_nvfp4_group_mm_configs();

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
                           int64_t cluster_n);

} // namespace paired_nvfp4
