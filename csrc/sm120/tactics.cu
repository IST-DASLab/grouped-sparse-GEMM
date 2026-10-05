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

// SM120 tactic table: the compiled tile variants (indexed by config_id) and cluster legality.

#include <torch/torch.h>

#include "../ops.h"

namespace paired_nvfp4 {

void run_group_mm_1sm_128x128(GroupMmParams const& p);
void run_group_mm_1sm_128x128_batched(GroupMmParams const& p);
void run_group_mm_1sm_128x64(GroupMmParams const& p);
void run_group_mm_1sm_256x128(GroupMmParams const& p);

// Names follow sparse_<sm mode>_<TileM>x<TileN>[_batched | _sk<splits>]; config_id is the index.
// Append new variants at the end so existing config_ids stay stable for cached autotuning results.
const GroupMmKind kGroupMmKinds[] = {
    {"sparse_1sm_128x128",         false, &run_group_mm_1sm_128x128},
    // Batched: every expert computes all max_n rows (kBatched, L = experts). Same bytes on valid
    // rows as the grouped variants; kept as a reference path and for fully packed experts.
    {"sparse_1sm_128x128_batched", false, &run_group_mm_1sm_128x128_batched},
    {"sparse_1sm_128x64",          false, &run_group_mm_1sm_128x64},
    {"sparse_1sm_256x128",         false, &run_group_mm_1sm_256x128},
    // Split-K (same kernels, K range cut into 2 or 4 work units): for decode-sized problems with
    // too few tiles to fill the SMs.
    {"sparse_1sm_128x64_sk2",      false, &run_group_mm_1sm_128x64,  2},
    {"sparse_1sm_128x64_sk4",      false, &run_group_mm_1sm_128x64,  4},
    {"sparse_1sm_128x128_sk2",     false, &run_group_mm_1sm_128x128, 2},
    {"sparse_1sm_128x128_sk4",     false, &run_group_mm_1sm_128x128, 4},
};
const int kNumGroupMmKinds = int(sizeof(kGroupMmKinds) / sizeof(kGroupMmKinds[0]));

void run_group_mm_swiglu_1sm_128x64(GroupMmParams const& p, SwigluFp4Args const& a);
void run_group_mm_swiglu_1sm_256x128(GroupMmParams const& p, SwigluFp4Args const& a);

// GEMM1 with the fused SwiGLU + FP4 epilogue (sm120/swiglu_epilogue.cuh): the decode default
// tile and the prefill one. Append new variants at the end, as above.
const SwigluMmKind kSwigluMmKinds[] = {
    {"fused_1sm_128x64",  false, &run_group_mm_swiglu_1sm_128x64},
    {"fused_1sm_256x128", false, &run_group_mm_swiglu_1sm_256x128},
};
const int kNumSwigluMmKinds = int(sizeof(kSwigluMmKinds) / sizeof(kSwigluMmKinds[0]));

// SM120 has no cluster multicast: the sparse kernel is built for a static 1x1x1 cluster, so a
// tactic asking for any other cluster (e.g. an SM100 default of 2x1) is rejected, not ignored.
void check_cluster(GroupMmKind const& kind, int cm, int cn) {
  TORCH_CHECK(cm == 1 && cn == 1,
              kind.name, ": SM120 kernels run with a 1x1 cluster only, got ", cm, "x", cn);
}

} // namespace paired_nvfp4
