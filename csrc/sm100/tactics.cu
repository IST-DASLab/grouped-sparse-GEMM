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

// SM100 tactic table: the compiled tile variants (indexed by config_id) and cluster legality.

#include <torch/torch.h>

#include "../ops.h"

namespace paired_nvfp4 {

void run_group_mm_2sm_256x128(GroupMmParams const& p);
void run_group_mm_2sm_256x256(GroupMmParams const& p);
void run_group_mm_1sm_128x128(GroupMmParams const& p);
void run_group_mm_1sm_128x256(GroupMmParams const& p);
void run_group_mm_2sm_256x64(GroupMmParams const& p);
void run_group_mm_1sm_128x64(GroupMmParams const& p);
void run_group_mm_2sm_256x64_k512(GroupMmParams const& p);
void run_group_mm_1sm_128x64_k512(GroupMmParams const& p);

// Names follow sparse_<sm mode>_<TileM>x<TileN>; config_id is the index. Append new variants at
// the end so existing config_ids stay stable for cached autotuning results.
const GroupMmKind kGroupMmKinds[] = {
    {"sparse_2sm_256x128", true,  &run_group_mm_2sm_256x128},
    {"sparse_2sm_256x256", true,  &run_group_mm_2sm_256x256},
    {"sparse_1sm_128x128", false, &run_group_mm_1sm_128x128},
    {"sparse_1sm_128x256", false, &run_group_mm_1sm_128x256},
    {"sparse_2sm_256x64", true, &run_group_mm_2sm_256x64},
    {"sparse_1sm_128x64", false, &run_group_mm_1sm_128x64},
    {"sparse_2sm_256x64_k512", true, &run_group_mm_2sm_256x64_k512},
    {"sparse_1sm_128x64_k512", false, &run_group_mm_1sm_128x64_k512},
};
const int kNumGroupMmKinds = int(sizeof(kGroupMmKinds) / sizeof(kGroupMmKinds[0]));

void run_group_mm_swiglu_2sm_256x64_k512(GroupMmParams const& p, SwigluFp4Args const& a);
void run_group_mm_swiglu_2sm_256x64(GroupMmParams const& p, SwigluFp4Args const& a);
void run_group_mm_swiglu_1sm_128x64(GroupMmParams const& p, SwigluFp4Args const& a);
void run_group_mm_swiglu_2sm_256x128(GroupMmParams const& p, SwigluFp4Args const& a);

const SwigluMmKind kSwigluMmKinds[] = {
    {"fused_2sm_256x64_k512", true, &run_group_mm_swiglu_2sm_256x64_k512},
    {"fused_2sm_256x64", true, &run_group_mm_swiglu_2sm_256x64},
    {"fused_1sm_128x64", false, &run_group_mm_swiglu_1sm_128x64},
    {"fused_2sm_256x128", true, &run_group_mm_swiglu_2sm_256x128},
};
const int kNumSwigluMmKinds = int(sizeof(kSwigluMmKinds) / sizeof(kSwigluMmKinds[0]));

// Block-scaled SM100 kernels take dynamic clusters with each dim a power of two <= 4 and at most
// 16 CTAs; a 2SM kernel pairs CTAs along M, so it needs an even cluster_m.
void check_cluster(GroupMmKind const& kind, int cm, int cn) {
  auto pow2_le4 = [](int x) { return x == 1 || x == 2 || x == 4; };
  TORCH_CHECK(pow2_le4(cm) && pow2_le4(cn) && cm * cn <= 16,
              kind.name, ": cluster dims must be powers of two <= 4 with product <= 16, got ",
              cm, "x", cn);
  TORCH_CHECK(!kind.two_sm || cm % 2 == 0,
              kind.name, " is a 2SM kernel and needs an even cluster_m, got ", cm);
}

} // namespace paired_nvfp4
