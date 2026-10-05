# Copyright (C) 2026 Kwanhee Lee and Dan Alistarh. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
"""Import-time checks that need no GPU."""

import torch

import paired_nvfp4_kernels as pnk

OPS = ("compress", "quant_act", "silu_mul_quant_act", "scatter_quant_act",
       "scatter_quant_act_planned", "dispatch_plan", "moe_finalize", "quant_rows",
       "scatter_fp4_planned", "group_mm",
       "group_mm_configs")


def test_ops_registered():
    ns = torch.ops.paired_nvfp4
    for op in OPS:
        assert hasattr(ns, op), op


def test_loaded_arch_is_built():
    assert pnk.loaded_arch() in pnk.built_archs()


def test_tactics():
    configs = pnk.group_mm_configs()
    assert configs and all(name.startswith("sparse_") for name in configs)
    tactics = pnk.all_tactics()
    assert pnk.DEFAULT_TACTIC in tactics
    for config_id, cm, cn in tactics:
        if pnk._is_2sm(configs[config_id]):
            assert cm % 2 == 0


def test_suggest_tactic_is_a_valid_tactic():
    tactics = pnk.all_tactics()
    for max_n in (1, 64, 767, 768, 4096):
        assert pnk.suggest_tactic(max_n) in tactics


def test_group_mm_schema_defaults_match_python():
    schema = torch.ops.paired_nvfp4.group_mm.default._schema
    defaults = {a.name: a.default_value for a in schema.arguments if a.default_value is not None}
    assert (defaults["config_id"], defaults["cluster_m"], defaults["cluster_n"]) == pnk.DEFAULT_TACTIC
