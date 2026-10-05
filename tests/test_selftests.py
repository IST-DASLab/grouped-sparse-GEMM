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
"""Gates backed by the compiled self-tests (csrc/testing/selftest.cu).

Each self-test builds its inputs on the host, writes them through the kernel's own CuTe layouts,
drives the production ops and returns error counters.
"""

import pytest
import torch

from conftest import GEMM_SHAPES, requires_gpu, requires_selftest, shape_id

pytestmark = [requires_gpu, requires_selftest]


def st():
    return torch.ops._paired_nvfp4_test


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shape", GEMM_SHAPES, ids=shape_id)
def test_group_mm_matches_dequant_reference(shape, seed):
    """prune -> NVFP4 quantize -> compress -> group_mm with distinct per-expert alphas, against
    alpha[e] * (Wdq @ Xdq) at atol = rtol = 1e-1."""
    max_abs, max_rel, ref_rms, num_bad = st().selftest(*shape, seed).tolist()
    assert ref_rms > 0.0
    assert num_bad == 0, f"max_abs={max_abs:.4f} max_rel={max_rel:.4f}"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shape", GEMM_SHAPES, ids=shape_id)
def test_quant_act_into_group_mm(shape, seed):
    """quant_act output fed to group_mm, against a host reference built from the stored product."""
    max_abs, max_rel, ref_rms, num_bad = st().selftest_quant_act(*shape, seed).tolist()
    assert ref_rms > 0.0
    assert num_bad == 0, f"max_abs={max_abs:.4f} max_rel={max_rel:.4f}"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("shape", GEMM_SHAPES, ids=shape_id)
def test_quant_act_bytes_match_host(shape, seed):
    """quant_act bytes (hardware e2m1 convert) equal a host replica (software convert)."""
    b_mis, s_mis, b_first, s_first, _, _, b_tot, s_tot = st().selftest_quant_act_diff(*shape, seed).tolist()
    assert b_tot > 0 and s_tot > 0
    assert b_mis == 0, f"b_act mismatch, first byte {int(b_first)}"
    assert s_mis == 0, f"sfb mismatch, first byte {int(s_first)}"


@pytest.mark.parametrize("shape", [s for s in GEMM_SHAPES if s != (1024, 256, 512, 8)], ids=shape_id)
def test_sf_stages_and_fused_quant(shape):
    """Scale values, SF swizzle, host/device layout evaluation, and fused-vs-legacy quant_act."""
    r = st().selftest_sf_debug(*shape, 0).tolist()
    lin_mis, swz_mis, off_mis = r[0], r[1], r[2]
    fused_mis = r[9]
    assert off_mis == 0, "SF layout indexes differently on host and device"
    assert lin_mis == 0, "scale values wrong"
    assert swz_mis == 0, "scatter_sf places scales wrong"
    assert fused_mis == 0, "fused quant_act diverges from the two-stage reference"


@pytest.mark.parametrize("shape", [
    (256, 128, 256, 1),
    (256, 256, 256, 2),
    (512, 128, 512, 4),
    (2048, 192, 768, 8),
], ids=shape_id)
def test_silu_mul_quant_is_byte_exact(shape):
    """Fused SwiGLU + quant equals silu_and_mul followed by quant_act, byte for byte."""
    b_mis, sfb_mis, b_first, sfb_first, b_tot, sfb_tot = st().selftest_silu_mul_quant(*shape, 0).tolist()
    assert b_tot > 0 and sfb_tot > 0
    assert b_mis == 0 and sfb_mis == 0, f"first b byte {int(b_first)}, first sfb byte {int(sfb_first)}"


@pytest.mark.parametrize("shape", [
    (64, 8, 256, 4, 64),        # cap below the likely per-expert load: overflow-heavy
    (128, 8, 2048, 128, 128),   # decode-like geometry with many experts
    (512, 8, 1024, 16, 512),
    (96, 4, 512, 8, 96),        # cap == T
], ids=shape_id)
def test_scatter_quant_is_byte_exact(shape):
    """Fused scatter + quant equals scatter -> quant_act on valid rows, with and without router
    weights, including EP-miss and over-capacity routings; every SFB byte is nonzero."""
    b_mis, s_mis, b_mis_w, s_mis_w, sfb_zero, valid = st().selftest_scatter_quant(*shape, 0).tolist()
    assert valid > 0
    assert (b_mis, s_mis, b_mis_w, s_mis_w, sfb_zero) == (0, 0, 0, 0, 0)
