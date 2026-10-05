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
"""CPU checks of the pure-torch weight-format reference."""

import torch

from paired_nvfp4_kernels import reference as ref


def test_e2m1_pack_roundtrip():
    vals = torch.tensor(ref.E2M1_VALUES + tuple(-v for v in ref.E2M1_VALUES[1:]) + (0.0,))
    packed = ref.pack_e2m1(vals[None])
    assert packed.shape == (1, vals.numel() // 2)
    assert torch.equal(ref.unpack_e2m1(packed)[0], vals)


def test_e2m1_rounding_saturates_and_is_nearest():
    x = torch.tensor([0.2, 0.3, 0.8, 1.3, 1.8, 2.6, 3.6, 5.1, 100.0, -0.3, -7.0])
    want = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 6.0, -0.5, -6.0])
    assert torch.equal(ref.round_to_e2m1(x), want)


def test_prune_is_paired_4of8():
    torch.manual_seed(0)
    w = ref.prune_paired_4of8(torch.randn(2, 16, 64))
    assert ref.is_paired_4of8(w)
    assert (w == 0).float().mean().item() == 0.5
    assert not ref.is_paired_4of8(torch.randn(2, 16, 64))


def test_quantize_dequantize_consistent():
    torch.manual_seed(0)
    w = ref.prune_paired_4of8(torch.randn(2, 8, 128))
    w_packed, w_sf, w_dq = ref.quantize_weight(w)
    assert w_packed.shape == (2, 8, 64) and w_sf.shape == (2, 8, 128 // ref.SF_BLOCK)
    assert torch.equal(ref.dequantize_weight(w_packed, w_sf), w_dq)
    assert ref.is_paired_4of8(w_dq)
    rel = (w_dq - w).norm() / w.norm()
    assert rel < 0.2
