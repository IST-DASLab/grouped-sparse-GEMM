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

import pytest
import torch

import paired_nvfp4_kernels as pnk

requires_gpu = pytest.mark.skipif(
    not pnk.is_available(),
    reason=f"needs a GPU matching the loaded arch module (SM{pnk.loaded_arch()})",
)


def _has_selftest(name="selftest"):
    return hasattr(torch.ops, "_paired_nvfp4_test") and hasattr(torch.ops._paired_nvfp4_test, name)


requires_selftest = pytest.mark.skipif(
    not _has_selftest(), reason="built without PAIRED_NVFP4_BUILD_TESTS=1"
)

# (features M, max_n, K, E) shared by the GEMM-level tests; covers a single expert, non-pow2
# max_n and K, and an unaligned SFB layout (max_n not a multiple of 128).
GEMM_SHAPES = [
    (256, 128, 256, 1),
    (256, 256, 256, 2),
    (512, 128, 512, 4),
    (1024, 256, 512, 8),
    (768, 192, 1024, 4),
]


def shape_id(shape):
    return "x".join(str(v) for v in shape)
