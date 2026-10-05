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
"""Pure-torch reference for the weight format `compress` consumes.

This is the executable spec a checkpoint producer follows. A weight W[E, M, K] is

  1. pruned to paired 4:8 along K: in every chunk of 8 elements (4 adjacent pairs) exactly 2
     pairs are kept, so the zeros come in aligned pairs of fp4 values;
  2. quantized to NVFP4 with one UE4M3 scale per 32 consecutive K elements:
         s   = ue4m3(amax(|W| over the block) / 6)
         fp4 = e2m1(W / float(s))                       (round to nearest, saturating to +-6)
  3. packed two e2m1 codes per byte, the lower K index in the low nibble.

The kernel additionally multiplies by a per-expert fp32 `alpha` in the epilogue, which is where
per-tensor (global) weight and activation scales are folded in.
"""

import torch

SF_BLOCK = 32   # K elements per scale: the sparse NVFP4 MMA's SFVecSize

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)   # magnitude of code 0..7; bit 3 is sign


def prune_paired_4of8(w):
    """Zero 2 of every 4 adjacent K-pairs, keeping the 2 pairs with the largest |w| sum.

    The pruning unit is a pair of fp4 values, which is what the SM100 sparse compressor and
    metadata encode. An element-wise 4-of-8 pattern is not representable and compress() would
    silently produce a wrong operand.
    """
    E, M, K = w.shape
    assert K % 8 == 0, K
    wp = w.reshape(E, M, K // 8, 4, 2)
    pair_mag = wp.abs().sum(-1)
    keep = pair_mag.argsort(dim=-1, descending=True)[..., :2]
    mask = torch.zeros_like(pair_mag, dtype=torch.bool).scatter_(-1, keep, True)
    return (wp * mask.unsqueeze(-1)).reshape(E, M, K)


def is_paired_4of8(w):
    """True iff every 8-element K chunk has at most 2 nonzero pairs."""
    E, M, K = w.shape
    nz_pairs = (w.reshape(E, M, K // 8, 4, 2) != 0).any(-1).sum(-1)
    return bool((nz_pairs <= 2).all())


def round_to_e2m1(x):
    """Round to the nearest e2m1 value, sign preserved, saturating at +-6."""
    sign = torch.sign(x)
    a = x.abs()
    out = torch.zeros_like(a)
    for thr, val in [(0.25, 0.5), (0.75, 1.0), (1.25, 1.5), (1.75, 2.0),
                     (2.5, 3.0), (3.5, 4.0), (5.0, 6.0)]:
        out = torch.where(a > thr, torch.full_like(a, val), out)
    return sign * out


def pack_e2m1(vals):
    """e2m1-valued floats [..., K] -> uint8 [..., K/2], lower K index in the low nibble."""
    a = vals.abs()
    code = torch.zeros(a.shape, dtype=torch.uint8, device=a.device)
    for c, v in enumerate(E2M1_VALUES):
        code = torch.where(a == v, torch.full_like(code, c), code)
    sign = ((vals < 0) & (code != 0)).to(torch.uint8) * 8   # never emit -0
    nib = code | sign
    return (nib[..., 0::2] | (nib[..., 1::2] << 4)).to(torch.uint8).contiguous()


def unpack_e2m1(packed):
    """uint8 [..., K/2] -> float32 [..., K]."""
    lut = torch.tensor(E2M1_VALUES, dtype=torch.float32, device=packed.device)
    nib = torch.stack([packed & 0xF, packed >> 4], dim=-1).reshape(*packed.shape[:-1], -1)
    mag = lut[(nib & 0x7).long()]
    return torch.where((nib & 0x8) != 0, -mag, mag)


def quantize_weight(w):
    """Paired-4:8-sparse w[E, M, K] -> (w_packed uint8 [E, M, K/2], w_blockscale uint8 [E, M, K/32],
    w_dequant float32 [E, M, K]).

    w_blockscale holds raw UE4M3 bytes in natural [E, M, K/32] order; compress() swizzles them.
    w_dequant is fp4 * scale, the weight the kernel effectively multiplies (up to alpha).
    """
    E, M, K = w.shape
    assert K % SF_BLOCK == 0, K
    wb = w.float().reshape(E, M, K // SF_BLOCK, SF_BLOCK)
    amax = wb.abs().amax(-1, keepdim=True)
    s_e4m3 = (amax / 6.0).clamp(min=1e-12, max=448.0).to(torch.float8_e4m3fn)
    s = s_e4m3.float()
    fp4 = round_to_e2m1(wb / s)
    w_dequant = (fp4 * s).reshape(E, M, K)
    w_packed = pack_e2m1(fp4.reshape(E, M, K))
    w_blockscale = s_e4m3.squeeze(-1).contiguous().view(torch.uint8)
    return w_packed, w_blockscale, w_dequant


def dequantize_weight(w_packed, w_blockscale):
    """Inverse of quantize_weight's packing: (uint8 [E, M, K/2], uint8 [E, M, K/32]) -> float32."""
    fp4 = unpack_e2m1(w_packed)
    E, M, K = fp4.shape
    s = w_blockscale.view(torch.float8_e4m3fn).float()
    return (fp4.reshape(E, M, K // SF_BLOCK, SF_BLOCK) * s.unsqueeze(-1)).reshape(E, M, K)
