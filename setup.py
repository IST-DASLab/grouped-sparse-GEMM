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
#
# Builds paired_nvfp4_kernels. setup.py imports torch, so build without isolation inside an
# environment that already has the CUDA-enabled torch you will run with:
#
#     git submodule update --init --depth 1
#     pip wheel . --no-deps --no-build-isolation -w dist
#
# Environment knobs
#   PAIRED_NVFP4_ARCHS        ';'-separated target archs, default "100a". One extension module
#                             (paired_nvfp4_kernels._C_sm<N>) is built per arch.
#   PAIRED_NVFP4_BUILD_TESTS  1 = also compile the self-test ops (torch.ops._paired_nvfp4_test.*)
#   PAIRED_NVFP4_PTXAS_V      1 = print per-kernel register / spill statistics
#   CUTLASS_DIR               CUTLASS checkout to build against, default third_party/cutlass
#   MAX_JOBS                  parallel nvcc jobs; >= 5 lets the tile variants compile concurrently

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
CSRC = os.path.join(HERE, "csrc")
VERSION = "0.13.0"

# Archs with a csrc/sm<N>/ configuration. Extend when porting (see docs/porting.md).
SUPPORTED_ARCHS = {"100a": 100, "120a": 120}

cutlass_dir = os.environ.get("CUTLASS_DIR", os.path.join(HERE, "third_party", "cutlass"))
cutlass_dir = os.path.abspath(cutlass_dir)
if not os.path.isfile(os.path.join(cutlass_dir, "include", "cutlass", "cutlass.h")):
    raise RuntimeError(
        f"CUTLASS not found at {cutlass_dir}. Run `git submodule update --init --depth 1` "
        "or set CUTLASS_DIR."
    )

archs = [a.strip() for a in os.environ.get("PAIRED_NVFP4_ARCHS", "100a").split(";") if a.strip()]
unknown = [a for a in archs if a not in SUPPORTED_ARCHS]
if unknown:
    raise RuntimeError(
        f"unsupported PAIRED_NVFP4_ARCHS entries {unknown}; supported: {sorted(SUPPORTED_ARCHS)}"
    )
build_tests = os.environ.get("PAIRED_NVFP4_BUILD_TESTS", "0") == "1"

# The override directory must come first: it shadows CUTLASS's stock SM100 and SM120 sparse GEMM
# kernel headers with their grouped variants.
include_dirs = [
    os.path.join(CSRC, "cutlass_overrides"),
    os.path.join(cutlass_dir, "include"),
    os.path.join(cutlass_dir, "tools", "util", "include"),
    CSRC,
]


def arch_sources(sm):
    arch_dir = os.path.join(CSRC, f"sm{sm}")
    srcs = [os.path.join(CSRC, "ops.cu")]
    srcs += sorted(os.path.join(arch_dir, f) for f in os.listdir(arch_dir) if f.endswith(".cu"))
    if build_tests:
        srcs.append(os.path.join(CSRC, "testing", "selftest.cu"))
    return srcs


def extension(arch):
    sm = SUPPORTED_ARCHS[arch]
    defines = [f"-DPAIRED_NVFP4_SM={sm}"]
    if sm == 100:
        # Grid dependency control in the CUTLASS GEMMs, for programmatic dependent launch
        # (csrc/pdl.cuh). The SM100 kernel override waits before its first dependent read.
        defines.append("-DCUTLASS_ENABLE_GDC_FOR_SM100=1")
    nvcc = [
        "-O3",
        "-std=c++17",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ] + defines
    if os.environ.get("PAIRED_NVFP4_PTXAS_V", "0") == "1":
        nvcc.append("-Xptxas=-v")
    return CUDAExtension(
        name=f"paired_nvfp4_kernels._C_sm{sm}",
        sources=arch_sources(sm),
        include_dirs=include_dirs,
        extra_compile_args={"cxx": ["-O3", "-std=c++17"] + defines, "nvcc": nvcc},
    )


setup(
    name="paired_nvfp4_kernels",
    version=VERSION,
    description="Paired-4:8 sparse NVFP4 (W4A4) grouped GEMM for MoE expert layers on Blackwell",
    long_description=open(os.path.join(HERE, "README.md")).read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    license_files=["LICENSE", "NOTICE"],
    url="https://github.com/IST-DASLab/grouped-sparse-GEMM",
    packages=["paired_nvfp4_kernels"],
    ext_modules=[extension(a) for a in archs],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
    python_requires=">=3.9",
)
