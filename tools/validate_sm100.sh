#!/usr/bin/env bash
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
# Re-validates the SM100 build on a B200 after changes to the shared layers:
#   1. builds the baseline (default: v0.10.0, the first standalone release) and this checkout for
#      SM100, each into its own directory;
#   2. tools/ab_compare.py: every op, and group_mm under every tactic, must give byte-identical
#      output in both builds;
#   3. the pytest suite against this checkout's build (with the self-test ops).
#
# Usage (from the repository root, inside the environment whose CUDA torch you build with):
#     tools/validate_sm100.sh [baseline-ref]
# Environment: MAX_JOBS (default 2; SM100 CUTLASS TUs need ~10+ GB of host memory each),
# WORK (scratch directory, default a new mktemp dir), CUTLASS_DIR (default third_party/cutlass).

set -euo pipefail

BASE_REF=${1:-34368ae}   # "grouped-sparse-GEMM: paired-4:8 sparse NVFP4 grouped GEMM" (v0.10.0)
REPO=$(git rev-parse --show-toplevel)
WORK=${WORK:-$(mktemp -d)}
export MAX_JOBS=${MAX_JOBS:-2}
export CUTLASS_DIR=${CUTLASS_DIR:-$REPO/third_party/cutlass}
PY=${PYTHON:-python}

cap=$($PY -c "import torch; print('%d%d' % torch.cuda.get_device_capability())")
if [ "$cap" != "100" ]; then
  echo "needs an SM100 GPU (found compute capability $cap)" >&2
  exit 1
fi
echo "work dir: $WORK   baseline: $BASE_REF   MAX_JOBS=$MAX_JOBS"

build() {   # build <source dir> <tag> <build tests 0|1>
  local src=$1 tag=$2 tests=$3
  mkdir -p "$WORK/dist_$tag"
  ( cd "$src" && PAIRED_NVFP4_ARCHS=100a PAIRED_NVFP4_BUILD_TESTS=$tests \
      $PY -m pip wheel . --no-deps --no-build-isolation -w "$WORK/dist_$tag" ) > "$WORK/build_$tag.log" 2>&1 \
    || { echo "build $tag failed, see $WORK/build_$tag.log" >&2; exit 1; }
  $PY -m pip install -q --no-deps --target "$WORK/pkg_$tag" "$WORK"/dist_"$tag"/*.whl
}

echo "== building baseline ($BASE_REF)"
git -C "$REPO" worktree add -q --detach "$WORK/base_src" "$BASE_REF"
trap 'git -C "$REPO" worktree remove --force "$WORK/base_src" 2>/dev/null || true' EXIT
build "$WORK/base_src" base 0

echo "== building this checkout ($(git -C "$REPO" rev-parse --short HEAD))"
build "$REPO" new 1

echo "== byte-level A/B (tools/ab_compare.py)"
# Both builds must run the same tactic list; ab_compare enumerates it from the loaded build.
PYTHONPATH="$WORK/pkg_base" $PY "$REPO/tools/ab_compare.py" run "$WORK/a.pt"
PYTHONPATH="$WORK/pkg_new"  $PY "$REPO/tools/ab_compare.py" run "$WORK/b.pt"
$PY "$REPO/tools/ab_compare.py" compare "$WORK/a.pt" "$WORK/b.pt"

echo "== test suite"
( cd "$WORK" && PYTHONPATH="$WORK/pkg_new" $PY -m pytest "$REPO/tests" -q )

echo "SM100 validation passed"
