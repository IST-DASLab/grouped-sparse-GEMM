#!/usr/bin/env bash
# End-to-end serving comparison on one 8-GPU node: paired48 sparse NVFP4 vs the FlashInfer dense
# NVFP4 MoE backends, in the TP layouts where those backends run (TP+EP and plain TP).
#
# Arms (name = results prefix):
#   A    paired48 sparse NVFP4 (ours)        TP8 + EP  backend auto (-> PAIRED48_NVFP4)
#   F    A + fused dispatch/combine          TP8 + EP  via bench/vllm_plugin + FUSED_KERNELS
#   V    A + fused dispatch/combine          TP8 + EP  via the patched vLLM in VLLM_SRC
#   V2   V with a second vLLM / kernel build TP8 + EP  VLLM_SRC2 + FUSED_KERNELS2
#   T    dense, FlashInfer TRTLLM-gen        TP8 + EP  flashinfer_trtllm
#   U    dense, FlashInfer CUTLASS           TP8 + EP  flashinfer_cutlass
#   S    dense, FlashInfer CuteDSL           TP8 + EP  flashinfer_cutedsl (contiguous)
#   B    dense, FlashInfer CuteDSL batched   TP8 + EP  flashinfer_cutedsl_batched
#   P    dense, vLLM default                 TP8       backend auto
#   DA   ours                                DP8 + EP  deepep_low_latency
#   DB   dense, CuteDSL batched              DP8 + EP  deepep_low_latency
#   DT2  dense, TRTLLM-gen                   DP8 + EP  flashinfer_nvlink_two_sided
#   DT1  dense, TRTLLM-gen                   DP8 + EP  flashinfer_nvlink_one_sided
#   DTH  dense, TRTLLM-gen                   DP8 + EP  deepep_high_throughput
#   DTA  dense, TRTLLM-gen                   DP8 + EP  allgather_reducescatter
#   DS2  ours, token-order experts           DP8 + EP  flashinfer_nvlink_two_sided, bf16 dispatch
#   DS2Q DS2 with FP4 dispatch (quantized before the all2all)
#   DS1  ours, token-order experts           DP8 + EP  flashinfer_nvlink_one_sided
#   DSH  ours, token-order experts           DP8 + EP  deepep_high_throughput
#   DSA  ours, token-order experts           DP8 + EP  allgather_reducescatter
#
# Every arm uses the same serving flags, KV dtype (bf16, pinned), workload and node, back to back
# in one job, so arm ratios cancel node-to-node drift. GREEDY_N > 0 also records the greedy
# completions of the first GREEDY_N GSM8K test questions, sent one at a time (fixed batch
# composition), so two arms of the same checkpoint can be compared token for token.
#
# Usage (inside an allocation with 8 GPUs):
#   ARMS="A T U S B P" OUT_DIR=/path/results bash bench/e2e_tp_serving.sh
# Knobs: FUSED_KERNELS (kernel build for F/V/DA), VLLM_SRC (patched vLLM tree for V/DA), DEEPEP
#        (DeepEP build for the DP arms), GREEDY_N, A_MODEL, DENSE_MODEL, CONCURRENCY,
#        NUM_PROMPTS, ISO_IN/ISO_OUT, PREFILL_CONC, MAX_NUM_BATCHED_TOKENS, GPU_UTIL,
#        MAX_MODEL_LEN, VENV.
set -uo pipefail

# Python environment with vllm, flashinfer and paired_nvfp4_kernels (default: the active one).
VENV=${VENV:-$(python3 -c 'import sys; print(sys.prefix)')}
A_MODEL=${A_MODEL:-ISTA-DASLab/Kimi-K2.5-P48NVFP4-MoESQ}
DENSE_MODEL=${DENSE_MODEL:-nvidia/Kimi-K2.5-NVFP4}
ARMS=${ARMS:-"A T U S B P"}
# Only needed by the arms that use them (F/V/V2/DA/DS* and the DP arms); checked per arm below.
FUSED_KERNELS=${FUSED_KERNELS:-}
VLLM_SRC=${VLLM_SRC:-}
DEEPEP=${DEEPEP:-}
GREEDY_N=${GREEDY_N:-0}
VLLM_SRC2=${VLLM_SRC2:-$VLLM_SRC}
FUSED_KERNELS2=${FUSED_KERNELS2:-$FUSED_KERNELS}
GSM8K_LIMIT=${GSM8K_LIMIT:-0}   # > 0: GSM8K 5-shot (lm-eval) on the first N test questions per arm
EXTRA_SERVE_ARGS=${EXTRA_SERVE_ARGS:-}   # appended to every vllm serve
PLUGIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vllm_plugin"
OUT_DIR=${OUT_DIR:?set OUT_DIR}
CONCURRENCY=${CONCURRENCY:-"16 64 256 512"}
NUM_PROMPTS=${NUM_PROMPTS:-500}
ISO_IN=${ISO_IN:-1024}; ISO_OUT=${ISO_OUT:-256}
PREFILL_IN=${PREFILL_IN:-4096}; PREFILL_CONC=${PREFILL_CONC:-"64"}; PREFILL_PROMPTS=${PREFILL_PROMPTS:-500}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4608}
GPU_UTIL=${GPU_UTIL:-0.90}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-bfloat16}
HOST=127.0.0.1; PORT=${PORT:-8123}
READY_TIMEOUT=${READY_TIMEOUT:-5400}

# Keep every compile / JIT / autotune cache off the (quota-limited) home directory: a full home
# makes torch.compile's cache writes fail and the server dies during its profile run.
CACHE=${CACHE:-$OUT_DIR/cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$CACHE/vllm}
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$CACHE}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE/triton}
export VLLM_NO_USAGE_STATS=1
# DeepEP-LL asserts nvshmem_qp_depth >= 2 * (dispatch tokens per rank + 1) on every dispatch;
# the default depth (1024) only admits MAX_NUM_BATCHED_TOKENS <= 511.
_qp=1024; while (( _qp < 2 * (MAX_NUM_BATCHED_TOKENS + 1) )); do _qp=$(( _qp * 2 )); done
export NVSHMEM_QP_DEPTH=${NVSHMEM_QP_DEPTH:-$_qp}
export PATH="$VENV/bin:$PATH"
# FlashInfer counts as available only with nvcc on PATH (no flashinfer-cubin wheel for this
# version); its TRTLLM-gen cubins are fetched at runtime and JIT modules land in ~/.cache/flashinfer.
command -v nvcc >/dev/null || echo "warning: nvcc not on PATH; FlashInfer arms may be unavailable" >&2
for _arm in $ARMS; do
  case "$_arm" in
    F) : "${FUSED_KERNELS:?arm F needs FUSED_KERNELS}" ;;
    V) : "${VLLM_SRC:?arm V needs VLLM_SRC}" "${FUSED_KERNELS:?arm V needs FUSED_KERNELS}" ;;
    V2) : "${VLLM_SRC2:?arm V2 needs VLLM_SRC2}" "${FUSED_KERNELS2:?arm V2 needs FUSED_KERNELS2}" ;;
    DA|DS*) : "${VLLM_SRC:?arm $_arm needs VLLM_SRC}" "${FUSED_KERNELS:?arm $_arm needs FUSED_KERNELS}" "${DEEPEP:?arm $_arm needs DEEPEP}" ;;
    D*) : "${DEEPEP:?arm $_arm needs DEEPEP}" ;;
  esac
done
mkdir -p "$OUT_DIR"
cd /tmp  # never import vllm from a source tree on sys.path

arm_model() { case "$1" in A|F|V|V2|DA|DS*) echo "$A_MODEL" ;; *) echo "$DENSE_MODEL" ;; esac; }
arm_pythonpath() {
  case "$1" in
    F) echo "$FUSED_KERNELS:$PLUGIN_DIR" ;;
    V) echo "$VLLM_SRC:$FUSED_KERNELS" ;;
    V2) echo "$VLLM_SRC2:$FUSED_KERNELS2" ;;
    DA|DS*) echo "$VLLM_SRC:$FUSED_KERNELS:$DEEPEP" ;;
    D*) echo "$DEEPEP" ;;
    *) echo "" ;;
  esac
}
arm_env() {  # extra environment for one arm
  case "$1" in
    DS2) echo "VLLM_PAIRED48_PREQUANT_DISPATCH=0" ;;
    *) echo "" ;;
  esac
}
arm_parallel() {
  local dp="--data-parallel-size 8 --enable-expert-parallel --api-server-count 1 --all2all-backend"
  case "$1" in
    P) echo "--tensor-parallel-size 8" ;;
    DA|DB) echo "$dp deepep_low_latency" ;;
    DT2|DS2|DS2Q) echo "$dp flashinfer_nvlink_two_sided" ;;
    DT1|DS1) echo "$dp flashinfer_nvlink_one_sided" ;;
    DTH|DSH) echo "$dp deepep_high_throughput" ;;
    DTA|DSA) echo "$dp allgather_reducescatter" ;;
    *) echo "--tensor-parallel-size 8 --enable-expert-parallel" ;;
  esac
}
arm_backend() {
  case "$1" in
    A|F|V|V2|P|DA|DS*) echo auto ;; T|DT*) echo flashinfer_trtllm ;; U) echo flashinfer_cutlass ;;
    S) echo flashinfer_cutedsl ;; B|DB) echo flashinfer_cutedsl_batched ;;
  esac
}

greedy() {  # <arm>: first GREEDY_N GSM8K test questions, one request at a time, temperature 0
  HF_DATASETS_OFFLINE=1 "$VENV/bin/python" - "$HOST" "$PORT" "$(arm_model "$1")" "$GREEDY_N" \
      "$OUT_DIR/${1}_greedy.jsonl" <<'PY'
import json, sys, urllib.request
import datasets
host, port, model, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
qs = datasets.load_dataset("openai/gsm8k", "main", split="test")["question"][:n]
with open(out, "w") as f:
    for i, q in enumerate(qs):
        body = json.dumps({"model": model, "prompt": f"Question: {q}\nAnswer:", "max_tokens": 128,
                           "temperature": 0, "seed": 0}).encode()
        req = urllib.request.Request(f"http://{host}:{port}/v1/completions", body,
                                     {"Content-Type": "application/json"})
        text = json.load(urllib.request.urlopen(req))["choices"][0]["text"]
        f.write(json.dumps({"i": i, "text": text}) + "\n")
PY
}

{
  echo "date: $(date -u +%FT%TZ)  host: $(hostname)"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
  "$VENV/bin/python" -c "import vllm, flashinfer, torch, paired_nvfp4_kernels as p; print('vllm', vllm.__version__, 'flashinfer', flashinfer.__version__, 'torch', torch.__version__, 'paired_nvfp4_kernels', getattr(p, '__version__', '?'))" 2>/dev/null
  echo "ARMS=$ARMS CONCURRENCY=$CONCURRENCY NUM_PROMPTS=$NUM_PROMPTS ISO=${ISO_IN}x${ISO_OUT} PREFILL=${PREFILL_IN}x1@{$PREFILL_CONC}"
  echo "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN=$MAX_MODEL_LEN GPU_UTIL=$GPU_UTIL KV_CACHE_DTYPE=$KV_CACHE_DTYPE"
  echo "A_MODEL=$A_MODEL"; echo "DENSE_MODEL=$DENSE_MODEL"
  echo "FUSED_KERNELS2=$FUSED_KERNELS2 VLLM_SRC2=$VLLM_SRC2"
  echo "FUSED_KERNELS=$FUSED_KERNELS VLLM_SRC=$VLLM_SRC DEEPEP=$DEEPEP NVSHMEM_QP_DEPTH=$NVSHMEM_QP_DEPTH"
} | tee "$OUT_DIR/meta.txt"

SERVER_PID=""
cleanup() {
  [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null
  sleep 5; pkill -f "vllm serve" 2>/dev/null; sleep 10; SERVER_PID=""
}
trap cleanup EXIT

wait_ready() {
  local t=0
  while (( t < READY_TIMEOUT )); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  [FAIL] server exited early"; return 1; }
    [[ "$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health")" == 200 ]] && return 0
    sleep 10; t=$((t + 10))
  done
  echo "  [FAIL] not ready in ${READY_TIMEOUT}s"; return 1
}

bench_serve() {  # <arm> <tag> <ilen> <olen> <num_prompts> <concurrency>
  local arm=$1 tag=$2
  vllm bench serve --backend openai --host "$HOST" --port "$PORT" --model "$(arm_model "$arm")" \
    --trust-remote-code --max-concurrency "$6" --request-rate inf \
    --dataset-name random --random-input-len "$3" --random-output-len "$4" --ignore-eos \
    --num-prompts "$5" --seed 0 \
    --percentile-metrics "ttft,tpot,itl,e2el" --metric-percentiles "50,99" \
    --save-result --result-filename "$OUT_DIR/${arm}_${tag}.json" \
    >"$OUT_DIR/${arm}_${tag}.log" 2>&1 || echo "  [warn] $arm/$tag nonzero exit"
}

for arm in $ARMS; do
  slog="$OUT_DIR/${arm}_server.log"
  echo "========== arm $arm: $(arm_backend "$arm") / $(arm_parallel "$arm") =========="
  # shellcheck disable=SC2046
  env $(arm_env "$arm") PYTHONPATH="$(arm_pythonpath "$arm")" vllm serve "$(arm_model "$arm")" --host "$HOST" --port "$PORT" --seed 0 --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_UTIL" --kv-cache-dtype "$KV_CACHE_DTYPE" \
    $(arm_parallel "$arm") --moe-backend "$(arm_backend "$arm")" $EXTRA_SERVE_ARGS >"$slog" 2>&1 &
  SERVER_PID=$!
  t0=$(date +%s)
  if ! wait_ready; then
    grep -m1 -A 30 -E "Traceback|Error" "$slog" | sed 's/^/  | /'; cleanup; continue
  fi
  echo "  ready after $(( $(date +%s) - t0 ))s"
  {
    grep -m1 "NvFp4 MoE backend" "$slog" || echo "(no nvfp4 backend line)"
    grep -m1 -E "PrepareAndFinalize" "$slog" || true
    grep -m1 -E "p48_fused_glue|fused dispatch plan and combine" "$slog" || true
    grep -m1 -E "FP4 all2all dispatch|keeping bf16 all2all" "$slog" || true
    grep -m1 "GPU KV cache size" "$slog" || true
    grep -m1 "Maximum concurrency" "$slog" || true
    # Output sanity: one greedy completion, so a broken backend cannot post a fast number.
    curl -s "http://$HOST:$PORT/v1/completions" -H 'Content-Type: application/json' \
      -d "{\"model\": \"$(arm_model "$arm")\", \"prompt\": \"The capital of France is\", \"max_tokens\": 16, \"temperature\": 0}" \
      | "$VENV/bin/python" -c "import json,sys; print('sanity:', repr(json.load(sys.stdin)['choices'][0]['text']))" \
      || echo "sanity: request failed"
  } | tee "$OUT_DIR/${arm}_info.txt"
  if (( GREEDY_N > 0 )); then
    greedy "$arm" && echo "  greedy: $(wc -l < "$OUT_DIR/${arm}_greedy.jsonl") completions" \
      || echo "  [warn] $arm greedy failed"
  fi
  if (( GSM8K_LIMIT > 0 )); then
    HF_DATASETS_OFFLINE=1 "$VENV/bin/python" -m lm_eval --model local-completions --tasks gsm8k \
      --model_args "model=$(arm_model "$arm"),base_url=http://$HOST:$PORT/v1/completions,num_concurrent=64,tokenizer=$(arm_model "$arm")" \
      --gen_kwargs "temperature=0,seed=42" --limit "$GSM8K_LIMIT" --trust_remote_code \
      --output_path "$OUT_DIR/${arm}_gsm8k" > "$OUT_DIR/${arm}_gsm8k.log" 2>&1 \
      || echo "  [warn] $arm gsm8k failed"
    echo "  gsm8k: $(grep -E 'strict-match' "$OUT_DIR/${arm}_gsm8k.log" | head -1)"
  fi
  bench_serve "$arm" warmup "$ISO_IN" 32 64 64  # absorbs first-request effects; not reported
  for c in $CONCURRENCY; do
    bench_serve "$arm" "conc$c" "$ISO_IN" "$ISO_OUT" "$NUM_PROMPTS" "$c"
    echo "  conc$c: $(grep -m1 'Output token throughput' "$OUT_DIR/${arm}_conc$c.log")"
  done
  for c in $PREFILL_CONC; do
    bench_serve "$arm" "prefill$c" "$PREFILL_IN" 1 "$PREFILL_PROMPTS" "$c"
    echo "  prefill$c: $(grep -m1 'Total [Tt]oken throughput' "$OUT_DIR/${arm}_prefill$c.log")"
  done
  cleanup
done

"$VENV/bin/python" - "$OUT_DIR" $ARMS <<'EOF'
import json, os, sys
out, arms = sys.argv[1], sys.argv[2:]
cells = sorted({f.split('_', 1)[1][:-5] for f in os.listdir(out)
                if f.endswith('.json') and not f.split('_', 1)[1].startswith('warmup')},
               key=lambda c: (c.startswith('prefill'), int(''.join(filter(str.isdigit, c)))))
def val(a, c):
    try:
        j = json.load(open(f'{out}/{a}_{c}.json'))
        return j['total_token_throughput' if c.startswith('prefill') else 'output_throughput']
    except Exception:
        return None
lines = ['# output tok/s (decode cells) / total tok/s (prefill cells)',
         'cell        ' + ''.join(f'{a:>10s}' for a in arms)]
for c in cells:
    lines.append(f'{c:12s}' + ''.join(f'{v:10.0f}' if (v := val(a, c)) else f'{"-":>10s}' for a in arms))
if 'A' in arms:
    lines.append('# A / arm (>1 = paired48 faster)')
    for c in cells:
        a = val('A', c)
        lines.append(f'{c:12s}' + ''.join(
            f'{a / v:10.3f}' if a and (v := val(x, c)) else f'{"-":>10s}' for x in arms))
def greedy(a):
    try:
        return [json.loads(l)['text'] for l in open(f'{out}/{a}_greedy.jsonl')]
    except Exception:
        return None
# Same checkpoint and same attention layout (TP8 vs DP8 shard attention differently).
for ref, others in (('A', ('F', 'V', 'V2')), ('V', ('V2',)), ('DA', ('DS2', 'DS1', 'DSH', 'DSA', 'DS2Q')),
                    ('DS2', ('DS2Q',))):
    ga = greedy(ref)
    for b in others:
        gb = greedy(b)
        if ga and gb:
            n = min(len(ga), len(gb))
            same = sum(x == y for x, y in zip(ga, gb))
            lines.append(f'# greedy {ref} vs {b}: {same}/{n} completions identical')
import glob
for a in arms:
    for f in glob.glob(f'{out}/{a}_gsm8k/**/results_*.json', recursive=True):
        r = json.load(open(f))['results']['gsm8k']
        lines.append(f"# gsm8k {a}: strict {r.get('exact_match,strict-match', float('nan')):.4f} "
                     f"flexible {r.get('exact_match,flexible-extract', float('nan')):.4f}")
open(f'{out}/SUMMARY.txt', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
EOF
