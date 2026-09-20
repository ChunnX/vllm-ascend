#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Sequential validation only; never builds, installs, or selects unallocated cards.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash tools/run_dspark_eager_validation.sh [options]
  --stage cpu|ops|model|all   Default: all. all stops before model if ops fail.
  --devices ID,ID,ID,ID      Four allocated cards; defaults to ASCEND_RT_VISIBLE_DEVICES.
  --python PATH             Interpreter in the installed NPU environment (default: python3).
  --log-dir PATH            New result directory (default: unique directory under /tmp).
  --help                    Show this help.
Model stages require VLLM_TEST_QWEN36_MODEL and VLLM_TEST_DSPARK_MODEL.
The ops stage uses only the first selected card. Model tests run sequentially at TP=4.
EOF
}

stage=all
devices=${ASCEND_RT_VISIBLE_DEVICES:-}
python_bin=python3
log_dir=
while (($#)); do
    case "$1" in
        --help) usage; exit 0 ;;
        --stage|--devices|--python|--log-dir)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --stage) stage=$2 ;;
                --devices) devices=$2 ;;
                --python) python_bin=$2 ;;
                --log-dir) log_dir=$2 ;;
            esac
            shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
case "$stage" in cpu|ops|model|all) ;; *) usage >&2; exit 2 ;; esac
command -v "$python_bin" >/dev/null
if [[ "$stage" != cpu ]]; then
    "$python_bin" - "$devices" <<'PY'
import sys
cards = sys.argv[1].split(",")
if len(cards) != 4 or len(set(cards)) != 4 or not all(c.isdigit() for c in cards):
    sys.exit("Specify exactly four distinct allocated card IDs with --devices")
PY
fi
if [[ "$stage" == model || "$stage" == all ]]; then
    : "${VLLM_TEST_QWEN36_MODEL:?Set the target checkpoint path}"
    : "${VLLM_TEST_DSPARK_MODEL:?Set the DSpark checkpoint path}"
fi
if [[ -z "$log_dir" ]]; then
    log_dir=$(mktemp -d "${TMPDIR:-/tmp}/dspark-eager.XXXXXX")
else
    # Do not overwrite a previous test receipt.
    mkdir "$log_dir"
fi
log_dir=$(cd "$log_dir" && pwd)
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTEST_ADDOPTS=
printf 'Logs: %s\n' "$log_dir"
{
    date -u
    git rev-parse HEAD
    git status --short
    "$python_bin" --version
    printf 'stage=%s devices=%s\n' "$stage" "$devices"
} > "$log_dir/source.txt"
printf 'stage\texit_code\n' > "$log_dir/summary.tsv"

run_stage() {
    local name=$1
    shift
    local result=0
    "$@" > "$log_dir/$name.log" 2>&1 || result=$?
    cat "$log_dir/$name.log"
    printf '%s\t%s\n' "$name" "$result" >> "$log_dir/summary.tsv"
    return "$result"
}

if [[ "$stage" == cpu || "$stage" == all ]]; then
    run_stage cpu env TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$python_bin" -m pytest \
        --noconftest -o addopts='' -sv --tb=short --junitxml="$log_dir/cpu.xml" \
        tests/ut/spec_decode/test_eager_survival_verification.py \
        tests/ut/ops/test_dcut_cpu_reference.py || exit 1
fi
if [[ "$stage" != cpu ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=${devices%%,*}
    run_stage environment "$python_bin" tools/inspect_dspark_eager_environment.py || exit 1
fi
if [[ "$stage" == ops || "$stage" == all ]]; then
    # Keep running all five cases after an assertion failure; do not use -x.
    if ! run_stage ops "$python_bin" -m pytest --noconftest -o addopts='' \
        -sv --tb=long --junitxml="$log_dir/ops.xml" \
        tests/e2e/nightly/single_node/ops/singlecard_ops/test_eager_gdn_varlen.py; then
        printf 'Operator gate FAILED; model stage not started. See %s\n' "$log_dir/summary.tsv"
        exit 1
    fi
fi
if [[ "$stage" == model || "$stage" == all ]]; then
    export ASCEND_RT_VISIBLE_DEVICES=$devices
    export VLLM_LOGGING_LEVEL=DEBUG
    # Model tests retain complete child-process logs alongside this stage log.
    run_stage model "$python_bin" -m pytest --noconftest -o addopts='' \
        -sv --tb=long --basetemp="$log_dir/model-processes" --junitxml="$log_dir/model.xml" \
        tests/e2e/nightly/single_node/spec_decode/test_qwen36_dspark_eager_survival.py || exit 1
fi
printf 'Requested stages completed. Receipt: %s\n' "$log_dir/summary.tsv"
