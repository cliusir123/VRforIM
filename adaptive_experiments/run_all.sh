#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
mkdir -p results/runs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TQDM_DISABLE=1
DATASETS=${DATASETS:-"ripley pima heart german boston pendulum"}
SCALE=()
[ -n "${COV_SCALE:-}" ] && SCALE=(--cov-scale "$COV_SCALE")

run() {
    local dev=$1 ds=$2 name=$3; shift 3
    local ckpt=()
    [ -f "checkpoints/$ds.npz" ] && ckpt=(--proposal-from "checkpoints/$ds.npz")
    JAX_PLATFORMS=$dev python3 run_adaptive.py --dataset "$ds" "${ckpt[@]}" "${SCALE[@]}" "$@" \
        --out "results/runs/$name.npz" > "results/runs/$name.log" 2>&1 &
}

for ds in $DATASETS; do
    case $ds in
    ripley|pima|heart|german)
        run cpu "$ds" "table3_${ds}"            --model logistic --mode after
        run cpu "$ds" "table4_${ds}_B50_l20"    --model logistic --mode during --batch-size 50 --n-batches 20
        run cpu "$ds" "table4_${ds}_B500_l200"  --model logistic --mode during --batch-size 500 --n-batches 200 ;;
    boston|pendulum)
        run cuda "$ds" "table5_${ds}_B50_l20"   --model gp --mode during --batch-size 50 --n-batches 20 ;;
    esac
done
wait
echo "all runs finished; outputs in results/runs/"
