#!/usr/bin/env bash
set -euo pipefail

sketcher=""
extra_overrides=()

ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# seeds=(0 1 2 3 4)
seeds=(0)

usage() {
    cat <<USAGE
Usage:
  scripts/run_zhuang_sweep.sh sketcher=NAME [HYDRA_OVERRIDES...]

Required:
  sketcher=NAME

All other arguments are passed through to Hydra unchanged.
The default sweep is 9 retention ratios x 5 seeds, so submit with --array 0-44.
USAGE
}

for arg in "$@"; do
    case "$arg" in
        sketcher=*)
            sketcher="${arg#sketcher=}"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            extra_overrides+=("$arg")
            ;;
    esac
done

if [[ -z "$sketcher" ]]; then
    echo "Error: sketcher=NAME is required." >&2
    usage >&2
    exit 2
fi

task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required; submit with --array 0-44}"
seed_count=${#seeds[@]}
ratio_idx=$((task_id / seed_count))
seed_idx=$((task_id % seed_count))

if (( ratio_idx >= ${#ratios[@]} )); then
    max_task=$((${#ratios[@]} * seed_count - 1))
    echo "Error: task id $task_id is out of range; expected 0 through $max_task." >&2
    exit 2
fi

ratio="${ratios[$ratio_idx]}"
seed="${seeds[$seed_idx]}"

retention_key="sketcher.args.retention_ratio"
case "$sketcher" in
    mog|mog_edge|mog_node)
        retention_key="sketcher.args.mog_args.retention_ratio"
        ;;
esac

echo "Zhuang sweep task: task_id=$task_id sketcher=$sketcher ratio=$ratio seed=$seed"

python scripts/evaluate.py \
    dataset=zhuang_subset \
    benchmark=zhuang \
    "+sketcher=$sketcher" \
    "$retention_key=$ratio" \
    "sketcher.args.random_seed=$seed" \
    trainer.args.epochs=100 \
    trainer.args.device=cuda \
    trainer.args.accelerator=gpu \
    trainer.args.run_wandb=True \
    benchmark.args.resume_from_checkpoint=False \
    "${extra_overrides[@]}"
