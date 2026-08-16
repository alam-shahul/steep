#!/bin/bash
#SBATCH -p gpu-small
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 50G
#SBATCH --gres=gpu:1
#SBATCH -t 1-23:00:00
#SBATCH --array=0-44   # 9 ratios x 5 seeds = 45 jobs
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Edge-mode MoG using the pre-merge sparsifier (steep/models/_sparsify_old.py),
# selected via `sketcher.args.mog_impl=old` in the mog_edge_old config.

name=mog_edge_old
OUTDIR="/work/magroup/xinyuelu/steep/experiments/zhuang_subset/records/$name"
mkdir -p "$OUTDIR"

ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
seeds=(0 1 2 3 4)

task_id=$SLURM_ARRAY_TASK_ID
ratio_idx=$((task_id / 5))
seed_idx=$((task_id % 5))

ratio=${ratios[$ratio_idx]}
seed=${seeds[$seed_idx]}

LOGFILE="$OUTDIR/run_r${ratio}_s${seed}.log"

# overwrite logfile each time, and send all following stdout/stderr into it
exec > "$LOGFILE" 2>&1

echo "Started at: $(date)"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Sketcher: ${name}"
echo "Retention ratio: ${ratio}"
echo "Random seed: ${seed}"
echo

cd /work/magroup/xinyuelu/steep || exit 1

uv run --no-sync python scripts/evaluate.py \
  dataset=zhuang_subset \
  benchmark=zhuang \
  +sketcher="$name" \
  sketcher.args.mog_args.retention_ratio="$ratio" \
  sketcher.args.random_seed="$seed" \
  trainer.args.epochs=100 \
  trainer.args.device=cuda \
  trainer.args.accelerator=gpu

echo
echo "Finished at: $(date)"
