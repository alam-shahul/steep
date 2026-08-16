#!/bin/bash
#SBATCH -p gpu-small
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 50G
#SBATCH --gres=gpu:1
#SBATCH -t 1-23:00:00
#SBATCH --array=0-44   # 9 ratios × 5 seeds = 45 jobs
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

name=random_subsample
OUTDIR="${STEEP_RECORDS_DIR:-experiments/zhuang_subset/records}/$name"
mkdir -p $OUTDIR

ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
seeds=(0 1 2 3 4)

task_id=$SLURM_ARRAY_TASK_ID
ratio_idx=$((task_id / 5))
seed_idx=$((task_id % 5))

ratio=${ratios[$ratio_idx]}
seed=${seeds[$seed_idx]}

LOGFILE=$OUTDIR/run_r${ratio}_s${seed}.log

uv run python scripts/evaluate.py \
  dataset=zhuang_subset \
  benchmark=zhuang \
  +sketcher=$name \
  sketcher.args.retention_ratio=$ratio \
  sketcher.args.random_seed=$seed \
  trainer.args.max_epochs=100 \
  trainer.args.accelerator=gpu \
  >> $LOGFILE 2>&1
