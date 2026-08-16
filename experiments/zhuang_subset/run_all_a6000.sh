#!/bin/bash
#SBATCH -p gpu-large
#SBATCH --gres=gpu:A6000:1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 50G
#SBATCH -t 1-00:00:00
#SBATCH --array=0-44
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Full 9 ratios x 5 seeds sweep for one sketcher, pinned to an A6000.
#
#   sbatch --export=ALL,SKETCHER=edge_hybrid,RATIO_ARG=retention_ratio run_all_a6000.sh
#   sbatch --export=ALL,SKETCHER=mog_edge_v3,RATIO_ARG=mog_args.retention_ratio run_all_a6000.sh
#
# gpu-large holds three node types (A100/A6000 on oven-0-13, A6000 on
# oven-0-17, H100 on oven-0-27), so the partition alone does NOT fix the
# hardware. `--gres=gpu:A6000:1` does, and there are 14 of them against the 7
# 2080Ti in gpu-small -- twice the concurrency on much faster cards.

: "${SKETCHER:?set SKETCHER=<sketcher config name>}"
: "${RATIO_ARG:=retention_ratio}"

OUTDIR="/work/magroup/xinyuelu/steep/experiments/zhuang_subset/records_a6000/$SKETCHER"
mkdir -p "$OUTDIR"

ratios=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
seeds=(0 1 2 3 4)

task_id=$SLURM_ARRAY_TASK_ID
ratio=${ratios[$((task_id / 5))]}
seed=${seeds[$((task_id % 5))]}

LOGFILE="$OUTDIR/run_r${ratio}_s${seed}.log"
exec > "$LOGFILE" 2>&1

echo "Started at: $(date)"
echo "Sketcher: ${SKETCHER}   ratio: ${ratio}   seed: ${seed}"
nvidia-smi --query-gpu=name --format=csv,noheader
echo

cd /work/magroup/xinyuelu/steep || exit 1

uv run --no-sync python scripts/evaluate.py \
  dataset=zhuang_subset \
  benchmark=zhuang \
  +sketcher="$SKETCHER" \
  "sketcher.args.${RATIO_ARG}=${ratio}" \
  sketcher.args.random_seed="$seed" \
  trainer.args.epochs=100 \
  trainer.args.device=cuda \
  trainer.args.accelerator=gpu

echo
echo "Finished at: $(date)"
