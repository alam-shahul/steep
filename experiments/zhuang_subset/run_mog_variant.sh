#!/bin/bash
#SBATCH -p gpu-small
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 50G
#SBATCH --gres=gpu:1
#SBATCH -t 1-00:00:00
#SBATCH --array=0-14
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Exploratory sweep for a new MoG variant: 5 ratios x 3 seeds.
# Coarser than the 9x5 grid the established methods use -- enough to rank the
# variants against each other before committing the full grid to a winner.
#
#   sbatch --export=ALL,SKETCHER=mog_edge_joint run_mog_variant.sh
#
# gpu-small on purpose: the cost metrics have to stay comparable with the
# baselines, which were all measured there.

: "${SKETCHER:?set SKETCHER=<sketcher config name>}"

OUTDIR="/work/magroup/xinyuelu/steep/experiments/zhuang_subset/records/$SKETCHER"
mkdir -p "$OUTDIR"

ratios=(0.1 0.3 0.5 0.7 0.9)
seeds=(0 1 2)

task_id=$SLURM_ARRAY_TASK_ID
ratio=${ratios[$((task_id / 3))]}
seed=${seeds[$((task_id % 3))]}

LOGFILE="$OUTDIR/run_r${ratio}_s${seed}.log"
exec > "$LOGFILE" 2>&1

echo "Started at: $(date)"
echo "Sketcher: ${SKETCHER}   ratio: ${ratio}   seed: ${seed}"
echo

cd /work/magroup/xinyuelu/steep || exit 1

uv run --no-sync python scripts/evaluate.py \
  dataset=zhuang_subset \
  benchmark=zhuang \
  +sketcher="$SKETCHER" \
  sketcher.args.mog_args.retention_ratio="$ratio" \
  sketcher.args.random_seed="$seed" \
  trainer.args.epochs=100 \
  trainer.args.device=cuda \
  trainer.args.accelerator=gpu

echo
echo "Finished at: $(date)"
