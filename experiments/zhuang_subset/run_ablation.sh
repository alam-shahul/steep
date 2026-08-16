#!/bin/bash
#SBATCH -p gpu-large
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 48G
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# MoG loss-term ablation, scored by a STAGATE trained on each sketch.
# Only NMI and edge purity are reported, so the partition does not matter for
# comparability (no timing or memory numbers come out of this job).

cd /work/magroup/xinyuelu/steep || exit 1
LOG=experiments/zhuang_subset/records/ablation_mog_loss.log
exec > "$LOG" 2>&1

echo "Started at: $(date)"
echo

uv run --no-sync python scripts/ablate_mog_loss.py \
  --num-slides 3 \
  --ratios 0.1 0.3 \
  --label-key subclass \
  --mog-epochs 50 \
  --stagate-epochs 100

echo
echo "Finished at: $(date)"
