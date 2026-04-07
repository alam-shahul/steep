#!/bin/bash
#SBATCH -p gpu-small
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem 50G
#SBATCH -t 1-23:00:00

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
  for seed in 0 1 2 3 4; do
    uv run python scripts/evaluate.py \
      dataset=zhuang_subset \
      benchmark=zhuang \
      +sketcher=spatialhopper \
      sketcher.args.retention_ratio=$ratio \
      sketcher.args.random_seed=$seed \
      trainer.args.epochs=100
      >> output_spatialhopper.txt 2>&1
  done
done
