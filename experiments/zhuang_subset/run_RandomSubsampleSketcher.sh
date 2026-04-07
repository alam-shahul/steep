#!/bin/bash
#SBATCH -p gpu-large
#SBATCH -n 1
#SBATCH -c 20
#SBATCH --mem 150G
#SBATCH -t 1-23:00:00

module load cuda-12.2
export LD_LIBRARY_PATH=/work/magroup/xinyuelu/steep/.venv/lib/python3.12/site-packages/torch/lib:$LD_LIBRARY_PATH

for ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7; do
  for seed in 0 1 2 3 4; do
    uv run python scripts/evaluate.py \
      dataset=zhuang_subset \
      benchmark=zhuang \
      +sketcher=random_subsample \
      sketcher.args.retention_ratio=$ratio \
      sketcher.args.random_seed=$seed \
      trainer.args.epochs=100
      >> output_random_subsample.txt 2>&1
  done
done
