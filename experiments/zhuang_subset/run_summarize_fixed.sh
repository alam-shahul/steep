#!/bin/bash
#SBATCH -p cpu
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Aggregates every method under records/ into CSVs and edge/node plots.
# Submitted with --dependency=afterany:<mog_edge_fixed>:<mog_node_fixed> so it
# fires once both benchmark arrays have finished, however they finish.

cd /work/magroup/xinyuelu/steep || exit 1

OUTDIR=experiments/zhuang_subset/records
LOG="$OUTDIR/summarize_fixed.log"

exec > "$LOG" 2>&1
echo "Started at: $(date)"
echo

uv run --no-sync python scripts/summarize_benchmark.py \
  --output-csv "$OUTDIR/summary_fixed.csv" \
  --latest-only

echo
echo "Finished at: $(date)"
