#!/bin/bash
# =============================================================================
# Phase 12 — Basin Water Balance Calibration (single node)
#
# Calibrates reservoir routing timescales (glaciated) and two-bucket
# parameters (non-glaciated) against GRDC observed discharge at Hunza
# outlet (station 2392800, Dainyor bridge) using differential_evolution + KGE.
#
# Prerequisites:
#   - Phase 10 array jobs must have completed (model_diagnostics.nc present)
#   - GRDC file downloaded to ~/data/grdc/2392800_Q_Day.Cmd.txt
#     (or omit --grdc_file to use synthetic data for testing)
#   - HydroBASINS Asia ZIP present at ~/oggm_dev_project/
#
# Submit:
#   sbatch local/sbatch_templates/12_basin_calibration.sh
#
# Or after Phase 10 array finishes:
#   sbatch --dependency=afterok:${PHASE10_JOB_ID} \
#          local/sbatch_templates/12_basin_calibration.sh
# =============================================================================

#SBATCH --job-name=oggm_wbc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/phase12_%j.out
#SBATCH --error=logs/phase12_%j.err
#SBATCH --partition=main

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oggm_env6

REPO=/home/users/mshafeeque/oggm_dev_project/oggmHydro
WORKDIR=~/hydro_analysis/phase10/hunza_run
OUTDIR=~/hydro_analysis/phase12
HYDROBASINS_DIR=~/oggm_dev_project
GRDC_FILE=~/data/grdc/2392800_Q_Day.Cmd.txt

mkdir -p logs "$OUTDIR"

cd "$REPO"

echo "=== Phase 12 Basin Water Balance Calibration  $(date) ==="
echo "    workdir     : $WORKDIR"
echo "    output_dir  : $OUTDIR"
echo "    grdc_file   : $GRDC_FILE"

# Build the grdc_file argument conditionally
GRDC_ARG=""
if [ -f "$GRDC_FILE" ]; then
    GRDC_ARG="--grdc_file $GRDC_FILE"
else
    echo "    WARNING: GRDC file not found — using synthetic discharge!"
fi

python local/run_phase12_calibration.py \
    --workdir "$WORKDIR" \
    --output_dir "$OUTDIR" \
    $GRDC_ARG \
    --filesuffix "_hunza" \
    --hydrobasins_dir "$HYDROBASINS_DIR" \
    --hydrobasins_level 8 \
    --bbox 73.5 35.5 76.5 37.5 \
    --ys 1982 \
    --ye 2016 \
    --metric KGE \
    --method differential_evolution \
    --seed 42

EXIT_CODE=$?
echo "=== Phase 12 exit code: $EXIT_CODE  $(date) ==="
exit $EXIT_CODE
