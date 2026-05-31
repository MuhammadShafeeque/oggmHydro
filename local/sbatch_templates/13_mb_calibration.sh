#!/bin/bash
# =============================================================================
# Phase 13 — Per-glacier MB Calibration from Discharge (single node)
#
# Constrains TI model parameters (melt_f, prcp_fac) with both Hugonnet 2021
# geodetic MB and observed glacier-only discharge derived from GRDC total
# discharge minus Phase 12 modelled non-glaciated Q (residual method), or
# from a fixed glacier fraction (fraction method, default).
#
# Prerequisites:
#   - Phase 10 array jobs completed (model_diagnostics.nc per glacier)
#   - Phase 12 calibration completed (optional, for residual discharge method)
#   - GRDC file downloaded (or omit for synthetic data)
#   - Hugonnet 2021 geodetic MB CSV (optional; uses defaults if absent)
#
# Submit:
#   sbatch local/sbatch_templates/13_mb_calibration.sh
#
# Or chain after Phase 12:
#   sbatch --dependency=afterok:${PHASE12_JOB_ID} \
#          local/sbatch_templates/13_mb_calibration.sh
# =============================================================================

#SBATCH --job-name=oggm_mbc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/phase13_%j.out
#SBATCH --error=logs/phase13_%j.err
#SBATCH --partition=main

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate oggm_env6

REPO=/home/users/mshafeeque/oggm_dev_project/oggmHydro
WORKDIR=~/hydro_analysis/phase10/hunza_run
OUTDIR=~/hydro_analysis/phase13
GRDC_FILE=~/data/grdc/2392800_Q_Day.Cmd.txt
GEODETIC_MB=~/data/hugonnet2021_rgi60.csv
PHASE12_JSON=~/hydro_analysis/phase12/basin_calib_params.json

mkdir -p logs "$OUTDIR"

cd "$REPO"

echo "=== Phase 13 MB Calibration from Discharge  $(date) ==="
echo "    workdir          : $WORKDIR"
echo "    output_dir       : $OUTDIR"
echo "    grdc_file        : $GRDC_FILE"
echo "    geodetic_mb_file : $GEODETIC_MB"
echo "    phase12_json     : $PHASE12_JSON"

# Build optional arguments
GRDC_ARG=""
if [ -f "$GRDC_FILE" ]; then
    GRDC_ARG="--grdc_file $GRDC_FILE"
else
    echo "    WARNING: GRDC file not found — using synthetic discharge!"
fi

GEODETIC_ARG=""
if [ -f "$GEODETIC_MB" ]; then
    GEODETIC_ARG="--geodetic_mb_file $GEODETIC_MB"
else
    echo "    WARNING: Hugonnet MB CSV not found — no geodetic constraint."
fi

PHASE12_ARG=""
if [ -f "$PHASE12_JSON" ]; then
    PHASE12_ARG="--phase12_json $PHASE12_JSON"
else
    echo "    INFO: Phase 12 JSON not found — using fraction method for Q_gl."
fi

python local/run_phase13_mb_calib.py \
    --workdir "$WORKDIR" \
    --output_dir "$OUTDIR" \
    $GRDC_ARG \
    $GEODETIC_ARG \
    $PHASE12_ARG \
    --glacier_frac 0.65 \
    --ys 2000 \
    --ye 2019 \
    --w_mb 0.6 \
    --w_Q 0.4 \
    --representative_frac 0.8 \
    --nprocesses 8

EXIT_CODE=$?
echo "=== Phase 13 exit code: $EXIT_CODE  $(date) ==="
exit $EXIT_CODE
