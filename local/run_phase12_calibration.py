"""Phase 12 — Basin Water Balance Calibration (Hunza basin).

Calibrates reservoir routing timescales (glaciated, 3-component) and
two-bucket parameters (non-glaciated) against observed gauge discharge at
the Hunza basin outlet (Dainyor bridge, GRDC station 2392800) using
differential_evolution + KGE.

Pre-requisite
-------------
  Phase 10 SLURM array must have completed:
    - OGGM working dir with per-glacier model_diagnostics.nc files
    - rgi_ids.npy in the output_dir

  HydroBASINS Asia ZIP must be present:
    ~/oggm_dev_project/hybas_lake_as_lev01-12_v1c.zip

  GRDC annual discharge file (optional):
    Download station 2392800 from https://www.bafg.de/GRDC/
    If omitted, synthetic observations are used (testing only).

Usage (cluster)
---------------
  cd /home/users/mshafeeque/oggm_dev_project/oggmHydro
  /home/users/mshafeeque/miniforge3/envs/oggm_env6/bin/python \\
      local/run_phase12_calibration.py \\
      --workdir ~/hydro_analysis/phase10/hunza_run \\
      --output_dir ~/hydro_analysis/phase12 \\
      [--grdc_file ~/data/grdc/2392800_Q_Day.Cmd.txt] \\
      [--filesuffix _hunza] \\
      [--ys 1982] [--ye 2016] \\
      [--metric KGE] [--cross_validate]

  Or via SLURM:
      sbatch local/sbatch_templates/12_basin_calibration.sh
"""

import argparse
import json
import logging
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# =============================================================================
# Defaults
# =============================================================================

_DEFAULT_WORKDIR    = os.path.expanduser('~/hydro_analysis/phase10/hunza_run')
_DEFAULT_OUTDIR     = os.path.expanduser('~/hydro_analysis/phase12')
_DEFAULT_FILESUFFIX = '_hunza'
_DEFAULT_BASINS_DIR = os.path.expanduser('~/oggm_dev_project')
_DEFAULT_YS         = 1982
_DEFAULT_YE         = 2016
_HUNZA_BBOX         = (73.5, 35.5, 76.5, 37.5)  # lon_min lat_min lon_max lat_max


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 12 — Calibrate Hunza basin water balance '
                    'against GRDC discharge')
    p.add_argument('--workdir', default=_DEFAULT_WORKDIR,
                   help='OGGM working directory with Phase 10 gdirs')
    p.add_argument('--output_dir', default=_DEFAULT_OUTDIR,
                   help='Output directory for JSON, figures, and log')
    p.add_argument('--grdc_file', default=None,
                   help='Path to GRDC file (ASCII or CSV). '
                        'Omit to use synthetic data (testing only).')
    p.add_argument('--filesuffix', default=_DEFAULT_FILESUFFIX,
                   help='model_diagnostics filesuffix from Phase 10')
    p.add_argument('--hydrobasins_dir', default=_DEFAULT_BASINS_DIR,
                   help='Directory containing hybas_lake_as_lev01-12_v1c.zip')
    p.add_argument('--hydrobasins_level', type=int, default=8)
    p.add_argument('--bbox', nargs=4, type=float,
                   default=list(_HUNZA_BBOX),
                   metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'))
    p.add_argument('--ys', type=int, default=_DEFAULT_YS,
                   help='Calibration start year (inclusive)')
    p.add_argument('--ye', type=int, default=_DEFAULT_YE,
                   help='Calibration end year (inclusive)')
    p.add_argument('--metric', default='KGE',
                   choices=['KGE', 'NSE', 'PBIAS'])
    p.add_argument('--method', default='differential_evolution',
                   choices=['differential_evolution', 'Nelder-Mead'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cross_validate', action='store_true',
                   help='Hold out last 25%% of years for validation')
    p.add_argument('--basin_prcp_fac', action='store_true',
                   help='Also calibrate a basin-wide precipitation factor')
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def _setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'phase12_calibration.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s  %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode='w'),
        ],
    )
    return log_path


def _load_gdirs(workdir, output_dir):
    """Load GlacierDirectory objects from Phase 10 working directory."""
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = workdir

    # rgi_ids.npy is written by run_hunza_preprocess.py into output_dir
    rgi_file = os.path.join(output_dir, 'rgi_ids.npy')
    if not os.path.exists(rgi_file):
        # Also try directly in workdir's parent
        alt = os.path.join(os.path.dirname(workdir), 'rgi_ids.npy')
        if os.path.exists(alt):
            rgi_file = alt
        else:
            raise FileNotFoundError(
                f'rgi_ids.npy not found at {rgi_file} or {alt}. '
                'Run run_hunza_preprocess.py first.'
            )

    rgi_ids = np.load(rgi_file, allow_pickle=True).tolist()
    log.info('rgi_ids.npy: %d RGI IDs', len(rgi_ids))

    gdirs = []
    missing = 0
    for rgi_id in rgi_ids:
        try:
            gdir = oggm.GlacierDirectory(rgi_id, base_dir=workdir)
            # Only include if model_diagnostics.nc exists
            if gdir.has_file('model_diagnostics'):
                gdirs.append(gdir)
            else:
                missing += 1
        except Exception:
            missing += 1

    log.info('Loaded %d gdirs with model_diagnostics (%d missing/skipped)',
             len(gdirs), missing)
    if not gdirs:
        raise RuntimeError(
            'No glacier directories with model_diagnostics.nc found. '
            'Ensure the Phase 10 SLURM array completed successfully.'
        )
    return gdirs


def _load_subbasins(hydrobasins_dir, level, bbox):
    """Load HydroBASINS sub-basins using the OGGM shop helper."""
    from oggm import cfg
    from oggm.shop.hydrobasins import get_hydrobasins

    cfg.PARAMS['hydrobasins_local_dir'] = hydrobasins_dir
    subbasins = get_hydrobasins(tuple(bbox), level=level,
                                region='as', use_lakes=True)
    log.info('Loaded %d sub-basins, total area = %.0f km²',
             len(subbasins), subbasins['SUB_AREA'].sum())
    return subbasins


def _make_synthetic_obs(ys, ye, seed=0):
    """Synthetic annual discharge for dry-run testing only."""
    log.warning(
        'No GRDC file provided — using SYNTHETIC discharge. '
        'Do NOT use for publication.'
    )
    rng = np.random.default_rng(seed)
    years = np.arange(ys, ye + 1)
    q = 420.0 + 0.5 * (years - years.mean()) + rng.normal(0, 80, len(years))
    q = np.maximum(q, 50.0)
    return pd.DataFrame({'year': years, 'q_m3s': q})


def _plot_results(result, obs_df, output_dir):
    """Plot calibrated vs observed annual discharge."""
    calib_years = np.asarray(result.get('calib_years', []))
    if len(calib_years) == 0:
        return

    obs_sub = obs_df[obs_df['year'].isin(calib_years)].set_index('year')

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(obs_sub.index, obs_sub['q_m3s'], 'k-o', ms=4,
            label='Observed (GRDC)')
    if 'Q_sim_m3s' in result:
        ax.plot(calib_years, result['Q_sim_m3s'], 'r--',
                label='Calibrated model')

    kge = result.get('KGE_calib', float('nan'))
    nse = result.get('NSE_calib', float('nan'))
    pbias = result.get('PBIAS_pct', float('nan'))
    ax.set_title(
        f'Hunza Basin — Water Balance Calibration\n'
        f'KGE={kge:.3f}  NSE={nse:.3f}  PBIAS={pbias:.1f}%'
    )
    ax.set_xlabel('Year')
    ax.set_ylabel('Annual discharge [m³ s⁻¹]')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(output_dir, 'phase12_calibrated_hydrograph.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info('Figure saved: %s', out_path)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log_path = _setup_logging(args.output_dir)

    log.info('=== Phase 12 — Basin Water Balance Calibration ===')
    log.info('workdir       : %s', args.workdir)
    log.info('output_dir    : %s', args.output_dir)
    log.info('grdc_file     : %s', args.grdc_file)
    log.info('filesuffix    : %s', args.filesuffix)
    log.info('period        : %d–%d', args.ys, args.ye)
    log.info('metric        : %s', args.metric)
    log.info('cross_validate: %s', args.cross_validate)

    from oggm.core.hydrology import (
        calibrate_basin_water_balance,
        read_grdc_data,
    )

    # ---- Load data ----
    gdirs = _load_gdirs(args.workdir, args.output_dir)
    subbasins = _load_subbasins(args.hydrobasins_dir,
                                args.hydrobasins_level,
                                args.bbox)

    if args.grdc_file and os.path.exists(args.grdc_file):
        log.info('Reading GRDC data from %s', args.grdc_file)
        obs_df = read_grdc_data(args.grdc_file, freq='annual',
                                ys=args.ys, ye=args.ye)
        log.info('  %d annual observations loaded', len(obs_df))
    else:
        if args.grdc_file:
            log.warning('GRDC file not found: %s', args.grdc_file)
        obs_df = _make_synthetic_obs(args.ys, args.ye, seed=args.seed)

    # ---- Calibrate ----
    log.info('Starting calibrate_basin_water_balance() ...')
    result = calibrate_basin_water_balance(
        gdirs=gdirs,
        subbasins_gdf=subbasins,
        obs_discharge=obs_df,
        obs_freq='annual',
        glacier_filesuffix=args.filesuffix,
        basin_prcp_fac=args.basin_prcp_fac,
        ys=args.ys,
        ye=args.ye,
        method=args.method,
        metric=args.metric,
        output_dir=args.output_dir,
        cross_validate=args.cross_validate,
        seed=args.seed,
    )

    # ---- Report ----
    log.info('=== Calibration Results ===')
    for key in ('k_rain_months', 'k_snow_months', 'k_ice_months',
                'k_snow_ngl', 'k_soil_months', 's_fc_mm',
                'basin_prcp_fac', 'KGE_calib', 'KGE_valid',
                'NSE_calib', 'PBIAS_pct', 'n_obs', 'convergence'):
        if key in result:
            v = result[key]
            log.info('  %-20s = %s', key,
                     f'{v:.4f}' if isinstance(v, float) else str(v))

    # Save final JSON with log path appended
    result['log_file'] = log_path
    json_path = os.path.join(args.output_dir, 'basin_calib_params.json')
    with open(json_path, 'w') as fh:
        json.dump(result, fh, indent=2, default=str)
    log.info('Parameters saved: %s', json_path)

    # ---- Plot ----
    _plot_results(result, obs_df, args.output_dir)

    log.info('=== Phase 12 calibration complete ===')
    return result


if __name__ == '__main__':
    main()
