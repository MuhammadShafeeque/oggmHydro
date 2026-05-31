"""Phase 13 — MB Calibration from Discharge (Hunza basin).

Runs per-glacier mass-balance calibration constrained by both geodetic MB
(Hugonnet 2021) and observed glacier-only discharge, using the
``mb_calibration_from_runoff()`` entity task and the basin-level wrapper
``mb_calibration_basin_from_discharge()``.

Scientific background
---------------------
The standard TI model calibration (``mb_calibration_from_geodetic_mb``)
fits {melt_f | prcp_fac | temp_bias} to a single scalar — under-determining
the 3-parameter system.  Adding observed glacier discharge as a second
constraint helps break the melt_f–prcp_fac degeneracy, which is important
in heavily glacierised basins like Karakoram where both melt and
accumulation are large.

This driver:
  1.  Loads Phase 10 glacier directories and estimates glacier-only discharge
  2.  Optionally loads per-glacier geodetic MB from a CSV (Hugonnet 2021)
  3.  Calls ``mb_calibration_basin_from_discharge()`` (parallel per glacier)
  4.  Saves per-glacier calibrated JSON files into each gdir
  5.  Produces summary statistics and plots

Estimating glacier-only discharge (three options)
--------------------------------------------------
  Option 1 (preferred, two nested gauges):
      Q_gl = Q_outlet - Q_non_glaciated_subbasins

  Option 2 (residual method, requires Phase 12 JSON):
      Q_gl = Q_obs_total - Q_ngl_modeled  (from basin_calib_params.json)

  Option 3 (fraction method, default):
      Q_gl = glacier_frac * Q_obs_total

  Pass --phase12_json to enable Option 2; otherwise Option 3 is used.

Pre-requisites
--------------
  - Phase 10 SLURM array completed (model_diagnostics.nc per glacier)
  - Phase 12 calibration (optional, for Option 2 residual method)
  - GRDC total discharge file (optional; synthetic data used if absent)
  - Hugonnet 2021 CSV (optional; uses default geodetic MB if absent)

Usage (cluster)
---------------
  cd /home/users/mshafeeque/oggm_dev_project/oggmHydro
  /home/users/mshafeeque/miniforge3/envs/oggm_env6/bin/python \\
      local/run_phase13_mb_calib.py \\
      --workdir ~/hydro_analysis/phase10/hunza_run \\
      --output_dir ~/hydro_analysis/phase13 \\
      [--grdc_file ~/data/grdc/2392800_Q_Day.Cmd.txt] \\
      [--geodetic_mb_file ~/data/hugonnet2021_rgi60.csv] \\
      [--phase12_json ~/hydro_analysis/phase12/basin_calib_params.json] \\
      [--glacier_frac 0.65] \\
      [--ys 2000] [--ye 2019] \\
      [--w_mb 0.6] [--w_Q 0.4] \\
      [--nprocesses 8]

  Or via SLURM:
      sbatch local/sbatch_templates/13_mb_calibration.sh
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
_DEFAULT_OUTDIR     = os.path.expanduser('~/hydro_analysis/phase13')
_DEFAULT_BASINS_DIR = os.path.expanduser('~/oggm_dev_project')
_DEFAULT_YS         = 2000   # Hugonnet 2021 geodetic MB period 2000-2019
_DEFAULT_YE         = 2019
_HUNZA_BBOX         = (73.5, 35.5, 76.5, 37.5)

# Default glacier fraction of total discharge for the Hunza basin
# (estimated from Phase 10 basin analysis; update after Phase 12)
_DEFAULT_GLACIER_FRAC = 0.65


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Phase 13 — Per-glacier MB calibration from discharge')
    p.add_argument('--workdir', default=_DEFAULT_WORKDIR,
                   help='OGGM working directory with Phase 10 gdirs')
    p.add_argument('--output_dir', default=_DEFAULT_OUTDIR,
                   help='Output directory for summary JSON, figures, and log')
    p.add_argument('--grdc_file', default=None,
                   help='GRDC total discharge file (ASCII or CSV). '
                        'Omit to use synthetic data (testing only).')
    p.add_argument('--geodetic_mb_file', default=None,
                   help='CSV with Hugonnet 2021 geodetic MB per glacier. '
                        'Required columns: rgi_id, dmdtda, err_dmdtda '
                        '[m w.e. yr-1]. Omit to use per-glacier defaults.')
    p.add_argument('--phase12_json', default=None,
                   help='Path to basin_calib_params.json from Phase 12. '
                        'When provided, residual method is used to estimate '
                        'glacier-only discharge (Option 2).')
    p.add_argument('--glacier_frac', type=float,
                   default=_DEFAULT_GLACIER_FRAC,
                   help='Glacierised fraction of total discharge (Option 3, '
                        'fraction method). Overridden if --phase12_json is '
                        'given and Q_ngl is available.')
    p.add_argument('--ys', type=int, default=_DEFAULT_YS,
                   help='Calibration period start year')
    p.add_argument('--ye', type=int, default=_DEFAULT_YE,
                   help='Calibration period end year')
    p.add_argument('--w_mb', type=float, default=0.6,
                   help='Weight for geodetic MB constraint (0–1)')
    p.add_argument('--w_Q', type=float, default=0.4,
                   help='Weight for discharge KGE constraint (0–1)')
    p.add_argument('--representative_frac', type=float, default=0.8,
                   help='Cumulative area fraction of representative glaciers '
                        'directly calibrated (rest inherit same params)')
    p.add_argument('--nprocesses', type=int, default=1,
                   help='Parallel OGGM processes for per-glacier calibration')
    p.add_argument('--filesuffix', default='',
                   help='Suffix for mb_calib output files in each gdir')
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def _setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'phase13_mb_calib.log')
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


def _load_gdirs(workdir, output_dir, filesuffix=''):
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = workdir

    rgi_file = os.path.join(output_dir, 'rgi_ids.npy')
    if not os.path.exists(rgi_file):
        alt = os.path.join(os.path.dirname(workdir), 'rgi_ids.npy')
        if os.path.exists(alt):
            rgi_file = alt
        else:
            raise FileNotFoundError(
                f'rgi_ids.npy not found at {rgi_file}. '
                'Run run_hunza_preprocess.py first.'
            )

    rgi_ids = np.load(rgi_file, allow_pickle=True).tolist()
    log.info('rgi_ids.npy: %d RGI IDs', len(rgi_ids))

    # Phase 10 stores gdirs under workdir/per_glacier/
    per_glacier_dir = os.path.join(workdir, 'per_glacier')
    base_dir = per_glacier_dir if os.path.isdir(per_glacier_dir) else workdir
    log.info('Using base_dir: %s', base_dir)

    gdirs, missing = [], 0
    for rgi_id in rgi_ids:
        try:
            gdir = oggm.GlacierDirectory(rgi_id, base_dir=base_dir)
            has_diag = (gdir.has_file('model_diagnostics', filesuffix=filesuffix)
                        if filesuffix
                        else gdir.has_file('model_diagnostics'))
            if has_diag:
                gdirs.append(gdir)
            else:
                missing += 1
        except Exception:
            missing += 1

    log.info('Loaded %d gdirs with model_diagnostics%s (%d missing)',
             len(gdirs), filesuffix, missing)
    if not gdirs:
        raise RuntimeError(
            'No gdirs with model_diagnostics.nc found. '
            'Ensure Phase 10 array completed successfully.'
        )
    return gdirs


def _load_grdc_annual(grdc_file, ys, ye):
    from oggm.core.hydrology import read_grdc_data
    return read_grdc_data(grdc_file, freq='annual', ys=ys, ye=ye)


def _make_synthetic_total_obs(ys, ye, seed=0):
    log.warning(
        'No GRDC file provided — using SYNTHETIC total discharge. '
        'Do NOT use for publication.'
    )
    rng = np.random.default_rng(seed)
    years = np.arange(ys, ye + 1)
    q = 420.0 + 0.5 * (years - years.mean()) + rng.normal(0, 80, len(years))
    return pd.DataFrame({'year': years, 'q_m3s': np.maximum(q, 50.0)})


def _estimate_glacier_discharge(total_obs_df, glacier_frac, phase12_json=None):
    """Estimate glacier-only annual discharge.

    Option 2 (residual): if phase12_json has Q_ngl_m3s.
    Option 3 (fraction): multiply total by glacier_frac.

    Returns
    -------
    obs_years : np.ndarray[int]
    q_gl_m3s  : np.ndarray[float]
    method    : str
    """
    years = total_obs_df['year'].values.astype(int)
    q_total = total_obs_df['q_m3s'].values.astype(float)

    if phase12_json and os.path.exists(phase12_json):
        with open(phase12_json) as fh:
            p12 = json.load(fh)
        if 'Q_ngl_m3s' in p12 and 'calib_years' in p12:
            p12_years = np.asarray(p12['calib_years'], dtype=int)
            q_ngl = np.asarray(p12['Q_ngl_m3s'], dtype=float)
            mask = np.isin(years, p12_years)
            if mask.sum() >= 3:
                q_gl = q_total[mask] - q_ngl[np.isin(p12_years, years[mask])]
                q_gl = np.maximum(q_gl, 0.0)
                log.info(
                    'Glacier discharge: residual method (Option 2), '
                    '%d years, mean=%.1f m3/s', mask.sum(), q_gl.mean()
                )
                return years[mask], q_gl, 'residual (Phase 12)'

    q_gl = glacier_frac * q_total
    log.info(
        'Glacier discharge: fraction method (Option 3, frac=%.2f), '
        '%d years, mean=%.1f m3/s', glacier_frac, len(years), q_gl.mean()
    )
    return years, q_gl, f'fraction (frac={glacier_frac:.2f})'


def _load_geodetic_mb(geodetic_mb_file):
    """Load Hugonnet 2021 CSV → dict {rgi_id: {'ref_mb', 'ref_mb_err'}} in kg m-2 yr-1."""
    df = pd.read_csv(geodetic_mb_file)
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {'dmdt': 'dmdtda', 'err_dmdt': 'err_dmdtda',
              'mb': 'dmdtda', 'mb_err': 'err_dmdtda'}
    df = df.rename(columns=rename)
    required = {'rgi_id', 'dmdtda', 'err_dmdtda'}
    if not required.issubset(df.columns):
        raise ValueError(
            f'geodetic_mb_file must have columns {required}. '
            f'Found: {list(df.columns)}'
        )
    result = {}
    for _, row in df.iterrows():
        result[str(row['rgi_id'])] = {
            'ref_mb':     float(row['dmdtda']) * 1000.0,
            'ref_mb_err': float(row['err_dmdtda']) * 1000.0,
        }
    log.info('Loaded geodetic MB for %d glaciers', len(result))
    return result


def _plot_summary(summary_df, output_dir):
    """Plot distribution of calibrated parameters across glaciers."""
    params = ['melt_f', 'prcp_fac', 'temp_bias']
    avail = [p for p in params if p in summary_df.columns]
    if not avail:
        return

    fig, axes = plt.subplots(1, len(avail), figsize=(4 * len(avail), 4))
    if len(avail) == 1:
        axes = [axes]

    labels = {'melt_f': 'melt_f [mm d⁻¹ K⁻¹]',
              'prcp_fac': 'prcp_fac [–]',
              'temp_bias': 'temp_bias [K]'}
    for ax, param in zip(axes, avail):
        data = summary_df[param].dropna()
        ax.hist(data, bins=20, edgecolor='white', color='steelblue')
        ax.axvline(data.median(), color='r', ls='--',
                   label=f'median={data.median():.2f}')
        ax.set_xlabel(labels.get(param, param))
        ax.set_ylabel('Count')
        ax.set_title(f'Calibrated {param}')
        ax.legend(fontsize=8)

    fig.suptitle('Phase 13 — Per-glacier MB calibration (discharge-constrained)',
                 fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(output_dir, 'phase13_param_distributions.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info('Figure saved: %s', out_path)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    log_path = _setup_logging(args.output_dir)

    log.info('=== Phase 13 — MB Calibration from Discharge ===')
    log.info('workdir          : %s', args.workdir)
    log.info('output_dir       : %s', args.output_dir)
    log.info('period           : %d–%d', args.ys, args.ye)
    log.info('w_mb / w_Q       : %.2f / %.2f', args.w_mb, args.w_Q)
    log.info('glacier_frac     : %.2f', args.glacier_frac)
    log.info('nprocesses       : %d', args.nprocesses)

    from oggm.core.hydrology import mb_calibration_basin_from_discharge

    # ---- Load glacier dirs ----
    # Phase 10 uses filesuffix _hunza; pass it so has_file check succeeds
    gdirs = _load_gdirs(args.workdir, args.output_dir, filesuffix='_hunza')

    # ---- Observed total discharge ----
    if args.grdc_file and os.path.exists(args.grdc_file):
        log.info('Reading GRDC data from %s', args.grdc_file)
        total_obs = _load_grdc_annual(args.grdc_file, args.ys, args.ye)
    else:
        if args.grdc_file:
            log.warning('GRDC file not found: %s', args.grdc_file)
        total_obs = _make_synthetic_total_obs(args.ys, args.ye)

    # ---- Estimate glacier-only discharge ----
    obs_years, q_gl_m3s, method = _estimate_glacier_discharge(
        total_obs, args.glacier_frac, args.phase12_json)
    log.info('Glacier Q method: %s', method)
    log.info('  %d years, mean=%.1f m3/s, range=[%.1f, %.1f]',
             len(obs_years), q_gl_m3s.mean(), q_gl_m3s.min(), q_gl_m3s.max())

    # ---- Geodetic MB ----
    ref_mb_df = None
    if args.geodetic_mb_file and os.path.exists(args.geodetic_mb_file):
        ref_mb_df = _load_geodetic_mb(args.geodetic_mb_file)
    elif args.geodetic_mb_file:
        log.warning('geodetic_mb_file not found: %s', args.geodetic_mb_file)
        log.warning('Proceeding without geodetic MB constraint (w_mb → 0).')
        args.w_mb = 0.0
        args.w_Q  = 1.0

    # ---- Run per-glacier calibration ----
    log.info('Starting mb_calibration_basin_from_discharge() ...')
    results = mb_calibration_basin_from_discharge(
        gdirs=gdirs,
        obs_glacier_discharge_m3s=q_gl_m3s,
        obs_years=obs_years,
        ref_mb_df=ref_mb_df,
        w_mb=args.w_mb,
        w_Q=args.w_Q,
        use_representative_glaciers=True,
        representative_frac=args.representative_frac,
        nprocesses=args.nprocesses,
        filesuffix=args.filesuffix,
    )

    log.info('Calibrated %d glaciers', len(results))

    # ---- Summary DataFrame ----
    rows = [{'rgi_id': rid, **calib} for rid, calib in results.items()]
    summary_df = pd.DataFrame(rows)

    summary_path = os.path.join(args.output_dir, 'phase13_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    log.info('Summary CSV: %s', summary_path)

    for param in ('melt_f', 'prcp_fac', 'temp_bias'):
        if param in summary_df.columns:
            col = summary_df[param].dropna()
            log.info('  %s: median=%.3f  mean=%.3f  std=%.3f',
                     param, col.median(), col.mean(), col.std())

    # ---- Metadata JSON ----
    meta = {
        'n_calibrated': len(results),
        'n_gdirs_total': len(gdirs),
        'obs_years': obs_years.tolist(),
        'glacier_discharge_method': method,
        'w_mb': args.w_mb,
        'w_Q': args.w_Q,
        'geodetic_mb_file': args.geodetic_mb_file,
        'grdc_file': args.grdc_file,
        'ys': args.ys,
        'ye': args.ye,
        'log_file': log_path,
    }
    meta_path = os.path.join(args.output_dir, 'phase13_metadata.json')
    with open(meta_path, 'w') as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info('Metadata: %s', meta_path)

    # ---- Plot ----
    _plot_summary(summary_df, args.output_dir)

    log.info('=== Phase 13 MB calibration complete ===')
    return results


if __name__ == '__main__':
    main()
