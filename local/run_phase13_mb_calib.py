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
    p.add_argument('--model_filesuffix', default='_hunza',
                   help='model_diagnostics filesuffix from Phase 10 (default _hunza)')
    p.add_argument('--outlet_lon', type=float, default=74.46)
    p.add_argument('--outlet_lat', type=float, default=35.90)
    p.add_argument('--no_delineate', action='store_true')
    return p.parse_args()


# =============================================================================
# Helpers
# =============================================================================

def _setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'phase13_mb_calib.log')
    # Attach handlers directly to the __main__ logger (not root) so that
    # cfg.initialize(logging_level='WARNING') cannot suppress our INFO messages.
    fmt = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s  %(message)s',
        datefmt='%H:%M:%S',
    )
    h_stream = logging.StreamHandler(sys.stdout)
    h_stream.setFormatter(fmt)
    h_file = logging.FileHandler(log_path, mode='w')
    h_file.setFormatter(fmt)

    main_log = logging.getLogger(__name__)
    main_log.setLevel(logging.INFO)
    main_log.addHandler(h_stream)
    main_log.addHandler(h_file)
    main_log.propagate = False  # Don't let root-logger level changes suppress us

    # Also configure the root logger so OGGM's own messages appear in stdout.
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s: %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        stream=sys.stdout,
    )
    return log_path


def _load_gdirs(workdir, output_dir, filesuffix='', representative_frac=0.85):
    """Load proper GlacierDirectory objects for representative large glaciers.

    Phase 13 requires proper GlacierDirectory objects (needs read_json,
    write_json, etc. for mb_calibration_from_runoff).  To avoid the
    ~90-second penalty of constructing 6551 gdirs (each reads outlines.tar.gz),
    we:
      1. Read glacier_list.csv for fast metadata (cenlon, cenlat, area_km2)
      2. Check for model_diagnostics NC via os.path.isfile() — no tar reads
      3. Sort by area, keep only top-N glaciers covering representative_frac
         of total glacier area (typically ~50 glaciers for Hunza)
      4. Construct proper GlacierDirectory only for those N glaciers

    The mb_calibration_basin_from_discharge() function is then called with
    use_representative_glaciers=False since we already pre-filtered.
    """
    import oggm
    from oggm import cfg

    cfg.initialize(logging_level='WARNING')
    cfg.PATHS['working_dir'] = workdir

    # --- locate rgi_ids.npy ---
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

    # --- locate glacier_list.csv ---
    csv_candidates = [
        os.path.join(output_dir, 'glacier_list.csv'),
        os.path.join(os.path.dirname(workdir), 'glacier_list.csv'),
    ]
    csv_file = next((p for p in csv_candidates if os.path.exists(p)), None)
    if csv_file is None:
        raise FileNotFoundError(
            f'glacier_list.csv not found in {csv_candidates}. '
            'Run run_hunza_preprocess.py first.'
        )
    meta_df = pd.read_csv(csv_file, index_col='rgi_id')
    log.info('glacier_list.csv: %d rows (%s)', len(meta_df), csv_file)

    # --- base_dir: Phase 10 stores gdirs under workdir/per_glacier/ ---
    per_glacier_dir = os.path.join(workdir, 'per_glacier')
    base_dir = per_glacier_dir if os.path.isdir(per_glacier_dir) else workdir
    log.info('Using base_dir: %s', base_dir)

    # --- fast file check: find matching glaciers + their areas ---
    nc_name = f'model_diagnostics{filesuffix}.nc'
    matching = []   # list of (rgi_id, area_km2)
    missing = 0
    for rgi_id in rgi_ids:
        if rgi_id not in meta_df.index:
            missing += 1
            continue
        row = meta_df.loc[rgi_id]
        gdir_dir = os.path.join(base_dir, rgi_id[:-6], rgi_id[:-3], rgi_id)
        nc_path = os.path.join(gdir_dir, nc_name)
        if os.path.isfile(nc_path):
            matching.append((rgi_id, float(row['area_km2'])))
        else:
            missing += 1

    log.info('Found %d glaciers with %s (%d without)',
             len(matching), nc_name, missing)

    # --- pre-filter to representative glaciers by area ---
    matching.sort(key=lambda x: x[1], reverse=True)
    total_area = sum(a for _, a in matching)
    cum_area = 0.0
    n_rep = 0
    for _, area in matching:
        cum_area += area
        n_rep += 1
        if cum_area / total_area >= representative_frac:
            break
    rep_ids = [rid for rid, _ in matching[:n_rep]]
    log.info(
        'Pre-filtered to %d representative glaciers covering %.0f%% '
        'of total glacier area (%.0f km² of %.0f km²)',
        n_rep,
        cum_area / total_area * 100,
        cum_area,
        total_area,
    )

    # --- construct proper GlacierDirectory only for the N rep glaciers ---
    gdirs = []
    for rgi_id in rep_ids:
        try:
            gdir = oggm.GlacierDirectory(rgi_id, base_dir=base_dir)
            gdirs.append(gdir)
        except Exception as exc:
            log.warning('Could not open gdir %s: %s', rgi_id, exc)

    log.info('Loaded %d proper GlacierDirectory objects', len(gdirs))
    if not gdirs:
        raise RuntimeError(
            'No representative glacier directories found. '
            'Ensure Phase 10 array completed successfully.'
        )
    return gdirs


def _load_grdc_annual(grdc_file, ys, ye):
    from oggm.core.hydrology import read_grdc_data
    return read_grdc_data(grdc_file, freq='annual', ys=ys, ye=ye)


def _make_synthetic_total_obs_from_model(gdirs, subbasins, ys, ye, seed=0):
    """Generate synthetic total obs identical to Phase 12 run 10.

    Routes annual glacier cache through linear reservoirs at default k values
    and adds NGL at default params + 5% noise.  This ensures Phase 13 uses
    the same obs as Phase 12, giving a coherent self-consistency test.
    """
    from oggm.core.hydrology import (
        _cache_basin_runoff_components,
        extract_subbasin_climate as _esc,
        compute_nonglaciated_runoff as _cnr,
        _linear_reservoir as _lr,
    )
    import oggm.cfg as _cfg

    log.warning(
        'No GRDC file provided — using SYNTHETIC total discharge from model '
        'truth + 5%% noise (same as Phase 12). Do NOT use for publication.'
    )

    cache = _cache_basin_runoff_components(
        gdirs, filesuffix='_hunza', ys=ys, ye=ye)
    years_all = cache['years']
    sel = (years_all >= ys) & (years_all <= ye)
    years_calib = years_all[sel]

    k_rain = float(_cfg.PARAMS.get('routing_k_rain_months', 0.5))
    k_snow = float(_cfg.PARAMS.get('routing_k_snow_months', 2.0))
    k_ice  = float(_cfg.PARAMS.get('routing_k_ice_months', 8.0))
    q_gl = (
        _lr(cache['rain_m3s'][sel], k=k_rain / 12.0, dt=1.0) +
        _lr(cache['snow_m3s'][sel], k=k_snow / 12.0, dt=1.0) +
        _lr(cache['ice_m3s'][sel],  k=k_ice  / 12.0, dt=1.0)
    )

    _total_km2 = float(subbasins['SUB_AREA'].sum())
    _gl_km2 = float(sum(g.rgi_area_km2 for g in gdirs))
    _ngl_frac = max(0.0, min(1.0, 1.0 - _gl_km2 / max(_total_km2, 1.0)))
    ngl_area = {int(r['HYBAS_ID']): float(r['SUB_AREA']) * _ngl_frac
                for _, r in subbasins.iterrows()}
    per_basin_clim = _esc(gdirs, subbasins, ys=ys, ye=ye)
    ngl_ds = _cnr(
        subbasins, per_basin_clim,
        k_snow_months=float(_cfg.PARAMS.get('nonglaciated_k_snow_months', 2.0)),
        k_soil_months=float(_cfg.PARAMS.get('nonglaciated_k_soil_months', 3.0)),
        s_fc_mm=float(_cfg.PARAMS.get('nonglaciated_s_fc_mm', 150.0)),
        nonglaciated_area_km2=ngl_area,
    )
    q_ngl_m = ngl_ds['Q_ngl_m3s'].sum('HYBAS_ID').values
    time_vals = ngl_ds['time'].values
    try:
        ngl_yrs = np.array([pd.Timestamp(t).year for t in time_vals])
    except Exception:
        ngl_yrs = np.array([int(t) for t in time_vals])
    q_ngl = np.array([q_ngl_m[ngl_yrs == yr].mean()
                      if (ngl_yrs == yr).any() else 0.0
                      for yr in years_calib], dtype=float)

    q_truth = q_gl + q_ngl
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.05 * float(q_truth.mean()), len(years_calib))
    log.info('Synthetic obs: mean=%.1f m³/s  (gl=%.1f, ngl=%.1f)',
             float(q_truth.mean()), float(q_gl.mean()), float(q_ngl.mean()))
    return pd.DataFrame({'year': years_calib,
                         'q_m3s': np.maximum(q_truth + noise, 10.0)})


def _estimate_glacier_discharge(total_obs_df, glacier_frac,
                                phase12_json=None,
                                gdirs=None, subbasins=None):
    """Estimate glacier-only annual discharge.

    Option 2 (residual): recompute Q_ngl at Phase 12 calibrated params.
    Option 3 (fraction): multiply total by glacier_frac.

    Returns
    -------
    obs_years : np.ndarray[int]
    q_gl_m3s  : np.ndarray[float]
    method    : str
    """
    years = total_obs_df['year'].values.astype(int)
    q_total = total_obs_df['q_m3s'].values.astype(float)
    ys, ye = int(years.min()), int(years.max())

    if phase12_json and os.path.exists(phase12_json) and gdirs and subbasins is not None:
        with open(phase12_json) as fh:
            p12 = json.load(fh)
        # Recompute Q_ngl at Phase 12 calibrated routing params
        try:
            from oggm.core.hydrology import (
                extract_subbasin_climate as _esc,
                compute_nonglaciated_runoff as _cnr,
            )
            import oggm.cfg as _cfg
            _total_km2 = float(subbasins['SUB_AREA'].sum())
            _gl_km2 = float(sum(g.rgi_area_km2 for g in gdirs))
            _ngl_frac = max(0.0, min(1.0, 1.0 - _gl_km2 / max(_total_km2, 1.0)))
            ngl_area = {int(r['HYBAS_ID']): float(r['SUB_AREA']) * _ngl_frac
                        for _, r in subbasins.iterrows()}
            pf = float(p12.get('basin_prcp_fac', 1.0))
            per_basin_clim = _esc(gdirs, subbasins, ys=ys, ye=ye)
            if pf != 1.0:
                per_basin_clim = {
                    hid: clim.assign({'prcp': clim['prcp'] * pf})
                    for hid, clim in per_basin_clim.items()
                }
            ngl_ds = _cnr(
                subbasins, per_basin_clim,
                k_snow_months=float(p12.get('k_snow_ngl', 2.0)),
                k_soil_months=float(p12.get('k_soil_months', 3.0)),
                s_fc_mm=float(p12.get('s_fc_mm', 150.0)),
                nonglaciated_area_km2=ngl_area,
            )
            q_ngl_m = ngl_ds['Q_ngl_m3s'].sum('HYBAS_ID').values
            time_vals = ngl_ds['time'].values
            try:
                ngl_yrs = np.array([pd.Timestamp(t).year for t in time_vals])
            except Exception:
                ngl_yrs = np.array([int(t) for t in time_vals])
            q_ngl_ann = np.array([q_ngl_m[ngl_yrs == yr].mean()
                                   if (ngl_yrs == yr).any() else 0.0
                                   for yr in years], dtype=float)
            q_gl = np.maximum(q_total - q_ngl_ann, 1.0)
            log.info(
                'Glacier discharge: residual method (Option 2, Phase 12 params), '
                '%d years, mean=%.1f m3/s  (NGL mean=%.1f)',
                len(years), q_gl.mean(), q_ngl_ann.mean()
            )
            return years, q_gl, 'residual (Phase 12 params)'
        except Exception as exc:
            log.warning('Residual method failed (%s); falling back to fraction.', exc)

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
    # Phase 13 reads from Phase 10 workdir; rgi_ids.npy may be in Phase 12 output dir
    phase12_dir = os.path.dirname(args.phase12_json) if args.phase12_json else args.output_dir
    gdirs = _load_gdirs(args.workdir, phase12_dir, filesuffix=args.model_filesuffix,
                        representative_frac=args.representative_frac)

    # ---- Load subbasins (needed for residual method and synthetic obs) ----
    from oggm import cfg
    from oggm.shop.hydrobasins import get_hydrobasins
    cfg.PARAMS['hydrobasins_local_dir'] = os.path.expanduser('~/oggm_dev_project')
    subbasins = get_hydrobasins(tuple(_HUNZA_BBOX), level=8, region='as', use_lakes=True)
    if not args.no_delineate:
        # Upstream delineation (same as Phase 12)
        from shapely.geometry import Point
        gauge_pt = Point(args.outlet_lon, args.outlet_lat)
        outlet_rows = subbasins[subbasins.geometry.contains(gauge_pt)]
        if len(outlet_rows) > 0:
            outlet_id = int(outlet_rows.iloc[0]['HYBAS_ID'])
            next_down = {int(r['HYBAS_ID']): int(r['NEXT_DOWN'])
                         for _, r in subbasins.iterrows()}
            upstream = set()
            q = {outlet_id}
            while q:
                c = q.pop()
                upstream.add(c)
                for hid, nd in next_down.items():
                    if nd == c and hid not in upstream:
                        q.add(hid)
            subbasins = subbasins[subbasins['HYBAS_ID'].isin(upstream)].copy()
            # Filter gdirs to basin
            try:
                bu = subbasins.geometry.union_all()
            except AttributeError:
                bu = subbasins.geometry.unary_union
            gdirs = [g for g in gdirs
                     if bu.contains(Point(g.cenlat, g.cenlon)) or
                        bu.contains(Point(g.cenlon, g.cenlat))]
        log.info('Basin: %d sub-basins, %.0f km², %d gdirs',
                 len(subbasins), subbasins['SUB_AREA'].sum(), len(gdirs))

    # ---- Observed total discharge ----
    if args.grdc_file and os.path.exists(args.grdc_file):
        log.info('Reading GRDC data from %s', args.grdc_file)
        total_obs = _load_grdc_annual(args.grdc_file, args.ys, args.ye)
    else:
        if args.grdc_file:
            log.warning('GRDC file not found: %s', args.grdc_file)
        total_obs = _make_synthetic_total_obs_from_model(
            gdirs, subbasins, args.ys, args.ye, seed=42)

    # ---- Estimate glacier-only discharge ----
    obs_years, q_gl_m3s, method = _estimate_glacier_discharge(
        total_obs, args.glacier_frac, args.phase12_json,
        gdirs=gdirs, subbasins=subbasins)
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
