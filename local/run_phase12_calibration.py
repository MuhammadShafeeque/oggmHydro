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
# Lightweight gdir proxy (no shapefile reads)
# =============================================================================

class _FastGdir:
    """Minimal GlacierDirectory substitute built from glacier_list.csv.

    Avoids the per-glacier outlines.tar.gz reads that make GlacierDirectory
    construction very slow (~14 ms per glacier × 6551 glaciers = ~90 s).
    Implements only the interface used by calibrate_basin_water_balance(),
    _cache_basin_runoff_components(), and extract_subbasin_climate():

        rgi_id, cenlon, cenlat, rgi_area_km2, get_filepath(), has_file()
    """
    __slots__ = ('rgi_id', 'cenlon', 'cenlat', 'rgi_area_km2', 'dir')

    def __init__(self, rgi_id, cenlon, cenlat, rgi_area_km2, base_dir):
        self.rgi_id = rgi_id
        self.cenlon = float(cenlon)
        self.cenlat = float(cenlat)
        self.rgi_area_km2 = float(rgi_area_km2)
        # OGGM directory layout: base_dir/O1/O1.O2/O1.O2.xxxxx/
        self.dir = os.path.join(base_dir, rgi_id[:-6], rgi_id[:-3], rgi_id)

    @property
    def rgi_area_m2(self):
        return self.rgi_area_km2 * 1e6

    def get_filepath(self, filename, filesuffix='', delete=False, **kwargs):
        from oggm import cfg
        fname = cfg.BASENAMES[filename]
        if filesuffix:
            parts = fname.rsplit('.', 1)
            fname = f'{parts[0]}{filesuffix}.{parts[1]}'
        out = os.path.join(self.dir, fname)
        if delete and os.path.isfile(out):
            os.remove(out)
        return out

    def has_file(self, filename, filesuffix='', **kwargs):
        return os.path.isfile(self.get_filepath(filename, filesuffix=filesuffix))

    def __repr__(self):
        return f'<_FastGdir {self.rgi_id}>'


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
    p.add_argument('--outlet-lon', type=float, default=74.46,
                   help='Longitude of basin outlet gauge [°E] '
                        '(default 74.46 = Dainyor bridge)')
    p.add_argument('--outlet-lat', type=float, default=35.90,
                   help='Latitude of basin outlet gauge [°N] '
                        '(default 35.90 = Dainyor bridge)')
    p.add_argument('--no-delineate', action='store_true',
                   help='Skip upstream delineation (use full bbox sub-basins)')
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


def _load_gdirs(workdir, output_dir, filesuffix=''):
    """Build fast _FastGdir proxies from glacier_list.csv (no shapefile reads).

    Reads glacier metadata (cenlon, cenlat, area_km2) from glacier_list.csv
    and checks for model_diagnostics NC files via os.path.isfile().  This
    avoids the per-glacier outlines.tar.gz reads that make vanilla
    GlacierDirectory construction very slow (~90 s for 6551 glaciers).
    """
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
                f'rgi_ids.npy not found at {rgi_file} or {alt}. '
                'Run run_hunza_preprocess.py first.'
            )
    rgi_ids = np.load(rgi_file, allow_pickle=True).tolist()
    log.info('rgi_ids.npy: %d RGI IDs', len(rgi_ids))

    # --- locate glacier_list.csv (cenlon, cenlat, area_km2) ---
    # Phase 10 writes glacier_list.csv to its output_dir (parent of workdir)
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

    # --- build _FastGdir objects; check NC file via os.path.isfile (fast) ---
    nc_name = f'model_diagnostics{filesuffix}.nc'
    gdirs = []
    missing = 0
    for rgi_id in rgi_ids:
        if rgi_id not in meta_df.index:
            missing += 1
            continue
        row = meta_df.loc[rgi_id]
        gdir = _FastGdir(
            rgi_id=rgi_id,
            cenlon=row['lon'],
            cenlat=row['lat'],
            rgi_area_km2=row['area_km2'],
            base_dir=base_dir,
        )
        if os.path.isfile(os.path.join(gdir.dir, nc_name)):
            gdirs.append(gdir)
        else:
            missing += 1

    log.info('Built %d fast gdirs with %s (%d missing/skipped)',
             len(gdirs), nc_name, missing)
    if not gdirs:
        raise RuntimeError(
            f'No glacier directories with {nc_name} found. '
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
    log.info('Loaded %d sub-basins (bbox), total area = %.0f km²',
             len(subbasins), subbasins['SUB_AREA'].sum())
    return subbasins


def _delineate_upstream_basin(subs_gdf, outlet_lon, outlet_lat):
    """Return only sub-basins that drain upstream of the gauge point.

    Uses the HydroBASINS ``NEXT_DOWN`` field to trace the drainage network
    upstream from the sub-basin containing ``(outlet_lon, outlet_lat)``.

    Parameters
    ----------
    subs_gdf : GeoDataFrame
        All candidate sub-basins (loaded from bbox).
    outlet_lon, outlet_lat : float
        WGS-84 coordinates of the gauge / basin outlet.

    Returns
    -------
    GeoDataFrame
        Subset of ``subs_gdf`` containing only the outlet sub-basin and all
        sub-basins draining into it.
    """
    from shapely.geometry import Point

    gauge_pt = Point(outlet_lon, outlet_lat)
    outlet_rows = subs_gdf[subs_gdf.geometry.contains(gauge_pt)]

    if len(outlet_rows) == 0:
        log.warning(
            'Outlet point (%.3f°E, %.3f°N) not found in any sub-basin; '
            'using all %d bbox sub-basins.',
            outlet_lon, outlet_lat, len(subs_gdf))
        return subs_gdf

    outlet_id = int(outlet_rows.iloc[0]['HYBAS_ID'])
    log.info('Outlet sub-basin HYBAS_ID: %d  (contains %.3f°E, %.3f°N)',
             outlet_id, outlet_lon, outlet_lat)

    # BFS upstream via NEXT_DOWN
    next_down_map = {
        int(r['HYBAS_ID']): int(r['NEXT_DOWN'])
        for _, r in subs_gdf.iterrows()
    }
    upstream_ids: set = set()
    queue: set = {outlet_id}
    while queue:
        current = queue.pop()
        upstream_ids.add(current)
        for hid, nd in next_down_map.items():
            if nd == current and hid not in upstream_ids:
                queue.add(hid)

    upstream_subs = subs_gdf[subs_gdf['HYBAS_ID'].isin(upstream_ids)].copy()
    log.info(
        'Upstream delineation: %d sub-basins, total area = %.0f km²  '
        '(was %d sub-basins, %.0f km²)',
        len(upstream_subs), upstream_subs['SUB_AREA'].sum(),
        len(subs_gdf), subs_gdf['SUB_AREA'].sum())
    return upstream_subs


def _filter_gdirs_to_basin(gdirs, basin_gdf):
    """Keep only glaciers whose centroid lies within the delineated basin."""
    from shapely.geometry import Point

    try:
        basin_union = basin_gdf.geometry.union_all()
    except AttributeError:  # geopandas < 0.14
        basin_union = basin_gdf.geometry.unary_union

    filtered = [
        gd for gd in gdirs
        if basin_union.contains(Point(gd.cenlon, gd.cenlat))
    ]

    area_all = sum(g.rgi_area_km2 for g in gdirs)
    area_filt = sum(g.rgi_area_km2 for g in filtered)
    log.info(
        'Glaciers in basin: %d / %d  (%.0f / %.0f km²)',
        len(filtered), len(gdirs), area_filt, area_all)

    if len(filtered) == 0:
        raise RuntimeError(
            'No glacier directories found within the delineated basin. '
            'Check --outlet-lat/--outlet-lon or pass --no-delineate.')
    return filtered


def _make_synthetic_obs(ys, ye, seed=0, q_mean=420.0):
    """Synthetic annual discharge for dry-run testing only.

    Parameters
    ----------
    q_mean : float
        Mean annual discharge [m³ s⁻¹].  Pass a model-derived estimate so
        that the synthetic obs are at the right scale for the optimizer to
        converge.  Default 420 m³ s⁻¹ (realistic for small basins only).
    """
    log.warning(
        'No GRDC file provided — using SYNTHETIC discharge. '
        'Do NOT use for publication.'
    )
    rng = np.random.default_rng(seed)
    years = np.arange(ys, ye + 1)
    trend = 0.005 * q_mean * (years - years.mean())
    noise = rng.normal(0, 0.1 * q_mean, len(years))
    q = q_mean + trend + noise
    q = np.maximum(q, 10.0)
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
        _cache_basin_runoff_components,
    )

    # ---- Load data ----
    gdirs = _load_gdirs(args.workdir, args.output_dir, filesuffix=args.filesuffix)
    subbasins = _load_subbasins(args.hydrobasins_dir,
                                args.hydrobasins_level,
                                args.bbox)

    # ---- Delineate upstream basin + filter gdirs ----
    if not args.no_delineate:
        subbasins = _delineate_upstream_basin(
            subbasins, args.outlet_lon, args.outlet_lat)
        gdirs = _filter_gdirs_to_basin(gdirs, subbasins)
    else:
        log.info('--no-delineate: using all %d bbox sub-basins and all %d gdirs',
                 len(subbasins), len(gdirs))

    if args.grdc_file and os.path.exists(args.grdc_file):
        log.info('Reading GRDC data from %s', args.grdc_file)
        obs_df = read_grdc_data(args.grdc_file, freq='annual',
                                ys=args.ys, ye=args.ye)
        log.info('  %d annual observations loaded', len(obs_df))
    else:
        if args.grdc_file:
            log.warning('GRDC file not found: %s', args.grdc_file)
        # Compute the *actual* model total discharge at default parameters so
        # the synthetic obs are centred on what the model really produces.
        # This avoids any hand-tuned scaling factor.
        log.info('Computing model-scale discharge at default params for synthetic obs ...')
        from oggm.core.hydrology import (
            extract_subbasin_climate as _esc,
            compute_nonglaciated_runoff as _cnr,
        )
        import oggm.cfg as _cfg

        # --- glacier component (pre-cached) ---
        _cache = _cache_basin_runoff_components(
            gdirs, filesuffix=args.filesuffix, ys=args.ys, ye=args.ye)
        q_gl_mean = float(np.mean(
            _cache['rain_m3s'] + _cache['snow_m3s'] + _cache['ice_m3s']))

        # --- NGL component at default params with corrected area ---
        _total_km2 = float(subbasins['SUB_AREA'].sum())
        _gl_km2 = float(sum(g.rgi_area_km2 for g in gdirs))
        _ngl_frac = max(0.0, min(1.0, 1.0 - _gl_km2 / max(_total_km2, 1.0)))
        _ngl_area_per_sub = {
            int(r['HYBAS_ID']): float(r['SUB_AREA']) * _ngl_frac
            for _, r in subbasins.iterrows()
        }
        _k_snow_def = float(_cfg.PARAMS.get('nonglaciated_k_snow_months', 2.0))
        _k_soil_def = float(_cfg.PARAMS.get('nonglaciated_k_soil_months', 3.0))
        _s_fc_def   = float(_cfg.PARAMS.get('nonglaciated_s_fc_mm', 150.0))

        _per_basin_clim = _esc(gdirs, subbasins, ys=args.ys, ye=args.ye)
        _ngl_ds = _cnr(
            subbasins, _per_basin_clim,
            k_snow_months=_k_snow_def,
            k_soil_months=_k_soil_def,
            s_fc_mm=_s_fc_def,
            nonglaciated_area_km2=_ngl_area_per_sub,
        )
        # Annual mean NGL: sum over sub-basins, average over time
        _q_ngl_monthly = _ngl_ds['Q_ngl_m3s'].sum('HYBAS_ID').values
        _time_vals = _ngl_ds['time'].values
        try:
            _yrs = np.array([pd.Timestamp(t).year for t in _time_vals])
        except Exception:
            _yrs = np.array([int(t) for t in _time_vals])
        _sel = (_yrs >= args.ys) & (_yrs <= args.ye)
        q_ngl_mean = float(np.mean(_q_ngl_monthly[_sel])) if _sel.any() else 0.0

        q_mean_est = max(q_gl_mean + q_ngl_mean, 50.0)
        log.info(
            '  q_gl_mean=%.1f m³/s  q_ngl_mean=%.1f m³/s  '
            '→ synthetic q_mean=%.1f m³/s  (ngl_frac=%.0f%%)',
            q_gl_mean, q_ngl_mean, q_mean_est, _ngl_frac * 100)
        obs_df = _make_synthetic_obs(args.ys, args.ye,
                                     seed=args.seed, q_mean=q_mean_est)

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
