"""
Glacier hydrological post-processing tasks.

Provides routing of OGGM run_with_hydro() output to simulate
discharge at the basin/glacier outlet.  Requires run_with_hydro()
to have been run first (model_diagnostics file must exist).

Phase 1: Single linear reservoir  — route_hydro_output()
Phase 2: Two-component reservoir  — route_hydro_output_2c()
Phase 3: Basin-level aggregation  — aggregate_basin_discharge()
Phase 4: KGE calibration          — calibrate_routing_params()
"""

import json
import logging
import os
import shutil

import numpy as np
import xarray as xr

from oggm import cfg
from oggm.utils import entity_task

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_DEFAULT_K_MONTHS = 3.0        # single-reservoir residence time [months]
_DEFAULT_K_FAST_MONTHS = 1.0   # fast-component (rain) residence time [months]
_DEFAULT_K_SLOW_MONTHS = 6.0   # slow-component (melt) residence time [months]
_RHO_WATER = 1000.0            # kg m-3
_SEC_PER_YEAR = 365.25 * 24 * 3600   # s yr-1
_SEC_PER_MONTH = _SEC_PER_YEAR / 12  # s month-1


# ---------------------------------------------------------------------------
# Phase 1 — core routing primitive (no OGGM dependency)
# ---------------------------------------------------------------------------

def _linear_reservoir(q_in, k, dt=1.0):
    """Single linear reservoir routing.

    Parameters
    ----------
    q_in : array-like, shape (N,)
        Input flux [any consistent unit].
    k : float
        Reservoir residence time [same unit as *dt*].  Must be > 0.
    dt : float
        Timestep size [same unit as *k*].  Must be > 0.

    Returns
    -------
    q_out : np.ndarray, shape (N,)
        Routed output flux.

    Notes
    -----
    Recurrence relation::

        alpha = exp(-dt / k)
        beta  = 1 - alpha
        q_out[t] = beta * q_in[t] + alpha * q_out[t-1]

    Initial condition: steady state, i.e. ``q_out[0] = q_in[0]``.
    """
    if k <= 0:
        raise ValueError(f'k must be positive, got {k}')
    if dt <= 0:
        raise ValueError(f'dt must be positive, got {dt}')

    q_in = np.asarray(q_in, dtype=float)
    alpha = np.exp(-dt / k)
    beta = 1.0 - alpha

    q_out = np.empty_like(q_in)
    q_out[0] = q_in[0]          # steady-state initial condition
    for t in range(1, len(q_in)):
        q_out[t] = beta * q_in[t] + alpha * q_out[t - 1]
    return q_out


# ---------------------------------------------------------------------------
# Phase 1 — entity task
# ---------------------------------------------------------------------------

@entity_task(log, writes=['model_diagnostics'])
def route_hydro_output(gdir, filesuffix='', k_months=None,
                       output_filesuffix=None):
    """Route glacier runoff to discharge using a single linear reservoir.

    Must be called **after** :func:`oggm.tasks.run_with_hydro`.  Reads
    the ``model_diagnostics`` netCDF file and appends two new variables:

    * ``discharge_m3s`` — routed discharge at the glacier outlet [m³ s⁻¹]
    * ``runoff_m3s``    — unrouted runoff (instantaneous) [m³ s⁻¹]

    Parameters
    ----------
    gdir : :class:`oggm.GlacierDirectory`
    filesuffix : str
        Suffix of the ``model_diagnostics`` file written by
        ``run_with_hydro()``.
    k_months : float, optional
        Reservoir residence time [months].  Defaults to
        ``cfg.PARAMS['routing_k_months']`` (fallback: 3.0).
    output_filesuffix : str, optional
        Filesuffix for the output ``model_diagnostics`` file.  If *None*,
        results are appended in-place to the input file.  If provided, the
        input file is copied to the new path and variables are appended
        there.
    """
    if k_months is None:
        k_months = cfg.PARAMS.get('routing_k_months', _DEFAULT_K_MONTHS)

    # --- Read existing hydro output ---
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    with xr.open_dataset(fpath) as ds:
        time = ds['time'].values
        runoff_kgyr = (
            ds['melt_on_glacier'].values +
            ds['melt_off_glacier'].values +
            ds['liq_prcp_on_glacier'].values +
            ds['liq_prcp_off_glacier'].values
        )

    # Identify valid (non-NaN) timesteps — OGGM convention: last year is NaN
    valid = ~np.isnan(runoff_kgyr)
    if not valid.any():
        raise RuntimeError(
            f'All runoff values are NaN in {fpath}. '
            'Did run_with_hydro() complete successfully?'
        )
    runoff_valid = runoff_kgyr[valid]

    # --- Unit conversion: kg yr⁻¹ → m³ s⁻¹ ---
    q_m3s = runoff_valid / _RHO_WATER / _SEC_PER_YEAR

    # --- Apply single linear reservoir ---
    # k in years for annual timestep (dt = 1 yr)
    k_years = k_months / 12.0
    q_routed = _linear_reservoir(q_m3s, k=k_years, dt=1.0)

    # Pad back to full time-axis length (NaN for invalid positions) so the
    # 'time' dimension size matches the existing file when appending.
    n_full = len(time)
    q_m3s_full = np.full(n_full, np.nan)
    q_routed_full = np.full(n_full, np.nan)
    q_m3s_full[valid] = q_m3s
    q_routed_full[valid] = q_routed

    # --- Build output dataset ---
    out_ds = xr.Dataset()
    out_ds.coords['time'] = time          # full length, matches existing file

    out_ds['discharge_m3s'] = ('time', q_routed_full)
    out_ds['discharge_m3s'].attrs = {
        'description': 'Routed glacier discharge at outlet',
        'units': 'm3 s-1',
        'routing_scheme': 'single_linear_reservoir',
        'k_months': float(k_months),
    }

    out_ds['runoff_m3s'] = ('time', q_m3s_full)
    out_ds['runoff_m3s'].attrs = {
        'description': 'Unrouted glacier runoff (instantaneous)',
        'units': 'm3 s-1',
    }

    # --- Write to file ---
    if output_filesuffix is not None:
        write_path = gdir.get_filepath('model_diagnostics',
                                       filesuffix=output_filesuffix)
        shutil.copy(fpath, write_path)
    else:
        write_path = fpath

    out_ds.to_netcdf(write_path, mode='a')
    log.debug('(%s) route_hydro_output done (k=%.1f months)', gdir.rgi_id,
              k_months)


# ---------------------------------------------------------------------------
# Phase 2 — two-component entity task
# ---------------------------------------------------------------------------

@entity_task(log, writes=['model_diagnostics'])
def route_hydro_output_2c(gdir, filesuffix='',
                          k_fast_months=None,
                          k_slow_months=None,
                          output_filesuffix=None):
    """Route runoff using a two-component linear reservoir model.

    Separates runoff into a *fast* rain component and a *slow* melt
    component, routes each through its own reservoir, then sums to total
    discharge.

    Component assignment (annual data)::

        fast = liq_prcp_on_glacier + liq_prcp_off_glacier   (rain → quick)
        slow = melt_on_glacier     + melt_off_glacier        (melt → delayed)

    Must be called **after** :func:`oggm.tasks.run_with_hydro`.

    Parameters
    ----------
    gdir : :class:`oggm.GlacierDirectory`
    filesuffix : str
        Suffix of the ``model_diagnostics`` file from ``run_with_hydro()``.
    k_fast_months : float, optional
        Residence time for rain component [months].
        Defaults to ``cfg.PARAMS['routing_k_fast_months']`` (fallback: 1.0).
    k_slow_months : float, optional
        Residence time for melt component [months].
        Defaults to ``cfg.PARAMS['routing_k_slow_months']`` (fallback: 6.0).
    output_filesuffix : str, optional
        Filesuffix for the output file.  If *None*, appends in-place.
    """
    if k_fast_months is None:
        k_fast_months = cfg.PARAMS.get('routing_k_fast_months',
                                       _DEFAULT_K_FAST_MONTHS)
    if k_slow_months is None:
        k_slow_months = cfg.PARAMS.get('routing_k_slow_months',
                                       _DEFAULT_K_SLOW_MONTHS)

    fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    with xr.open_dataset(fpath) as ds:
        time = ds['time'].values
        rain_kgyr = (ds['liq_prcp_on_glacier'].values +
                     ds['liq_prcp_off_glacier'].values)
        melt_kgyr = (ds['melt_on_glacier'].values +
                     ds['melt_off_glacier'].values)

    # Identify valid (non-NaN) timesteps
    valid = ~np.isnan(rain_kgyr + melt_kgyr)
    if not valid.any():
        raise RuntimeError(
            f'All runoff values are NaN in {fpath}. '
            'Did run_with_hydro() complete successfully?'
        )
    rain_valid = rain_kgyr[valid]
    melt_valid = melt_kgyr[valid]

    # Unit conversion: kg yr⁻¹ → m³ s⁻¹
    rain_m3s = rain_valid / _RHO_WATER / _SEC_PER_YEAR
    melt_m3s = melt_valid / _RHO_WATER / _SEC_PER_YEAR

    # Routing (k in years for annual dt=1 yr)
    q_fast = _linear_reservoir(rain_m3s, k=k_fast_months / 12.0, dt=1.0)
    q_slow = _linear_reservoir(melt_m3s, k=k_slow_months / 12.0, dt=1.0)
    q_total = q_fast + q_slow

    # Pad back to full length so 'time' dimension matches the existing file
    n_full = len(time)
    def _pad(arr):
        out = np.full(n_full, np.nan)
        out[valid] = arr
        return out

    # Build output dataset
    out_ds = xr.Dataset()
    out_ds.coords['time'] = time          # full length

    out_ds['discharge_m3s'] = ('time', _pad(q_total))
    out_ds['discharge_m3s'].attrs = {
        'description': 'Total routed discharge (fast + slow components)',
        'units': 'm3 s-1',
        'routing_scheme': 'two_component_linear_reservoir',
        'k_fast_months': float(k_fast_months),
        'k_slow_months': float(k_slow_months),
    }
    out_ds['discharge_fast_m3s'] = ('time', _pad(q_fast))
    out_ds['discharge_fast_m3s'].attrs = {
        'description': 'Routed rain (fast) component',
        'units': 'm3 s-1',
    }
    out_ds['discharge_slow_m3s'] = ('time', _pad(q_slow))
    out_ds['discharge_slow_m3s'].attrs = {
        'description': 'Routed melt (slow) component',
        'units': 'm3 s-1',
    }
    out_ds['rain_m3s'] = ('time', _pad(rain_m3s))
    out_ds['rain_m3s'].attrs = {
        'description': 'Unrouted rain runoff',
        'units': 'm3 s-1',
    }
    out_ds['melt_m3s'] = ('time', _pad(melt_m3s))
    out_ds['melt_m3s'].attrs = {
        'description': 'Unrouted melt runoff',
        'units': 'm3 s-1',
    }

    if output_filesuffix is not None:
        write_path = gdir.get_filepath('model_diagnostics',
                                       filesuffix=output_filesuffix)
        shutil.copy(fpath, write_path)
    else:
        write_path = fpath

    out_ds.to_netcdf(write_path, mode='a')
    log.debug('(%s) route_hydro_output_2c done (k_fast=%.1f, k_slow=%.1f)',
              gdir.rgi_id, k_fast_months, k_slow_months)


# ---------------------------------------------------------------------------
# Phase 3 — basin-level aggregation (not an entity_task — operates on a list)
# ---------------------------------------------------------------------------

def aggregate_basin_discharge(gdirs, filesuffix='', basin_id=None):
    """Aggregate routed discharge from all glaciers in a basin.

    Sums ``discharge_m3s`` across all provided glacier directories.  All
    glaciers must have been processed by :func:`route_hydro_output` (or
    :func:`route_hydro_output_2c`) with the same *filesuffix* and must share
    an identical ``time`` coordinate.

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
        All glaciers in the basin.
    filesuffix : str
        Filesuffix of the ``model_diagnostics`` file to read.
    basin_id : str, optional
        Label for the basin (stored in output attributes).

    Returns
    -------
    ds_basin : :class:`xarray.Dataset`
        Dataset with ``basin_discharge_m3s`` on the shared time axis.

    Raises
    ------
    RuntimeError
        If ``discharge_m3s`` is missing from any glacier's file (i.e.
        routing has not yet been applied).
    ValueError
        If time axes are not identical across glaciers.
    """
    all_q = []
    time_ref = None

    for gdir in gdirs:
        fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
        with xr.open_dataset(fpath) as ds:
            if 'discharge_m3s' not in ds:
                raise RuntimeError(
                    f'discharge_m3s not found in {fpath}. '
                    f'Run route_hydro_output() first for {gdir.rgi_id}.'
                )
            q = ds['discharge_m3s'].values
            t = ds['time'].values

        if time_ref is None:
            time_ref = t
        elif not np.array_equal(t, time_ref):
            raise ValueError(
                f'Time axis mismatch for {gdir.rgi_id}. '
                'All glaciers must use the same ys, ye, and filesuffix.'
            )
        all_q.append(q)

    q_total = np.nansum(np.stack(all_q, axis=0), axis=0)

    ds_basin = xr.Dataset()
    ds_basin.coords['time'] = time_ref
    ds_basin['basin_discharge_m3s'] = ('time', q_total)
    ds_basin['basin_discharge_m3s'].attrs = {
        'description': 'Total basin discharge (sum over all glaciers)',
        'units': 'm3 s-1',
        'n_glaciers': len(gdirs),
        'basin_id': str(basin_id) if basin_id is not None else 'unknown',
    }
    return ds_basin


# ---------------------------------------------------------------------------
# Phase 4 — Kling-Gupta calibration
# ---------------------------------------------------------------------------

def _kling_gupta_efficiency(q_sim, q_obs):
    """Kling-Gupta Efficiency (KGE; Gupta et al. 2009).

    .. math::

        KGE = 1 - \\sqrt{(r-1)^2 + (\\alpha-1)^2 + (\\beta-1)^2}

    where :math:`r` is the Pearson correlation coefficient,
    :math:`\\alpha = \\sigma_{sim}/\\sigma_{obs}` the variability ratio,
    and :math:`\\beta = \\mu_{sim}/\\mu_{obs}` the bias ratio.

    Parameters
    ----------
    q_sim : array-like, shape (N,)
        Simulated discharge [any consistent unit].
    q_obs : array-like, shape (N,)
        Observed discharge [same unit].

    Returns
    -------
    kge : float
        KGE value in (−∞, 1].  Perfect score = 1.0.
        Returns ``-np.inf`` if fewer than 2 finite paired values exist.
    """
    q_sim = np.asarray(q_sim, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)
    mask = np.isfinite(q_sim) & np.isfinite(q_obs)
    if mask.sum() < 2:
        return -np.inf
    qs, qo = q_sim[mask], q_obs[mask]
    r = np.corrcoef(qs, qo)[0, 1]
    alpha = qs.std() / qo.std() if qo.std() > 0 else np.inf
    beta = qs.mean() / qo.mean() if qo.mean() > 0 else np.inf
    return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def _extract_model_years(time_values):
    """Extract integer calendar years from an OGGM time coordinate.

    Handles both :class:`cftime.datetime` objects (the common OGGM case)
    and numpy/pandas datetime64 values.

    Parameters
    ----------
    time_values : array-like
        Time coordinate values from an OGGM ``model_diagnostics`` dataset.

    Returns
    -------
    years : np.ndarray of int
    """
    years = []
    for t in time_values:
        if hasattr(t, 'year'):                                # cftime / pandas Timestamp
            years.append(int(t.year))
        elif isinstance(t, (int, float, np.integer, np.floating)):
            # OGGM stores annual time as bare numeric year (int or float like 1985.0)
            years.append(int(round(float(t))))
        else:                                                 # numpy datetime64
            import pandas as pd
            years.append(int(pd.Timestamp(t).year))
    return np.array(years, dtype=int)


def calibrate_routing_params(gdir, obs_discharge_m3s, obs_years,
                              filesuffix='', scheme='single',
                              method='Nelder-Mead',
                              output_filesuffix=None):
    """Calibrate reservoir residence time(s) against observed annual discharge.

    Minimises ``1 − KGE`` with respect to the routing residence time(s)
    using :func:`scipy.optimize.minimize`.  Model runoff is read from the
    ``model_diagnostics`` netCDF **once**; routing is performed purely in
    memory during optimisation (no file I/O per iteration).

    The calibrated parameters are written to
    ``<gdir.dir>/hydro_calib_params.json`` and returned as a dict.

    Parameters
    ----------
    gdir : :class:`oggm.GlacierDirectory`
    obs_discharge_m3s : array-like, shape (N,)
        Observed **annual** discharge [m³ s⁻¹].
    obs_years : array-like of int, shape (N,)
        Calendar years corresponding to *obs_discharge_m3s*.
    filesuffix : str
        Suffix of the ``model_diagnostics`` file from ``run_with_hydro()``.
    scheme : {'single', 'two_component'}
        Routing model to calibrate.  ``'single'`` optimises one parameter
        (``k_months``); ``'two_component'`` optimises two
        (``k_fast_months``, ``k_slow_months``).
    method : str
        :func:`scipy.optimize.minimize` method (default: ``'Nelder-Mead'``).
    output_filesuffix : str, optional
        If provided, the calibrated routing result is written to a new
        ``model_diagnostics`` file with this suffix.

    Returns
    -------
    calib_result : dict
        Contains: ``'scheme'``, ``'kge'``, ``'n_obs'``, ``'success'``,
        ``'obs_years_range'``, and either ``'k_months'`` (single) or
        ``'k_fast_months'`` + ``'k_slow_months'`` (two_component).

    Raises
    ------
    ValueError
        If *scheme* is unknown, arrays lengths differ, or fewer than 3
        overlapping years exist between model output and observations.
    ImportError
        If :mod:`scipy` is not installed.
    """
    try:
        from scipy.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            'scipy is required for calibration.  '
            'Install with: conda install scipy'
        ) from exc

    obs_discharge_m3s = np.asarray(obs_discharge_m3s, dtype=float)
    obs_years = np.asarray(obs_years, dtype=int)

    if len(obs_discharge_m3s) != len(obs_years):
        raise ValueError(
            'obs_discharge_m3s and obs_years must have the same length '
            f'(got {len(obs_discharge_m3s)} and {len(obs_years)})'
        )
    if scheme not in ('single', 'two_component'):
        raise ValueError(
            f"scheme must be 'single' or 'two_component', got {scheme!r}"
        )

    # --- Read model runoff once ---
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    with xr.open_dataset(fpath) as ds:
        time = ds['time'].values
        rain_kgyr = (ds['liq_prcp_on_glacier'].values +
                     ds['liq_prcp_off_glacier'].values)
        melt_kgyr = (ds['melt_on_glacier'].values +
                     ds['melt_off_glacier'].values)

    runoff_kgyr = rain_kgyr + melt_kgyr
    model_years = _extract_model_years(time)
    valid_model = ~np.isnan(runoff_kgyr)

    # --- Align obs and model to overlapping valid years ---
    common_years = np.intersect1d(obs_years, model_years[valid_model])
    if len(common_years) < 3:
        raise ValueError(
            f'Only {len(common_years)} overlapping valid year(s) between '
            f'model ({model_years[valid_model].min()}–'
            f'{model_years[valid_model].max()}) and observations '
            f'({obs_years.min()}–{obs_years.max()}).  Need ≥ 3.'
        )

    obs_mask = np.isin(obs_years, common_years)
    mod_mask = np.isin(model_years, common_years) & valid_model

    q_obs = obs_discharge_m3s[obs_mask]
    rain_m3s = rain_kgyr[mod_mask] / _RHO_WATER / _SEC_PER_YEAR
    melt_m3s = melt_kgyr[mod_mask] / _RHO_WATER / _SEC_PER_YEAR
    runoff_m3s = rain_m3s + melt_m3s

    # --- Define cost function (routing in memory, no I/O) ---
    if scheme == 'single':
        def _cost(params):
            k = float(params[0])
            if k <= 0.01:
                return 2.0           # penalty for non-physical k
            q_sim = _linear_reservoir(runoff_m3s, k=k / 12.0, dt=1.0)
            return 1.0 - _kling_gupta_efficiency(q_sim, q_obs)

        x0 = [cfg.PARAMS.get('routing_k_months', _DEFAULT_K_MONTHS)]
        bounds = [(0.1, 120.0)]

    else:  # two_component
        def _cost(params):
            kf, ks = float(params[0]), float(params[1])
            if kf <= 0.01 or ks <= 0.01:
                return 2.0
            q_fast = _linear_reservoir(rain_m3s, k=kf / 12.0, dt=1.0)
            q_slow = _linear_reservoir(melt_m3s, k=ks / 12.0, dt=1.0)
            return 1.0 - _kling_gupta_efficiency(q_fast + q_slow, q_obs)

        x0 = [cfg.PARAMS.get('routing_k_fast_months', _DEFAULT_K_FAST_MONTHS),
              cfg.PARAMS.get('routing_k_slow_months', _DEFAULT_K_SLOW_MONTHS)]
        bounds = [(0.1, 24.0), (1.0, 120.0)]

    # --- Run optimiser ---
    opt = minimize(_cost, x0, method=method,
                   options={'xatol': 0.01, 'fatol': 0.001, 'maxiter': 500})
    kge_val = float(1.0 - opt.fun)

    # --- Build result dict ---
    if scheme == 'single':
        calib_result = {
            'scheme': 'single',
            'k_months': float(opt.x[0]),
            'kge': kge_val,
            'n_obs': int(len(common_years)),
            'success': bool(opt.success),
            'obs_years_range': [int(common_years.min()), int(common_years.max())],
        }
    else:
        calib_result = {
            'scheme': 'two_component',
            'k_fast_months': float(opt.x[0]),
            'k_slow_months': float(opt.x[1]),
            'kge': kge_val,
            'n_obs': int(len(common_years)),
            'success': bool(opt.success),
            'obs_years_range': [int(common_years.min()), int(common_years.max())],
        }

    log.info('(%s) calibrate_routing_params: KGE=%.3f, success=%s, result=%s',
             gdir.rgi_id, kge_val, opt.success,
             {k: v for k, v in calib_result.items()
              if k not in ('success', 'n_obs', 'obs_years_range', 'scheme')})

    # --- Persist to gdir directory ---
    calib_path = os.path.join(gdir.dir, 'hydro_calib_params.json')
    with open(calib_path, 'w') as fh:
        json.dump(calib_result, fh, indent=2)

    # --- Optionally apply calibrated routing and write output ---
    if output_filesuffix is not None:
        if scheme == 'single':
            route_hydro_output(gdir, filesuffix=filesuffix,
                               k_months=calib_result['k_months'],
                               output_filesuffix=output_filesuffix)
        else:
            route_hydro_output_2c(gdir, filesuffix=filesuffix,
                                  k_fast_months=calib_result['k_fast_months'],
                                  k_slow_months=calib_result['k_slow_months'],
                                  output_filesuffix=output_filesuffix)

    return calib_result


# ---------------------------------------------------------------------------
# Phase 5 — five-component entity task
# ---------------------------------------------------------------------------

_DEFAULT_K_RAIN_MONTHS = 0.5   # rain on/off glacier [months]
_DEFAULT_K_SNOW_MONTHS = 2.0   # snowmelt on/off glacier [months]
_DEFAULT_K_ICE_MONTHS = 8.0    # ice melt (subglacial drainage) [months]


@entity_task(log, writes=['model_diagnostics'])
def route_hydro_output_5c(gdir, filesuffix='',
                           k_rain_months=None,
                           k_snow_months=None,
                           k_ice_months=None,
                           output_filesuffix=None):
    """Route runoff using a five-component linear reservoir model.

    Separates runoff into five physically distinct components, routes each
    through its own linear reservoir, then sums to total discharge.

    Requires :func:`oggm.tasks.run_with_hydro` to have been run **after**
    the Phase 5 update to ``flowline.py`` that adds per-band SWE tracking.
    The ``model_diagnostics`` file must contain ``snowmelt_on_glacier`` and
    ``icemelt_on_glacier``.

    Component assignment
    --------------------
    * ``liq_prcp_on_glacier``  → fast rain-on-ice reservoir  (k_rain_months)
    * ``liq_prcp_off_glacier`` → fast rain-off-ice reservoir (k_rain_months)
    * ``snowmelt_on_glacier``  → medium snowmelt reservoir   (k_snow_months)
    * ``melt_off_glacier``     → off-glacier snowmelt        (k_snow_months * 1.5)
    * ``icemelt_on_glacier``   → slow ice-melt reservoir     (k_ice_months)

    Parameters
    ----------
    gdir : :class:`oggm.GlacierDirectory`
    filesuffix : str
        Suffix of the ``model_diagnostics`` file from ``run_with_hydro()``.
    k_rain_months : float, optional
        Residence time for rain components [months].
        Defaults to ``cfg.PARAMS['routing_k_rain_months']`` (fallback: 0.5).
    k_snow_months : float, optional
        Residence time for on-glacier snowmelt [months].
        Defaults to ``cfg.PARAMS['routing_k_snow_months']`` (fallback: 2.0).
    k_ice_months : float, optional
        Residence time for ice melt [months].
        Defaults to ``cfg.PARAMS['routing_k_ice_months']`` (fallback: 8.0).
    output_filesuffix : str, optional
        Filesuffix for the output file.  If *None*, appends in-place.

    Raises
    ------
    RuntimeError
        If ``snowmelt_on_glacier`` or ``icemelt_on_glacier`` are missing
        (Phase 5 SWE split not available in the diagnostics file).
    """
    if k_rain_months is None:
        k_rain_months = cfg.PARAMS.get('routing_k_rain_months',
                                       _DEFAULT_K_RAIN_MONTHS)
    if k_snow_months is None:
        k_snow_months = cfg.PARAMS.get('routing_k_snow_months',
                                       _DEFAULT_K_SNOW_MONTHS)
    if k_ice_months is None:
        k_ice_months = cfg.PARAMS.get('routing_k_ice_months',
                                      _DEFAULT_K_ICE_MONTHS)

    fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    with xr.open_dataset(fpath) as ds:
        if 'snowmelt_on_glacier' not in ds or 'icemelt_on_glacier' not in ds:
            raise RuntimeError(
                'snowmelt_on_glacier / icemelt_on_glacier not found in '
                f'{fpath}. Re-run run_with_hydro() after the Phase 5 '
                'update to flowline.py, and ensure snowmelt_on_glacier '
                'and icemelt_on_glacier are in store_diagnostic_variables.'
            )
        time = ds['time'].values
        rain_on_kgyr = ds['liq_prcp_on_glacier'].values
        rain_off_kgyr = ds['liq_prcp_off_glacier'].values
        snowmelt_on_kgyr = ds['snowmelt_on_glacier'].values
        snowmelt_off_kgyr = ds['melt_off_glacier'].values
        icemelt_kgyr = ds['icemelt_on_glacier'].values

    # Identify valid (non-NaN) timesteps
    combined = (rain_on_kgyr + rain_off_kgyr + snowmelt_on_kgyr +
                snowmelt_off_kgyr + icemelt_kgyr)
    valid = ~np.isnan(combined)
    if not valid.any():
        raise RuntimeError(
            f'All runoff values are NaN in {fpath}. '
            'Did run_with_hydro() complete successfully?'
        )

    # Unit conversion: kg yr-1 → m3 s-1
    def _to_m3s(arr):
        return arr[valid] / _RHO_WATER / _SEC_PER_YEAR

    rain_on_m3s = _to_m3s(rain_on_kgyr)
    rain_off_m3s = _to_m3s(rain_off_kgyr)
    snowmelt_on_m3s = _to_m3s(snowmelt_on_kgyr)
    snowmelt_off_m3s = _to_m3s(snowmelt_off_kgyr)
    icemelt_m3s = _to_m3s(icemelt_kgyr)

    # Route each component (k in years for annual dt=1 yr)
    q_rain_on = _linear_reservoir(
        rain_on_m3s, k=k_rain_months / 12.0, dt=1.0)
    q_rain_off = _linear_reservoir(
        rain_off_m3s, k=k_rain_months / 12.0, dt=1.0)
    q_snowmelt_on = _linear_reservoir(
        snowmelt_on_m3s, k=k_snow_months / 12.0, dt=1.0)
    q_snowmelt_off = _linear_reservoir(
        snowmelt_off_m3s, k=(k_snow_months * 1.5) / 12.0, dt=1.0)
    q_icemelt = _linear_reservoir(
        icemelt_m3s, k=k_ice_months / 12.0, dt=1.0)
    q_total = (q_rain_on + q_rain_off + q_snowmelt_on +
               q_snowmelt_off + q_icemelt)

    # Pad back to full time-axis length (NaN for invalid positions)
    n_full = len(time)

    def _pad(arr):
        out = np.full(n_full, np.nan)
        out[valid] = arr
        return out

    # Build output dataset
    out_ds = xr.Dataset()
    out_ds.coords['time'] = time

    out_ds['discharge_5c_m3s'] = ('time', _pad(q_total))
    out_ds['discharge_5c_m3s'].attrs = {
        'description': 'Total routed discharge (5-component)',
        'units': 'm3 s-1',
        'routing_scheme': 'five_component_linear_reservoir',
        'k_rain_months': float(k_rain_months),
        'k_snow_months': float(k_snow_months),
        'k_ice_months': float(k_ice_months),
    }
    out_ds['discharge_rain_on_m3s'] = ('time', _pad(q_rain_on))
    out_ds['discharge_rain_on_m3s'].attrs = {
        'description': 'Routed rain-on-glacier component',
        'units': 'm3 s-1',
    }
    out_ds['discharge_rain_off_m3s'] = ('time', _pad(q_rain_off))
    out_ds['discharge_rain_off_m3s'].attrs = {
        'description': 'Routed rain-off-glacier component',
        'units': 'm3 s-1',
    }
    out_ds['discharge_snowmelt_on_m3s'] = ('time', _pad(q_snowmelt_on))
    out_ds['discharge_snowmelt_on_m3s'].attrs = {
        'description': 'Routed on-glacier snowmelt component',
        'units': 'm3 s-1',
    }
    out_ds['discharge_snowmelt_off_m3s'] = ('time', _pad(q_snowmelt_off))
    out_ds['discharge_snowmelt_off_m3s'].attrs = {
        'description': 'Routed off-glacier snowmelt component',
        'units': 'm3 s-1',
    }
    out_ds['discharge_icemelt_m3s'] = ('time', _pad(q_icemelt))
    out_ds['discharge_icemelt_m3s'].attrs = {
        'description': 'Routed ice-melt component',
        'units': 'm3 s-1',
    }

    if output_filesuffix is not None:
        write_path = gdir.get_filepath('model_diagnostics',
                                       filesuffix=output_filesuffix)
        shutil.copy(fpath, write_path)
    else:
        write_path = fpath

    out_ds.to_netcdf(write_path, mode='a')
    log.debug(
        '(%s) route_hydro_output_5c done (k_rain=%.1f, k_snow=%.1f, '
        'k_ice=%.1f months)',
        gdir.rgi_id, k_rain_months, k_snow_months, k_ice_months,
    )


# ---------------------------------------------------------------------------
# Phase 7 — Muskingum-Cunge channel routing entity task
# ---------------------------------------------------------------------------

def _compute_node_contributions(G, acc_arr):
    """Compute fractional drainage contribution for each stream network node.

    Each node's contribution is its flow accumulation minus the sum of
    accumulations at all immediate upstream (predecessor) network nodes.
    This gives the number of cells that drain *directly* to the node without
    first passing through another stream node.

    Parameters
    ----------
    G : nx.DiGraph
        Stream network from
        :func:`~oggm.core.terrain_routing.build_stream_network`.
    acc_arr : np.ndarray, shape (nrows, ncols)
        Flow accumulation grid aligned with the network's DEM.

    Returns
    -------
    fractions : dict
        ``{(r, c): float}`` — normalised fractional contribution for each
        node, summing to 1.0 across all nodes.
    """
    local_acc = {}
    for node in G.nodes():
        r, c = node
        contrib = float(acc_arr[r, c])
        for pred in G.predecessors(node):
            pr, pc = pred
            contrib -= float(acc_arr[pr, pc])
        local_acc[node] = max(contrib, 1.0)  # at least 1 cell

    total = max(sum(local_acc.values()), 1.0)
    return {k: v / total for k, v in local_acc.items()}


def _write_trivial_routing(gdir, filesuffix='', output_filesuffix=None):
    """Copy hillslope discharge directly to terrain routing output.

    Fallback used when the glacier is too small to build a stream network
    or when no stream cells are found above the delineation threshold.
    The Phase 5 (or Phase 2 / Phase 1) discharge is written as
    ``discharge_terrain_m3s`` without any further modification.
    """
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    with xr.open_dataset(fpath) as ds:
        time = ds['time'].values
        if 'discharge_5c_m3s' in ds:
            q_passthrough = ds['discharge_5c_m3s'].values.copy()
        elif 'discharge_m3s' in ds:
            q_passthrough = ds['discharge_m3s'].values.copy()
        else:
            q_passthrough = np.full(len(time), np.nan)

    out_ds = xr.Dataset()
    out_ds.coords['time'] = time
    out_ds['discharge_terrain_m3s'] = ('time', q_passthrough)
    out_ds['discharge_terrain_m3s'].attrs = {
        'description': (
            'Glacier discharge (trivial pass-through; '
            'no stream network available for channel routing)'
        ),
        'units': 'm3 s-1',
        'routing_scheme': 'trivial',
    }

    if output_filesuffix is not None:
        write_path = gdir.get_filepath('model_diagnostics',
                                       filesuffix=output_filesuffix)
        shutil.copy(fpath, write_path)
    else:
        write_path = fpath

    out_ds.to_netcdf(write_path, mode='a')


@entity_task(log, writes=['model_diagnostics'])
def compute_channel_routing(gdir, filesuffix='',
                             stream_threshold_cells=None,
                             muskingum_X=None,
                             celerity_m_per_s=None,
                             output_filesuffix=None):
    """Route per-glacier discharge through the terrain stream network.

    Builds a DEM-derived stream network (Phase 6) from the glacier's
    ``gridded_data`` and routes the prior hillslope-routed discharge
    (Phase 5 or Phase 2) through the channel network using
    Muskingum-Cunge channel routing (Phase 7).

    Must be called **after** :func:`route_hydro_output_5c` (or
    :func:`route_hydro_output`), which provides the hillslope-to-channel
    routed discharge in ``model_diagnostics``.

    Parameters
    ----------
    gdir : :class:`oggm.GlacierDirectory`
    filesuffix : str
        Suffix of the ``model_diagnostics`` file containing the prior
        hillslope-routing output (``discharge_5c_m3s`` or
        ``discharge_m3s``).
    stream_threshold_cells : int, optional
        Minimum upstream cell count to define a stream.
        Defaults to ``cfg.PARAMS.get('stream_threshold_cells', 5)``.
    muskingum_X : float, optional
        Muskingum attenuation coefficient [0, 0.5].
        X = 0 → maximum attenuation; X = 0.5 → pure translation.
        Defaults to ``cfg.PARAMS.get('muskingum_X', 0.25)``.
    celerity_m_per_s : float, optional
        Kinematic wave celerity [m s⁻¹] for reach travel-time
        K = L / c_k.
        Defaults to ``cfg.PARAMS.get('channel_celerity_m_per_s', 1.5)``.
    output_filesuffix : str, optional
        Filesuffix for the output ``model_diagnostics`` file.
        If *None*, results are appended in-place to the input file.

    Writes
    ------
    ``discharge_terrain_m3s`` : m³ s⁻¹
        Muskingum-Cunge routed discharge at the glacier stream outlet.
        Metadata attributes include routing parameters and network
        statistics.

    Notes
    -----
    Lateral inflow to each stream reach is proportional to its local
    drainage contribution: the node's flow accumulation minus the sum of
    immediate upstream network predecessors' accumulations (computed via
    :func:`_compute_node_contributions`).

    At the annual timestep (typical OGGM output) the Muskingum-Cunge
    travel-time correction is very small (K ≈ seconds–hours vs
    Δt = 1 year), so ``discharge_terrain_m3s ≈ discharge_5c_m3s``.
    The network structure is preserved for sub-annual routing (Phase 9)
    and multi-glacier basin aggregation (Phase 8).

    If the glacier DEM produces no stream cells above
    *stream_threshold_cells*, a trivial pass-through is applied and a
    warning is logged (see :func:`_write_trivial_routing`).
    """
    from oggm.core.terrain_routing import (
        compute_flow_direction,
        compute_flow_accumulation,
        delineate_streams,
        build_stream_network,
        route_stream_network,
    )

    # --- Parameters ---
    if stream_threshold_cells is None:
        stream_threshold_cells = int(
            cfg.PARAMS.get('stream_threshold_cells', 5))
    if muskingum_X is None:
        muskingum_X = float(cfg.PARAMS.get('muskingum_X', 0.25))
    if celerity_m_per_s is None:
        celerity_m_per_s = float(
            cfg.PARAMS.get('channel_celerity_m_per_s', 1.5))

    # --- Read DEM from gridded_data ---
    gdata_path = gdir.get_filepath('gridded_data')
    with xr.open_dataset(gdata_path) as ds:
        dem_arr = ds['topo'].values.astype(float)
        if 'glacier_mask' in ds:
            glacier_mask = ds['glacier_mask'].values.astype(bool)
        else:
            glacier_mask = np.ones(dem_arr.shape, dtype=bool)

    # Restrict to glacier domain; non-glacier cells are set to NaN
    dem_glacier = np.where(glacier_mask, dem_arr, np.nan)

    # Cell size in metres (abs because some grids have negative dy)
    cellsize_m = abs(float(gdir.grid.dx))

    # --- Phase 6: terrain analysis ---
    fdir = compute_flow_direction(dem_glacier, cellsize_m,
                                  fill_pits_first=True)
    acc = compute_flow_accumulation(fdir)
    streams = delineate_streams(acc, threshold_cells=stream_threshold_cells)

    if not streams.any():
        log.warning(
            '(%s) No stream cells found (threshold=%d cells); '
            'applying trivial pass-through routing.',
            gdir.rgi_id, stream_threshold_cells,
        )
        _write_trivial_routing(gdir, filesuffix=filesuffix,
                               output_filesuffix=output_filesuffix)
        return

    G = build_stream_network(streams, fdir, dem_glacier, cellsize_m,
                             acc_arr=acc)

    if G.number_of_edges() == 0:
        log.warning(
            '(%s) Stream network has no edges (glacier too small); '
            'applying trivial pass-through routing.',
            gdir.rgi_id,
        )
        _write_trivial_routing(gdir, filesuffix=filesuffix,
                               output_filesuffix=output_filesuffix)
        return

    # --- Read hillslope-routed discharge from model_diagnostics ---
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    with xr.open_dataset(fpath) as ds:
        time = ds['time'].values
        if 'discharge_5c_m3s' in ds:
            q_hillslope_full = ds['discharge_5c_m3s'].values.copy()
        elif 'discharge_m3s' in ds:
            q_hillslope_full = ds['discharge_m3s'].values.copy()
        else:
            raise RuntimeError(
                f'No routed discharge found in {fpath}. '
                'Run route_hydro_output_5c() or route_hydro_output() first.'
            )

    valid = ~np.isnan(q_hillslope_full)
    if not valid.any():
        raise RuntimeError(
            f'All discharge values are NaN in {fpath}. '
            'Did run_with_hydro() and a routing step complete successfully?'
        )
    q_hillslope = q_hillslope_full[valid]   # shape (n_valid,)

    # --- Build lateral inflow dict ---
    # Distribute total discharge across nodes proportional to each node's
    # local drainage contribution (= its accumulation minus predecessors').
    fractions = _compute_node_contributions(G, acc)
    lateral_inflow_dict = {
        node: q_hillslope * frac
        for node, frac in fractions.items()
        if frac > 0
    }

    # --- Muskingum-Cunge routing through the stream network ---
    q_outlet, _ = route_stream_network(
        G,
        lateral_inflow_dict=lateral_inflow_dict,
        dt_seconds=_SEC_PER_YEAR,
        muskingum_X=muskingum_X,
        celerity_m_per_s=celerity_m_per_s,
    )

    # Pad back to full time length (NaN at invalid timesteps)
    n_full = len(time)
    q_terrain_full = np.full(n_full, np.nan)
    q_terrain_full[valid] = q_outlet

    # --- Build output dataset ---
    out_ds = xr.Dataset()
    out_ds.coords['time'] = time

    out_ds['discharge_terrain_m3s'] = ('time', q_terrain_full)
    out_ds['discharge_terrain_m3s'].attrs = {
        'description': (
            'Muskingum-Cunge channel-routed discharge at glacier outlet'
        ),
        'units': 'm3 s-1',
        'routing_scheme': 'muskingum_cunge',
        'muskingum_X': float(muskingum_X),
        'celerity_m_per_s': float(celerity_m_per_s),
        'stream_threshold_cells': int(stream_threshold_cells),
        'n_stream_nodes': int(G.number_of_nodes()),
        'n_stream_reaches': int(G.number_of_edges()),
    }

    # --- Write to file ---
    if output_filesuffix is not None:
        write_path = gdir.get_filepath('model_diagnostics',
                                       filesuffix=output_filesuffix)
        shutil.copy(fpath, write_path)
    else:
        write_path = fpath

    out_ds.to_netcdf(write_path, mode='a')
    log.debug(
        '(%s) compute_channel_routing done (%d nodes, %d edges, '
        'X=%.2f, c=%.1f m/s)',
        gdir.rgi_id, G.number_of_nodes(), G.number_of_edges(),
        muskingum_X, celerity_m_per_s,
    )
