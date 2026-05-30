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
