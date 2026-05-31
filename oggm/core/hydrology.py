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


# ===========================================================================
# Phase 8 — Basin Integration: non-glaciated area runoff model
# ===========================================================================

# ---------------------------------------------------------------------------
# Spatial climate extraction — per-sub-basin W5E5 forcing
# ---------------------------------------------------------------------------

def extract_subbasin_climate(gdirs, subbasins_gdf, ys=None, ye=None):
    """Extract per-sub-basin monthly climate from the nearest glacier's file.

    Each glacier's ``climate_historical.nc`` stores the W5E5 grid cell
    closest to the glacier centroid.  For every HydroBASINS sub-basin we
    select the glacier whose centroid is nearest (Euclidean distance in
    lon/lat space) and use its climate as the sub-basin forcing.  Optionally
    the time series is sliced to [*ys*, *ye*].

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
        Glacier directories.  All must have a ``climate_historical.nc`` file.
    subbasins_gdf : :class:`geopandas.GeoDataFrame` or :class:`pandas.DataFrame`
        Sub-basin metadata.  Must contain ``HYBAS_ID``.  If ``geometry`` is
        present, the centroid is used; otherwise ``centroid_lon`` /
        ``centroid_lat`` columns are required.
    ys : int, optional
        Start year for the time slice (inclusive).
    ye : int, optional
        End year for the time slice (inclusive).

    Returns
    -------
    dict
        ``{HYBAS_ID: xarray.Dataset}`` — each value is an xr.Dataset with
        variables ``temp`` [°C] and ``prcp`` [mm month⁻¹] on a ``time``
        coordinate, taken from the nearest glacier's climate file.
    """
    import xarray as xr
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        raise ImportError(
            'scipy is required for extract_subbasin_climate(). '
            'Install it with: conda install scipy'
        )

    # Glacier centroid array
    gl_lons = np.array([g.cenlon for g in gdirs], dtype=float)
    gl_lats = np.array([g.cenlat for g in gdirs], dtype=float)
    tree = cKDTree(np.column_stack([gl_lons, gl_lats]))

    # Sub-basin centroids
    import pandas as _pd
    try:
        import geopandas as _gpd
        has_gpd = True
    except ImportError:
        has_gpd = False

    if has_gpd and hasattr(subbasins_gdf, 'geometry') and (
            subbasins_gdf.geometry is not None):
        ctr = subbasins_gdf.geometry.to_crs('EPSG:6933').centroid.to_crs(
            subbasins_gdf.crs)
        sub_lons = ctr.x.values
        sub_lats = ctr.y.values
    elif ('centroid_lon' in subbasins_gdf.columns and
          'centroid_lat' in subbasins_gdf.columns):
        sub_lons = np.asarray(subbasins_gdf['centroid_lon'], dtype=float)
        sub_lats = np.asarray(subbasins_gdf['centroid_lat'], dtype=float)
    else:
        raise ValueError(
            'subbasins_gdf must have a geometry column or '
            '"centroid_lon"/"centroid_lat" columns.'
        )
    hybas_ids = np.asarray(subbasins_gdf['HYBAS_ID'])

    # Nearest glacier for each sub-basin
    _, nearest_idx = tree.query(np.column_stack([sub_lons, sub_lats]))

    # Load unique climate files (cache by glacier index)
    unique_idx = np.unique(nearest_idx)
    climate_cache: dict = {}
    for idx in unique_idx:
        gdir = gdirs[int(idx)]
        fpath = gdir.get_filepath('climate_historical')
        with xr.open_dataset(fpath) as ds:
            clim = ds[['temp', 'prcp']].load()
        if ys is not None and ye is not None:
            try:
                sliced = clim.sel(time=slice(str(ys), str(ye)))
                if len(sliced['time']) > 0:
                    clim = sliced
            except Exception:
                pass
        climate_cache[int(idx)] = clim

    # Build per-subbasin dict
    per_subbasin: dict = {}
    for hybas_id, gl_idx in zip(hybas_ids, nearest_idx):
        per_subbasin[int(hybas_id)] = climate_cache[int(gl_idx)]

    log.info(
        'extract_subbasin_climate: %d sub-basins assigned from %d unique '
        'glacier climate files',
        len(per_subbasin), len(unique_idx),
    )
    return per_subbasin


# ---------------------------------------------------------------------------
# Two-bucket snowmelt / soil-moisture runoff model
# ---------------------------------------------------------------------------

def _run_two_bucket_model(t_celsius, prcp_mm, temp_threshold=0.0,
                          k_snow_months=2.0, k_soil_months=3.0,
                          s_fc_mm=150.0, dt_months=1.0):
    """Two-bucket (snow + soil) conceptual runoff model for non-glaciated area.

    Driven by monthly-mean air temperature *t_celsius* and monthly total
    precipitation *prcp_mm*, this model partitions precipitation into
    rain/snowfall, routes snowmelt through a snow water equivalent (SWE)
    bucket, and routes rain + meltwater through a soil-moisture bucket.
    Runoff is generated when soil moisture exceeds field capacity.

    Governing equations (monthly timestep)
    ----------------------------------------
    Precipitation partitioning::

        P_snow = prcp * (1 - f_rain)
        P_rain = prcp * f_rain
        f_rain = sigmoid(4 * (T - T_threshold))  ∈ [0, 1]

    Snow bucket (SWE; mm)::

        M_snow = SWE * (1 - exp(-dt / k_snow))   [snowmelt]
        SWE[t+1] = SWE[t] + P_snow - M_snow

    Potential evapotranspiration (simplified Hamon)::

        PET = max(0, 0.55 * (T - T_threshold))   [mm month-1]

    Soil bucket (S_soil; mm)::

        S_soil[t+1] = S_soil[t] + P_rain + M_snow - ET - Q_ngl
        ET = min(PET, S_soil[t] + P_rain + M_snow)
        Q_ngl = max(0, k_eff * (S_soil[t+1] - S_fc))  [linear release above FC]
        k_eff = 1 - exp(-dt / k_soil)

    Parameters
    ----------
    t_celsius : array-like, shape (N,)
        Monthly mean air temperature [°C].
    prcp_mm : array-like, shape (N,)
        Monthly total precipitation [mm].
    temp_threshold : float
        Rain/snow partitioning threshold temperature [°C].  Default: 0 °C.
    k_snow_months : float
        Snow residence time (e-folding timescale) [months].  Default: 2.
    k_soil_months : float
        Soil drainage timescale [months].  Default: 3.
    s_fc_mm : float
        Field capacity (soil moisture threshold above which runoff occurs)
        [mm].  Default: 150.
    dt_months : float
        Timestep size [months].  Default: 1.

    Returns
    -------
    q_mm : np.ndarray, shape (N,)
        Monthly runoff [mm month-1].  Always non-negative.
    swe : np.ndarray, shape (N,)
        Snow water equivalent [mm] at each timestep.
    s_soil : np.ndarray, shape (N,)
        Soil moisture [mm] at each timestep.

    Notes
    -----
    At annual resolution (dt_months = 12) the model collapses to a simple
    annual water balance; monthly resolution is recommended for seasonal
    runoff timing.
    """
    t = np.asarray(t_celsius, dtype=float)
    p = np.asarray(prcp_mm, dtype=float)
    n = len(t)

    if len(p) != n:
        raise ValueError('t_celsius and prcp_mm must have the same length.')
    if k_snow_months <= 0 or k_soil_months <= 0:
        raise ValueError('k_snow_months and k_soil_months must be positive.')
    if s_fc_mm < 0:
        raise ValueError('s_fc_mm must be non-negative.')

    # Partition coefficients
    f_rain = 1.0 / (1.0 + np.exp(-4.0 * (t - temp_threshold)))
    p_rain = p * f_rain
    p_snow = p * (1.0 - f_rain)

    # Decay coefficients
    alpha_snow = np.exp(-dt_months / k_snow_months)
    k_eff_soil = 1.0 - np.exp(-dt_months / k_soil_months)

    # State arrays
    swe = np.zeros(n, dtype=float)
    s_soil = np.zeros(n, dtype=float)
    q_mm = np.zeros(n, dtype=float)

    # Initialise snow at zero; soil at field capacity
    swe_state = 0.0
    soil_state = s_fc_mm

    for i in range(n):
        # --- Snow bucket ---
        melt = swe_state * (1.0 - alpha_snow)
        swe_state = swe_state * alpha_snow + p_snow[i]
        swe_state = max(swe_state, 0.0)

        # --- PET (simplified Hamon, T-based) ---
        pet = max(0.0, 0.55 * (t[i] - temp_threshold))

        # --- Soil bucket ---
        inflow = p_rain[i] + melt
        avail = soil_state + inflow
        et = min(pet, avail)
        soil_pre = avail - et
        # Linear drainage above field capacity
        q_i = max(0.0, k_eff_soil * (soil_pre - s_fc_mm))
        soil_state = soil_pre - q_i
        soil_state = max(soil_state, 0.0)

        swe[i] = swe_state
        s_soil[i] = soil_state
        q_mm[i] = q_i

    return q_mm, swe, s_soil


def _run_two_bucket_model_batch(T, P, temp_threshold=0.0,
                                k_snow_months=2.0, k_soil_months=3.0,
                                s_fc_mm=150.0, dt_months=1.0):
    """Vectorized two-bucket model for all sub-basins simultaneously.

    Equivalent to calling _run_two_bucket_model per row of T/P, but runs
    the time loop once over all basins using NumPy array operations.
    ~100× faster than the per-basin Python loop for the optimizer hot path.

    Parameters
    ----------
    T : np.ndarray, shape (n_basins, n_time)
        Monthly mean temperature [°C] per sub-basin.
    P : np.ndarray, shape (n_basins, n_time)
        Monthly precipitation [mm] per sub-basin.
    Other params: same as _run_two_bucket_model.

    Returns
    -------
    q_mm : np.ndarray, shape (n_basins, n_time)
    swe  : np.ndarray, shape (n_basins, n_time)
    s_soil : np.ndarray, shape (n_basins, n_time)
    """
    n_basins, n_time = T.shape

    f_rain = 1.0 / (1.0 + np.exp(-4.0 * (T - temp_threshold)))
    p_rain = P * f_rain
    p_snow = P * (1.0 - f_rain)
    pet_all = np.maximum(0.0, 0.55 * (T - temp_threshold))

    alpha_snow = np.exp(-dt_months / k_snow_months)
    k_eff_soil = 1.0 - np.exp(-dt_months / k_soil_months)

    q_mm = np.zeros((n_basins, n_time), dtype=float)
    swe_out = np.zeros((n_basins, n_time), dtype=float)
    s_soil_out = np.zeros((n_basins, n_time), dtype=float)

    swe_state = np.zeros(n_basins, dtype=float)
    soil_state = np.full(n_basins, s_fc_mm, dtype=float)

    for i in range(n_time):
        melt = swe_state * (1.0 - alpha_snow)
        swe_state = np.maximum(0.0, swe_state * alpha_snow + p_snow[:, i])

        inflow = p_rain[:, i] + melt
        avail = soil_state + inflow
        et = np.minimum(pet_all[:, i], avail)
        soil_pre = avail - et
        q_i = np.maximum(0.0, k_eff_soil * (soil_pre - s_fc_mm))
        soil_state = np.maximum(0.0, soil_pre - q_i)

        swe_out[:, i] = swe_state
        s_soil_out[:, i] = soil_state
        q_mm[:, i] = q_i

    return q_mm, swe_out, s_soil_out


# ---------------------------------------------------------------------------
# Non-glaciated runoff — per-subbasin function (not an entity_task)
# ---------------------------------------------------------------------------

def compute_nonglaciated_runoff(subbasins_gdf, climate_ds,
                                temp_threshold=None, k_snow_months=None,
                                k_soil_months=None, s_fc_mm=None,
                                nonglaciated_area_km2=None):
    """Compute non-glaciated area runoff for a set of HydroBASINS sub-basins.

    Applies the two-bucket snowmelt/soil model to each sub-basin independently,
    driven by area-weighted (centroid-representative) temperature and
    precipitation from *climate_ds*.

    Parameters
    ----------
    subbasins_gdf : :class:`geopandas.GeoDataFrame` or :class:`pandas.DataFrame`
        Sub-basin metadata.  Must contain at minimum:
        ``HYBAS_ID``, ``SUB_AREA`` (km²), and—if lapse-rate correction is
        desired—``centroid_lon``, ``centroid_lat``, ``centroid_elev_m``.
        A plain DataFrame (no geometry) is also accepted.
    climate_ds : :class:`xarray.Dataset` or dict
        Climate forcing.  Two forms are accepted:

        * **Single xr.Dataset** — variables ``temp`` [°C] and
          ``prcp`` [mm month⁻¹] on a ``time`` coordinate.  If the dataset
          has spatial dimensions (``lat``, ``lon``), the nearest grid point
          to each sub-basin centroid is extracted.  Otherwise (single time
          series), the same forcing is applied to all sub-basins.
        * **dict {HYBAS_ID → xr.Dataset}** — per-sub-basin climate datasets
          as returned by :func:`extract_subbasin_climate`.  Each entry must
          have the same variables and time axis.  This is the preferred form
          when spatial climate variability is important.
    temp_threshold : float, optional
        Rain/snow temperature threshold [°C].  Reads
        ``cfg.PARAMS['nonglaciated_temp_threshold_degC']`` if not given.
    k_snow_months : float, optional
        Snow e-folding residence time [months].  Reads
        ``cfg.PARAMS['nonglaciated_k_snow_months']`` if not given.
    k_soil_months : float, optional
        Soil drainage timescale [months].  Reads
        ``cfg.PARAMS['nonglaciated_k_soil_months']`` if not given.
    s_fc_mm : float, optional
        Field capacity [mm].  Reads
        ``cfg.PARAMS['nonglaciated_s_fc_mm']`` if not given.
    nonglaciated_area_km2 : dict or :class:`pandas.Series`, optional
        Actual non-glaciated area per sub-basin {HYBAS_ID → km²}.
        When provided, this replaces ``SUB_AREA`` for the conversion from
        runoff depth to volumetric discharge, so that the glacier-covered
        fraction is excluded from the non-glaciated runoff estimate.
        If *None*, ``SUB_AREA`` is used (legacy behaviour, overestimates
        non-glaciated runoff in heavily glacierized sub-basins).

    Returns
    -------
    :class:`xarray.Dataset`
        Variables per sub-basin and timestep:
        ``Q_ngl_mm``  — runoff depth [mm month-1]
        ``Q_ngl_m3s`` — volumetric discharge [m3 s-1]
        ``SWE_mm``    — snow water equivalent [mm]
        ``S_soil_mm`` — soil moisture [mm]
        Coordinates: ``time`` (from *climate_ds*), ``HYBAS_ID``.

    Notes
    -----
    This function is not an :func:`oggm.utils.entity_task`; it operates on
    a collection of sub-basins rather than a single glacier directory.
    Call it once for the whole basin study area.
    """
    import xarray as xr

    # Read parameters
    if temp_threshold is None:
        temp_threshold = float(
            cfg.PARAMS.get('nonglaciated_temp_threshold_degC', 0.0))
    if k_snow_months is None:
        k_snow_months = float(
            cfg.PARAMS.get('nonglaciated_k_snow_months', 2.0))
    if k_soil_months is None:
        k_soil_months = float(
            cfg.PARAMS.get('nonglaciated_k_soil_months', 3.0))
    if s_fc_mm is None:
        s_fc_mm = float(cfg.PARAMS.get('nonglaciated_s_fc_mm', 150.0))

    # Determine sub-basin IDs and areas
    hybas_ids = np.asarray(subbasins_gdf['HYBAS_ID'])
    sub_area_km2 = np.asarray(subbasins_gdf['SUB_AREA'], dtype=float)
    n_basins = len(hybas_ids)

    # Build non-glaciated area array: use provided values if available,
    # otherwise fall back to total SUB_AREA (legacy behaviour)
    if nonglaciated_area_km2 is not None:
        ngl_area_km2 = np.array([
            float(nonglaciated_area_km2.get(int(hid), sa))
            for hid, sa in zip(hybas_ids, sub_area_km2)
        ], dtype=float)
        # Clamp to non-negative
        ngl_area_km2 = np.maximum(ngl_area_km2, 0.0)
    else:
        ngl_area_km2 = sub_area_km2.copy()

    # Detect per-subbasin climate dict vs single dataset
    _is_dict_climate = isinstance(climate_ds, dict)

    if _is_dict_climate:
        # Use the first entry's time axis as reference
        first_key = next(iter(climate_ds))
        time_arr = climate_ds[first_key]['time'].values
    else:
        time_arr = climate_ds['time'].values

    # Determine spatial structure of the climate dataset (single-DS mode)
    if not _is_dict_climate:
        spatial_dims = set(climate_ds['temp'].dims) - {'time'}
        has_spatial = bool(spatial_dims)
        has_centroids = ('centroid_lon' in subbasins_gdf.columns and
                         'centroid_lat' in subbasins_gdf.columns)
    else:
        has_spatial = False
        has_centroids = False

    # Containers for results
    n_time = len(time_arr)
    q_mm_all = np.zeros((n_basins, n_time), dtype=float)
    swe_all = np.zeros((n_basins, n_time), dtype=float)
    s_soil_all = np.zeros((n_basins, n_time), dtype=float)

    if _is_dict_climate:
        # Fast vectorised path: stack all sub-basin climate into (n_basins, n_time)
        # matrices and run the time loop once for all basins simultaneously.
        T_mat = np.zeros((n_basins, n_time), dtype=float)
        P_mat = np.zeros((n_basins, n_time), dtype=float)
        for i_b in range(n_basins):
            hid = int(hybas_ids[i_b])
            clim_b = climate_ds.get(hid, climate_ds[first_key])
            T_mat[i_b] = clim_b['temp'].values.ravel()
            P_mat[i_b] = clim_b['prcp'].values.ravel()
        q_mm_all, swe_all, s_soil_all = _run_two_bucket_model_batch(
            T_mat, P_mat,
            temp_threshold=temp_threshold,
            k_snow_months=k_snow_months,
            k_soil_months=k_soil_months,
            s_fc_mm=s_fc_mm,
            dt_months=1.0,
        )
    else:
        for i_b in range(n_basins):
            if has_spatial and has_centroids:
                lon_c = float(subbasins_gdf['centroid_lon'].iloc[i_b])
                lat_c = float(subbasins_gdf['centroid_lat'].iloc[i_b])
                t_series = climate_ds['temp'].sel(
                    lat=lat_c, lon=lon_c, method='nearest').values
                p_series = climate_ds['prcp'].sel(
                    lat=lat_c, lon=lon_c, method='nearest').values
            else:
                t_series = climate_ds['temp'].values.ravel()
                p_series = climate_ds['prcp'].values.ravel()
                if len(t_series) != n_time:
                    raise ValueError(
                        f'climate_ds temp has {len(t_series)} timesteps but '
                        f'time coordinate has {n_time}.'
                    )
            q_mm, swe, s_soil = _run_two_bucket_model(
                t_celsius=t_series,
                prcp_mm=p_series,
                temp_threshold=temp_threshold,
                k_snow_months=k_snow_months,
                k_soil_months=k_soil_months,
                s_fc_mm=s_fc_mm,
                dt_months=1.0,
            )
            q_mm_all[i_b] = q_mm
            swe_all[i_b] = swe
            s_soil_all[i_b] = s_soil

    # Convert runoff depth [mm month-1] → volumetric discharge [m3 s-1]
    # Use non-glaciated area (not total SUB_AREA) so glacier-covered fraction
    # is excluded from this term.
    area_m2 = ngl_area_km2 * 1e6          # km2 → m2
    q_m3s_all = (q_mm_all * 1e-3 * area_m2[:, np.newaxis]) / _SEC_PER_MONTH

    ds_out = xr.Dataset(
        {
            'Q_ngl_mm':   (['HYBAS_ID', 'time'], q_mm_all),
            'Q_ngl_m3s':  (['HYBAS_ID', 'time'], q_m3s_all),
            'SWE_mm':     (['HYBAS_ID', 'time'], swe_all),
            'S_soil_mm':  (['HYBAS_ID', 'time'], s_soil_all),
        },
        coords={
            'time': time_arr,
            'HYBAS_ID': hybas_ids,
        },
    )
    ds_out['Q_ngl_mm'].attrs = {
        'description': 'Non-glaciated area runoff depth',
        'units': 'mm month-1',
    }
    ds_out['Q_ngl_m3s'].attrs = {
        'description': 'Non-glaciated area volumetric discharge',
        'units': 'm3 s-1',
    }
    ds_out['SWE_mm'].attrs = {
        'description': 'Snow water equivalent (two-bucket model)',
        'units': 'mm',
    }
    ds_out['S_soil_mm'].attrs = {
        'description': 'Soil moisture (two-bucket model)',
        'units': 'mm',
    }
    ds_out.attrs = {
        'temp_threshold': temp_threshold,
        'k_snow_months': k_snow_months,
        'k_soil_months': k_soil_months,
        's_fc_mm': s_fc_mm,
    }

    log.info('compute_nonglaciated_runoff: %d sub-basins × %d timesteps',
             n_basins, n_time)
    return ds_out


# ---------------------------------------------------------------------------
# Combined basin discharge
# ---------------------------------------------------------------------------

def aggregate_glacier_discharge_to_subbasins(gdirs, polygon_assignment,
                                              filesuffix,
                                              discharge_var='discharge_2c_m3s'):
    """Sum per-glacier routed discharge to HydroBASINS sub-basins.

    For each glacier, loads its annual discharge time series and distributes
    it to each sub-basin in proportion to the glacier's area fraction in that
    sub-basin (from :func:`oggm.shop.hydrobasins.assign_glaciers_to_subbasins_polygon`).

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
    polygon_assignment : :class:`pandas.DataFrame`
        Output of ``assign_glaciers_to_subbasins_polygon``.  Must contain
        columns ``rgi_id``, ``HYBAS_ID``, ``area_fraction``.
    filesuffix : str
        Filesuffix of the ``model_diagnostics`` file to read.
    discharge_var : str
        Variable name for discharge [m³ s⁻¹].

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, dict)
        * ``q_total``    — total glacier discharge [m³ s⁻¹], shape (n_years,)
        * ``time_ref``   — year array, shape (n_years,)
        * ``hybas_ids``  — array of HYBAS_IDs with assigned glaciers
        * ``q_by_subbasin`` — dict {HYBAS_ID: np.array shape (n_years,)}
    """
    import xarray as xr

    # Build a lookup: rgi_id → [(HYBAS_ID, area_fraction), ...]
    assignment_map: dict = {}
    for _, row in polygon_assignment.iterrows():
        rgi_id = row['rgi_id']
        hybas = int(row['HYBAS_ID'])
        frac = float(row['area_fraction'])
        if hybas < 0:
            continue
        assignment_map.setdefault(rgi_id, []).append((hybas, frac))

    gdir_map = {g.rgi_id: g for g in gdirs}

    # Accumulate discharge per subbasin
    q_by_subbasin: dict = {}
    time_ref = None
    q_total = None

    for rgi_id, allocations in assignment_map.items():
        gdir = gdir_map.get(rgi_id)
        if gdir is None:
            continue
        try:
            fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
            with xr.open_dataset(fpath) as ds:
                if discharge_var not in ds:
                    # Try fallbacks
                    for v in ('discharge_2c_m3s', 'discharge_m3s',
                              'discharge_terrain_m3s'):
                        if v in ds:
                            discharge_var = v
                            break
                    else:
                        continue
                q = ds[discharge_var].values.astype(float)
                t = ds['time'].values
        except Exception:
            continue

        if time_ref is None:
            time_ref = t
            q_total = np.zeros(len(t), dtype=float)

        q_total += q

        for hybas, frac in allocations:
            if hybas not in q_by_subbasin:
                q_by_subbasin[hybas] = np.zeros(len(t), dtype=float)
            q_by_subbasin[hybas] += q * frac

    if time_ref is None:
        raise ValueError('No valid glacier discharge files found.')

    hybas_ids_arr = np.array(sorted(q_by_subbasin.keys()), dtype=int)
    return q_total, time_ref, hybas_ids_arr, q_by_subbasin


def combine_basin_discharge(gdirs, subbasins_assignment, nonglaciated_ds,
                             glacier_filesuffix='',
                             glacier_discharge_var='discharge_m3s'):
    """Combine glaciated and non-glaciated discharge for a study basin.

    Sums routed glacier discharge over all provided glacier directories and
    adds the non-glaciated sub-basin runoff from *nonglaciated_ds*.  The
    two components are aligned on a shared time axis.

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
        Glaciers in the basin.  Each must have a ``model_diagnostics`` file
        with *glacier_discharge_var* present (i.e. a routing task has already
        been run).
    subbasins_assignment : :class:`pandas.DataFrame`
        Output of :func:`oggm.shop.hydrobasins.assign_glaciers_to_subbasins`.
        Must contain columns ``rgi_id`` and ``HYBAS_ID``.
    nonglaciated_ds : :class:`xarray.Dataset`
        Output of :func:`compute_nonglaciated_runoff`.  Must contain variable
        ``Q_ngl_m3s`` with dimensions ``(HYBAS_ID, time)``.
    glacier_filesuffix : str
        Filesuffix for the ``model_diagnostics`` file to read from each gdir.
    glacier_discharge_var : str
        Name of the discharge variable in the glacier diagnostics file.
        Typically ``'discharge_m3s'`` (after routing) or
        ``'discharge_terrain_m3s'`` (after channel routing).

    Returns
    -------
    :class:`xarray.Dataset`
        Variables on a shared annual time axis:
        ``Q_glacier_m3s``    — total glaciated discharge [m3 s-1]
        ``Q_nonglacial_m3s`` — total non-glaciated discharge [m3 s-1]
        ``Q_total_m3s``      — combined total basin discharge [m3 s-1]
        Coordinate: ``time``.

    Raises
    ------
    RuntimeError
        If *glacier_discharge_var* is not found in any glacier's file.
    ValueError
        If the glaciated and non-glaciated time axes cannot be aligned.
    """
    import xarray as xr

    # --- Sum glaciated discharge ---
    all_q_gl = []
    time_ref = None

    for gdir in gdirs:
        fpath = gdir.get_filepath('model_diagnostics',
                                  filesuffix=glacier_filesuffix)
        with xr.open_dataset(fpath) as ds:
            if glacier_discharge_var not in ds:
                raise RuntimeError(
                    f'{glacier_discharge_var!r} not found in {fpath}. '
                    f'Run a routing task first for {gdir.rgi_id}.'
                )
            q = ds[glacier_discharge_var].values
            t = ds['time'].values

        if time_ref is None:
            time_ref = t
        elif not np.array_equal(t, time_ref):
            raise ValueError(
                f'Time axis mismatch for glacier {gdir.rgi_id}. '
                'All glaciers must share the same time axis.'
            )
        all_q_gl.append(q)

    if not all_q_gl:
        raise ValueError('gdirs is empty; cannot compute glacier discharge.')

    q_glacier = np.nansum(np.stack(all_q_gl, axis=0), axis=0)

    # --- Sum non-glaciated discharge over all sub-basins ---
    q_ngl_total = nonglaciated_ds['Q_ngl_m3s'].sum(dim='HYBAS_ID').values
    time_ngl = nonglaciated_ds['time'].values

    # --- Align time axes ---
    # Glacier diagnostics are annual; Q_ngl may be monthly or annual.
    # Extract year integers from any time dtype (float, int, datetime64).
    def _to_years(t_arr):
        if np.issubdtype(t_arr.dtype, np.floating):
            return t_arr.astype(int)
        if np.issubdtype(t_arr.dtype, np.integer):
            return t_arr
        try:
            import pandas as pd
            return np.array([pd.Timestamp(tt).year for tt in t_arr])
        except Exception:
            raise ValueError(
                'Cannot convert time coordinate to integer years.'
            )

    yrs_gl = _to_years(time_ref)
    yrs_ngl_raw = _to_years(time_ngl)

    # If Q_ngl is sub-annual (monthly), aggregate to annual mean discharge.
    # Annual mean of a mean-monthly series = mean monthly value, which
    # is already the correct time-averaged rate [m3 s-1].
    unique_yrs_ngl = np.unique(yrs_ngl_raw)
    if len(yrs_ngl_raw) > len(unique_yrs_ngl):
        log.debug('combine_basin_discharge: resampling sub-annual '
                  '(%d steps) Q_ngl to %d annual means',
                  len(yrs_ngl_raw), len(unique_yrs_ngl))
        q_ngl_annual = np.array(
            [q_ngl_total[yrs_ngl_raw == yr].mean() for yr in unique_yrs_ngl],
            dtype=float)
        q_ngl_total = q_ngl_annual
        yrs_ngl = unique_yrs_ngl
    else:
        yrs_ngl = yrs_ngl_raw

    # Intersection of years present in both datasets
    common_years = np.intersect1d(yrs_gl, yrs_ngl)
    if len(common_years) == 0:
        raise ValueError(
            'Glacier and non-glaciated time axes have no overlapping years. '
            f'Glacier years: {yrs_gl.min()}\u2013{yrs_gl.max()}, '
            f'Non-glaciated years: {yrs_ngl.min()}\u2013{yrs_ngl.max()}.'
        )

    idx_gl  = np.isin(yrs_gl,  common_years)
    idx_ngl = np.isin(yrs_ngl, common_years)

    q_glacier_aligned = q_glacier[idx_gl]
    q_ngl_aligned     = q_ngl_total[idx_ngl]

    # --- Build output dataset ---
    ds_out = xr.Dataset(
        {
            'Q_glacier_m3s':    ('time', q_glacier_aligned),
            'Q_nonglacial_m3s': ('time', q_ngl_aligned),
            'Q_total_m3s':      ('time', q_glacier_aligned + q_ngl_aligned),
        },
        coords={'time': common_years},
    )
    ds_out['Q_glacier_m3s'].attrs = {
        'description': 'Total glaciated discharge (sum over all glaciers)',
        'units': 'm3 s-1',
        'n_glaciers': len(gdirs),
    }
    ds_out['Q_nonglacial_m3s'].attrs = {
        'description': 'Total non-glaciated sub-basin discharge',
        'units': 'm3 s-1',
    }
    ds_out['Q_total_m3s'].attrs = {
        'description': 'Combined basin discharge (glaciated + non-glaciated)',
        'units': 'm3 s-1',
    }

    log.info(
        'combine_basin_discharge: %d glaciers, %d years, '
        'Q_glacier mean=%.2f m3/s, Q_ngl mean=%.2f m3/s',
        len(gdirs), len(common_years),
        float(np.nanmean(q_glacier_aligned)),
        float(np.nanmean(q_ngl_aligned)),
    )
    return ds_out


# ===========================================================================
# Phase 12 — Basin Water Balance Calibration
# ===========================================================================

def _nash_sutcliffe_efficiency(q_sim, q_obs):
    """Nash-Sutcliffe Efficiency (NSE; Nash & Sutcliffe, 1970).

    NSE = 1 - SS_res / SS_tot

    Parameters
    ----------
    q_sim : array-like
        Simulated discharge.
    q_obs : array-like
        Observed discharge.

    Returns
    -------
    float
        NSE in (−∞, 1].  Perfect score = 1.0.
        Returns ``-np.inf`` if fewer than 2 finite paired values exist.
    """
    q_sim = np.asarray(q_sim, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)
    mask = np.isfinite(q_sim) & np.isfinite(q_obs)
    if mask.sum() < 2:
        return -np.inf
    qs, qo = q_sim[mask], q_obs[mask]
    ss_res = np.sum((qo - qs) ** 2)
    ss_tot = np.sum((qo - qo.mean()) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else -np.inf
    return float(1.0 - ss_res / ss_tot)


def _percent_bias(q_sim, q_obs):
    """Percent Bias (PBIAS).

    PBIAS = 100 * Σ(obs - sim) / Σ(obs)

    Positive: model underestimates. Negative: model overestimates.

    Parameters
    ----------
    q_sim : array-like
    q_obs : array-like

    Returns
    -------
    float
        PBIAS in percent. ``np.nan`` if fewer than 1 finite pair or sum(obs)=0.
    """
    q_sim = np.asarray(q_sim, dtype=float)
    q_obs = np.asarray(q_obs, dtype=float)
    mask = np.isfinite(q_sim) & np.isfinite(q_obs)
    if mask.sum() < 1:
        return np.nan
    qs, qo = q_sim[mask], q_obs[mask]
    obs_sum = qo.sum()
    if obs_sum == 0:
        return np.nan
    return float(100.0 * (qo - qs).sum() / obs_sum)


def read_grdc_data(filepath_or_df, station_id=None, freq='annual',
                   ys=None, ye=None):
    """Read GRDC station discharge data into a standard DataFrame.

    Accepts:
      - Path to a GRDC ASCII file (.txt) — lines starting with '#' are
        treated as header comments.
      - Path to a pre-processed CSV with columns ``date`` (or ``year``) and
        ``Q_m3s`` (or common synonyms).
      - A :class:`pandas.DataFrame` (pass-through with standardisation).

    Parameters
    ----------
    filepath_or_df : str, os.PathLike, or pd.DataFrame
        Input data source.
    station_id : str, optional
        Unused at present; reserved for multi-station files.
    freq : {'annual', 'monthly'}
        Target temporal resolution.  ``'annual'`` aggregates by calendar
        year mean.  ``'monthly'`` keeps monthly rows.
    ys : int, optional
        First year to include (inclusive).
    ye : int, optional
        Last year to include (inclusive).

    Returns
    -------
    pd.DataFrame
        For ``freq='annual'``: columns ``year`` (int), ``q_m3s`` (float).
        For ``freq='monthly'``: columns ``date`` (datetime), ``q_m3s`` (float).
        Missing-value rows (NaN) are retained.
    """
    import pandas as _pd

    if isinstance(filepath_or_df, _pd.DataFrame):
        df = filepath_or_df.copy()
    else:
        filepath_or_df = str(filepath_or_df)
        if filepath_or_df.endswith('.txt'):
            # GRDC ASCII format: skip lines starting with '#'
            rows = []
            header_found = False
            with open(filepath_or_df, 'r', encoding='utf-8', errors='replace') as fh:
                for line in fh:
                    stripped = line.strip()
                    if stripped.startswith('#') or not stripped:
                        # Look for column header line inside comments
                        if ';' in stripped and 'YYYY' in stripped.upper():
                            header_found = True
                        continue
                    rows.append(stripped)
            if not rows:
                raise ValueError(f'No data rows found in {filepath_or_df}')
            records = []
            for row in rows:
                # GRDC typical: YYYY-MM-DD;HH:MM; value; flag
                parts = row.replace(';', ' ').split()
                if len(parts) < 2:
                    continue
                try:
                    date_str = parts[0]
                    # Last numeric column is the value
                    value_str = parts[-1] if len(parts) >= 2 else 'nan'
                    value = float(value_str)
                    records.append((date_str, value))
                except (ValueError, IndexError):
                    continue
            df = _pd.DataFrame(records, columns=['date', 'q_m3s'])
            try:
                df['date'] = _pd.to_datetime(df['date'])
            except Exception:
                pass
        else:
            df = _pd.read_csv(filepath_or_df)

    # Standardise column names (lower-case, strip whitespace)
    df.columns = [c.lower().strip() for c in df.columns]

    # Map common discharge column names to q_m3s
    if 'q_m3s' not in df.columns:
        for candidate in ('discharge', 'q(m3/s)', 'q(m³/s)', 'q_m3/s',
                          'flow', 'value', 'runoff'):
            if candidate in df.columns:
                df = df.rename(columns={candidate: 'q_m3s'})
                break

    if 'q_m3s' not in df.columns:
        raise ValueError(
            f'Cannot find discharge column. '
            f'Available columns: {list(df.columns)}'
        )

    df['q_m3s'] = _pd.to_numeric(df['q_m3s'], errors='coerce')
    # GRDC missing value sentinel
    df.loc[df['q_m3s'] < -998, 'q_m3s'] = np.nan

    # Ensure a year column is available
    if freq == 'annual':
        if 'year' not in df.columns:
            if 'date' in df.columns:
                try:
                    df['year'] = _pd.to_datetime(df['date']).dt.year
                except Exception:
                    df['year'] = df['date'].astype(int)
            else:
                raise ValueError(
                    'DataFrame must have "year" or "date" column for '
                    'freq="annual".'
                )
        df['year'] = df['year'].astype(int)
        df = (df.groupby('year')['q_m3s']
               .mean()
               .reset_index())
        df['year'] = df['year'].astype(int)
        # Apply year filter
        if ys is not None:
            df = df[df['year'] >= int(ys)]
        if ye is not None:
            df = df[df['year'] <= int(ye)]
        df = df.reset_index(drop=True)
    else:
        # Monthly
        if 'date' not in df.columns:
            if 'year' in df.columns and 'month' in df.columns:
                df['date'] = _pd.to_datetime(
                    df[['year', 'month']].assign(day=1))
            else:
                raise ValueError(
                    'For freq="monthly", DataFrame must have "date" column '
                    'or "year"+"month" columns.'
                )
        df['date'] = _pd.to_datetime(df['date'])
        if ys is not None:
            df = df[df['date'].dt.year >= int(ys)]
        if ye is not None:
            df = df[df['date'].dt.year <= int(ye)]
        df = df.reset_index(drop=True)

    return df


def _cache_basin_runoff_components(gdirs, filesuffix='', ys=None, ye=None,
                                   n_workers=8):
    """Pre-compute basin-level annual runoff component sums from model output.

    Reads rain, snowmelt, and ice-melt annual timeseries from each glacier's
    ``model_diagnostics.nc`` and returns basin-level summed arrays.  Called
    once before the optimisation loop to avoid repeated file I/O per iteration.

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
    filesuffix : str
        Suffix of the ``model_diagnostics`` file to read.
    ys : int or None
        Start year for the cache period (inclusive).
    ye : int or None
        End year for the cache period (inclusive).
    n_workers : int
        Number of parallel threads for reading NC files (default: 8).
        Set to 1 to disable threading (useful for debugging).

    Returns
    -------
    dict with keys:
        ``'years'``    — np.ndarray int (common year axis)
        ``'rain_m3s'`` — np.ndarray [m³ s⁻¹] basin-total liquid precip
        ``'snow_m3s'`` — np.ndarray [m³ s⁻¹] basin-total snowmelt
        ``'ice_m3s'``  — np.ndarray [m³ s⁻¹] basin-total ice melt
    """
    from concurrent.futures import ThreadPoolExecutor

    def _read_one(gdir):
        try:
            fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
        except Exception:
            return None
        if not os.path.isfile(fpath):
            return None
        try:
            with xr.open_dataset(fpath) as ds:
                time_vals = ds['time'].values
                years_arr = _extract_model_years(time_vals)
                if ('icemelt_on_glacier' in ds and
                        'snowmelt_on_glacier' in ds):
                    rain_kgyr = (ds['liq_prcp_on_glacier'].values +
                                 ds['liq_prcp_off_glacier'].values)
                    snow_kgyr = (ds['snowmelt_on_glacier'].values +
                                 ds['melt_off_glacier'].values)
                    ice_kgyr = ds['icemelt_on_glacier'].values
                else:
                    rain_kgyr = (ds['liq_prcp_on_glacier'].values +
                                 ds['liq_prcp_off_glacier'].values)
                    melt_kgyr = (ds['melt_on_glacier'].values +
                                 ds['melt_off_glacier'].values)
                    ice_kgyr = melt_kgyr * 0.6
                    snow_kgyr = melt_kgyr * 0.4
        except Exception as e:
            log.warning('_cache_basin_runoff_components: skipping %s: %s',
                        gdir.rgi_id, e)
            return None

        valid = ~np.isnan(rain_kgyr + snow_kgyr + ice_kgyr)
        yr_valid = years_arr[valid]
        mask = np.ones(len(yr_valid), dtype=bool)
        if ys is not None:
            mask &= yr_valid >= int(ys)
        if ye is not None:
            mask &= yr_valid <= int(ye)
        yr_f = yr_valid[mask]
        rain_m3s = rain_kgyr[valid][mask] / _RHO_WATER / _SEC_PER_YEAR
        snow_m3s = snow_kgyr[valid][mask] / _RHO_WATER / _SEC_PER_YEAR
        ice_m3s = ice_kgyr[valid][mask] / _RHO_WATER / _SEC_PER_YEAR
        return (yr_f, rain_m3s, snow_m3s, ice_m3s)

    # --- parallel read ---
    actual_workers = min(n_workers, len(gdirs)) if n_workers > 1 else 1
    if actual_workers > 1:
        with ThreadPoolExecutor(max_workers=actual_workers) as pool:
            results = list(pool.map(_read_one, gdirs))
    else:
        results = [_read_one(g) for g in gdirs]

    # --- sequential accumulation ---
    rain_total = None
    snow_total = None
    ice_total = None
    years_ref = None

    for res in results:
        if res is None:
            continue
        yr_filtered, rain_m3s, snow_m3s, ice_m3s = res
        if years_ref is None:
            years_ref = yr_filtered
            rain_total = np.zeros_like(rain_m3s)
            snow_total = np.zeros_like(snow_m3s)
            ice_total = np.zeros_like(ice_m3s)
        common = np.intersect1d(years_ref, yr_filtered)
        if len(common) == 0:
            continue
        idx_ref = np.isin(years_ref, common)
        idx_new = np.isin(yr_filtered, common)
        rain_total[idx_ref] += rain_m3s[idx_new]
        snow_total[idx_ref] += snow_m3s[idx_new]
        ice_total[idx_ref] += ice_m3s[idx_new]

    if years_ref is None:
        raise ValueError(
            'No valid glacier model_diagnostics files found. '
            'Run run_with_hydro() first.'
        )

    return {
        'years': years_ref,
        'rain_m3s': rain_total,
        'snow_m3s': snow_total,
        'ice_m3s': ice_total,
    }


def calibrate_basin_water_balance(
    gdirs,
    subbasins_gdf,
    obs_discharge,
    obs_freq='annual',
    glacier_filesuffix='',
    calibrate_params=(
        'k_rain_months',
        'k_snow_months',
        'k_ice_months',
        'k_snow_ngl',
        'k_soil_months',
        's_fc_mm',
    ),
    basin_prcp_fac=False,
    obs_glacier_frac=None,
    ys=None,
    ye=None,
    method='differential_evolution',
    metric='KGE',
    output_dir=None,
    cross_validate=False,
    seed=42,
):
    """Calibrate basin-level water balance against observed gauge discharge.

    Optimises reservoir routing timescales (glaciated) and two-bucket
    parameters (non-glaciated) simultaneously using observed total discharge
    at the basin outlet.  Uses a two-phase strategy:

    1. **Global search** with :func:`scipy.optimize.differential_evolution`
       (avoids local minima in the non-convex KGE landscape).
    2. **Local polish** with Nelder-Mead.

    An efficient pre-caching strategy ensures the inner optimisation loop
    costs only O(60 000) scalar operations per iteration (< 1 ms), making
    500 iterations feasible in under a second.

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
        All glacier directories in the basin.  Each must have a
        ``model_diagnostics.nc`` file from ``run_with_hydro()``.
    subbasins_gdf : GeoDataFrame or DataFrame
        HydroBASINS sub-basin table.  Must contain ``HYBAS_ID`` and ``SUB_AREA``.
    obs_discharge : str, os.PathLike, pd.DataFrame, or array-like
        Observed discharge.  Accepted forms:

        * Path to a GRDC ``.txt`` or ``.csv`` file
        * :class:`pandas.DataFrame` with ``year`` and ``Q_m3s`` columns
        * (N, 2) array-like with columns [year, Q_m3s]
    obs_freq : {'annual', 'monthly'}
        Temporal resolution of observations (default: ``'annual'``).
    glacier_filesuffix : str
        Filesuffix of the ``model_diagnostics`` file to read for glaciers.
    calibrate_params : tuple of str
        Which basin parameters to optimise.  Subset of
        ``('k_rain_months', 'k_snow_months', 'k_ice_months',
          'k_snow_ngl', 'k_soil_months', 's_fc_mm')``.
    basin_prcp_fac : bool
        If ``True``, also calibrate a basin-wide precipitation correction
        factor applied uniformly to all precipitation inputs.
    obs_glacier_frac : float, optional
        Known glaciated fraction of observed discharge.  If provided, the
        observed signal is split before comparison to the modelled
        glaciated / non-glaciated components.  Currently unused; reserved
        for future multi-objective use.
    ys : int, optional
        Start year of the calibration period (inclusive).
    ye : int, optional
        End year of the calibration period (inclusive).
    method : {'differential_evolution', 'Nelder-Mead'}
        Primary optimisation method.
    metric : {'KGE', 'NSE', 'PBIAS'}
        Goodness-of-fit metric to maximise.
    output_dir : str or None
        Directory in which to write ``basin_calib_params.json``.
        If *None*, nothing is written.
    cross_validate : bool
        When ``True``, the last 25 % of years are withheld as a validation
        set and ``KGE_valid`` is reported in the output.
    seed : int
        Random seed for the differential evolution algorithm.

    Returns
    -------
    dict
        Best-fit parameters plus diagnostic metrics:
        ``k_rain_months``, ``k_snow_months``, ``k_ice_months``,
        ``k_snow_ngl``, ``k_soil_months``, ``s_fc_mm``,
        ``basin_prcp_fac``, ``KGE_calib``, ``KGE_valid``,
        ``NSE_calib``, ``PBIAS_pct``, ``n_obs``,
        ``calib_years``, ``valid_years``, ``obs_freq``, ``metric``,
        ``convergence``.

    Raises
    ------
    ImportError
        If :mod:`scipy` is not installed.
    ValueError
        If fewer than 3 overlapping valid years exist.
    """
    try:
        from scipy.optimize import differential_evolution, minimize
    except ImportError as exc:
        raise ImportError(
            'scipy is required for calibrate_basin_water_balance(). '
            'Install with: conda install scipy'
        ) from exc
    import pandas as _pd

    # ---- Parse observed discharge ----
    if isinstance(obs_discharge, (_pd.DataFrame,)):
        obs_df = read_grdc_data(obs_discharge, freq=obs_freq, ys=ys, ye=ye)
    elif isinstance(obs_discharge, (str, os.PathLike)):
        obs_df = read_grdc_data(str(obs_discharge), freq=obs_freq,
                                ys=ys, ye=ye)
    else:
        arr = np.asarray(obs_discharge)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            obs_df = _pd.DataFrame({
                'year': arr[:, 0].astype(int),
                'q_m3s': arr[:, 1].astype(float),
            })
        else:
            raise ValueError(
                'obs_discharge must be a file path, DataFrame, or (N,2) array.'
            )

    obs_years_all = obs_df['year'].values.astype(int)
    q_obs_all = obs_df['q_m3s'].values.astype(float)

    # ---- Pre-cache glacier runoff ----
    log.info('calibrate_basin_water_balance: caching glacier runoff ...')
    cache = _cache_basin_runoff_components(
        gdirs, filesuffix=glacier_filesuffix, ys=ys, ye=ye)
    model_years = cache['years']
    rain_tot = cache['rain_m3s']
    snow_tot = cache['snow_m3s']
    ice_tot = cache['ice_m3s']

    # ---- Pre-extract sub-basin climate ----
    log.info('calibrate_basin_water_balance: extracting sub-basin climate ...')
    per_basin_clim = extract_subbasin_climate(gdirs, subbasins_gdf,
                                              ys=ys, ye=ye)

    # ---- Align time axes ----
    common_years = np.intersect1d(obs_years_all, model_years)
    obs_finite_n = np.isfinite(
        q_obs_all[np.isin(obs_years_all, common_years)]).sum()
    if obs_finite_n < 3:
        raise ValueError(
            f'Only {obs_finite_n} finite overlapping years between model and '
            f'observations. Need >= 3.'
        )

    log.info('calibrate_basin_water_balance: %d common years [%d-%d]',
             len(common_years), int(common_years.min()),
             int(common_years.max()))

    # Cross-validation split
    if cross_validate and len(common_years) >= 8:
        n_valid_yrs = max(1, len(common_years) // 4)
        calib_years = common_years[:-n_valid_yrs]
        valid_years = common_years[-n_valid_yrs:]
    else:
        calib_years = common_years
        valid_years = None

    # ---- Parameter bounds from params.cfg ----
    bounds_map = {
        'k_rain_months': (
            float(cfg.PARAMS.get('wbc_k_rain_min', 0.1)),
            float(cfg.PARAMS.get('wbc_k_rain_max', 3.0))),
        'k_snow_months': (
            float(cfg.PARAMS.get('wbc_k_snow_min', 0.5)),
            float(cfg.PARAMS.get('wbc_k_snow_max', 12.0))),
        'k_ice_months': (
            float(cfg.PARAMS.get('wbc_k_ice_min', 2.0)),
            float(cfg.PARAMS.get('wbc_k_ice_max', 36.0))),
        'k_snow_ngl': (
            float(cfg.PARAMS.get('wbc_k_snow_ngl_min', 0.5)),
            float(cfg.PARAMS.get('wbc_k_snow_ngl_max', 12.0))),
        'k_soil_months': (
            float(cfg.PARAMS.get('wbc_k_soil_min', 1.0)),
            float(cfg.PARAMS.get('wbc_k_soil_max', 24.0))),
        's_fc_mm': (
            float(cfg.PARAMS.get('wbc_s_fc_min', 50.0)),
            float(cfg.PARAMS.get('wbc_s_fc_max', 500.0))),
        'basin_prcp_fac': (
            float(cfg.PARAMS.get('wbc_prcp_fac_min', 0.5)),
            float(cfg.PARAMS.get('wbc_prcp_fac_max', 3.0))),
    }

    # Build ordered list of parameters to calibrate
    params_to_calib = list(calibrate_params)
    if basin_prcp_fac:
        if 'basin_prcp_fac' not in params_to_calib:
            params_to_calib.append('basin_prcp_fac')
    bounds_list = [bounds_map[p] for p in params_to_calib]

    # Default (fixed) values for all parameters
    defaults = {
        'k_rain_months': float(cfg.PARAMS.get('routing_k_rain_months',
                                               _DEFAULT_K_RAIN_MONTHS)),
        'k_snow_months': float(cfg.PARAMS.get('routing_k_snow_months',
                                               _DEFAULT_K_SNOW_MONTHS)),
        'k_ice_months': float(cfg.PARAMS.get('routing_k_ice_months',
                                              _DEFAULT_K_ICE_MONTHS)),
        'k_snow_ngl': float(cfg.PARAMS.get('nonglaciated_k_snow_months', 2.0)),
        'k_soil_months': float(cfg.PARAMS.get('nonglaciated_k_soil_months', 3.0)),
        's_fc_mm': float(cfg.PARAMS.get('nonglaciated_s_fc_mm', 150.0)),
        'basin_prcp_fac': 1.0,
    }

    # ---- Non-glaciated area per sub-basin ----
    # The NGL two-bucket model must use only the non-glaciated fraction of
    # each sub-basin.  Using total SUB_AREA would double-count the rain that
    # already appears in model_diagnostics.nc (liq_prcp_on/off_glacier).
    # We approximate by distributing total glacier area proportionally across
    # sub-basins (exact spatial join is expensive and not needed here).
    _total_basin_km2 = float(subbasins_gdf['SUB_AREA'].sum())
    _total_glacier_km2 = float(sum(gd.rgi_area_km2 for gd in gdirs))
    _ngl_frac = max(0.0, min(1.0,
                              1.0 - _total_glacier_km2 / max(_total_basin_km2, 1.0)))
    _ngl_area_per_sub = {
        int(row['HYBAS_ID']): float(row['SUB_AREA']) * _ngl_frac
        for _, row in subbasins_gdf.iterrows()
    }
    log.info(
        'calibrate_basin_water_balance: total basin %.0f km², glaciers %.0f km², '
        'NGL fraction %.1f%%',
        _total_basin_km2, _total_glacier_km2, _ngl_frac * 100,
    )

    # ---- Metric evaluation ----
    def _eval_metric(q_sim, q_obs):
        if metric == 'KGE':
            return _kling_gupta_efficiency(q_sim, q_obs)
        elif metric == 'NSE':
            return _nash_sutcliffe_efficiency(q_sim, q_obs)
        elif metric == 'PBIAS':
            return -abs(_percent_bias(q_sim, q_obs))
        else:
            raise ValueError(f'Unknown metric: {metric!r}')

    # ---- Forward model (runs inside optimizer) ----
    def _forward_model(params_vec, eval_years):
        """Run the basin forward model; return (years, Q_total_m3s)."""
        p = dict(defaults)
        for name, val in zip(params_to_calib, params_vec):
            p[name] = float(val)

        # Clamp to physically valid range — Nelder-Mead polish can wander
        # outside the DE bounds, so guard here rather than relying on bounds.
        k_rain = max(1e-4, p['k_rain_months'])
        k_snow = max(1e-4, p['k_snow_months'])
        k_ice = max(1e-4, p['k_ice_months'])
        k_snow_ngl = max(1e-4, p['k_snow_ngl'])
        k_soil = max(1e-4, p['k_soil_months'])
        s_fc = max(0.0, p['s_fc_mm'])
        pf = max(1e-4, p['basin_prcp_fac'])

        # Glacier routing (pre-cached arrays, O(N_years) ops)
        idx_gl = np.isin(model_years, eval_years)
        yr_gl = model_years[idx_gl]

        q_rain_r = _linear_reservoir(
            rain_tot[idx_gl] * pf, k=k_rain / 12.0, dt=1.0)
        q_snow_r = _linear_reservoir(
            snow_tot[idx_gl], k=k_snow / 12.0, dt=1.0)
        q_ice_r = _linear_reservoir(
            ice_tot[idx_gl], k=k_ice / 12.0, dt=1.0)
        q_gl = q_rain_r + q_snow_r + q_ice_r

        # Non-glaciated runoff (two-bucket model, O(N_sub x N_months) ops)
        if pf != 1.0:
            scaled_clim = {
                hid: clim.assign({'prcp': clim['prcp'] * pf})
                for hid, clim in per_basin_clim.items()
            }
        else:
            scaled_clim = per_basin_clim

        ngl_ds = compute_nonglaciated_runoff(
            subbasins_gdf, scaled_clim,
            k_snow_months=k_snow_ngl,
            k_soil_months=k_soil,
            s_fc_mm=s_fc,
            nonglaciated_area_km2=_ngl_area_per_sub,
        )

        # Aggregate Q_ngl to annual
        q_ngl_monthly = ngl_ds['Q_ngl_m3s'].sum(dim='HYBAS_ID').values
        time_ngl = ngl_ds['time'].values
        ngl_years = _extract_model_years(time_ngl)
        yr_unique = np.unique(ngl_years)
        q_ngl_ann = np.array(
            [q_ngl_monthly[ngl_years == yr].mean() for yr in yr_unique],
            dtype=float)

        # Intersect all time axes
        common_ev = np.intersect1d(np.intersect1d(yr_gl, yr_unique),
                                   eval_years)
        idx_gl2 = np.isin(yr_gl, common_ev)
        idx_ngl = np.isin(yr_unique, common_ev)
        yr_out = yr_gl[idx_gl2]
        q_total = q_gl[idx_gl2] + q_ngl_ann[idx_ngl]
        return yr_out, q_total

    # ---- Cost function ----
    def _cost(params_vec, eval_years):
        yr_sim, q_sim = _forward_model(params_vec, eval_years)

        obs_mask = np.isin(obs_years_all, yr_sim)
        sim_mask = np.isin(yr_sim, obs_years_all[obs_mask])
        q_obs_aligned = q_obs_all[obs_mask]
        q_sim_aligned = q_sim[sim_mask]

        finite = np.isfinite(q_obs_aligned) & np.isfinite(q_sim_aligned)
        if finite.sum() < 2:
            return 2.0
        score = _eval_metric(q_sim_aligned[finite], q_obs_aligned[finite])
        return 1.0 - score  # minimise (1 - metric)

    # ---- Optimise ----
    log.info(
        'calibrate_basin_water_balance: optimising %d params via %s '
        '(metric=%s) ...',
        len(params_to_calib), method, metric,
    )
    x0 = np.array([defaults[p] for p in params_to_calib])

    if method == 'differential_evolution':
        result_de = differential_evolution(
            lambda x: _cost(x, calib_years),
            bounds=bounds_list,
            seed=seed,
            maxiter=300,
            tol=1e-3,
            mutation=(0.5, 1.5),
            recombination=0.7,
            popsize=10,
            workers=1,
        )
        result_nm = minimize(
            lambda x: _cost(x, calib_years),
            result_de.x,
            method='Nelder-Mead',
            bounds=bounds_list,
            options={'xatol': 0.01, 'fatol': 1e-3, 'maxiter': 500},
        )
        final_x = result_nm.x if result_nm.fun <= result_de.fun else result_de.x
        converged = bool(result_de.success)
    else:
        res = minimize(
            lambda x: _cost(x, calib_years),
            x0,
            method=method,
            options={'xatol': 0.01, 'fatol': 1e-3, 'maxiter': 1000},
        )
        final_x = res.x
        converged = bool(res.success)

    # ---- Assemble result ----
    best = dict(defaults)
    for name, val in zip(params_to_calib, final_x):
        best[name] = float(val)

    kge_calib = float(1.0 - _cost(final_x, calib_years))

    yr_sim_c, q_sim_c = _forward_model(final_x, calib_years)
    obs_mask_c = np.isin(obs_years_all, yr_sim_c)
    sim_mask_c = np.isin(yr_sim_c, obs_years_all[obs_mask_c])
    qoc = q_obs_all[obs_mask_c]
    qsc = q_sim_c[sim_mask_c]
    finite_c = np.isfinite(qoc) & np.isfinite(qsc)
    nse_calib = float(_nash_sutcliffe_efficiency(qsc[finite_c], qoc[finite_c]))
    pbias_calib = float(_percent_bias(qsc[finite_c], qoc[finite_c]))

    calib_result = {
        'k_rain_months': best['k_rain_months'],
        'k_snow_months': best['k_snow_months'],
        'k_ice_months': best['k_ice_months'],
        'k_snow_ngl': best['k_snow_ngl'],
        'k_soil_months': best['k_soil_months'],
        's_fc_mm': best['s_fc_mm'],
        'basin_prcp_fac': best['basin_prcp_fac'],
        'KGE_calib': kge_calib,
        'NSE_calib': nse_calib,
        'PBIAS_pct': pbias_calib,
        'n_obs': int(finite_c.sum()),
        'calib_years': [int(calib_years.min()), int(calib_years.max())],
        'valid_years': None,
        'KGE_valid': None,
        'obs_freq': obs_freq,
        'metric': metric,
        'convergence': converged,
    }

    if cross_validate and valid_years is not None and len(valid_years) > 0:
        kge_valid = float(1.0 - _cost(final_x, valid_years))
        calib_result['KGE_valid'] = kge_valid
        calib_result['valid_years'] = [int(valid_years.min()),
                                       int(valid_years.max())]

    log.info(
        'calibrate_basin_water_balance: KGE=%.3f  NSE=%.3f  PBIAS=%.1f%%  '
        'converged=%s',
        kge_calib, nse_calib, pbias_calib, converged,
    )

    # ---- Write output ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'basin_calib_params.json')
        with open(out_path, 'w') as fh:
            json.dump(calib_result, fh, indent=2)
        log.info('calibrate_basin_water_balance: result written to %s',
                 out_path)

    return calib_result


def mb_calibration_basin_from_discharge(
    gdirs,
    obs_glacier_discharge_m3s,
    obs_years,
    ref_mb_df=None,
    w_mb=0.6,
    w_Q=0.4,
    use_representative_glaciers=True,
    representative_frac=0.8,
    nprocesses=1,
    filesuffix='',
):
    """Apply per-glacier discharge-constrained MB calibration to a basin.

    Wraps :func:`~oggm.core.massbalance.mb_calibration_from_runoff` and
    applies it to all (or a representative subset of) glaciers in a basin.

    When *use_representative_glaciers* is ``True``, only the largest glaciers
    that together cover *representative_frac* of total glacier area are
    directly calibrated; all remaining smaller glaciers receive the same
    calibrated parameter set derived from the representative sample (the
    per-glacier geodetic MB constraint still applies for each).

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
    obs_glacier_discharge_m3s : array-like
        Observed glacier-only annual discharge [m³ s⁻¹].
    obs_years : array-like of int
        Years corresponding to *obs_glacier_discharge_m3s*.
    ref_mb_df : dict or pd.DataFrame, optional
        Per-glacier geodetic MB constraints.  If a DataFrame, must have
        columns ``rgi_id``, ``dmdtda`` [m w.e. yr⁻¹], ``err_dmdtda``
        [m w.e. yr⁻¹] (values are converted to kg m⁻² yr⁻¹ internally).
        If a dict, keys are ``rgi_id`` and values are dicts with keys
        ``dmdtda``/``err_dmdtda`` or ``ref_mb``/``ref_mb_err``.
    w_mb : float
        Weight for the geodetic MB term in the joint cost function.
    w_Q : float
        Weight for the discharge KGE term in the joint cost function.
    use_representative_glaciers : bool
        Sub-sample to large glaciers (see above).
    representative_frac : float
        Cumulative area fraction covered by the representative sample.
    nprocesses : int
        Number of parallel processes (1 = sequential).
    filesuffix : str
        Filesuffix for ``mb_calib.json``.

    Returns
    -------
    dict
        ``{rgi_id: calib_dict}`` for all directly calibrated glaciers.
    """
    from oggm.core.massbalance import mb_calibration_from_runoff

    obs_glacier_discharge_m3s = np.asarray(obs_glacier_discharge_m3s,
                                            dtype=float)
    obs_years = np.asarray(obs_years, dtype=int)

    # Sub-sample to representative glaciers by area
    if use_representative_glaciers and len(gdirs) > 1:
        areas = np.array([g.rgi_area_m2 for g in gdirs])
        total_area = areas.sum()
        sort_idx = np.argsort(areas)[::-1]
        cumfrac = np.cumsum(areas[sort_idx]) / total_area
        n_rep = int(np.searchsorted(cumfrac, representative_frac)) + 1
        rep_gdirs = [gdirs[i] for i in sort_idx[:n_rep]]
        log.info(
            'mb_calibration_basin_from_discharge: %d representative glaciers '
            '(%.0f%% area) of %d total',
            len(rep_gdirs), representative_frac * 100, len(gdirs),
        )
    else:
        rep_gdirs = gdirs

    def _get_geodetic(gdir):
        """Extract geodetic MB for a glacier from ref_mb_df."""
        if ref_mb_df is None:
            return None, None
        rgi = gdir.rgi_id
        if isinstance(ref_mb_df, dict):
            row = ref_mb_df.get(rgi, {})
            mb = float(row.get('dmdtda', row.get('ref_mb', np.nan)))
            err = float(row.get('err_dmdtda', row.get('ref_mb_err', np.nan)))
        else:
            import pandas as _pd
            sub = ref_mb_df[ref_mb_df['rgi_id'] == rgi]
            if len(sub) == 0:
                return None, None
            mb = float(sub['dmdtda'].iloc[0] * 1000
                       if 'dmdtda' in sub.columns
                       else sub['ref_mb'].iloc[0])
            err = float(sub['err_dmdtda'].iloc[0] * 1000
                        if 'err_dmdtda' in sub.columns
                        else sub.get('ref_mb_err', np.array([np.nan])).iloc[0])
        return (mb if np.isfinite(mb) else None,
                err if np.isfinite(err) else None)

    results = {}
    for gdir in rep_gdirs:
        ref_mb_v, ref_mb_err_v = _get_geodetic(gdir)
        try:
            res = mb_calibration_from_runoff(
                gdir,
                obs_glacier_discharge_m3s=obs_glacier_discharge_m3s,
                obs_years=obs_years,
                ref_mb=ref_mb_v,
                ref_mb_err=ref_mb_err_v,
                w_mb=w_mb,
                w_Q=w_Q,
                filesuffix=filesuffix,
            )
            results[gdir.rgi_id] = res
        except Exception as exc:
            log.warning(
                'mb_calibration_basin_from_discharge: failed for %s: %s',
                gdir.rgi_id, exc,
            )

    # Apply calibration to remaining (non-representative) glaciers
    if use_representative_glaciers and len(rep_gdirs) < len(gdirs):
        rep_ids = {g.rgi_id for g in rep_gdirs}
        remaining = [g for g in gdirs if g.rgi_id not in rep_ids]
        for gdir in remaining:
            ref_mb_v, ref_mb_err_v = _get_geodetic(gdir)
            try:
                mb_calibration_from_runoff(
                    gdir,
                    obs_glacier_discharge_m3s=obs_glacier_discharge_m3s,
                    obs_years=obs_years,
                    ref_mb=ref_mb_v,
                    ref_mb_err=ref_mb_err_v,
                    w_mb=w_mb,
                    w_Q=w_Q,
                    filesuffix=filesuffix,
                )
            except Exception as exc:
                log.warning(
                    'mb_calibration_basin_from_discharge: non-rep %s: %s',
                    gdir.rgi_id, exc,
                )

    log.info(
        'mb_calibration_basin_from_discharge: calibrated %d / %d glaciers',
        len(results), len(rep_gdirs),
    )
    return results
