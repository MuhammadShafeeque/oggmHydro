"""
Glacier hydrological post-processing tasks.

Provides routing of OGGM run_with_hydro() output to simulate
discharge at the basin/glacier outlet.  Requires run_with_hydro()
to have been run first (model_diagnostics file must exist).

Phase 1: Single linear reservoir  — route_hydro_output()
Phase 2: Two-component reservoir  — route_hydro_output_2c()
Phase 3: Basin-level aggregation  — aggregate_basin_discharge()
"""

import logging
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
