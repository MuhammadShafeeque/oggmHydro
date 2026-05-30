"""
Sandbox prototype for glacier discharge routing.

Phase 0 of the discharge routing development (branch: oggm-hydro-1).
Provides a linear reservoir routing function and a helper to apply it
directly to OGGM's ``run_with_hydro()`` output stored in
``model_diagnostics`` netCDF files.

This module is intentionally self-contained and has no hard OGGM
run-time dependency (beyond reading the netCDF output).  Once the
approach is validated here it will be promoted to
``oggm.core.hydrology``.

Usage example
-------------
>>> from oggm import cfg, workflow, tasks
>>> from oggm.sandbox.hydro_dev import route_glacier_runoff, linear_reservoir
>>>
>>> cfg.initialize()
>>> cfg.PARAMS['store_model_geometry'] = True
>>> gdir = workflow.init_glacier_directories(['RGI60-11.00897'])[0]
>>> # ... run full OGGM pipeline ...
>>> tasks.run_with_hydro(gdir, run_task=tasks.run_from_climate_data,
...                      ys=1985, ye=2015, store_monthly_hydro=True)
>>> df = route_glacier_runoff(gdir, k=3.0)
>>> print(df.head())
"""

# Builtins
import logging

# External libs
import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_RHO_WATER = 1000.0          # kg m-3  (liquid water density)
_SEC_PER_YEAR = 365.25 * 24 * 3600   # seconds per year
_SEC_PER_MONTH = 365.25 / 12 * 24 * 3600   # seconds per month (avg)


# ---------------------------------------------------------------------------
# Core routing math — no OGGM dependency
# ---------------------------------------------------------------------------

def linear_reservoir(q_in, k, dt=1.0):
    """Route a runoff time series through a single linear reservoir.

    Solves the linear storage equation analytically for each discrete
    timestep:

    .. math::

        Q_{out}(t + \\Delta t) = \\beta \\, Q_{in}(t) + \\alpha \\, Q_{out}(t)

    where :math:`\\alpha = e^{-\\Delta t / k}` (decay factor) and
    :math:`\\beta = 1 - \\alpha` (drainage fraction).

    Parameters
    ----------
    q_in : array-like, shape (N,)
        Input runoff time series.  Any consistent unit (e.g. m³ s⁻¹,
        kg yr⁻¹) — the output will be in the same unit.
    k : float
        Reservoir residence time.  Must be in the *same unit* as ``dt``.
        For example, if ``dt=1`` represents one year and you want a
        3-month residence time, pass ``k=3/12=0.25``.
    dt : float, optional
        Timestep size (default 1.0).  Must be in the same unit as ``k``.

    Returns
    -------
    q_out : np.ndarray, shape (N,)
        Routed discharge time series in the same unit as ``q_in``.

    Notes
    -----
    Initial condition: steady state, i.e. ``q_out[0] = q_in[0]``.

    Limiting behaviour:

    * ``k → 0``  ⟹  ``alpha → 0``, ``beta → 1``  ⟹  ``q_out ≈ q_in``
      (no smoothing, instantaneous routing)
    * ``k → ∞``  ⟹  ``alpha → 1``, ``beta → 0``  ⟹  ``q_out`` converges
      to the running mean (full smoothing)
    """
    q_in = np.asarray(q_in, dtype=float)
    if k <= 0:
        raise ValueError(f"Residence time k must be positive, got k={k}")
    if dt <= 0:
        raise ValueError(f"Timestep dt must be positive, got dt={dt}")

    alpha = np.exp(-dt / k)   # decay factor (fraction remaining each step)
    beta = 1.0 - alpha        # fraction draining each step

    q_out = np.empty_like(q_in)
    q_out[0] = q_in[0]        # steady-state initial condition
    for t in range(1, len(q_in)):
        q_out[t] = beta * q_in[t] + alpha * q_out[t - 1]

    return q_out


def two_component_reservoir(q_fast_in, q_slow_in, k_fast, k_slow, dt=1.0):
    """Route fast (rain) and slow (melt) components through separate reservoirs.

    Parameters
    ----------
    q_fast_in : array-like, shape (N,)
        Fast-component runoff (liquid precipitation on/off glacier).
    q_slow_in : array-like, shape (N,)
        Slow-component runoff (melt on/off glacier).
    k_fast : float
        Residence time for the fast (rain) reservoir [same unit as dt].
    k_slow : float
        Residence time for the slow (melt) reservoir [same unit as dt].
    dt : float, optional
        Timestep size (default 1.0).

    Returns
    -------
    q_total : np.ndarray, shape (N,)
        Total routed discharge = fast + slow components.
    q_fast_out : np.ndarray, shape (N,)
        Routed fast component.
    q_slow_out : np.ndarray, shape (N,)
        Routed slow component.
    """
    q_fast_out = linear_reservoir(q_fast_in, k=k_fast, dt=dt)
    q_slow_out = linear_reservoir(q_slow_in, k=k_slow, dt=dt)
    q_total = q_fast_out + q_slow_out
    return q_total, q_fast_out, q_slow_out


# ---------------------------------------------------------------------------
# OGGM integration helpers
# ---------------------------------------------------------------------------

def _read_hydro_vars(gdir, filesuffix=''):
    """Read hydrology variables from a model_diagnostics netCDF file.

    Parameters
    ----------
    gdir : GlacierDirectory
    filesuffix : str
        Filesuffix of the model_diagnostics file.

    Returns
    -------
    ds : xr.Dataset (open — caller must close or use as context manager)
    """
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    return xr.open_dataset(fpath)


def route_glacier_runoff(gdir, filesuffix='', k=3.0):
    """Apply single-reservoir routing to ``run_with_hydro()`` output.

    Reads the ``model_diagnostics`` netCDF produced by
    ``oggm.core.flowline.run_with_hydro()``, computes total runoff,
    converts to m³ s⁻¹, and routes through a linear reservoir with
    residence time ``k`` (months).

    Parameters
    ----------
    gdir : GlacierDirectory
        The glacier directory to process.
    filesuffix : str, optional
        Filesuffix of the ``model_diagnostics`` file (default ``''``).
    k : float, optional
        Reservoir residence time in **months** (default 3.0).
        Internally converted to years to match the annual OGGM timestep.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns:

        * ``runoff_kg_yr``   — raw total runoff [kg yr⁻¹]
        * ``runoff_m3s``     — runoff converted to [m³ s⁻¹]
        * ``discharge_m3s``  — routed discharge [m³ s⁻¹]

    Notes
    -----
    The last timestep in OGGM's ``model_diagnostics`` is always NaN
    (convention) and is dropped before routing.

    The routine assumes **annual** OGGM output (``store_monthly_hydro``
    can also be present but is not used here — see
    ``route_glacier_runoff_monthly`` for monthly routing).
    """
    with _read_hydro_vars(gdir, filesuffix=filesuffix) as ds:
        # Total runoff = melt + liquid precipitation (kg yr-1, annual)
        runoff_kgyr = (
            ds['melt_on_glacier'] +
            ds['melt_off_glacier'] +
            ds['liq_prcp_on_glacier'] +
            ds['liq_prcp_off_glacier']
        ).values

    # Drop the trailing NaN year (OGGM convention: last year is always NaN)
    if np.isnan(runoff_kgyr[-1]):
        runoff_kgyr = runoff_kgyr[:-1]

    # Convert kg yr-1 → m³ s-1
    q_m3s = runoff_kgyr / _RHO_WATER / _SEC_PER_YEAR

    # Route: k in months → convert to years for annual dt=1 year
    k_years = k / 12.0
    q_routed = linear_reservoir(q_m3s, k=k_years, dt=1.0)

    df = pd.DataFrame({
        'runoff_kg_yr': runoff_kgyr,
        'runoff_m3s': q_m3s,
        'discharge_m3s': q_routed,
    })
    return df


def route_glacier_runoff_monthly(gdir, filesuffix='',
                                  k_months=3.0):
    """Apply single-reservoir routing to monthly ``run_with_hydro()`` output.

    Requires ``run_with_hydro()`` to have been called with
    ``store_monthly_hydro=True``.  Uses the ``*_monthly`` variables and
    dt = 1 month, which gives more realistic routing than annual data for
    k values < 12 months.

    Parameters
    ----------
    gdir : GlacierDirectory
    filesuffix : str, optional
    k_months : float, optional
        Residence time in **months** (default 3.0).  Because dt = 1 month,
        this is passed directly as ``k`` to :func:`linear_reservoir`.

    Returns
    -------
    df : pd.DataFrame
        Monthly DataFrame with columns ``runoff_m3s`` and
        ``discharge_m3s``.
    """
    monthly_vars = [
        'melt_on_glacier_monthly',
        'melt_off_glacier_monthly',
        'liq_prcp_on_glacier_monthly',
        'liq_prcp_off_glacier_monthly',
    ]
    with _read_hydro_vars(gdir, filesuffix=filesuffix) as ds:
        missing = [v for v in monthly_vars if v not in ds]
        if missing:
            raise RuntimeError(
                f"Monthly hydro variables not found in model_diagnostics "
                f"({missing}). Re-run run_with_hydro() with "
                f"store_monthly_hydro=True."
            )
        runoff_kgmonth = (
            ds['melt_on_glacier_monthly'] +
            ds['melt_off_glacier_monthly'] +
            ds['liq_prcp_on_glacier_monthly'] +
            ds['liq_prcp_off_glacier_monthly']
        ).values.ravel()

    # Drop trailing NaNs
    valid = ~np.isnan(runoff_kgmonth)
    runoff_kgmonth = runoff_kgmonth[valid]

    # Convert kg month-1 → m³ s-1
    q_m3s = runoff_kgmonth / _RHO_WATER / _SEC_PER_MONTH

    # Route — dt=1 month, k in months
    q_routed = linear_reservoir(q_m3s, k=k_months, dt=1.0)

    df = pd.DataFrame({
        'runoff_m3s': q_m3s,
        'discharge_m3s': q_routed,
    })
    return df


def route_glacier_runoff_2c(gdir, filesuffix='',
                             k_fast=1.0, k_slow=6.0):
    """Two-component routing: separate fast (rain) and slow (melt) reservoirs.

    Parameters
    ----------
    gdir : GlacierDirectory
    filesuffix : str, optional
    k_fast : float, optional
        Residence time for the fast (rain) component in **months**
        (default 1.0).
    k_slow : float, optional
        Residence time for the slow (melt) component in **months**
        (default 6.0).

    Returns
    -------
    df : pd.DataFrame
        Annual DataFrame with columns:

        * ``rain_m3s``       — liquid precip [m³ s⁻¹]
        * ``melt_m3s``       — melt runoff [m³ s⁻¹]
        * ``discharge_fast`` — routed rain discharge [m³ s⁻¹]
        * ``discharge_slow`` — routed melt discharge [m³ s⁻¹]
        * ``discharge_m3s``  — total routed discharge [m³ s⁻¹]
    """
    with _read_hydro_vars(gdir, filesuffix=filesuffix) as ds:
        rain_kgyr = (
            ds['liq_prcp_on_glacier'] +
            ds['liq_prcp_off_glacier']
        ).values
        melt_kgyr = (
            ds['melt_on_glacier'] +
            ds['melt_off_glacier']
        ).values

    # Drop trailing NaN year
    if np.isnan(rain_kgyr[-1]):
        rain_kgyr = rain_kgyr[:-1]
        melt_kgyr = melt_kgyr[:-1]

    rain_m3s = rain_kgyr / _RHO_WATER / _SEC_PER_YEAR
    melt_m3s = melt_kgyr / _RHO_WATER / _SEC_PER_YEAR

    k_fast_yr = k_fast / 12.0
    k_slow_yr = k_slow / 12.0

    q_total, q_fast_out, q_slow_out = two_component_reservoir(
        rain_m3s, melt_m3s,
        k_fast=k_fast_yr, k_slow=k_slow_yr, dt=1.0
    )

    df = pd.DataFrame({
        'rain_m3s': rain_m3s,
        'melt_m3s': melt_m3s,
        'discharge_fast': q_fast_out,
        'discharge_slow': q_slow_out,
        'discharge_m3s': q_total,
    })
    return df


# ---------------------------------------------------------------------------
# Validation helpers (run interactively to check math)
# ---------------------------------------------------------------------------

def validate_linear_reservoir(n=50, k=3.0, verbose=True):
    """Quick numerical validation of :func:`linear_reservoir`.

    Checks three limiting-behaviour properties:

    1. **Mass conservation** — long-run sum of output ≈ sum of input
       (holds exactly in steady periodic forcing).
    2. **k → 0 limit** — output ≈ input (no smoothing).
    3. **k → ∞ limit** — output approaches a constant (full smoothing).

    Parameters
    ----------
    n : int
        Length of synthetic runoff signal (years).
    k : float
        Nominal residence time to test (years).
    verbose : bool
        If True, print a summary.

    Returns
    -------
    results : dict
        Dictionary with boolean ``passed`` flags for each check.
    """
    # Synthetic seasonal-like runoff (sine wave with positive offset)
    t = np.arange(n, dtype=float)
    q_in = 10.0 + 8.0 * np.sin(2 * np.pi * t / 5)  # 5-year cycle

    results = {}

    # 1. Mass conservation check (after warm-up, use last 80% of series)
    q_out_nominal = linear_reservoir(q_in, k=k, dt=1.0)
    warm = int(0.2 * n)
    ratio = q_out_nominal[warm:].sum() / q_in[warm:].sum()
    results['mass_conservation'] = abs(ratio - 1.0) < 0.15  # within 15%

    # 2. k → 0 gives q_out ≈ q_in
    q_out_fast = linear_reservoir(q_in, k=1e-4, dt=1.0)
    results['k_zero_limit'] = np.allclose(q_out_fast, q_in, rtol=1e-2)

    # 3. k → ∞ gives nearly constant output
    q_out_slow = linear_reservoir(q_in, k=1e4, dt=1.0)
    cv_out = q_out_slow[warm:].std() / q_out_slow[warm:].mean()
    results['k_inf_limit'] = cv_out < 0.01  # coefficient of variation < 1%

    results['all_passed'] = all(results.values())

    if verbose:
        print("Linear reservoir validation:")
        for key, val in results.items():
            status = "PASS" if val else "FAIL"
            print(f"  [{status}] {key}")

    return results


def initialize_routing_params():
    """Initialize default routing parameters in cfg.PARAMS.

    Call this if you want to use the routing sandbox functions
    before the params are added to params.cfg.  Mirrors the calving
    sandbox pattern (see ``calving_jan.initialize_calving_params``).
    """
    try:
        from oggm import cfg
    except ImportError:
        return  # allow import without full OGGM install

    defaults = {
        'routing_k_months': 3.0,
        'routing_k_fast_months': 1.0,
        'routing_k_slow_months': 6.0,
        'routing_scheme': 'single_reservoir',
    }
    for key, val in defaults.items():
        if key not in cfg.PARAMS:
            cfg.PARAMS[key] = val
