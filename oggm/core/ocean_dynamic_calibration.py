"""Calibration of the calving constant through the dynamic model.

A constant fitted on the inverted front, ``k = Q_obs / flux(k=1)``
(:func:`oggm.core.ocean_inversion.fit_calving_k`), is not the constant at
which the dynamic model returns ``Q_obs``: the run starts from a front the
measured bed thinned, and thickens it again.
:func:`run_dynamic_calving_k_calibration` corrects it one pass at a time, each
pass a full inversion, dynamic spinup and control run
(:func:`dynamic_calving_k_run_with_dynamic_spinup`), until the control's mean
frontal ablation over the reference period is within a tolerance of the
target. The update rule (:func:`propose_calving_k`):

- ``k <- k * (Q_target / Q_dyn) ** (1 / e)`` while every pass is on one side
  of the target, with ``e`` the response ``d ln Q / d ln k`` of the two passes
  closest to the target where it lies in [0.2, 3] (``rule = 'response'``),
  else 1 (``'ratio'``), and the factor bounded by ``max_step``;
- log-interpolated between the narrowest pair of neighbouring constants that
  straddles the target, once one exists (``'bracket'``).

A front that does not close ends on the constant of its closest pass
(``'best_pass'``, :func:`best_calving_k`). Glaciers without a target take the
geometric mean of the calibrated constants
(:func:`calving_k_for_glaciers_without_target`).
"""
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from oggm import entity_task, utils
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

log = logging.getLogger(__name__)

# Errors of the dynamic spinup and the flowline model a pass reacts to
STALLED = ('Not able to minimise', 'Could not find mismatch')
ICE_FREE = 'ice free'
CFL = 'CFL error'

# The fronts best_calving_k may still move
OPEN_REASONS = ('', 'step_bounded')


def _history_frame(history):
    """(k, q) pairs as a frame, in the order they were measured."""
    a = np.asarray(history, dtype=float).reshape(-1, 2)
    return pd.DataFrame({'calving_k': a[:, 0], 'q_dyn': a[:, 1]})


def propose_calving_k(k_now, q_now, target, history, max_step=4.):
    """The next calving constant of one front, and the rule that gave it.

    Parameters
    ----------
    k_now : float
        the constant of the last control run, yr-1
    q_now : float
        the frontal ablation that run returned, > 0
    target : float
        the frontal ablation to match, > 0, in the unit of ``q_now``
    history : list of (k, q) pairs
        every pass measured on the front, the last one included
    max_step : float
        the largest factor between two constants outside a bracket

    Returns
    -------
    (float, str)
        the constant and ``'bracket'``, ``'response'`` or ``'ratio'``
    """
    cols = ['calving_k', 'q_dyn']
    seen = _history_frame(history)
    seen = seen[(seen['q_dyn'] > 0) & (seen['calving_k'] > 0)]
    # The narrowest straddling pair of neighbours, so that a response that is
    # not monotonic never steps out of a bracket already found. Stable sorts:
    # passes at the same constant stay in the order they were measured.
    s = seen.sort_values('calving_k', kind='stable')[cols].to_numpy()
    pairs = [(a, b) for a, b in zip(s[:-1], s[1:])
             if a[0] < b[0] and (a[1] - target) * (b[1] - target) < 0]
    if pairs:
        (k0, q0), (k1, q1) = min(pairs, key=lambda p: p[1][0] / p[0][0])
        w = np.log(target / q0) / np.log(q1 / q0)
        k = float(np.exp(np.log(k0) + w * (np.log(k1) - np.log(k0))))
        if np.isclose(seen['calving_k'], k, rtol=1e-3).any():
            k = float(np.sqrt(k0 * k1))
        return k, 'bracket'
    # All on one side: the response the two passes closest to the target
    # measured, where it is sane
    two_sided = ((seen['q_dyn'] < target).any() and
                 (seen['q_dyn'] > target).any())
    near = seen.assign(miss=np.abs(np.log(seen['q_dyn'] / target)))
    near = near.sort_values('miss', kind='stable')
    e = 1.
    if len(near) > 1 and not two_sided:
        (k0, q0), (k1, q1) = (near[cols].iloc[0].tolist(),
                              near[cols].iloc[1].tolist())
        if k0 != k1 and q0 != q1:
            slope = np.log(q1 / q0) / np.log(k1 / k0)
            e = slope if 0.2 <= slope <= 3 else 1.
    factor = float(np.clip((target / q_now) ** (1 / e), 1 / max_step,
                           max_step))
    return k_now * factor, 'ratio' if e == 1 else 'response'


def calving_k_step(k_prev, q_dyn, target, target_err=None, history=(),
                   tolerance=0.05, max_step=4.):
    """One correction pass of one front.

    Parameters
    ----------
    k_prev : float
        the constant the control ran with, yr-1
    q_dyn : float or None
        the frontal ablation the control returned; None or NaN when it
        did not run
    target : float or None
        the frontal ablation to match; None or NaN for no target
    target_err : float or None
        the error of the target. A front within the larger of
        ``tolerance`` and ``target_err / target`` is ``within_error``.
    history : list of (k, q) pairs
        the passes measured on the front before this one
    tolerance : float
        relative mismatch within which the front is converged
    max_step : float
        see :func:`propose_calving_k`

    Returns
    -------
    dict
        ``calving_k`` (the next constant), ``rule``, ``reason`` (``''`` or
        one of ``no_target``, ``target_not_positive``, ``no_control_run``,
        ``control_flux_zero``, ``step_bounded``), ``ratio``, ``converged``,
        ``within_error`` and ``moved``
    """
    def num(v):
        return np.nan if v is None else float(v)

    tgt, dyn, err = num(target), num(q_dyn), num(target_err)
    gated = not np.isnan(tgt)
    ratio = dyn / tgt if tgt > 0 else np.nan
    rel_err = err / tgt if tgt > 0 else np.nan
    wide = max(tolerance, 0. if np.isnan(rel_err) else rel_err)
    converged = bool(abs(ratio - 1) <= tolerance)
    within_error = bool(abs(ratio - 1) <= wide)
    reason = ''
    if not gated:
        reason = 'no_target'
    elif not tgt > 0:
        reason = 'target_not_positive'
    elif np.isnan(dyn):
        reason = 'no_control_run'
    elif dyn <= 0:
        reason = 'control_flux_zero'
    moved = reason == '' and not converged
    k, rule = float(k_prev), ''
    if moved:
        k, rule = propose_calving_k(k_prev, dyn, tgt,
                                    list(history) + [(k_prev, dyn)],
                                    max_step=max_step)
        if (rule != 'bracket' and
                not 1 / max_step < k / k_prev < max_step):
            reason = 'step_bounded'
    return dict(calving_k=k, rule=rule, reason=reason, ratio=ratio,
                converged=converged, within_error=within_error, moved=moved)


def best_calving_k(history, target):
    """The pass whose control came closest to the target, in log space.

    Parameters
    ----------
    history : list of (k, q) pairs
        every pass measured on the front
    target : float
        the frontal ablation to match

    Returns
    -------
    (float, float) or None
        the (k, q) of that pass, or None without a target > 0 or a
        positive control
    """
    h = _history_frame(history)
    h = h[h['q_dyn'] > 0]
    if not (target is not None and target > 0) or h.empty:
        return None
    i = int(np.argmin(np.abs(np.log(h['q_dyn'] / target)).to_numpy()))
    return float(h['calving_k'].iloc[i]), float(h['q_dyn'].iloc[i])


def mean_frontal_ablation(gdir, period, filesuffix=''):
    """Mean frontal ablation of a run over ``period``, Gt yr-1.

    From the cumulative ``calving_m3`` of ``model_diagnostics<filesuffix>``,
    at ``gdir.settings['ice_density']``.

    Returns
    -------
    float
        NaN when the run does not cover the period or has no calving output
    """
    y0, y1 = period
    fp = gdir.get_filepath('model_diagnostics', filesuffix=filesuffix)
    if not os.path.exists(fp):
        return np.nan
    with xr.open_dataset(fp) as ds:
        t = ds['time'].values
        if 'calving_m3' not in ds or t.min() > y0 or t.max() < y1:
            return np.nan
        c = ds['calving_m3'].sel(time=[y0, y1]).values
    rho = gdir.settings['ice_density'] / 1000.
    return float(c[1] - c[0]) / (y1 - y0) / 1e9 * rho


def _settings_over(gdir, settings_filesuffix, name, **values):
    """A fresh settings file over ``settings_filesuffix`` holding ``values``;
    returns its filesuffix."""
    sfx = f'{settings_filesuffix}{name}'
    path = gdir.get_filepath('settings', filesuffix=sfx)
    if os.path.exists(path):
        os.remove(path)
    ms = utils.ModelSettings(gdir, filesuffix=sfx,
                             parent_filesuffix=settings_filesuffix)
    for key, val in values.items():
        ms.set(key, val)
    return sfx


def _error(task, gdir, **kwargs):
    """Run an entity task; '' or the error it raised, truncated."""
    try:
        task(gdir, continue_on_error=False, **kwargs)
    except Exception as e:  # noqa: BLE001
        return f'{type(e).__name__}: {e}'[:200]
    return ''


def dynamic_calving_k_run_with_dynamic_spinup(
        gdir, calving_k=None, ref_period=(2000, 2010), settings_filesuffix='',
        output_filesuffix='_dyn_k', spinup_filesuffix=None, glen_a=None,
        fs=None, init_yr=1990, ye=2020,
        climate_filename='climate_historical', climate_input_filesuffix='',
        precision_absolute=3., precision_percent=1., maxiter=60,
        stalled_precision=(8., 2.), spinup_cfl_min_dt=(10., 1.),
        run_cfl_min_dt=(10.,), kwargs_spinup=None):
    """One pass of :func:`run_dynamic_calving_k_calibration`.

    The default ``run_function``. With the calving constant already in the
    settings: the calving inversion at a fixed Glen A
    (:func:`oggm.core.ocean_inversion.find_inversion_calving_from_bathymetry`
    when ``inversion_calving_from_bathymetry`` is set, else
    :func:`oggm.core.inversion.find_inversion_calving_from_any_mb`),
    :func:`oggm.core.flowline.init_present_time_glacier`, on the bathymetry
    route :func:`oggm.core.bedmachine_flowline.bedmachine_terminus_bed`, the
    dynamic spinup matching the area and starting no later than ``init_yr``,
    then the control: a constant-k run from the spinup's ``init_yr`` state to
    ``ye``. Its mean frontal ablation over ``ref_period`` is returned.

    A spinup search that stalls is run again at ``stalled_precision``, one
    stopped by the CFL criterion at each ``spinup_cfl_min_dt`` in turn. A
    spinup that still fails, or starts after ``init_yr``, leaves the control
    to start cold from the calibrated glacier in ``init_yr``. A control
    stopped by the CFL criterion is run again at each ``run_cfl_min_dt``, and
    if a spun-up one still stops, cold. The start is written to the settings
    as ``calving_k_dyn_start`` (``'spinup'`` or ``'cold'``), the spinup's
    last error as ``calving_k_dyn_spinup_error``.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    calving_k : float
        the constant of this pass, yr-1, for the record: the caller has
        already written it to the settings
    ref_period : tuple of two years
        the window of the mean frontal ablation
    settings_filesuffix : str
        the settings to run with
    output_filesuffix : str
        the filesuffix of the control run
    spinup_filesuffix : str, optional
        the filesuffix of the spinup run. Default: ``output_filesuffix``
        + ``'_spinup'``.
    glen_a, fs : float, optional
        the inversion's. Default: the settings'.
    init_yr : int
        the year the control starts in
    ye : int
        the year the control ends in
    climate_filename, climate_input_filesuffix : str
        the climate of the control
    precision_absolute, precision_percent, maxiter :
        passed to :func:`oggm.core.dynamic_spinup.run_dynamic_spinup`
    stalled_precision : (float, float)
        ``precision_absolute`` and ``precision_percent`` of the retry of a
        stalled spinup search
    spinup_cfl_min_dt, run_cfl_min_dt : tuple of float
        the ``cfl_min_dt`` of the CFL retries, s
    kwargs_spinup : dict, optional
        more keyword arguments for the spinup

    Returns
    -------
    float
        the control's mean frontal ablation over ``ref_period``, Gt yr-1
    """
    from oggm.core.bedmachine_flowline import bedmachine_terminus_bed
    from oggm.core.dynamic_spinup import run_dynamic_spinup
    from oggm.core.flowline import (init_present_time_glacier,
                                    run_from_climate_data)
    from oggm.core.inversion import find_inversion_calving_from_any_mb
    from oggm.core.ocean_inversion import (
        find_inversion_calving_from_bathymetry)

    if spinup_filesuffix is None:
        spinup_filesuffix = f'{output_filesuffix}_spinup'
    sfx = dict(settings_filesuffix=settings_filesuffix)

    gdir.settings_filesuffix = settings_filesuffix
    bathymetry = gdir.settings['inversion_calving_from_bathymetry']
    if bathymetry:
        find_inversion_calving_from_bathymetry(gdir, glen_a=glen_a, fs=fs,
                                               continue_on_error=False, **sfx)
    else:
        find_inversion_calving_from_any_mb(gdir, glen_a=glen_a, fs=fs,
                                           continue_on_error=False, **sfx)
    init_present_time_glacier(gdir, continue_on_error=False, **sfx)
    if bathymetry:
        gdir.settings_filesuffix = settings_filesuffix
        depth = gdir.settings['terminus_water_depth']
        # The terminus bed edits its kept copy when there is one, which is
        # the flowline of an earlier pass
        gdir.get_filepath('model_flowlines', filesuffix='_synthetic',
                          delete=True)
        d = list(bedmachine_terminus_bed(gdir, water_depth=depth,
                                         continue_on_error=False).values())[-1]
        if not d['water_depth_after'] > 0:
            raise InvalidWorkflowError(f'({gdir.rgi_id}) front still on land '
                                       'after the terminus bed')

    kw = dict(minimise_for='area', output_filesuffix=spinup_filesuffix,
              allow_calving=True, store_model_geometry=True,
              ignore_errors=False, precision_absolute=precision_absolute,
              precision_percent=precision_percent, maxiter=maxiter,
              spinup_start_yr_max=init_yr,
              climate_input_filesuffix=climate_input_filesuffix,
              **(kwargs_spinup or {}))
    spin_sfx = settings_filesuffix
    err = _error(run_dynamic_spinup, gdir, settings_filesuffix=spin_sfx, **kw)
    if any(s in err for s in STALLED) and ICE_FREE not in err:
        p_abs, p_pct = stalled_precision
        err = _error(run_dynamic_spinup, gdir, settings_filesuffix=spin_sfx,
                     **{**kw, 'precision_absolute': p_abs,
                        'precision_percent': p_pct})
    elif CFL in err:
        for dt in spinup_cfl_min_dt:
            spin_sfx = _settings_over(gdir, settings_filesuffix,
                                      f'_cfl{dt:g}', cfl_min_dt=dt)
            err = _error(run_dynamic_spinup, gdir,
                         settings_filesuffix=spin_sfx, **kw)
            if CFL not in err:
                break
    spun = err == ''
    if spun:
        gdir.settings_filesuffix = spin_sfx
        spun = bool(gdir.settings['run_dynamic_spinup_success'])
    if spun:
        fp = gdir.get_filepath('model_geometry', filesuffix=spinup_filesuffix)
        with xr.open_dataset(fp) as ds:
            if float(ds['time'][0]) > init_yr:
                spun = False
                err = f'spinup starts in {float(ds["time"][0]):.0f}'

    def control(spun, run_sfx):
        for name in ('model_diagnostics', 'model_geometry', 'fl_diagnostics'):
            gdir.get_filepath(name, filesuffix=output_filesuffix, delete=True)
        init = (dict(init_model_filesuffix=spinup_filesuffix,
                     init_model_yr=init_yr) if spun else {})
        return _error(run_from_climate_data, gdir, ys=init_yr, ye=ye,
                      climate_filename=climate_filename,
                      climate_input_filesuffix=climate_input_filesuffix,
                      output_filesuffix=output_filesuffix,
                      settings_filesuffix=run_sfx, **init)

    def control_with_cfl_retry(spun):
        run_err = control(spun, settings_filesuffix)
        for dt in run_cfl_min_dt:
            if CFL not in run_err:
                break
            run_err = control(spun, _settings_over(
                gdir, settings_filesuffix, f'_cfl{dt:g}', cfl_min_dt=dt))
        return run_err

    run_err = control_with_cfl_retry(spun)
    if spun and CFL in run_err:
        spun = False
        run_err = control_with_cfl_retry(spun)

    gdir.settings_filesuffix = settings_filesuffix
    gdir.settings['calving_k_dyn_start'] = 'spinup' if spun else 'cold'
    gdir.settings['calving_k_dyn_spinup_error'] = err
    if run_err:
        raise RuntimeError(f'control run failed: {run_err}')
    return mean_frontal_ablation(gdir, ref_period,
                                 filesuffix=output_filesuffix)


def _set_calving_k(gdir, k):
    # The inversion and the forward model read separate keys
    gdir.settings['inversion_calving_k'] = float(k)
    gdir.settings['calving_k'] = float(k)


def _record(gdir, settings_filesuffix, out):
    gdir.settings_filesuffix = settings_filesuffix
    for key, val in out.items():
        gdir.settings[key] = val
        if settings_filesuffix == '':
            gdir.add_to_diagnostics(key, val)


def _num(v):
    return np.nan if v is None else float(v)


@entity_task(log, writes=['inversion_output', 'model_flowlines',
                          'model_geometry', 'model_diagnostics'])
def run_dynamic_calving_k_calibration(
        gdir, settings_filesuffix='', ref_fa=None, ref_fa_err=None,
        ref_period=(2000, 2010), k_first_guess=None, glen_a=None, fs=None,
        tolerance=0.05, max_step=4., maxiter=10, ignore_errors=False,
        output_filesuffix='_dyn_k',
        run_function=dynamic_calving_k_run_with_dynamic_spinup,
        kwargs_run_function=None):
    """Calibrate the calving constant so that the dynamic model returns the
    observed frontal ablation.

    Each pass writes the constant to ``calving_k`` and
    ``inversion_calving_k`` and calls ``run_function``, which returns the
    mean frontal ablation of a control run over ``ref_period``; the next
    constant follows from :func:`calving_k_step`. The search stops within
    ``tolerance`` of ``ref_fa``, when a pass cannot be corrected (``reason``),
    or after ``maxiter`` passes. A front still outside the tolerance then
    goes back to the constant of its closest pass (:func:`best_calving_k`),
    run once more so that the directory holds that run.

    The mass balance is not recalibrated between passes: calibrate it with
    ``ref_fa`` as the frontal ablation first, so that ``melt_f`` does not
    depend on the constant.

    The constant and the record are written to the settings (and to the
    diagnostics with the default settings): ``calving_k_static`` (the first
    guess), ``calving_k_dyn_reason``, ``_rule`` (``'best_pass'`` for the
    closest pass, else the rule of the last correction, ``''`` if the front
    converged), ``_rules`` (one per correction), ``_passes`` (the number of
    corrections), ``_converged``, ``_within_error``, ``_q_target``,
    ``_q_target_err``, ``_q_dyn`` (the control at the final constant),
    ``_history`` (every ``[k, q]`` run) and ``_error``. A glacier without a
    target, or with one not > 0, is not run.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    settings_filesuffix : str
        You can use a different set of settings by providing a filesuffix.
        This is useful for sensitivity experiments. Code-wise the
        settings_filesuffix is set in the @entity-task decorater.
    ref_fa : float, optional
        the observed frontal ablation, Gt yr-1. None: no target.
    ref_fa_err : float, optional
        its error, Gt yr-1, for ``within_error``
    ref_period : tuple of two years
        the period ``ref_fa`` is the mean of
    k_first_guess : float, optional
        yr-1. Default: ``gdir.settings['calving_k']``.
    glen_a, fs : float, optional
        the inversion's, held fixed over the passes. Default: the settings'
        at the start.
    tolerance : float
        relative mismatch within which the front is converged
    max_step : float
        the largest factor between two constants outside a bracket
    maxiter : int
        the largest number of passes, before the rerun of the closest one
    ignore_errors : bool
        if True, a pass whose ``run_function`` raises ends the search as
        ``no_control_run``, the constant of that pass kept; else the error
        is raised
    output_filesuffix : str
        passed to ``run_function``
    run_function : callable
        ``run_function(gdir, calving_k=, ref_period=, settings_filesuffix=,
        output_filesuffix=, glen_a=, fs=, **kwargs_run_function)`` returns
        the control's mean frontal ablation over ``ref_period`` in Gt yr-1.
        Default: :func:`dynamic_calving_k_run_with_dynamic_spinup`.
    kwargs_run_function : dict, optional
        more keyword arguments for ``run_function``

    Returns
    -------
    dict
        the record, or None for a glacier that is not tidewater
    """
    if maxiter < 1:
        raise InvalidParamsError('maxiter must be at least 1')
    if not gdir.is_tidewater:
        _record(gdir, settings_filesuffix,
                {'calving_k_dyn_reason': 'not_tidewater'})
        return None

    k_static = float(gdir.settings['calving_k'] if k_first_guess is None
                     else k_first_guess)
    tgt, err = _num(ref_fa), _num(ref_fa_err)
    out = {'calving_k_static': k_static, 'calving_k_dyn_q_target': tgt,
           'calving_k_dyn_q_target_err': err}
    if np.isnan(tgt) or not tgt > 0:
        out.update(calving_k_dyn_reason=('no_target' if np.isnan(tgt) else
                                         'target_not_positive'),
                   calving_k_dyn_passes=0, calving_k_dyn_history=[])
        _record(gdir, settings_filesuffix, out)
        return out

    if glen_a is None:
        glen_a = gdir.settings['inversion_glen_a']
    if fs is None:
        fs = gdir.settings['inversion_fs']
    kw = dict(ref_period=list(ref_period), output_filesuffix=output_filesuffix,
              glen_a=glen_a, fs=fs, **(kwargs_run_function or {}))
    errors = []

    def one_pass(k):
        gdir.settings_filesuffix = settings_filesuffix
        _set_calving_k(gdir, k)
        try:
            q = run_function(gdir, calving_k=k,
                             settings_filesuffix=settings_filesuffix, **kw)
        except Exception as e:  # noqa: BLE001
            if not ignore_errors:
                raise
            errors.append(f'{type(e).__name__}: {e}'[:200])
            q = np.nan
        history.append([float(k), _num(q)])
        return _num(q)

    history, rules, passes, k = [], [], 0, k_static
    for _ in range(maxiter):
        q = one_pass(k)
        step = calving_k_step(k, q, tgt, err, history[:-1],
                              tolerance=tolerance, max_step=max_step)
        if not step['moved']:
            break
        passes += 1
        rules.append(step['rule'])
        k = step['calving_k']

    rule = step['rule']
    k_final, q_final = history[-1]
    if step['reason'] in OPEN_REASONS and not step['converged']:
        k_final, q_final = best_calving_k(history, tgt)
        rule = 'best_pass'
        if k_final != history[-1][0]:
            q_final = one_pass(k_final)
    gdir.settings_filesuffix = settings_filesuffix
    _set_calving_k(gdir, k_final)
    final = calving_k_step(k_final, q_final, tgt, err, tolerance=tolerance)
    reason = step['reason']
    if errors and np.isnan(q_final):
        reason = 'no_control_run'

    out.update(calving_k_dyn_reason=reason, calving_k_dyn_rule=rule,
               calving_k_dyn_rules=rules, calving_k_dyn_passes=passes,
               calving_k_dyn_converged=final['converged'],
               calving_k_dyn_within_error=final['within_error'],
               calving_k_dyn_q_dyn=q_final, calving_k_dyn_history=history,
               calving_k_dyn_error='; '.join(errors))
    _record(gdir, settings_filesuffix, out)
    out['calving_k'] = k_final
    return out


def calving_k_for_glaciers_without_target(gdirs, exclude=(),
                                          settings_filesuffix=''):
    """Give the tidewater glaciers without a target the geometric mean of the
    calibrated constants.

    The constants are those :func:`run_dynamic_calving_k_calibration` left
    on the glaciers with a target (``calving_k_dyn_q_target``), those in
    ``exclude`` left out of the mean; it is written to both calving
    constants of every other tidewater glacier, with
    ``calving_k_dyn_rule = 'geometric_mean'``.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        the glacier directories
    exclude : list of str
        RGI ids that keep their own constant but do not inform the mean
    settings_filesuffix : str
        the settings to read and write

    Returns
    -------
    float
        the mean, or None if no glacier has a calibrated constant
    """
    fit, rest = [], []
    for gdir in gdirs:
        if not gdir.is_tidewater:
            continue
        gdir.settings_filesuffix = settings_filesuffix
        try:
            tgt = _num(gdir.settings['calving_k_dyn_q_target'])
        except KeyError:
            tgt = np.nan
        if np.isnan(tgt):
            rest.append(gdir)
            continue
        k = gdir.settings['calving_k']
        if k > 0 and gdir.rgi_id not in exclude:
            fit.append(k)
    if not fit:
        return None
    k_mean = float(np.exp(np.log(pd.Series(fit, dtype=float)).mean()))
    for gdir in rest:
        _set_calving_k(gdir, k_mean)
        _record(gdir, settings_filesuffix,
                {'calving_k_dyn_rule': 'geometric_mean'})
    return k_mean
