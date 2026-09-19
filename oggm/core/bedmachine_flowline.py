"""The measured bed under and beyond the calving front.

``init_present_time_glacier`` continues a tidewater flowline past its terminus over a
bed it invents: ``calving_line_extension`` points (30) deepening linearly at
``calving_front_slope`` (tan alpha = 0.05), at a width that is the mean of the last
five inversion widths. ``params.cfg`` calls both "arbitrary". That fabricated
bathymetry is not inert: every calving law is proportional to the water depth ``d``
and the width ``w`` at the last cell above water level, so as soon as the front leaves
the inverted domain both factors are made up.

At a site with retrograde beds the sign is wrong as well as the magnitude, so this
module replaces the extension with BedMachine bed sampled along the flowline itself,
and keeps the synthetic version beside it under its own filesuffix so the two can be
run against each other. Nothing OGGM owns is modified: this is an edit of
``model_flowlines`` applied after ``init_present_time_glacier``.

:py:func:`calving_front_width_check` is the same geometry read for OGGM issue #875 --
the width the run uses at the front is not the ``calving_front_width`` the inversion
recorded, and every law is linear in it.
"""
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage import map_coordinates

from oggm import cfg, entity_task, global_task
from oggm.core.ocean_params import ocean_param
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

log = logging.getLogger(__name__)

WIDTH_METHODS = ('inversion', 'terminus', 'mean5')


def _purge_lazy(obj):
    """Drop cached lazy properties, which are stale once the bed has moved."""
    for name in [k for k in vars(obj) if k.startswith('_lazy_')]:
        delattr(obj, name)


def extension_slice(gdir, fl, atol=1e-6):
    """The slice of ``fl`` holding the synthetic calving extension, or None.

    Returns None unless the last ``calving_line_extension`` points are exactly what
    ``init_present_time_glacier`` builds: ice-free, rectangular, one constant width,
    and a bed falling linearly at ``calving_front_slope`` from the terminus bed. That
    is the guard against overwriting a real part of the glacier, or an extension this
    module has already replaced.
    """

    n = int(gdir.settings['calving_line_extension'])
    if n < 2 or fl.nx <= n:
        return None
    sl = slice(fl.nx - n, fl.nx)

    is_rect = getattr(fl, 'is_rectangular', None)
    if is_rect is None or not (np.all(fl.thick[sl] == 0)
                               and np.all(is_rect[sl])):
        return None
    w = fl._w0_m[sl]
    if not np.allclose(w, w[0], rtol=1e-9):
        return None

    deepening = n * fl.dx_meter * gdir.settings['calving_front_slope']
    expected = np.linspace(fl.bed_h[fl.nx - n - 1],
                           fl.bed_h[fl.nx - n - 1] - deepening, n)
    if not np.allclose(fl.bed_h[sl], expected, atol=atol):
        return None
    return sl


def sample_gridded_on_line(gdir, line, varname, interp='linear', sl=None):
    """Sample a ``gridded_data`` variable at the flowline's grid coordinates.

    Points outside the local grid come back as NaN rather than being extrapolated:
    the calving extension is ``calving_line_extension * dx`` pixels beyond the
    terminus and routinely runs off a glacier directory built with a small border.
    """

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        if varname not in ds:
            raise InvalidWorkflowError(
                f'{varname!r} is not in gridded_data. Run '
                'tasks.bedmachine_bed_to_gdir first.')
        data = np.asarray(ds[varname].data, dtype=float)

    x, y = (np.asarray(c, dtype=float) for c in line.coords.xy)
    if sl is not None:
        x, y = x[sl], y[sl]

    ny, nx = data.shape
    inside = (x >= 0) & (x <= nx - 1) & (y >= 0) & (y <= ny - 1)
    out = np.full(x.shape, np.nan)
    if np.any(inside):
        order = 1 if interp == 'linear' else 0
        out[inside] = map_coordinates(data, [y[inside], x[inside]],
                                      order=order, mode='nearest')
    return out


def _inversion_calving_front_width(gdir):
    """``calving_front_width`` from the settings, falling back to diagnostics."""
    # ModelSettings.get takes no default and raises on a missing key.
    if 'calving_front_width' in gdir.settings:
        w = gdir.settings['calving_front_width']
    else:
        w = gdir.get_diagnostics().get('calving_front_width', None)
    return None if w is None else float(w)


@entity_task(log, writes=['model_flowlines'])
def bedmachine_calving_extension(gdir, bed_var='bedmachine_bed',
                                 width_method=None, match_terminus=None,
                                 filesuffix='', synthetic_filesuffix='_synthetic'):
    """Replace the synthetic calving extension with the measured bed.

    Run after ``init_present_time_glacier``. The untouched flowlines are written to
    ``model_flowlines`` under ``synthetic_filesuffix`` before anything is changed, so
    a run over the synthetic bed stays reproducible; if that file already exists it is
    used as the input, which makes the task re-runnable with different options.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    bed_var : str
        the ``gridded_data`` variable to sample. Default ``'bedmachine_bed'``.
    width_method : str
        ``'inversion'`` (the ``calving_front_width`` the inversion recorded and the
        calibration saw -- this is also the OGGM #875 mitigation), ``'terminus'``
        (the width of the last cell holding ice) or ``'mean5'`` (leave OGGM's
        five-cell mean alone). Default: ``cfg.PARAMS['bed_extension_width_method']``.
    match_terminus : bool
        shift the measured bed so that it joins the inverted bed continuously at the
        terminus. The offset is reported either way. Default:
        ``cfg.PARAMS['bed_extension_match_terminus']``.
    filesuffix : str
        the ``model_flowlines`` to edit
    synthetic_filesuffix : str
        where to keep the untouched copy

    Returns
    -------
    a dict, one entry per edited flowline, of what changed
    """

    if not gdir.is_tidewater:
        raise InvalidWorkflowError(f'({gdir.rgi_id}) not tidewater: there is no '
                                   'calving extension to replace.')

    width_method = width_method or ocean_param('bed_extension_width_method')
    if width_method not in WIDTH_METHODS:
        raise InvalidParamsError(f'width_method must be one of {WIDTH_METHODS}')
    if match_terminus is None:
        match_terminus = ocean_param('bed_extension_match_terminus')

    keep = gdir.has_file('model_flowlines', filesuffix=synthetic_filesuffix)
    fls = gdir.read_pickle('model_flowlines',
                           filesuffix=synthetic_filesuffix if keep else filesuffix)
    if not keep:
        gdir.write_pickle(fls, 'model_flowlines',
                          filesuffix=synthetic_filesuffix)

    w_inv = _inversion_calving_front_width(gdir)
    out = {}
    for i, fl in enumerate(fls):
        sl = extension_slice(gdir, fl)
        if sl is None:
            continue

        i0 = sl.start
        bed_new = sample_gridded_on_line(gdir, fl.line, bed_var, sl=sl)
        if not np.all(np.isfinite(bed_new)):
            n_out = int(np.sum(~np.isfinite(bed_new)))
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) {n_out} of {len(bed_new)} calving-extension '
                f'points have no {bed_var}. The extension reaches '
                f'{gdir.settings["calving_line_extension"] * fl.dx:.0f} pixels past '
                'the terminus, so the glacier directory needs a border at least '
                'that wide.')

        # The inversion's bed at the terminus against the measured one, at the same
        # point. This is the join, and it is a measurement of the inversion.
        bed_at_terminus = sample_gridded_on_line(
            gdir, fl.line, bed_var, sl=slice(i0 - 1, i0))[0]
        offset = float(fl.bed_h[i0 - 1] - bed_at_terminus)
        if match_terminus:
            bed_new = bed_new + offset

        bed_syn = fl.bed_h[sl].copy()
        w_syn = float(fl._w0_m[i0])
        w_term = float(fl.widths_m[i0 - 1])
        if width_method == 'inversion' and w_inv is None:
            log.warning(f'({gdir.rgi_id}) no calving_front_width from the '
                        'inversion; falling back to the terminus width.')
        w_new = {'inversion': w_inv if w_inv is not None else w_term,
                 'terminus': w_term,
                 'mean5': w_syn}[width_method]

        fl.bed_h[sl] = bed_new
        fl._w0_m[sl] = w_new
        _purge_lazy(fl)

        wl = fl.water_level if fl.water_level is not None else 0.
        out[f'fl_{i}'] = {
            'bed_terminus_inverted': float(fl.bed_h[i0 - 1]),
            'bed_terminus_measured': float(bed_at_terminus),
            'bed_terminus_offset': offset,
            'bed_extension_synthetic_mean': float(np.mean(bed_syn)),
            'bed_extension_measured_mean': float(np.mean(bed_new)),
            'bed_extension_synthetic_end': float(bed_syn[-1]),
            'bed_extension_measured_end': float(bed_new[-1]),
            'depth_extension_synthetic_mean': float(np.mean(np.clip(wl - bed_syn, 0, None))),
            'depth_extension_measured_mean': float(np.mean(np.clip(wl - bed_new, 0, None))),
            'n_above_water_measured': int(np.sum(bed_new > wl)),
            'measured_is_monotonic_deepening': bool(np.all(np.diff(bed_new) <= 0)),
            'width_method': width_method,
            'width_synthetic': w_syn,
            'width_used': float(w_new),
            'width_inversion': w_inv,
            'width_terminus': w_term,
        }

    if not out:
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) no synthetic calving extension found in '
            'model_flowlines. Run init_present_time_glacier first (and note that '
            'this task refuses to touch an extension it does not recognise).')

    gdir.write_pickle(fls, 'model_flowlines', filesuffix=filesuffix)
    for k, v in out.items():
        gdir.add_to_diagnostics(f'bed_extension_{k}', v)
    return out


@entity_task(log)
def calving_front_width_check(gdir, rtol=None, raise_on_fail=False,
                              filesuffix=''):
    """OGGM #875: does the run's frontal width match the inversion's?

    ``calving_front_width`` is the inversion flowline's terminus width, and it is what
    the calibrated ``k`` was fitted with. After ``init_present_time_glacier`` the run
    uses the width at the last cell above water level, which is that same width while
    the front sits in the inverted domain and the mean of the last five inversion
    widths as soon as it moves into the extension. Since every law is proportional to
    the width, the second of those is an unrecorded rescaling of the calibrated flux.

    Reports both, so the defect is visible as what it is: exact at the terminus,
    latent until the front advances.
    """

    rtol = ocean_param('calving_front_width_rtol') if rtol is None else rtol
    fls = gdir.read_pickle('model_flowlines', filesuffix=filesuffix)
    w_inv = _inversion_calving_front_width(gdir)

    d = {'rgi_id': gdir.rgi_id,
         'width_inversion': w_inv,
         'width_terminus': np.nan,
         'width_extension': np.nan,
         'rel_diff_terminus': np.nan,
         'rel_diff_extension': np.nan,
         'passes': None}

    # The tail is the ice-free, rectangular extension, whether or not it still
    # carries the synthetic bed: the width is what this check is about.
    fl = fls[-1]
    has_ice = np.nonzero(fl.thick > 0)[0]
    if has_ice.size:
        d['width_terminus'] = float(fl.widths_m[has_ice[-1]])
        tail = has_ice[-1] + 1
        is_rect = getattr(fl, 'is_rectangular', None)
        if tail < fl.nx and is_rect is not None and is_rect[tail]:
            d['width_extension'] = float(fl._w0_m[tail])

    if w_inv:
        for key in ('terminus', 'extension'):
            w = d[f'width_{key}']
            if np.isfinite(w):
                d[f'rel_diff_{key}'] = float(abs(w - w_inv) / w_inv)
        diffs = [v for v in (d['rel_diff_terminus'], d['rel_diff_extension'])
                 if np.isfinite(v)]
        d['passes'] = bool(diffs and max(diffs) < rtol)

    if d['passes'] is False:
        msg = (f'({gdir.rgi_id}) frontal width disagrees with the inversion: '
               f'{d["width_terminus"]:.0f} m at the terminus and '
               f'{d["width_extension"]:.0f} m over the extension against '
               f'{w_inv:.0f} m recorded by the inversion '
               f'({100 * d["rel_diff_extension"]:.1f}% over the extension). '
               'Every calving law is proportional to this width.')
        if raise_on_fail:
            raise InvalidWorkflowError(msg)
        log.warning(msg)

    return d


@entity_task(log)
def bed_extension_statistics(gdir, filesuffix='',
                             synthetic_filesuffix='_synthetic'):
    """Synthetic against measured calving extension, one row per glacier."""

    d = {'rgi_id': gdir.rgi_id,
         'rgi_area_km2': gdir.rgi_area_km2,
         'is_tidewater': gdir.is_tidewater}
    try:
        meas = gdir.read_pickle('model_flowlines', filesuffix=filesuffix)
        syn = gdir.read_pickle('model_flowlines', filesuffix=synthetic_filesuffix)
    except FileNotFoundError:
        return d

    for fl_s, fl_m in zip(syn, meas):
        sl = extension_slice(gdir, fl_s)
        if sl is None:
            continue
        wl = fl_s.water_level if fl_s.water_level is not None else 0.
        bed_s, bed_m = fl_s.bed_h[sl], fl_m.bed_h[sl]
        d.update({
            'n_extension': int(sl.stop - sl.start),
            'dx_meter': float(fl_s.dx_meter),
            'extension_length_m': float((sl.stop - sl.start) * fl_s.dx_meter),
            'bed_terminus': float(fl_s.bed_h[sl.start - 1]),
            'depth_terminus': float(max(wl - fl_s.bed_h[sl.start - 1], 0.)),
            'bed_end_synthetic': float(bed_s[-1]),
            'bed_end_measured': float(bed_m[-1]),
            'depth_mean_synthetic': float(np.mean(np.clip(wl - bed_s, 0, None))),
            'depth_mean_measured': float(np.mean(np.clip(wl - bed_m, 0, None))),
            'depth_ratio': float(np.mean(np.clip(wl - bed_m, 0, None)) /
                                 max(np.mean(np.clip(wl - bed_s, 0, None)), 1e-9)),
            'bed_rmse': float(np.sqrt(np.mean((bed_m - bed_s) ** 2))),
            'slope_synthetic': float(-np.polyfit(
                np.arange(len(bed_s)) * fl_s.dx_meter, bed_s, 1)[0]),
            'slope_measured': float(-np.polyfit(
                np.arange(len(bed_m)) * fl_m.dx_meter, bed_m, 1)[0]),
            'n_above_water_measured': int(np.sum(bed_m > wl)),
            'width_synthetic': float(fl_s._w0_m[sl.start]),
            'width_measured': float(fl_m._w0_m[sl.start]),
        })
        break
    return d


@global_task(log)
def compile_bed_extension_statistics(gdirs, filesuffix='', path=True, **kwargs):
    """Gather :py:func:`bed_extension_statistics` over a list of glaciers."""
    from oggm.workflow import execute_entity_task

    out = pd.DataFrame(execute_entity_task(bed_extension_statistics, gdirs,
                                           **kwargs)).set_index('rgi_id')
    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                f'bed_extension_statistics{filesuffix}.csv')
        out.to_csv(path)
    return out


@global_task(log)
def compile_calving_front_width_check(gdirs, filesuffix='', path=True, **kwargs):
    """Gather :py:func:`calving_front_width_check` over a list of glaciers."""
    from oggm.workflow import execute_entity_task

    out = pd.DataFrame(execute_entity_task(calving_front_width_check, gdirs,
                                           **kwargs)).set_index('rgi_id')
    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                f'calving_front_width_check{filesuffix}.csv')
        out.to_csv(path)
    return out
