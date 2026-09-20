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

Elevation-band flowlines carry no geometry: ``fixed_dx_elevation_band_flowline``
builds its ``Centerline`` with ``line=None`` and ``Flowline.__init__`` then invents a
straight line along the first grid row. Sampling a gridded field along *that* is not
sampling the glacier, so :py:func:`sample_gridded_on_line` is refused on it and
:py:func:`sample_gridded_by_band` -- the same field binned by surface elevation, which
is what a band flowline's cells are -- is used instead.

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

# A bed step at the junction this large is worth saying out loud (m).
BED_STEP_WARN = 50.


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


def has_line_geometry(fl):
    """False when ``fl.line`` is the straight line ``Flowline`` invents for a band.

    ``Flowline.__init__`` builds ``x = arange(nx) * dx, y = 0`` whenever it is given
    no line, which is every elevation-band flowline. It is a valid LineString and it
    indexes the first row of the grid, so a sample along it returns numbers.
    """

    if fl.line is None:
        return False
    x, y = (np.asarray(c, dtype=float) for c in fl.line.coords.xy)
    return not (np.all(y == 0) and
                np.allclose(x, np.arange(len(x)) * fl.dx))


def _gridded(gdir, *varnames):
    """One or more ``gridded_data`` variables as float arrays."""
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        out = []
        for varname in varnames:
            if varname not in ds:
                raise InvalidWorkflowError(
                    f'({gdir.rgi_id}) {varname!r} is not in gridded_data. Run '
                    'tasks.bedmachine_bed_to_gdir first.')
            out.append(np.asarray(ds[varname].data, dtype=float))
    return out[0] if len(out) == 1 else out


def sample_gridded_by_band(gdir, z_eval, varname, bsize=None,
                           topo_var='topo_smoothed'):
    """A ``gridded_data`` variable binned by surface elevation, read at ``z_eval``.

    The band-flowline analogue of :py:func:`sample_gridded_on_line`: a band's cells
    are elevation intervals over the glacier mask, so ``elevation_band_flowline``'s
    own binning is what puts a gridded field onto them. Bands holding no glacier
    pixel are interpolated over.
    """

    data, topo, mask = _gridded(gdir, varname, topo_var, 'glacier_mask')
    mask = mask == 1
    data, topo = data[mask], topo[mask]
    ok = np.isfinite(data) & np.isfinite(topo)
    data, topo = data[ok], topo[ok]
    if data.size < 3:
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) only {data.size} glacier cells carry {varname!r}.')

    bsize = bsize or cfg.PARAMS['elevation_band_flowline_binsize']
    bins = np.arange(np.floor(topo.min() / bsize) * bsize,
                     np.ceil(topo.max() / bsize) * bsize + 0.01, bsize)
    i = np.clip(np.digitize(topo, bins) - 1, 0, max(len(bins) - 2, 0))
    n = np.bincount(i, minlength=len(bins) - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.bincount(i, weights=data, minlength=len(bins) - 1) / n
    zc = 0.5 * (bins[:-1] + bins[1:])
    return np.interp(np.asarray(z_eval, dtype=float), zc[n > 0], mean[n > 0])


def offshore_bed_profile(gdir, n_points, bed_var='bedmachine_bed', min_pixels=3):
    """Median sub-sea-level bed at each distance ring outside the glacier mask.

    The measured counterpart of the linear deepening the calving extension invents.
    Like :py:func:`oggm.core.ocean_inversion.terminus_water_depth_from_bed` it reads
    the mask rather than a terminus coordinate, so it needs no line geometry; like
    it, it is a median over the whole ice-ocean contact at that distance and not
    over the fjord ahead of one front.

    Rings holding fewer than ``min_pixels`` wet cells come back as NaN.
    """
    from scipy import ndimage

    bed, mask = _gridded(gdir, bed_var, 'glacier_mask')
    dist = ndimage.distance_transform_edt(mask != 1) * gdir.grid.dx
    wet = np.isfinite(bed) & (bed < 0)
    dx = gdir.grid.dx
    out = np.full(int(n_points), np.nan)
    for j in range(int(n_points)):
        ring = wet & (dist > j * dx) & (dist <= (j + 1) * dx)
        if ring.sum() >= min_pixels:
            out[j] = float(np.median(bed[ring]))
    return out


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
        if not has_line_geometry(fl):
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) this flowline has no geometry, so there is no '
                'line to sample the bed along. Elevation-band flowlines are built '
                'with line=None and Flowline invents a straight one along the '
                'first grid row. Use bedmachine_terminus_bed instead, which reads '
                'the mask.')

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
        elif abs(offset) > BED_STEP_WARN:
            log.warning(
                f'({gdir.rgi_id}) the inverted bed at the terminus is {offset:.0f} m '
                f'from the measured one, so the measured extension starts with a step '
                'of that size. A large step is a surface-gradient spike and can fail '
                'the run on CFL; match_terminus=True removes it.')

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


def _rectangular_tail(fl, i0):
    """How many cells of ice immediately above ``i0`` are rectangular.

    ``fixed_dx_elevation_band_flowline`` marks the last five bins of a tidewater
    glacier rectangular, so their width is ``_w0_m`` and does not move with the
    thickness. Correcting the bed inside that tail therefore changes the ice column
    and nothing else, which is why it is the default blend window.
    """
    is_rect = getattr(fl, 'is_rectangular', None)
    if is_rect is None:
        return 1
    n = 0
    while n < i0 and is_rect[i0 - n]:
        n += 1
    return max(n, 1)


@entity_task(log, writes=['model_flowlines'])
def bedmachine_terminus_bed(gdir, water_depth=None, n_blend=None,
                            bed_var='bedmachine_bed', filesuffix='',
                            synthetic_filesuffix='_synthetic'):
    """Put the prescribed calving-front depth on the run's flowline.

    :py:func:`oggm.core.ocean_inversion.find_inversion_calving_from_bathymetry`
    fixes the water depth the *inversion* uses, but the terminus thickness the
    inversion returns comes from the SIA and the surface mass balance, so
    ``bed_h = surface_h - thick`` puts the run's front far deeper than the depth
    that was prescribed -- measured at 1.8x to 5.1x over the twelve FIIC divides.
    Since every calving law reads its water depth as ``-bed_h`` at the last cell
    above water level, the run then calves against a depth nothing constrained.

    This sets the bed at the front to ``water_level - water_depth``, ramps the
    correction out over ``n_blend`` cells so no surface-gradient step is created,
    and continues past the terminus over the measured offshore bed
    (:py:func:`offshore_bed_profile`) instead of the invented linear deepening. The
    DEM surface is held fixed, so the ice thickness absorbs the change.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    water_depth : float
        metres, positive down. Default: the ``terminus_water_depth`` written by
        :py:func:`oggm.core.ocean_inversion.terminus_water_depth_from_bed`, which
        is the depth ``k`` was fitted at.
    n_blend : int
        cells to ramp the correction over, the front included. Default: the
        rectangular tail, over which the width does not follow the thickness.
    bed_var : str
        the ``gridded_data`` variable holding the measured bed
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
                                   'calving front to place.')

    if water_depth is None:
        if 'terminus_water_depth' not in gdir.settings:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) no terminus_water_depth in the settings; run '
                'terminus_water_depth_from_bed first, or pass one.')
        water_depth = float(gdir.settings['terminus_water_depth'])
    if not (water_depth > 0):
        raise InvalidParamsError(f'({gdir.rgi_id}) water_depth = {water_depth} '
                                 'is not a depth below sea level.')

    keep = gdir.has_file('model_flowlines', filesuffix=synthetic_filesuffix)
    fls = gdir.read_pickle('model_flowlines',
                           filesuffix=synthetic_filesuffix if keep else filesuffix)
    if not keep:
        gdir.write_pickle(fls, 'model_flowlines', filesuffix=synthetic_filesuffix)

    out = {}
    for i, fl in enumerate(fls):
        has_ice = np.nonzero(fl.thick > 0)[0]
        if not has_ice.size:
            continue
        i0 = int(has_ice[-1])
        wl = fl.water_level if fl.water_level is not None else 0.
        if fl.surface_h[i0] <= wl:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) the last cell holding ice is already below the '
                'water level; there is no front to place.')

        n = int(n_blend) if n_blend else _rectangular_tail(fl, i0)
        n = int(min(max(n, 1), i0 + 1))
        bed_old = fl.bed_h.copy()
        surf = fl.surface_h.copy()
        vol_before = float(np.sum(fl.section) * fl.dx_meter)

        # The ice: the full correction at the front, dying out n cells upglacier.
        target = wl - water_depth
        delta = target - bed_old[i0]
        ramp = np.linspace(1., n, n) / n
        bed_new = bed_old.copy()
        bed_new[i0 - n + 1:i0 + 1] += ramp * delta

        # Beyond it: the measured bed, ring by ring, with the prescribed depth
        # where a ring holds too little water to take a median of.
        n_ext = fl.nx - i0 - 1
        if n_ext > 0:
            off = offshore_bed_profile(gdir, n_ext, bed_var=bed_var)
            miss = ~np.isfinite(off)
            if miss.all():
                off[:] = target
            elif miss.any():
                j = np.arange(n_ext)
                off[miss] = np.interp(j[miss], j[~miss], off[~miss])
            bed_new[i0 + 1:] = np.minimum(off, target)

        # The DEM surface is an observation: the thickness takes the change. Beyond
        # the terminus the surface is the bed, as init_present_time_glacier has it.
        surf[i0 + 1:] = bed_new[i0 + 1:]
        fl.bed_h = bed_new
        fl.thick = surf - bed_new
        _purge_lazy(fl)

        if fl.thick[i0] <= 0:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) raising the bed to {target:.0f} m leaves no ice '
                f'at the front (surface {surf[i0]:.0f} m). The prescribed water '
                'depth and the DEM disagree about where sea level is.')

        rho_o = ocean_param('ocean_water_density')
        out[f'fl_{i}'] = {
            'terminus_index': i0,
            'n_blend': n,
            'water_depth_prescribed': float(water_depth),
            'water_depth_before': float(max(wl - bed_old[i0], 0.)),
            'water_depth_after': float(max(wl - bed_new[i0], 0.)),
            'bed_terminus_before': float(bed_old[i0]),
            'bed_terminus_after': float(bed_new[i0]),
            'thick_terminus_before': float(surf[i0] - bed_old[i0]),
            'thick_terminus_after': float(fl.thick[i0]),
            'thick_flotation': float(water_depth * rho_o /
                                     cfg.PARAMS['ice_density']),
            'width_terminus': float(fl.widths_m[i0]),
            'bed_extension_measured_mean': (float(np.mean(bed_new[i0 + 1:]))
                                            if n_ext > 0 else np.nan),
            'bed_extension_synthetic_mean': (float(np.mean(bed_old[i0 + 1:]))
                                             if n_ext > 0 else np.nan),
            'volume_before_m3': vol_before,
            'volume_after_m3': float(np.sum(fl.section) * fl.dx_meter),
        }

    if not out:
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no flowline holds ice.')

    gdir.write_pickle(fls, 'model_flowlines', filesuffix=filesuffix)
    for k, v in out.items():
        gdir.add_to_diagnostics(f'terminus_bed_{k}', v)
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


@entity_task(log)
def calving_vs_bed_extension(gdir, run_task=None, filesuffix='',
                             synthetic_filesuffix='_synthetic',
                             output_filesuffix='_bedext', **run_kwargs):
    """What the calving extension is worth, in calving.

    Two runs of the same glacier under the same mass balance and the same ``k``,
    differing only in the bed beyond the present terminus. The ratio is the answer to
    whether the synthetic extension is defensible: at one it never mattered, and
    away from one it set the frontal ablation of the run.

    Parameters
    ----------
    run_task : func
        the run to use. Default
        :py:func:`oggm.core.flowline.run_from_climate_data`.
    run_kwargs : dict
        passed on to it (``ys``, ``ye``, ``calving_law`` ...)
    """

    from oggm.core.flowline import run_from_climate_data
    # Not run_constant_climate: its ConstantMassBalance interpolates over a fixed
    # height table built from the real glacier, and the synthetic extension puts
    # surface elevations below it -- the run raises before it can be compared.
    run_task = run_task or run_from_climate_data

    out = {'rgi_id': gdir.rgi_id}
    for tag, fsx in (('synthetic', synthetic_filesuffix), ('measured', filesuffix)):
        suffix = f'{output_filesuffix}_{tag}'
        run_task(gdir, model_flowlines_filesuffix=fsx, output_filesuffix=suffix,
                 **run_kwargs)
        with xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                               filesuffix=suffix)) as ds:
            out[f'calving_m3_{tag}'] = float(ds.calving_m3[-1])
            out[f'volume_m3_{tag}'] = float(ds.volume_m3[-1])
            out[f'length_m_{tag}'] = float(ds.length_m[-1])

    syn = out['calving_m3_synthetic']
    out['calving_ratio'] = (out['calving_m3_measured'] / syn if syn > 0
                            else np.nan)
    out['calving_diff_m3'] = out['calving_m3_measured'] - syn
    out['volume_ratio'] = (out['volume_m3_measured'] /
                           max(out['volume_m3_synthetic'], 1e-9))
    return out


@global_task(log)
def compile_calving_vs_bed_extension(gdirs, filesuffix='', path=True, **kwargs):
    """Gather :py:func:`calving_vs_bed_extension` over a list of glaciers."""
    from oggm.workflow import execute_entity_task

    out = pd.DataFrame(execute_entity_task(calving_vs_bed_extension, gdirs,
                                           **kwargs)).set_index('rgi_id')
    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                f'calving_vs_bed_extension{filesuffix}.csv')
        out.to_csv(path)
    return out


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
