"""The inversion side of ocean-forced frontal ablation.

The dynamical run can take a time-varying calving constant; the inversion cannot.
``calving_mb`` returns a scalar and the mass-balance calibration refuses to run at
all once it is non-zero, so the inversion gets one ``k`` per glacier, computed from
the period-mean thermal forcing:

``k_inv = k_ref * (mean(TF, period) / TF_ref) ** gamma``

That forces the order of operations: calibrate the mass balance first, set the
calving rate second, and never re-calibrate. There is no loop between ``melt_f``
and ``k``, which is a limitation to state rather than one to hide;
:func:`calving_vs_geodetic_residual` is what puts a number on it.

Unlike OGGM v1.6.3, where ``inversion_calving_k`` was a global and mutating it inside
a worker leaked to the next glacier, this version resolves it per glacier through
``gdir.settings``, so no task needs vendoring.
"""
import logging

import numpy as np

from oggm import cfg
from oggm import entity_task
from oggm.core.ocean_calving import _band_names
from oggm.core.ocean_params import ocean_param
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.utils import DisableLogger

log = logging.getLogger(__name__)


def ocean_tf_mean(gdir, band=None, period=None, ocean_filesuffix=''):
    """Period-mean thermal forcing in one band, degC."""
    import xarray as xr

    if not gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no ocean_data file; run '
                                   'process_ocean_data first.')
    band = band or ocean_param('ocean_tf_band')
    period = period or ocean_param('ocean_tf_ref_period')

    with xr.open_dataset(gdir.get_filepath('ocean_data',
                                           filesuffix=ocean_filesuffix)) as ds:
        ds = ds.load()
    names = _band_names(ds)
    if band not in names:
        raise InvalidParamsError(f'band {band!r} not in {names}')
    tf = ds['thermal_forcing'].isel(band=names.index(band))
    return float(tf.sel(time=slice(*period)).mean('time'))


@entity_task(log)
def set_inversion_k_from_ocean(gdir, k_ref=None, gamma=None, band=None,
                               period=None, tf_ref=None, ocean_filesuffix=''):
    """Scale the inversion's calving constant by the period-mean thermal forcing.

    Writes ``inversion_calving_k`` to the glacier's settings, so the stock
    :func:`oggm.core.inversion.find_inversion_calving_from_any_mb` picks it up
    without being modified.

    Parameters
    ----------
    k_ref : float
        the calving constant at the reference forcing, a-1. Defaults to the
        glacier's current ``inversion_calving_k``.
    gamma : float
        the thermal-forcing exponent. Defaults to
        ``cfg.PARAMS['calving_tf_exponent']``.
    period : tuple of str
        the period whose mean forcing scales k.
    tf_ref : float
        the reference forcing. Defaults to the period mean itself, which makes
        ``k_inv == k_ref`` and is the no-op control.
    """
    if not gdir.is_tidewater:
        log.warning('(%s) not tidewater, inversion k untouched', gdir.rgi_id)
        return None

    band = band or ocean_param('ocean_tf_band')
    period = period or ocean_param('ocean_tf_ref_period')
    gamma = ocean_param('calving_tf_exponent') if gamma is None else gamma
    k_ref = gdir.settings['inversion_calving_k'] if k_ref is None else k_ref

    tf_mean = ocean_tf_mean(gdir, band=band, period=period,
                            ocean_filesuffix=ocean_filesuffix)
    if tf_ref is None:
        tf_ref = ocean_param('ocean_tf_ref')
    if tf_ref is None:
        tf_ref = tf_mean
    if not np.isfinite(tf_ref) or tf_ref <= 0:
        raise InvalidParamsError(
            f'({gdir.rgi_id}) TF_ref = {tf_ref} is not usable; set '
            "cfg.PARAMS['ocean_tf_ref'] explicitly or pick a different band.")

    k_inv = float(k_ref * (max(tf_mean, 0.) / tf_ref) ** gamma)
    gdir.write_to_settings({'inversion_calving_k': k_inv}, overwrite=True)
    gdir.add_to_diagnostics('ocean_inversion_k', k_inv)
    gdir.add_to_diagnostics('ocean_inversion_tf_mean', tf_mean)
    gdir.add_to_diagnostics('ocean_inversion_tf_ref', float(tf_ref))
    gdir.add_to_diagnostics('ocean_inversion_band', band)
    return k_inv


@entity_task(log)
def calving_vs_geodetic_residual(gdir, ref_mb=None, ref_mb_err=None):
    """How much of the geodetic target the inverted frontal ablation consumes.

    Two things at once. It is the check against the degenerate case in which
    frontal ablation eats the whole mass budget, which the forced order makes
    possible because the mass balance is calibrated with no calving at all. And the
    ratio it returns is the measure of what the missing loop costs: small against
    ``ref_mb_err`` means the open loop is defensible, large means it is not.

    Parameters
    ----------
    ref_mb : float
        the geodetic target the mass balance was calibrated on, kg m-2 yr-1.
    ref_mb_err : float
        its uncertainty, same units.
    """
    rate = gdir.inversion_calving_rate  # km3 a-1
    rho = gdir.settings['ice_density']
    cmb = rate * 1e9 * rho / gdir.rgi_area_m2  # kg m-2 yr-1

    out = {'calving_specific_mb': cmb, 'calving_rate_km3_per_a': rate}
    if ref_mb is not None:
        out['calving_over_geodetic'] = (cmb / abs(ref_mb) if ref_mb else np.inf)
        if ref_mb_err:
            out['calving_over_geodetic_err'] = cmb / abs(ref_mb_err)
        if cmb > abs(ref_mb):
            log.warning('(%s) frontal ablation %.1f kg m-2 yr-1 exceeds the '
                        'geodetic target %.1f: exclude from the calibration',
                        gdir.rgi_id, cmb, ref_mb)
            out['calving_exceeds_geodetic'] = True
    for k, v in out.items():
        gdir.add_to_diagnostics(k, v if isinstance(v, bool) else float(v))
    return out


def partition_calving_constant(k_total, melt_rate, thick, lam=None):
    """The residual calving constant that keeps the total frontal ablation fixed.

    ``k`` is calibrated against an *observed* frontal ablation, so it already
    contains the calving that submarine melt drives. Adding a melt term on top of it
    therefore counts the melt twice, and the default ``k_c = k`` does exactly that.
    Both terms share the submerged area, so the constraint
    ``k_c*h + lam*mdot == k_total*h`` is linear and solves in closed form.

    Parameters
    ----------
    k_total : float
        the calibrated calving constant, a-1.
    melt_rate : float
        the reference submarine melt rate, m s-1, as the laws return it.
    thick : float
        the terminus ice thickness the constant acts on, m.
    lam : float
        the undercut efficiency. Defaults to
        ``cfg.PARAMS['calving_undercut_efficiency']``.

    Returns
    -------
    (k_c, melt_fraction) : the residual constant in a-1, and the melt share of the
    unchanged total.
    """
    lam = ocean_param('calving_undercut_efficiency') if lam is None else lam
    if thick <= 0:
        raise InvalidParamsError(f'thick = {thick} is not a terminus thickness')
    u_total = k_total / cfg.SEC_IN_YEAR * thick
    u_melt = lam * melt_rate
    if u_total <= 0:
        return 0., 0.
    if u_melt >= u_total:
        log.warning('submarine melt alone (%.3e m s-1) matches or exceeds the '
                    'calibrated frontal ablation speed (%.3e): the residual calving '
                    'constant is zero, and the melt term is the whole flux',
                    u_melt, u_total)
        return 0., 1.
    k_c = (u_total - u_melt) / thick * cfg.SEC_IN_YEAR
    return float(k_c), float(u_melt / u_total)


def frontal_ablation_corrected_mb(ref_mb, area_m2, calving_flux_km3=None,
                                  below_wl_flux_km3=None, f_bwl=0.75,
                                  rho=None):
    """Malles et al. (2023) Eq. (18): a geodetic target corrected for calving.

    A geodetic elevation change sees only what is above water, so a marine glacier's
    target has to have the frontal ablation and the below-waterline retreat added
    back before the mass balance is calibrated on it. OGGM cannot do this inside the
    calibration, which refuses to run on a calving glacier, so it is done here and
    the corrected value is passed in as ``ref_mb``.

    Parameters
    ----------
    ref_mb : float
        the uncorrected geodetic mass balance, kg m-2 yr-1.
    area_m2 : float
        the glacier area the target refers to.
    calving_flux_km3, below_wl_flux_km3 : float
        observed frontal ablation and the part of it from below-waterline retreat,
        km3 a-1.
    f_bwl : float
        the fraction of the below-waterline term that is a real mass loss.
    """
    rho = cfg.PARAMS['ice_density'] if rho is None else rho
    corr = 0.
    if calving_flux_km3:
        corr += calving_flux_km3 * 1e9 * rho / area_m2
    if below_wl_flux_km3:
        corr += f_bwl * below_wl_flux_km3 * 1e9 * rho / area_m2
    return ref_mb - corr


# --- the bathymetry route ---------------------------------------------------
#
# The stock inversion solves for the water depth that makes the calving law and the
# SIA agree, taking the free-board from the DEM. Where the DEM puts a marine terminus
# at sea level the free-board is ~0, the solved depth collapses onto the full ice
# thickness, and the flux scales as thickness squared. Fixing the depth from measured
# bathymetry replaces one unknown with an observation and leaves `k` as the only
# parameter, which is what the frontal-ablation observations constrain.


@entity_task(log)
def terminus_water_depth_from_bed(gdir, bed_var='bedmachine_bed',
                                  dilate=2, min_pixels=5):
    """Measured water depth at the calving front, metres, positive down.

    The median of the sub-sea-level bed over the ocean cells that touch the glacier
    mask, i.e. the ice-ocean contact on the glacier's own grid. Taken from the mask
    rather than from a terminus coordinate because elevation-band flowlines carry no
    geometry, and taken as a median over the contact rather than one cell because
    neither the outline nor the bed grid is accurate to a single 150 m cell.

    Writes ``terminus_water_depth`` and ``terminus_water_depth_n`` to the settings.
    """
    import xarray as xr
    from scipy import ndimage

    if not gdir.is_tidewater:
        return None

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        if bed_var not in ds:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) no {bed_var} in gridded_data; run '
                'bedmachine_bed_to_gdir first.')
        bed = ds[bed_var].values
        mask = ds['glacier_mask'].values.astype(bool)

    ring = ndimage.binary_dilation(mask, iterations=int(dilate)) & ~mask
    wet = bed[ring & np.isfinite(bed) & (bed < 0)]
    if wet.size < min_pixels:
        # No ocean touching the outline: fall back to the deepest water in the
        # domain, which is the fjord the front drains into.
        wet = bed[np.isfinite(bed) & (bed < 0)]
    if wet.size < min_pixels:
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) only {wet.size} sub-sea-level bed cells in the '
            f'domain; cannot set a water depth.')

    depth = float(-np.median(wet))
    gdir.settings['terminus_water_depth'] = depth
    gdir.settings['terminus_water_depth_n'] = int(wet.size)
    return depth


def _setting(gdir, key, default=None):
    """One setting, or `default`. `gdir.settings.get` raises on a missing key."""
    return gdir.settings[key] if key in gdir.settings else default


def flotation_thickness(water_depth, rho_ocean=None, rho_ice=None):
    """Ice thickness a front of this water depth carries at flotation, metres."""
    rho_ocean = rho_ocean or ocean_param('ocean_water_density')
    rho_ice = rho_ice or cfg.PARAMS['ice_density']
    return float(water_depth) * rho_ocean / rho_ice


def calving_law_flux(gdir, water_depth=None, k=None, thick=None,
                     input_filesuffix=''):
    """``k * thick * water_depth * width`` in km3 yr-1, at a prescribed depth.

    Deliberately not :func:`oggm.core.inversion.calving_flux_from_depth`: that one
    derives the thickness from the DEM free-board, which is the quantity this route
    exists to stop using.
    """
    if water_depth is None:
        water_depth = gdir.settings['terminus_water_depth']
    if k is None:
        k = gdir.settings['inversion_calving_k']
    if thick is None:
        thick = flotation_thickness(water_depth)
    fl = gdir.read_pickle('inversion_flowlines', filesuffix=input_filesuffix)[-1]
    width = fl.widths[-1] * gdir.grid.dx
    return dict(flux=max(k * thick * water_depth * width / 1e9, 0.),
                width=width, thick=thick, water_depth=water_depth,
                inversion_calving_k=k, free_board=thick - water_depth)


@entity_task(log)
def find_inversion_calving_from_bathymetry(gdir, water_depth=None, k=None,
                                           mb_model=None, mb_years=None,
                                           glen_a=None, fs=None,
                                           settings_filesuffix='',
                                           input_filesuffix='',
                                           output_filesuffix=''):
    """Calving inversion with the water depth prescribed rather than solved.

    Drop-in for :func:`oggm.core.inversion.find_inversion_calving_from_any_mb` when
    bathymetry is available. The flux follows from the measured depth and ``k``; the
    thickness inversion is then re-run against it, exactly as the stock task does
    once it has found its own flux.
    """
    from oggm.core import massbalance
    from oggm.core.inversion import prepare_for_inversion, mass_conservation_inversion

    if not gdir.is_tidewater:
        return None

    if water_depth is None:
        water_depth = _setting(gdir, 'terminus_water_depth')
    if water_depth is None:
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) no terminus_water_depth; run '
            'terminus_water_depth_from_bed first, or pass one.')

    # Volume without calving, for the same statistic the stock task records.
    gdir.inversion_calving_rate = 0
    with DisableLogger():
        massbalance.apparent_mb_from_any_mb(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=input_filesuffix,
            output_filesuffix=output_filesuffix,
            mb_model=mb_model, mb_years=mb_years)
        prepare_for_inversion(gdir, settings_filesuffix=settings_filesuffix,
                              input_filesuffix=input_filesuffix,
                              output_filesuffix=output_filesuffix)
        v_ref = mass_conservation_inversion(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=output_filesuffix,
            output_filesuffix=output_filesuffix, glen_a=glen_a, fs=fs)
    gdir.settings['volume_before_calving'] = v_ref

    out = calving_law_flux(gdir, water_depth=water_depth, k=k,
                           input_filesuffix=output_filesuffix)
    gdir.inversion_calving_rate = out['flux']

    with DisableLogger():
        massbalance.apparent_mb_from_any_mb(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=input_filesuffix,
            output_filesuffix=output_filesuffix,
            mb_model=mb_model, mb_years=mb_years)
        prepare_for_inversion(gdir, settings_filesuffix=settings_filesuffix,
                              input_filesuffix=output_filesuffix,
                              output_filesuffix=output_filesuffix)
        mass_conservation_inversion(
            gdir, settings_filesuffix=settings_filesuffix,
            input_filesuffix=output_filesuffix,
            output_filesuffix=output_filesuffix,
            water_level=0., glen_a=glen_a, fs=fs)

    fl = gdir.read_pickle('inversion_flowlines', filesuffix=output_filesuffix)[-1]
    f_calving = (fl.flux[-1] * (gdir.grid.dx ** 2) * 1e-9
                 / gdir.settings['ice_density'])

    odf = {'calving_flux': f_calving,
           'calving_law_flux': out['flux'],
           'calving_rate_myr': f_calving * 1e9 / (out['thick'] * out['width']),
           'calving_water_level': 0.,
           'calving_inversion_k': out['inversion_calving_k'],
           'calving_front_water_depth': out['water_depth'],
           'calving_front_free_board': out['free_board'],
           'calving_front_thick': out['thick'],
           'calving_front_width': out['width'],
           'calving_depth_source': 'bathymetry'}
    for key, val in odf.items():
        gdir.settings[key] = val
    return odf


def fit_calving_k(gdirs, observed_flux, water_depth=None, input_filesuffix=''):
    """Pooled ``k`` matching the summed observed frontal ablation.

    Pooled rather than per-divide because `k` is fitted across the divide set, the
    same argument that makes the Glen A fit a global task. Returns the pooled value
    and the per-divide values it is pooled from, which are the spread to report.

    Parameters
    ----------
    observed_flux : dict
        rgi_id -> observed frontal ablation in km3 yr-1 of ice. Divides absent
        from it take no part in the fit.
    """
    num, den, per = 0., 0., {}
    for gdir in gdirs:
        obs = observed_flux.get(gdir.rgi_id)
        if obs is None or not gdir.is_tidewater:
            continue
        d = water_depth or _setting(gdir, 'terminus_water_depth')
        shape = calving_law_flux(gdir, water_depth=d, k=1.,
                                 input_filesuffix=input_filesuffix)['flux']
        if shape <= 0:
            continue
        num += obs
        den += shape
        per[gdir.rgi_id] = obs / shape
    if den <= 0:
        raise InvalidWorkflowError('no divide contributed to the k fit')
    return num / den, per
