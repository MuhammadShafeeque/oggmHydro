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
