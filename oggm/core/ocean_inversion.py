"""The inversion side of ocean-forced frontal ablation.

The dynamical run can take a time-varying calving constant; the inversion cannot.
``calving_mb`` returns a scalar and the mass-balance calibration refuses to run at
all once it is non-zero, so the inversion gets one ``k`` per glacier, computed from
the period-mean thermal forcing:

``k_inv = k_ref * (mean(TF, period) / TF_ref) ** gamma``

That forced the order of operations: calibrate the mass balance first, set the
calving rate second, and never re-calibrate. ``mb_calibration_from_geodetic_mb`` no
longer refuses a calving glacier -- it shifts the surface target by ``calving_mb``,
since the geodetic observation is the total mass change and the surface balance is
not -- so the two can now be calibrated together.

There is still no *loop*, and on the bathymetry route there need not be one: the
calving law's shape is the DEM free board, the measured bed and the inversion-flowline
width, none of which the climate sets, so ``k = observed / shape`` is exact on the
first pass. :func:`fit_calving_k_model` is the per-divide replacement for the pooled
constant, and the one term of the parameterization the atmosphere does set --
subglacial discharge, via :func:`subglacial_discharge_from_mb` -- is what lets ``k``
differ between climate baselines at all.

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


# BedMachine's own mask: 0 ocean, 1 ice-free land, 2 grounded ice, 3 floating ice.
BEDMACHINE_OCEAN, BEDMACHINE_FLOATING = 0, 3


def ocean_cells(ds, mask, bed_var='bedmachine_bed'):
    """Cells outside the glacier that are ocean, with the reason recorded.

    ``bedmachine_mask`` is used where the file carries it. "Bed below sea level
    outside the divide" is not the same thing at 81 N, where most of the bed around
    one divide is below sea level *under the neighbouring ice*: over the twelve FIIC
    divides that test admits two to six times as many cells as BedMachine's own
    ocean class, and it changes the median depth by up to a factor of three.
    """
    bed = np.asarray(ds[bed_var].values, dtype=float)
    wet = np.isfinite(bed) & (bed < 0) & ~mask
    if 'bedmachine_mask' in ds:
        cls = np.round(np.asarray(ds['bedmachine_mask'].values, dtype=float))
        return wet & np.isin(cls, (BEDMACHINE_OCEAN, BEDMACHINE_FLOATING)), 'mask'
    return wet, 'bed_below_sea_level'


@entity_task(log)
def terminus_water_depth_from_bed(gdir, bed_var='bedmachine_bed',
                                  dilate=2, min_pixels=5, fallback_trim=None):
    """Measured water depth at the calving front, metres, positive down.

    The median of the sub-sea-level bed over the ocean cells that touch the glacier
    mask, i.e. the ice-ocean contact on the glacier's own grid. Taken from the mask
    rather than from a terminus coordinate because elevation-band flowlines carry no
    geometry, and taken as a median over the contact rather than one cell because
    neither the outline nor the bed grid is accurate to a single 150 m cell.

    Writes ``terminus_water_depth`` and ``terminus_water_depth_n`` to the settings.

    ``fallback_trim`` cells are cut from each edge of the map before the fallback
    searches it, so a directory built with a wider border than OGGM's tidewater 10 falls
    back to the same water. Default: the ``terminus_depth_fallback_trim`` setting, else 0.
    """
    import xarray as xr
    from scipy import ndimage

    if not gdir.is_tidewater:
        return None
    if fallback_trim is None:
        fallback_trim = int(_setting(gdir, 'terminus_depth_fallback_trim', 0))

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        if bed_var not in ds:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) no {bed_var} in gridded_data; run '
                'bedmachine_bed_to_gdir first.')
        bed = ds[bed_var].values
        mask = ds['glacier_mask'].values.astype(bool)
        ocean, source = ocean_cells(ds, mask, bed_var=bed_var)

    ring = ndimage.binary_dilation(mask, iterations=int(dilate)) & ~mask
    wet = bed[ring & ocean]
    if wet.size < min_pixels:
        # No ocean touching the outline: fall back to the whole ocean in the
        # domain, which is the fjord the front drains into.
        sea = ocean.copy()
        if fallback_trim:
            t = fallback_trim
            sea[:t, :] = sea[-t:, :] = sea[:, :t] = sea[:, -t:] = False
        wet = bed[sea]
    if wet.size < min_pixels:
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) only {wet.size} sub-sea-level bed cells in the '
            f'domain; cannot set a water depth.')

    depth = float(-np.median(wet))
    gdir.settings['terminus_water_depth'] = depth
    gdir.settings['terminus_water_depth_n'] = int(wet.size)
    gdir.settings['terminus_water_depth_ocean_source'] = source
    return depth


def _setting(gdir, key, default=None):
    """One setting, or `default`. `gdir.settings.get` raises on a missing key."""
    return gdir.settings[key] if key in gdir.settings else default


def flotation_thickness(water_depth, rho_ocean=None, rho_ice=None):
    """Ice thickness a front of this water depth carries at flotation, metres."""
    rho_ocean = rho_ocean or ocean_param('ocean_water_density')
    rho_ice = rho_ice or cfg.PARAMS['ice_density']
    return float(water_depth) * rho_ocean / rho_ice


def front_thickness(gdir, water_depth, input_filesuffix=''):
    """Terminus thickness at the prescribed depth: free board plus depth.

    Flotation is the *minimum* a front of this depth can carry, not what it does
    carry. Over the twelve FIIC divides the DEM free board puts eleven above it, by
    a median factor of 1.39, so they are grounded and flotation understates them.
    It is kept as the floor, which is what the twelfth needs.

    This is also the thickness the run has: ``bedmachine_terminus_bed`` holds the
    DEM surface and sets ``bed_h = water_level - water_depth``, so the forward
    model's front carries exactly this. Calibrating ``k`` against anything else
    makes the run miss the observation at t = 0 by the ratio between the two.
    """
    fl = gdir.read_pickle('inversion_flowlines', filesuffix=input_filesuffix)[-1]
    free_board = max(float(fl.surface_h[-1]), 0.)
    return max(free_board + float(water_depth), flotation_thickness(water_depth))


def calving_law_flux(gdir, water_depth=None, k=None, thick=None,
                     input_filesuffix=''):
    """``k * thick * water_depth * width`` in km3 yr-1, at a prescribed depth.

    Deliberately not :func:`oggm.core.inversion.calving_flux_from_depth`: that one
    *solves* for the depth and takes the thickness from it, which is the step this
    route exists to replace. Here the depth is the measurement and the thickness
    follows from it and the free board (:func:`front_thickness`).
    """
    if water_depth is None:
        water_depth = gdir.settings['terminus_water_depth']
    if k is None:
        k = gdir.settings['inversion_calving_k']
    if thick is None:
        thick = front_thickness(gdir, water_depth,
                                input_filesuffix=input_filesuffix)
    fl = gdir.read_pickle('inversion_flowlines', filesuffix=input_filesuffix)[-1]
    width = fl.widths[-1] * gdir.grid.dx
    return dict(flux=max(k * thick * water_depth * width / 1e9, 0.),
                width=width, thick=thick, water_depth=water_depth,
                inversion_calving_k=k, free_board=thick - water_depth,
                thick_flotation=flotation_thickness(water_depth))


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
           'calving_front_thick_flotation': out['thick_flotation'],
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


def subglacial_discharge_from_mb(gdir, period=None, mb_model=None,
                                 input_filesuffix=''):
    """Monthly runoff leaving the glacier, m3 s-1, from the calibrated mass balance.

    The melt parameterization's ``q_sg`` has to come from the same climate as the run
    it forces, and no dynamical run exists when the inversion needs it. This reads
    melt and liquid precipitation straight off the calibrated mass-balance model, which
    is the same decomposition ``run_with_hydro`` reports -- ``melt_f * tmelt`` for melt
    and ``prcp - prcpsol`` for rain -- one stage earlier.

    It is potential runoff on a fixed geometry: there is no snow-bucket accounting and
    no refreezing, so it is an upper bound that is monotone in the melt energy. As a
    covariate that is what is wanted; as a water budget it is not.

    Returns ``(floatyears, discharge)`` monthly, or ``(None, None)`` when the climate
    file does not cover the period.
    """
    import xarray as xr
    from oggm.core.massbalance import MultipleFlowlineMassBalance, MonthlyTIModel
    from oggm.utils import date_to_floatyear

    fls = gdir.read_pickle('inversion_flowlines', filesuffix=input_filesuffix)
    if mb_model is None:
        mb_model = MultipleFlowlineMassBalance(gdir, fls=fls,
                                               mb_model_class=MonthlyTIModel)
    with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
        y0c, y1c = int(ds.time.dt.year[0]), int(ds.time.dt.year[-1])
    y0, y1 = (int(period[0]), int(period[1])) if period else (y0c, y1c)
    y0, y1 = max(y0, y0c + 1), min(y1, y1c)
    if y1 < y0:
        return None, None

    years, q = [], []
    for yr in range(y0, y1 + 1):
        for m in range(1, 13):
            t = date_to_floatyear(yr, m)
            total_kg = 0.
            for fl, mbm in zip(fls, mb_model.flowline_mb_models):
                _, _, tmelt, prcp, prcpsol = mbm.get_monthly_mb(
                    fl.surface_h, year=t, add_climate=True)
                runoff = mbm.melt_f * tmelt + (prcp - prcpsol)   # kg m-2 month-1
                area = fl.widths_m * fl.dx_meter                 # m2 per node
                total_kg += float(np.sum(np.clip(runoff, 0., None) * area))
            sec = mbm.sec_in_month(year=t)
            years.append(yr + (m - 0.5) / 12)
            q.append(total_kg / 1000. / sec)                     # m3 s-1 of water
    return np.asarray(years), np.asarray(q)


@entity_task(log)
def write_subglacial_discharge(gdir, period=None, ocean_filesuffix='',
                               input_filesuffix=''):
    """Add ``subglacial_discharge`` to ``ocean_data.nc`` so the melt law can use it.

    Without it :class:`~oggm.core.ocean_calving.MeltPlusCalving` runs in Rignot's
    no-discharge limit, where the law collapses to ``B * TF**beta`` and the ``h``
    and ``q`` terms drop out entirely.
    """
    import netCDF4
    import numpy as np

    if not gdir.is_tidewater:
        return None
    if not gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no ocean_data file')

    yrs, q = subglacial_discharge_from_mb(gdir, period=period,
                                          input_filesuffix=input_filesuffix)
    if yrs is None:
        raise InvalidWorkflowError(f'({gdir.rgi_id}) climate does not cover {period}')

    path = gdir.get_filepath('ocean_data', filesuffix=ocean_filesuffix)
    with netCDF4.Dataset(path, 'a') as nc:
        t = nc.variables['time']
        target = (nc.variables['time'][:].astype(float))
        # The ocean file's own axis decides the length; discharge is interpolated onto
        # it rather than the reverse, because the thermal forcing is the measurement.
        import cftime
        dates = cftime.num2date(target, t.units,
                                getattr(t, 'calendar', 'standard'))
        tgt_yrs = np.array([d.year + (d.month - 0.5) / 12 for d in dates])
        vals = np.interp(tgt_yrs, yrs, q, left=np.nan, right=np.nan)
        if 'subglacial_discharge' in nc.variables:
            v = nc.variables['subglacial_discharge']
        else:
            v = nc.createVariable('subglacial_discharge', 'f4', ('time',))
            v.units = 'm3 s-1'
            v.long_name = 'subglacial discharge'
        v[:] = vals
    gdir.add_to_diagnostics('subglacial_discharge_mean',
                            float(np.nanmean(vals)))
    return float(np.nanmean(vals))


def submarine_melt_rate(water_depth, q_sg, tf, A=None, B=None, alpha=None, beta=None):
    """Rignot et al. (2016) Eqn 1 as Slater et al. (2020) Eqn 2 applies it, m d-1.

    ``mdot = (A * h * q**alpha + B) * TF**beta`` with ``h`` the grounding-line water
    depth in m, ``q`` the subglacial runoff normalised by calving-front area in m d-1
    and ``TF`` the thermal forcing in degC. The scalar twin of
    :meth:`~oggm.core.ocean_calving.MeltPlusCalving.melt_rate`, for use on the
    inversion side where there is one value per divide rather than a series.
    """
    A = ocean_param('calving_melt_A') if A is None else A
    B = ocean_param('calving_melt_B') if B is None else B
    alpha = ocean_param('calving_melt_alpha') if alpha is None else alpha
    beta = ocean_param('calving_melt_beta') if beta is None else beta
    if not np.isfinite(water_depth) or not np.isfinite(tf) or tf <= 0:
        return np.nan
    q = 0. if not np.isfinite(q_sg) else max(q_sg, 0.)
    return (A * max(water_depth, 0.) * q ** alpha + B) * tf ** beta


def calving_covariates(gdir, observed_flux=None, ocean_filesuffix='', band=None,
                       period=None, input_filesuffix=''):
    """Every candidate predictor of one divide's calving constant.

    The geometric terms are the ones the calving law already carries plus the shape
    around them; ``q_sg`` and ``melt_rate`` are the part of the melt parameterization
    that has never entered this calibration. ``k`` is included when an observed flux is
    given, so the same call builds both sides of a fit.
    """
    out = {'rgi_id': gdir.rgi_id, 'is_tidewater': bool(gdir.is_tidewater)}
    if not gdir.is_tidewater:
        return out

    shape = calving_law_flux(gdir, k=1., input_filesuffix=input_filesuffix)
    cls = gdir.read_pickle('inversion_input', filesuffix=input_filesuffix)[-1]
    fl = gdir.read_pickle('inversion_flowlines', filesuffix=input_filesuffix)[-1]
    out.update(shape_km3=shape['flux'],
               water_depth=shape['water_depth'],
               front_thickness=shape['thick'],
               free_board=shape['free_board'],
               thick_flotation=shape['thick_flotation'],
               width=shape['width'],
               area_km2=gdir.rgi_area_km2,
               terminus_slope=float(cls['slope_angle'][-1]),
               terminus_elev=float(fl.surface_h[-1]),
               flowline_length=float(fl.dx_meter * len(fl.surface_h)))
    # How close the front is to floating. The law floors the thickness at flotation, so
    # this is also which divides that floor is binding on.
    out['flotation_ratio'] = (out['front_thickness'] / out['thick_flotation']
                              if out['thick_flotation'] > 0 else np.nan)
    out['depth_fraction'] = out['water_depth'] / out['front_thickness']
    out['aspect_ratio'] = out['width'] / out['front_thickness']

    front_area = out['width'] * out['front_thickness']
    yrs, q = subglacial_discharge_from_mb(gdir, period=period,
                                          input_filesuffix=input_filesuffix)
    # m3 s-1 over the front area, expressed per day, which is the unit Eqn 2 wants.
    out['q_sg'] = (float(np.nanmean(q)) * 86400. / front_area
                   if yrs is not None and front_area > 0 else np.nan)

    # `ocean_data` is registered in cfg.BASENAMES at shop import time, and a process
    # that only called init_ocean_params() has not done it. Importing here rather than
    # asking every caller to remember is what keeps a missing forcing distinguishable
    # from an unregistered basename.
    from oggm.shop import ocean as _ocean_basenames  # noqa: F401
    if gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        out['tf'] = ocean_tf_mean(gdir, band=band, period=period,
                                  ocean_filesuffix=ocean_filesuffix)
        out['melt_rate'] = submarine_melt_rate(out['water_depth'], out['q_sg'],
                                               out['tf'])
    if observed_flux is not None and gdir.rgi_id in observed_flux:
        obs = observed_flux[gdir.rgi_id]
        out['obs_flux_km3'] = obs
        out['k'] = obs / shape['flux'] if shape['flux'] > 0 else np.nan
    return out


# Terms that enter a model in logs: positive, scale free, and multiplicative in the
# parameterization the model is trying to recover.
_LOG_TERMS = frozenset(('water_depth', 'front_thickness', 'width', 'area_km2',
                        'free_board', 'aspect_ratio', 'flowline_length', 'q_sg',
                        'tf', 'melt_rate', 'shape_km3'))


def _design(rows, terms):
    cols = [np.ones(len(rows))]
    for t in terms:
        v = np.array([r[t] for r in rows], dtype=float)
        cols.append(np.log(v) if t in _LOG_TERMS else v)
    return np.column_stack(cols)


def fit_calving_k_model(gdirs, observed_flux, terms=(), covariates=None, **kwargs):
    """``log k`` regressed on covariates across the divide set.

    Global rather than per glacier, for the same reason the Glen A fit is: the
    coefficients are shared, and fitting them per divide is what makes the model
    unfalsifiable. With no ``terms`` this is an intercept alone, i.e. the geometric
    mean ``k`` -- the honest baseline for :func:`fit_calving_k`'s flux-weighted pool.

    Returns a dict with the coefficients, the terms, the in-sample and leave-one-out
    log residuals, and the per-divide ``k`` the model implies.
    """
    rows = covariates if covariates is not None else [
        calving_covariates(g, observed_flux=observed_flux, **kwargs) for g in gdirs]
    rows = [r for r in rows if r.get('k') is not None
            and np.isfinite(r.get('k', np.nan))
            and all(np.isfinite(r.get(t, np.nan)) for t in terms)]
    if len(rows) <= len(terms) + 1:
        raise InvalidWorkflowError(
            f'{len(rows)} usable divides against {len(terms) + 1} coefficients: the '
            'model would interpolate rather than fit.')

    y = np.log(np.array([r['k'] for r in rows], dtype=float))
    X = _design(rows, terms)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    loo = np.full(len(rows), np.nan)
    for i in range(len(rows)):
        keep = np.arange(len(rows)) != i
        if np.linalg.matrix_rank(X[keep]) < X.shape[1]:
            continue
        b, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
        loo[i] = X[i] @ b - y[i]

    return dict(terms=list(terms), coefficients=beta,
                rgi_ids=[r['rgi_id'] for r in rows],
                k_model={r['rgi_id']: float(np.exp(v))
                         for r, v in zip(rows, X @ beta)},
                residual=X @ beta - y, loo_residual=loo,
                loo_sd=float(np.nanstd(loo)),
                n=len(rows))


@entity_task(log)
def set_inversion_k_from_model(gdir, model=None, k_fallback=None, **kwargs):
    """Write the calving constant a fitted covariate model implies for one divide.

    Sets both ``inversion_calving_k`` and ``calving_k``: OGGM keeps the inversion's
    constant and the forward model's apart, and writing only the first leaves every
    run at the default while every report names the fitted value.
    """
    if not gdir.is_tidewater:
        return None
    if model is None:
        raise InvalidParamsError('set_inversion_k_from_model needs a model from '
                                 'fit_calving_k_model')
    k = model['k_model'].get(gdir.rgi_id)
    if k is None:
        row = calving_covariates(gdir, **kwargs)
        terms = model['terms']
        if all(np.isfinite(row.get(t, np.nan)) for t in terms):
            k = float(np.exp(_design([row], terms) @ model['coefficients']))
        elif k_fallback is not None:
            k = float(k_fallback)
        else:
            raise InvalidWorkflowError(
                f'({gdir.rgi_id}) is missing {terms} and no k_fallback was given; a '
                'silent default here is a factor on every frontal flux.')
    gdir.write_to_settings({'inversion_calving_k': k, 'calving_k': k},
                           overwrite=True)
    gdir.add_to_diagnostics('covariate_inversion_k', float(k))
    return float(k)
