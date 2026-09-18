"""Ocean-forced frontal ablation laws for the OGGM dynamical model.

Four laws, as a nested family rooted at the stock model:

===================  =========================================================
``constant``         :class:`ConstantK`, identical to
                     :func:`oggm.core.flowline.k_calving_law`. The control.
``tf_power``         :class:`TFPower`, ``k(t) = k0 (TF/TF_ref)**gamma``.
                     Reduces to the control when ``TF == TF_ref``.
``melt_calving``     :class:`MeltPlusCalving`, an explicit submarine-melt term
                     after Rignot et al. (2016) plus a residual calving term.
``sea_ice``          :class:`SeaIceModulated`, the melt term gated by open-water
                     fraction. Reduces to ``melt_calving`` at ``delta = 0``.
===================  =========================================================

Each is a drop-in for the ``calving_law=`` kwarg of ``FluxBasedModel`` and
``SemiImplicitModel``, so nothing in OGGM is modified. They are nested on purpose:
comparing them is then a comparison inside one model rather than between four.

Two traps the classes exist to close. ``model.calving_k`` is in s-1 while
``cfg.PARAMS['calving_k']`` is in a-1, and a law must return m3 s-1. And a negative
thermal forcing raised to a fractional power is a NaN that propagates silently into
``calving_m3``, so every law clips the forcing at zero before the power.
"""
import logging

import numpy as np

from oggm import cfg
from oggm import entity_task
from oggm import utils
from oggm.core.ocean_params import ocean_param
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

log = logging.getLogger(__name__)


class _OceanCalvingLaw:
    """Base for time-varying calving laws driven by an ocean_data file.

    Sub-classes implement :meth:`frontal_speed`, returning a front-normal speed in
    m s-1. This class multiplies by the submerged area, which is the algebra of the
    stock law: ``k*d*h*w == (k*h) * (d*w)``.
    """

    name = 'ocean'

    def __init__(self, years, tf, open_water=None, q_sg=None, tf_ref=None):
        self.years = np.asarray(years, dtype=float)
        self.tf = np.asarray(tf, dtype=float)
        self.open_water = None if open_water is None else np.asarray(open_water, float)
        self.q_sg = None if q_sg is None else np.asarray(q_sg, dtype=float)
        self.tf_ref = float(tf_ref) if tf_ref is not None else float(np.nanmean(self.tf))
        if not np.isfinite(self.tf_ref) or self.tf_ref <= 0:
            # (TF/TF_ref)**gamma is meaningless for TF_ref <= 0. This is the
            # freezing-point case, and it has to be a deliberate modelling decision
            # rather than a division.
            raise InvalidParamsError(
                f'TF_ref = {self.tf_ref} is not usable; set '
                "cfg.PARAMS['ocean_tf_ref'] explicitly or pick a different band.")
        self._model_k = None

    def _at(self, arr, yr):
        """Clamped linear interpolation, so a run past the forcing freezes it."""
        if arr is None:
            return None
        return float(np.interp(yr, self.years, arr, left=arr[0], right=arr[-1]))

    def _tf_at(self, yr):
        return max(self._at(self.tf, yr), 0.)  # clip before any power, as ISSM does

    def frontal_speed(self, h, d, w, yr):
        raise NotImplementedError

    def __call__(self, model, flowline, last_above_wl):
        h = flowline.thick[last_above_wl]
        d = h - (flowline.surface_h[last_above_wl] - model.water_level)
        if d <= 0 or h <= 0:
            return 0.
        self._model_k = model.calving_k  # s-1
        w = flowline.widths_m[last_above_wl]
        return utils.clip_min(self.frontal_speed(h, d, w, model.yr) * d * w, 0.)

    def _k(self, value):
        """A calving constant in s-1, from `value` in a-1 or from the model."""
        if value is not None:
            return value / cfg.SEC_IN_YEAR
        if self._model_k is None:
            raise InvalidWorkflowError('no calving constant: pass one to the law or '
                                       'let the model provide calving_k')
        return self._model_k


class ConstantK(_OceanCalvingLaw):
    """The control. Bit-identical to :func:`oggm.core.flowline.k_calving_law`.

    Written out rather than delegating so that all four laws share one geometry, and
    so the equivalence is something a test asserts rather than something assumed.
    Unlike its siblings it does *not* clip at zero, because the stock law does not.
    """

    name = 'constant'

    def __init__(self, years=None, tf=None, k=None, **kwargs):
        self.k = k  # a-1, or None to take the model's calving_k
        self.years = None if years is None else np.asarray(years, dtype=float)
        self.tf = None if tf is None else np.asarray(tf, dtype=float)
        self._model_k = None

    def __call__(self, model, flowline, last_above_wl):
        h = flowline.thick[last_above_wl]
        d = h - (flowline.surface_h[last_above_wl] - model.water_level)
        k = model.calving_k if self.k is None else self.k / cfg.SEC_IN_YEAR
        return k * d * h * flowline.widths_m[last_above_wl]


class TFPower(_OceanCalvingLaw):
    """Oerlemans-Nick with an ocean-scaled proportionality constant.

    ``k(t) = k0 * (max(TF(t), 0) / TF_ref) ** gamma``

    ``gamma`` defaults to Rignot et al. (2016) beta, on the argument that the thermal
    dependence of frontal ablation inherits that of submarine melt. That is an
    assumption, not a derivation: Rignot's beta describes an undercutting rate, and
    the two coincide only if frontal ablation is melt-limited.
    """

    name = 'tf_power'

    def __init__(self, *args, k0=None, gamma=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.k0 = k0
        self.gamma = ocean_param('calving_tf_exponent') if gamma is None else gamma

    def frontal_speed(self, h, d, w, yr):
        k = self._k(self.k0)
        return k * (self._tf_at(yr) / self.tf_ref) ** self.gamma * h


class MeltPlusCalving(_OceanCalvingLaw):
    """Frontal ablation as a sum of front-normal speeds.

    ``Q_f = w * d * (k_c * h + lam * mdot)``, with ``mdot`` from Rignot et al. (2016)
    Eq. (1): ``mdot = (A * h_w * q_sg**alpha + B) * TF**beta`` in m d-1.

    ``lam`` is there because Rignot's ``mdot`` is the horizontally averaged *maximum*
    melt rate, not the area-averaged one. Multiplying it by the whole submerged area
    overstates the melt mass flux, so the product is defensible as a calving driver
    rather than as a melt flux. ``lam = 1`` is the naive additive case.

    ``h_w`` follows ISSM in being the water depth at the front, not the ice thickness.
    """

    name = 'melt_calving'

    def __init__(self, *args, k_c=None, lam=None, water_depth=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = ocean_param('calving_melt_A')
        self.B = ocean_param('calving_melt_B')
        self.alpha = ocean_param('calving_melt_alpha')
        self.beta = ocean_param('calving_melt_beta')
        self.lam = ocean_param('calving_undercut_efficiency') if lam is None else lam
        self.k_c = k_c  # a-1
        self.water_depth = water_depth  # m, from bathymetry when known

    def melt_rate(self, d, w, yr):
        """Rignot et al. (2016) Eq. (1), in m s-1."""
        tf = self._tf_at(yr)
        hw = d if self.water_depth is None else self.water_depth
        if self.q_sg is None:
            # Rignot's own no-subglacial-discharge limit, and the defensible
            # default at a cold-based ice cap: the law reduces to B * TF**beta.
            q = 0.
        else:
            area = max(hw * w, 1e-3)
            q = max(self._at(self.q_sg, yr), 0.) * 86400. / area  # m3 s-1 -> m d-1
        mdot = (self.A * max(hw, 0.) * q ** self.alpha + self.B) * tf ** self.beta
        return mdot / 86400.

    def frontal_speed(self, h, d, w, yr):
        return self._k(self.k_c) * h + self.lam * self.melt_rate(d, w, yr)


class SeaIceModulated(MeltPlusCalving):
    """Frontal ablation gated by open water rather than by ocean heat alone.

    ``Q_f = w * d * (k_ice * h + f_ow(t)**delta * lam * mdot(TF))``

    Motivated by the only near-terminus hydrography at Flade Isblink: water within
    0.005 degC of the freezing point at 44-75 m, and roughly 10 m of melt per month
    in the August open-water period against 1.2 m per year subsurface. If that holds,
    open-water duration is the operative predictor and ocean heat is not. It is also
    the term Malles et al. (2023) wrote into their Eq. (5) and then declined.

    ``delta = 0`` collapses this to :class:`MeltPlusCalving`, which is what makes
    ``delta`` testable rather than assumed.
    """

    name = 'sea_ice'

    def __init__(self, *args, k_ice=None, delta=None, **kwargs):
        super().__init__(*args, k_c=k_ice, **kwargs)
        self.delta = (ocean_param('calving_openwater_exponent') if delta is None
                      else delta)

    def frontal_speed(self, h, d, w, yr):
        if self.open_water is None:
            raise InvalidWorkflowError('SeaIceModulated needs open_water_frac in '
                                       'ocean_data.nc')
        f = max(self._at(self.open_water, yr), 0.)
        melt = self.lam * self.melt_rate(d, w, yr)
        return self._k(self.k_c) * h + (f ** self.delta) * melt


LAWS = {'constant': ConstantK, 'tf_power': TFPower,
        'melt_calving': MeltPlusCalving, 'sea_ice': SeaIceModulated}


def ocean_calving_law(gdir, calving_law=None, band=None, ocean_filesuffix='',
                      tf_ref=None, **law_kwargs):
    """Build a calving law for one glacier from its ocean_data file.

    Separate from the run task so a law can be built, inspected and tested without
    running the model.
    """
    import xarray as xr

    calving_law = calving_law or ocean_param('calving_law')
    if calving_law not in LAWS:
        raise InvalidParamsError(f'unknown calving law {calving_law!r}; '
                                 f'available: {sorted(LAWS)}')
    if calving_law == 'constant':
        return ConstantK(**law_kwargs)

    if not gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no ocean_data file; run '
                                   'process_ocean_data first.')

    band = band or ocean_param('ocean_tf_band')
    fp = gdir.get_filepath('ocean_data', filesuffix=ocean_filesuffix)
    with xr.open_dataset(fp) as ds:
        ds = ds.load()
    if 'band' not in ds.dims:
        raise InvalidWorkflowError('ocean_data has no band dimension')
    names = [str(n) for n in _band_names(ds)]
    if band not in names:
        raise InvalidParamsError(f'band {band!r} not in {names}')
    i = names.index(band)

    yrs = (ds['time.year'].values + (ds['time.month'].values - 0.5) / 12)
    tf_ref = ocean_param('ocean_tf_ref') if tf_ref is None else tf_ref
    return LAWS[calving_law](
        yrs, ds['thermal_forcing'].values[:, i],
        open_water=(ds['open_water_frac'].values if 'open_water_frac' in ds
                    else None),
        q_sg=(ds['subglacial_discharge'].values if 'subglacial_discharge' in ds
              else None),
        tf_ref=tf_ref, **law_kwargs)


def _band_names(ds):
    """Band names, whether they are a coordinate or the char variable we write."""
    if 'band_name' in ds:
        raw = ds['band_name'].values
        if raw.ndim == 2:  # char array
            return [b''.join(r).decode().strip('\x00') for r in raw]
        return [str(r) for r in raw]
    return [str(b) for b in ds['band'].values]


@entity_task(log)
def run_with_ocean_forcing(gdir, calving_law=None, band=None, ocean_filesuffix='',
                           tf_ref=None, law_kwargs=None,
                           climate_filename='gcm_data',
                           climate_input_filesuffix='',
                           output_filesuffix='', **kwargs):
    """Run the dynamical model with an ocean-forced calving law.

    The law is built inside the worker, so no forcing array crosses the
    multiprocessing pickle boundary.

    Parameters
    ----------
    calving_law : str
        one of 'constant', 'tf_power', 'melt_calving', 'sea_ice'.
    band : str
        which depth band of the ocean file the law reads.
    law_kwargs : dict
        passed to the law's constructor (k0, gamma, lam, delta, ...).
    """
    from oggm.core.flowline import run_from_climate_data

    law = ocean_calving_law(gdir, calving_law=calving_law, band=band,
                            ocean_filesuffix=ocean_filesuffix, tf_ref=tf_ref,
                            **(law_kwargs or {}))
    gdir.add_to_diagnostics('ocean_calving_law', law.name)
    if band:
        gdir.add_to_diagnostics('ocean_calving_band', band)

    return run_from_climate_data(gdir, calving_law=law,
                                 climate_filename=climate_filename,
                                 climate_input_filesuffix=climate_input_filesuffix,
                                 output_filesuffix=output_filesuffix, **kwargs)
