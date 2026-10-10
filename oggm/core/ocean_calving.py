"""Ocean-forced frontal ablation laws for the flowline models.

Five laws, as a nested family rooted at the stock model:

===================  =========================================================
``constant``         :class:`ConstantK`, identical to
                     :func:`oggm.core.flowline.k_calving_law`. The control.
``tf_power``         :class:`TFPower`,
                     ``k(t) = k0 ((1 - a) + a (TF/TF_ref)**gamma)``.
                     Reduces to the control when ``TF == TF_ref`` or ``a = 0``.
``melt_calving``     :class:`MeltPlusCalving`, an explicit submarine-melt term
                     after Rignot et al. (2016) plus a calving term.
``sea_ice``          :class:`SeaIceModulated`, the melt term gated by the
                     open-water fraction. Reduces to ``melt_calving`` at
                     ``delta = 0``.
``larger_of``        :class:`LargerOf`, the larger of the calving and the
                     submarine-melt speed. Reduces to the control at
                     ``lam = 0``.
===================  =========================================================

Each is passed as the ``calving_law`` of ``FluxBasedModel`` or
``SemiImplicitModel``, and reads the ``ocean_data`` file written by
:mod:`oggm.shop.ocean`.

Each law also reports the split of what it returns into calving and submarine
melt, which :func:`write_frontal_components` writes beside the run output. The
split is a model diagnostic: melt undercutting drives calving, so a calving
constant calibrated against an observed frontal ablation already contains the
melt-driven part (see ``partition`` in :func:`ocean_calving_law`).

``model.calving_k`` is in s-1 while ``cfg.PARAMS['calving_k']`` is in yr-1, and
a law returns m3 s-1. The thermal forcing is clipped at zero before any power.
"""
import logging
import os

import numpy as np
import xarray as xr

from oggm import cfg
from oggm import entity_task, global_task
from oggm import utils
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

log = logging.getLogger(__name__)


def band_names(ds):
    """The depth-band names of an ``ocean_data`` dataset, in file order."""
    if 'band_name' not in ds:
        return [str(b) for b in ds['band'].values]
    raw = np.asarray(ds['band_name'].values)
    if raw.ndim == 2:  # netCDF char array, if xarray did not join it
        raw = [b''.join(r) for r in raw]
    return [(r.decode() if isinstance(r, bytes) else str(r)).strip('\x00')
            for r in raw]


def tf_power_mean(tf, gamma):
    """The reference thermal forcing at which ``tf_power`` leaves k unchanged.

    The power mean ``mean(max(TF, 0)**gamma)**(1/gamma)``, i.e. the ``TF_ref``
    for which ``(TF/TF_ref)**gamma`` averages to one over ``tf``.

    Parameters
    ----------
    tf : array
        thermal forcing, degC
    gamma : float
        the exponent of the law

    Returns
    -------
    float
        the reference, or nan when ``tf`` holds no finite value
    """
    tf = np.asarray(tf, dtype=float)
    tf = np.clip(tf[np.isfinite(tf)], 0., None)
    if not tf.size:
        return np.nan
    if gamma == 0:
        return float(np.mean(tf))
    return float(np.mean(tf ** gamma) ** (1. / gamma))


def _melt_fraction(u_calving, u_melt):
    """The melt share of a front-normal speed, bounded to [0, 1]."""
    total = u_calving + u_melt
    if total <= 0:
        return 0.
    return float(min(max(u_melt / total, 0.), 1.))


class _OceanCalvingLaw:
    """Base for time-varying calving laws driven by an ocean_data file.

    Sub-classes implement :meth:`frontal_speed_components`, returning
    front-normal speeds in m s-1. This class multiplies by the submerged area,
    which is the algebra of the stock law: ``k*d*h*w == (k*h) * (d*w)``.

    Parameters
    ----------
    years : array
        the time axis of the forcing, in float years
    tf : array
        thermal forcing on that axis, degC
    open_water : array, optional
        open-water fraction on that axis
    q_sg : array, optional
        subglacial discharge on that axis, m3 s-1
    """

    name = 'ocean'
    # law kwarg -> the parameter it defaults to
    settings_keys = {}

    def __init__(self, years, tf, open_water=None, q_sg=None):
        self._init_accounting()
        self.years = np.asarray(years, dtype=float)
        self.tf = np.asarray(tf, dtype=float)
        self.open_water = (None if open_water is None
                           else np.asarray(open_water, dtype=float))
        self.q_sg = None if q_sg is None else np.asarray(q_sg, dtype=float)
        self._model_k = None

    def _at(self, arr, yr):
        """Clamped linear interpolation, so a run past the forcing freezes it."""
        if arr is None:
            return None
        return float(np.interp(yr, self.years, arr, left=arr[0], right=arr[-1]))

    def _tf_at(self, yr):
        return max(self._at(self.tf, yr), 0.)  # clip before any power

    def frontal_speed(self, h, d, w, yr):
        """The total front-normal speed, m s-1."""
        return sum(self.frontal_speed_components(h, d, w, yr))

    def frontal_speed_components(self, h, d, w, yr):
        """``(calving, submarine melt)`` front-normal speeds, m s-1.

        They sum to :meth:`frontal_speed`. A law with no explicit melt term
        puts everything in the first slot.
        """
        raise NotImplementedError

    def __call__(self, model, flowline, last_above_wl):
        h = flowline.thick[last_above_wl]
        d = h - (flowline.surface_h[last_above_wl] - model.water_level)
        if d <= 0 or h <= 0:
            self._account(model, flowline, 0., 0.)
            return 0.
        self._model_k = model.calving_k  # s-1
        w = flowline.widths_m[last_above_wl]
        u_c, u_m = self.frontal_speed_components(h, d, w, model.yr)
        q = utils.clip_min((u_c + u_m) * d * w, 0.)
        self._account(model, flowline, q, _melt_fraction(u_c, u_m))
        return q

    def _k(self, value):
        """A calving constant in s-1, from `value` in yr-1 or from the model."""
        if value is not None:
            return value / cfg.SEC_IN_YEAR
        if self._model_k is None:
            raise InvalidWorkflowError('no calving constant: pass one to the '
                                       'law or let the model provide '
                                       'calving_k')
        return self._model_k

    # --- the melt/calving split ------------------------------------------------
    #
    # The evolution models add what the law returns times ``dt`` to one
    # counter and never see the two summands, so the law keeps its own books.
    # It cannot see ``dt`` either (``model.t`` advances after the calving
    # block), so each step is closed on the next call, and the last open step
    # is closed against the model's own total by :meth:`components_m3`.

    def _init_accounting(self):
        self.frontal_ablation_m3 = 0.
        self.submarine_melt_m3 = 0.
        self._t_prev = None
        self._pending = {}
        self._series = []

    def reset_accounting(self):
        """Start the books again, e.g. before reusing a law for a second run."""
        self._init_accounting()

    def _account(self, model, flowline, q, f_melt):
        t = getattr(model, 't', None)
        if t is None:
            return  # no model clock: a bare call, not a run
        if self._t_prev is None:
            self._t_prev = t
        elif t > self._t_prev:
            self._close(t - self._t_prev, getattr(model, 'yr', None))
            self._t_prev = t
        self._pending[id(flowline)] = (q, f_melt)

    def _close(self, dt, yr=None):
        """Integrate the previous step, whose length is only now known."""
        for q, f in self._pending.values():
            self.frontal_ablation_m3 += q * dt
            self.submarine_melt_m3 += q * f * dt
        self._pending = {}
        if yr is None:
            return
        # one record per month bounds the memory
        if not self._series or int(yr * 12) > int(self._series[-1][0] * 12):
            self._series.append((float(yr), self.frontal_ablation_m3,
                                 self.submarine_melt_m3))

    def _pending_melt_fraction(self):
        tot = sum(q for q, _ in self._pending.values())
        if tot <= 0:
            return 0.
        return sum(q * f for q, f in self._pending.values()) / tot

    def components_m3(self, model=None):
        """``(calving, submarine melt)`` since the run started, m3.

        With a model they sum to its ``calving_m3_since_y0``: the step still
        open when the run stopped is closed here at its own melt fraction.
        """
        total, melt = self.frontal_ablation_m3, self.submarine_melt_m3
        ref = getattr(model, 'calving_m3_since_y0', None)
        if ref is not None and ref > total:
            melt += (ref - total) * self._pending_melt_fraction()
            total = ref
        return total - melt, melt

    def component_series(self, model=None):
        """Monthly cumulative (years, frontal ablation, submarine melt), m3."""
        rec = list(self._series)
        if model is not None:
            yr = getattr(model, 'yr', None)
            calving, melt = self.components_m3(model)
            if yr is not None and (not rec or yr > rec[-1][0]):
                rec.append((float(yr), calving + melt, melt))
        if not rec:
            return (np.zeros(0), np.zeros(0), np.zeros(0))
        a = np.asarray(rec, dtype=float)
        return a[:, 0], a[:, 1], a[:, 2]


class ConstantK(_OceanCalvingLaw):
    """The control: :func:`oggm.core.flowline.k_calving_law` with accounting.

    Like the stock law, and unlike its siblings, it does not clip at zero.

    Parameters
    ----------
    k : float, optional
        the calving constant in yr-1. Default: the model's ``calving_k``.
    """

    name = 'constant'

    def __init__(self, years=None, tf=None, k=None, **kwargs):
        self._init_accounting()
        self.k = k
        self.years = None if years is None else np.asarray(years, dtype=float)
        self.tf = None if tf is None else np.asarray(tf, dtype=float)
        self._model_k = None

    def frontal_speed_components(self, h, d, w, yr):
        return self._k(self.k) * h, 0.

    def __call__(self, model, flowline, last_above_wl):
        h = flowline.thick[last_above_wl]
        d = h - (flowline.surface_h[last_above_wl] - model.water_level)
        k = model.calving_k if self.k is None else self.k / cfg.SEC_IN_YEAR
        q = k * d * h * flowline.widths_m[last_above_wl]
        self._account(model, flowline, q, 0.)
        return q


class TFPower(_OceanCalvingLaw):
    """The stock law with a calving constant scaled by the thermal forcing.

    ``k(t) = k0 * ((1 - a) + a * (max(TF(t), 0) / TF_ref) ** gamma)``

    The default ``gamma`` is the thermal-forcing exponent of the submarine
    melt rate of Rignot et al. (2016), which assumes that frontal ablation
    inherits the thermal dependence of the melt. ``a`` is the share of the
    calving constant that follows the ocean; the rest is a background term
    that calves at the freezing point too.

    Parameters
    ----------
    tf_ref : float
        the reference thermal forcing, degC. Must be positive:
        :func:`tf_power_mean` over a reference period makes the law return
        ``k0`` on average over that period.
    k0 : float, optional
        the calving constant at ``TF_ref``, yr-1. Default: the model's
        ``calving_k``.
    gamma : float, optional
        default: ``cfg.PARAMS['calving_tf_exponent']``
    tf_fraction : float, optional
        ``a``, in [0, 1]. Default: ``cfg.PARAMS['calving_tf_fraction']``
    """

    name = 'tf_power'
    settings_keys = {'gamma': 'calving_tf_exponent',
                     'tf_fraction': 'calving_tf_fraction'}

    def __init__(self, *args, tf_ref=None, k0=None, gamma=None,
                 tf_fraction=None, **kwargs):
        super().__init__(*args, **kwargs)
        if tf_ref is None or not np.isfinite(tf_ref) or tf_ref <= 0:
            raise InvalidParamsError(
                f'TF_ref = {tf_ref} is not usable: the tf_power law needs a '
                'positive reference thermal forcing.')
        self.tf_ref = float(tf_ref)
        self.k0 = k0
        self.gamma = (cfg.PARAMS['calving_tf_exponent'] if gamma is None
                      else gamma)
        self.tf_fraction = (cfg.PARAMS['calving_tf_fraction']
                            if tf_fraction is None else tf_fraction)
        if not 0 <= self.tf_fraction <= 1:
            raise InvalidParamsError(
                f'tf_fraction = {self.tf_fraction} is not in [0, 1]')

    def scaling(self, yr):
        """``k(t) / k0`` at ``yr``."""
        s = (self._tf_at(yr) / self.tf_ref) ** self.gamma
        if self.tf_fraction == 1:
            return s
        return (1 - self.tf_fraction) + self.tf_fraction * s

    def frontal_speed_components(self, h, d, w, yr):
        k = self._k(self.k0)
        return k * self.scaling(yr) * h, 0.


class MeltPlusCalving(_OceanCalvingLaw):
    """Frontal ablation as a sum of front-normal speeds.

    ``Q_f = w * d * (k_c * h + lam * mdot)``, with ``mdot`` from Rignot et
    al. (2016) Eq. (1): ``mdot = (A * h_w * q_sg**alpha + B) * TF**beta`` in
    m d-1, ``h_w`` being the water depth at the front.

    ``mdot`` is the horizontally averaged maximum melt rate, not the
    area-averaged one, so ``lam`` scales its action on the whole submerged
    front. Without a subglacial discharge the law is ``B * TF**beta``.

    Parameters
    ----------
    k_c : float, optional
        the calving constant, yr-1. Default: the model's ``calving_k``, in
        which case the melt term adds to a constant that may already contain
        it (see ``partition`` in :func:`ocean_calving_law`).
    lam : float, optional
        default: ``cfg.PARAMS['calving_undercut_efficiency']``
    water_depth : float, optional
        the water depth of the melt rate, m. Default: the model's own.
    A, B, alpha, beta : float, optional
        default: ``cfg.PARAMS['calving_melt_A']`` and so on
    """

    name = 'melt_calving'
    settings_keys = {'A': 'calving_melt_A', 'B': 'calving_melt_B',
                     'alpha': 'calving_melt_alpha',
                     'beta': 'calving_melt_beta',
                     'lam': 'calving_undercut_efficiency'}

    def __init__(self, *args, k_c=None, lam=None, water_depth=None, A=None,
                 B=None, alpha=None, beta=None, **kwargs):
        super().__init__(*args, **kwargs)
        for kwarg, value in (('A', A), ('B', B), ('alpha', alpha),
                             ('beta', beta), ('lam', lam)):
            if value is None:
                value = cfg.PARAMS[self.settings_keys[kwarg]]
            setattr(self, kwarg, value)
        self.k_c = k_c
        self.water_depth = water_depth

    def melt_rate(self, d, w, yr):
        """Rignot et al. (2016) Eq. (1), in m s-1."""
        tf = self._tf_at(yr)
        hw = d if self.water_depth is None else self.water_depth
        if self.q_sg is None:
            q = 0.
        else:
            area = max(hw * w, 1e-3)
            # m3 s-1 -> m d-1
            q = max(self._at(self.q_sg, yr), 0.) * cfg.SEC_IN_DAY / area
        mdot = ((self.A * max(hw, 0.) * q ** self.alpha + self.B)
                * tf ** self.beta)
        return mdot / cfg.SEC_IN_DAY

    def melt_speed(self, d, w, yr):
        """The front-normal melt speed the law applies, m s-1."""
        return self.lam * self.melt_rate(d, w, yr)

    def frontal_speed_components(self, h, d, w, yr):
        return self._k(self.k_c) * h, self.melt_speed(d, w, yr)


class SeaIceModulated(MeltPlusCalving):
    """:class:`MeltPlusCalving` with the melt term gated by open water.

    ``Q_f = w * d * (k_ice * h + f_ow(t)**delta * lam * mdot(TF))``

    The open-water fraction ``f_ow`` acts on the melt term only; the calving
    term is left as it is. ``delta = 0`` is :class:`MeltPlusCalving`.

    Parameters
    ----------
    k_ice : float, optional
        the calving constant, yr-1 (``k_c`` of :class:`MeltPlusCalving`)
    delta : float, optional
        default: ``cfg.PARAMS['calving_openwater_exponent']``
    """

    name = 'sea_ice'
    settings_keys = dict(MeltPlusCalving.settings_keys,
                         delta='calving_openwater_exponent')

    def __init__(self, *args, k_ice=None, delta=None, **kwargs):
        kwargs.setdefault('k_c', k_ice)
        super().__init__(*args, **kwargs)
        self.delta = (cfg.PARAMS['calving_openwater_exponent']
                      if delta is None else delta)

    def melt_speed(self, d, w, yr):
        if self.open_water is None:
            raise InvalidWorkflowError('SeaIceModulated needs '
                                       'open_water_frac in ocean_data.nc')
        f = max(self._at(self.open_water, yr), 0.)
        melt = self.lam * self.melt_rate(d, w, yr)
        return (f ** self.delta) * melt


class LargerOf(MeltPlusCalving):
    """The larger of the calving and the submarine-melt speed.

    ``Q_f = w * d * max(k_c * h, lam * mdot)``, after Malles et al. (2025).
    The whole flux is reported as the larger term. ``lam = 0`` is the control.
    With ``partition``, ``k_c`` is lowered until the period mean of the larger
    term equals the calibrated total.
    """

    name = 'larger_of'

    def frontal_speed_components(self, h, d, w, yr):
        u_c, u_m = self._k(self.k_c) * h, self.melt_speed(d, w, yr)
        return (u_c, 0.) if u_c >= u_m else (0., u_m)


LAWS = {'constant': ConstantK, 'tf_power': TFPower,
        'melt_calving': MeltPlusCalving, 'sea_ice': SeaIceModulated,
        'larger_of': LargerOf}


def partition_law_k(gdir, law, period=None):
    """Give a melt-bearing law the residual calving constant, keeping the total.

    A calving constant fitted to an observed frontal ablation already contains
    the calving that submarine melt drives, so a melt term added on top of it
    counts that melt twice.
    :func:`oggm.core.ocean_inversion.partition_calving_constant` solves
    ``k_c h + u_melt = k h`` for ``k_c`` at the front the inversion
    prescribed, with the melt speed the law applies (its open-water gate
    included) averaged over ``period``. The total is then unchanged over that
    period and only its split moves. For :class:`LargerOf`,
    :func:`oggm.core.ocean_inversion.larger_of_calving_constant` solves
    ``mean(max(k_c h, u_melt)) = k h`` instead.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory, inverted with
        :func:`oggm.core.ocean_inversion.find_inversion_calving_from_bathymetry`
    law : :class:`MeltPlusCalving`
        the law, whose ``k_c`` is set in place
    period : tuple of two years, optional
        default: ``gdir.settings['ocean_tf_ref_period']``

    Returns
    -------
    (k_c, melt_fraction) : the residual constant in yr-1 and the melt share of
        the total, or ``(None, 0.)`` for a law without a melt term
    """
    from oggm.core.ocean_inversion import (_setting, larger_of_calving_constant,
                                          partition_calving_constant)

    if not hasattr(law, 'melt_rate'):
        return None, 0.
    k_total = _setting(gdir, 'calving_k', _setting(gdir, 'inversion_calving_k'))
    thick = _setting(gdir, 'calving_front_thick')
    depth = _setting(gdir, 'terminus_water_depth')
    width = _setting(gdir, 'calving_front_width')
    if None in (k_total, thick, depth, width):
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) partitioning needs calving_k, '
            'calving_front_thick, terminus_water_depth and calving_front_width '
            'in the settings; run find_inversion_calving_from_bathymetry '
            'first.')

    period = period or gdir.settings['ocean_tf_ref_period']
    y0, y1 = (float(y) for y in period)
    yrs = np.asarray(law.years, dtype=float)
    sel = (yrs >= y0) & (yrs < y1 + 1)
    if not sel.any():
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) the ocean record does not cover '
            f'{y0:.0f}-{y1:.0f}, so the reference melt rate cannot be formed. '
            'Give the law an explicit k_c, or pick a period the record '
            'covers.')
    u_melt = [law.melt_speed(depth, width, y) for y in yrs[sel]]
    if isinstance(law, LargerOf):
        k_c, frac = larger_of_calving_constant(k_total, u_melt, thick)
    else:
        k_c, frac = partition_calving_constant(k_total, float(np.mean(u_melt)),
                                               thick, lam=1.)
    law.k_c = k_c
    return k_c, frac


def ocean_calving_law(gdir, calving_law=None, band=None, ocean_filesuffix='',
                      tf_ref=None, partition=False, partition_period=None,
                      **law_kwargs):
    """Build a calving law for one glacier from its ocean_data file.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    calving_law : str, optional
        'constant', 'tf_power', 'melt_calving', 'sea_ice' or 'larger_of'.
        Default: ``gdir.settings['calving_law']``.
    band : str, optional
        the depth band of the ocean file the law reads. Default:
        ``gdir.settings['ocean_tf_band']``.
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    tf_ref : float, optional
        the reference thermal forcing of 'tf_power', ignored by the other
        laws. Default: ``gdir.settings['ocean_tf_ref']``, and when that is not
        set either, :func:`tf_power_mean` of the glacier's own record over
        ``gdir.settings['ocean_tf_ref_period']``, so that the law returns the
        calving constant on average over that period.
    partition : bool
        for a melt-bearing law, set ``k_c`` from :func:`partition_law_k` so
        that the law keeps the calibrated total over ``partition_period``
        instead of adding melt to it.
    partition_period : tuple of two years, optional
        the period over which the partition keeps the total. Default:
        ``gdir.settings['ocean_tf_ref_period']``.
    **law_kwargs
        passed to the law (``k0``, ``gamma``, ``lam``, ``delta`` ...). The
        ones not given are read from the glacier's settings.

    Returns
    -------
    the law, to be passed as ``calving_law`` to a flowline model
    """
    calving_law = calving_law or gdir.settings['calving_law']
    if calving_law not in LAWS:
        raise InvalidParamsError(f'unknown calving law {calving_law!r}; '
                                 f'available: {sorted(LAWS)}')
    if calving_law == 'constant':
        return ConstantK(**law_kwargs)

    if not gdir.has_file('ocean_data', filesuffix=ocean_filesuffix):
        raise InvalidWorkflowError(f'({gdir.rgi_id}) no ocean_data file; run '
                                   'process_ocean_data first.')

    band = band or gdir.settings['ocean_tf_band']
    fp = gdir.get_filepath('ocean_data', filesuffix=ocean_filesuffix)
    with xr.open_dataset(fp) as ds:
        ds = ds.load()
    if 'band' not in ds.dims:
        raise InvalidWorkflowError('ocean_data has no band dimension')
    names = band_names(ds)
    if band not in names:
        raise InvalidParamsError(f'band {band!r} not in {names}')
    tf = ds['thermal_forcing'].values[:, names.index(band)]
    yrs = ds['time.year'].values + (ds['time.month'].values - 0.5) / 12

    cls = LAWS[calving_law]
    for kwarg, key in cls.settings_keys.items():
        if law_kwargs.get(kwarg) is None:
            law_kwargs[kwarg] = gdir.settings[key]

    if cls is TFPower:
        if tf_ref is None:
            tf_ref = gdir.settings['ocean_tf_ref']
        if tf_ref is None:
            y0, y1 = (int(y) for y in gdir.settings['ocean_tf_ref_period'])
            sel = (ds['time.year'].values >= y0) & (ds['time.year'].values <= y1)
            if not sel.any():
                raise InvalidWorkflowError(
                    f'({gdir.rgi_id}) the ocean record does not cover '
                    f'{y0}-{y1}, so there is no reference thermal forcing. '
                    'Pass tf_ref or set ocean_tf_ref.')
            tf_ref = tf_power_mean(tf[sel], law_kwargs['gamma'])
        law_kwargs['tf_ref'] = tf_ref

    law = cls(yrs, tf,
              open_water=(ds['open_water_frac'].values
                          if 'open_water_frac' in ds else None),
              q_sg=(ds['subglacial_discharge'].values
                    if 'subglacial_discharge' in ds else None),
              **law_kwargs)
    if partition:
        k_c, frac = partition_law_k(gdir, law, period=partition_period)
        if k_c is not None:
            gdir.add_to_diagnostics('calving_k_residual', float(k_c))
            gdir.add_to_diagnostics('calving_melt_fraction_reference',
                                    float(frac))
    return law


def frontal_ablation_components(model):
    """``(calving, submarine melt)`` of a finished run, m3, from its model."""
    law = getattr(model, 'calving_law', None)
    if not hasattr(law, 'components_m3'):
        # the stock law, or any other callable: nothing attributable to melt
        return float(getattr(model, 'calving_m3_since_y0', 0.)), 0.
    return law.components_m3(model)


def write_frontal_components(gdir, law, model=None, output_filesuffix=''):
    """Write the melt/calving split of a finished run to its own file.

    The two series are a share of the ``calving_m3`` of the run's
    ``model_diagnostics`` file, which is left untouched, so they sum to it at
    every step.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    law : the calving law the run used
    model : the flowline model of the run, optional
        closes the last open step of the split
    output_filesuffix : str
        the filesuffix of the run

    Returns
    -------
    (calving, submarine melt) : the totals of the run, m3
    """
    calving, melt = law.components_m3(model)
    total = calving + melt
    frac = float(melt / total) if total > 0 else 0.
    gdir.add_to_diagnostics('submarine_melt_m3', float(melt))
    gdir.add_to_diagnostics('calving_only_m3', float(calving))
    gdir.add_to_diagnostics('submarine_melt_fraction', frac)

    fp = gdir.get_filepath('model_diagnostics', filesuffix=output_filesuffix)
    if not os.path.exists(fp):
        return calving, melt
    with xr.open_dataset(fp) as ds:
        ds = ds.load()
    if 'calving_m3' not in ds:
        log.warning("(%s) no calving_m3 in the diagnostics: add 'calving' to "
                    "store_diagnostic_variables to get the split", gdir.rgi_id)
        return calving, melt

    yrs, cum_total, cum_melt = law.component_series(model)
    cum = ds['calving_m3'].values
    if len(yrs) == 0:
        f_t = np.full(cum.shape, frac)
    else:
        f = np.where(cum_total > 0,
                     cum_melt / np.where(cum_total > 0, cum_total, 1.), 0.)
        f_t = np.interp(ds['time'].values.astype(float), yrs, f,
                        left=f[0], right=f[-1])

    out = xr.Dataset(coords={c: ds[c] for c in ds.coords})
    out['calving_m3'] = ds['calving_m3']
    out['submarine_melt_m3'] = ('time', cum * f_t)
    out['submarine_melt_m3'].attrs = {
        'description': 'Accumulated submarine melt part of the frontal '
                       'ablation',
        'unit': 'm 3'}
    out['calving_only_m3'] = ('time', cum * (1 - f_t))
    out['calving_only_m3'].attrs = {
        'description': 'Accumulated iceberg calving part of the frontal '
                       'ablation',
        'unit': 'm 3'}
    out.attrs.update({'rgi_id': gdir.rgi_id,
                      'cenlon': float(gdir.cenlon),
                      'cenlat': float(gdir.cenlat),
                      'calving_law': law.name,
                      'submarine_melt_fraction': frac,
                      'partition': 'modelled, not observed'})
    out.to_netcdf(gdir.get_filepath('frontal_ablation_diagnostics',
                                    filesuffix=output_filesuffix, delete=True))
    return calving, melt


@global_task(log)
def compile_frontal_components(gdirs, input_filesuffix='', path=True):
    """Compile the frontal ablation split of several glaciers into one file.

    The counterpart of :func:`oggm.utils.compile_run_output` for the
    ``frontal_ablation_diagnostics`` files.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    input_filesuffix : str
        the filesuffix of the runs
    path : str or bool
        where to store the file (default is on the working dir). Set to
        `False` to disable disk storage.

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        with dimensions (time, rgi_id), or None when no glacier has the file
    """
    dss = []
    for gdir in gdirs:
        fp = gdir.get_filepath('frontal_ablation_diagnostics',
                               filesuffix=input_filesuffix)
        if not os.path.exists(fp):
            log.warning('(%s) no frontal_ablation_diagnostics%s, skipped',
                        gdir.rgi_id, input_filesuffix)
            continue
        with xr.open_dataset(fp) as ds:
            ds = ds.load()
        ds = ds[['calving_m3', 'submarine_melt_m3', 'calving_only_m3']]
        ds = ds.expand_dims(rgi_id=[gdir.rgi_id])
        ds.coords['lon'] = ('rgi_id', [float(gdir.cenlon)])
        ds.coords['lat'] = ('rgi_id', [float(gdir.cenlat)])
        dss.append(ds)
    if not dss:
        return None
    out = xr.concat(dss, dim='rgi_id')
    out.attrs['partition'] = 'modelled, not observed'
    if path:
        if path is True:
            path = os.path.join(
                cfg.PATHS['working_dir'],
                f'frontal_ablation_compiled{input_filesuffix}.nc')
        out.to_netcdf(path)
    return out


@entity_task(log, writes=['frontal_ablation_diagnostics'])
def run_with_ocean_forcing(gdir, calving_law=None, band=None,
                           ocean_filesuffix='', tf_ref=None, law_kwargs=None,
                           climate_filename='gcm_data',
                           climate_input_filesuffix='',
                           output_filesuffix='', **kwargs):
    """Run the flowline model with an ocean-forced calving law.

    A wrapper around :func:`oggm.core.flowline.run_from_climate_data`, which
    also writes the calving and submarine melt parts of the frontal ablation
    (see :func:`write_frontal_components`).

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    calving_law : str, optional
        'constant', 'tf_power', 'melt_calving', 'sea_ice' or 'larger_of'.
        Default: ``gdir.settings['calving_law']``.
    band : str, optional
        the depth band of the ocean file the law reads. Default:
        ``gdir.settings['ocean_tf_band']``.
    ocean_filesuffix : str
        the filesuffix of the ``ocean_data`` file
    tf_ref : float, optional
        the reference thermal forcing of 'tf_power' (see
        :func:`ocean_calving_law`)
    law_kwargs : dict, optional
        passed to :func:`ocean_calving_law` (``partition``, ``k0``, ``gamma``,
        ``lam``, ``delta`` ...)
    climate_filename : str
        name of the climate file, e.g. 'climate_historical' or 'gcm_data'
    climate_input_filesuffix : str
        filesuffix of the climate file
    output_filesuffix : str
        filesuffix of the output files
    **kwargs
        passed to :func:`oggm.core.flowline.run_from_climate_data`

    Returns
    -------
    the flowline model, as returned by ``run_from_climate_data``
    """
    from oggm.core.flowline import run_from_climate_data

    # A law that cannot be built must not leave an earlier run's files behind
    for name in ('model_diagnostics', 'frontal_ablation_diagnostics'):
        gdir.get_filepath(name, filesuffix=output_filesuffix, delete=True)

    law = ocean_calving_law(gdir, calving_law=calving_law, band=band,
                            ocean_filesuffix=ocean_filesuffix, tf_ref=tf_ref,
                            **(law_kwargs or {}))
    gdir.add_to_diagnostics('ocean_calving_law', law.name)
    if band:
        gdir.add_to_diagnostics('ocean_calving_band', band)

    model = run_from_climate_data(
        gdir, calving_law=law, climate_filename=climate_filename,
        climate_input_filesuffix=climate_input_filesuffix,
        output_filesuffix=output_filesuffix, **kwargs)
    write_frontal_components(gdir, law, model=model,
                             output_filesuffix=output_filesuffix)
    return model
