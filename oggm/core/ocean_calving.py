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

Each law also reports the **split** of what it returns: ``frontal_speed_components``
gives ``(calving, submarine melt)`` separately, and the base class keeps an exact
running total of each, which :func:`write_frontal_components` appends to the run's
sidecar ``frontal_ablation_diagnostics.nc`` as ``calving_only_m3`` and
``submarine_melt_m3`` (``compile_run_output`` rejects unknown variables). It is a partition
the model asserts, not one any observation constrains -- melt undercutting drives
calving, so the two are not independently forced, and a calving constant calibrated
against an observed total already contains the melt-driven part.

Two traps the classes exist to close. ``model.calving_k`` is in s-1 while
``cfg.PARAMS['calving_k']`` is in a-1, and a law must return m3 s-1. And a negative
thermal forcing raised to a fractional power is a NaN that propagates silently into
``calving_m3``, so every law clips the forcing at zero before the power.
"""
import logging
import os

import numpy as np

from oggm import cfg
from oggm import entity_task
from oggm import utils
from oggm.core.ocean_params import ocean_param
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

log = logging.getLogger(__name__)

# A sidecar, because compile_run_output raises on any variable it does not know.
cfg.add_to_basenames(
    'frontal_ablation_diagnostics', 'frontal_ablation_diagnostics.nc',
    'The submarine melt and calving parts of the frontal ablation of a run, on '
    'the time axis of its model_diagnostics file.')


def _melt_fraction(u_calving, u_melt):
    """The melt share of a front-normal speed, bounded to [0, 1]."""
    total = u_calving + u_melt
    if total <= 0:
        return 0.
    return float(min(max(u_melt / total, 0.), 1.))


class _OceanCalvingLaw:
    """Base for time-varying calving laws driven by an ocean_data file.

    Sub-classes implement :meth:`frontal_speed`, returning a front-normal speed in
    m s-1. This class multiplies by the submerged area, which is the algebra of the
    stock law: ``k*d*h*w == (k*h) * (d*w)``.
    """

    name = 'ocean'

    def __init__(self, years, tf, open_water=None, q_sg=None, tf_ref=None):
        self._init_accounting()
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
        """The total front-normal speed, m s-1."""
        return sum(self.frontal_speed_components(h, d, w, yr))

    def frontal_speed_components(self, h, d, w, yr):
        """``(calving, submarine melt)`` front-normal speeds, m s-1.

        They sum to :meth:`frontal_speed`. A law with no explicit melt term puts
        everything in the first slot, which is the honest statement: its calving
        constant already contains the melt.
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
        """A calving constant in s-1, from `value` in a-1 or from the model."""
        if value is not None:
            return value / cfg.SEC_IN_YEAR
        if self._model_k is None:
            raise InvalidWorkflowError('no calving constant: pass one to the law or '
                                       'let the model provide calving_k')
        return self._model_k

    # --- the melt/calving split ------------------------------------------------
    #
    # The evolution models call the law, multiply what it returns by ``dt`` and add
    # the product to one counter. They never see the two summands and there is no
    # diagnostic hook for a second one, so the law keeps its own books. It cannot
    # see ``dt`` either -- ``self.t`` advances *after* the calving block -- so each
    # step is closed on the next call, and the final open step is closed against
    # the model's own total by :meth:`components_m3`. The split is then exact.

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
        # one record per month is enough to carry a series and bounds the memory
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

        With a model they sum exactly to its ``calving_m3_since_y0``: the step still
        open when the run stopped is closed here at that step's own melt fraction.
        """
        total, melt = self.frontal_ablation_m3, self.submarine_melt_m3
        ref = getattr(model, 'calving_m3_since_y0', None)
        if ref is not None and ref > total:
            melt += (ref - total) * self._pending_melt_fraction()
            total = ref
        return total - melt, melt

    def component_series(self, model=None):
        """Monthly cumulative ``(years, frontal ablation, submarine melt)``, m3."""
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
    """The control. Bit-identical to :func:`oggm.core.flowline.k_calving_law`.

    Written out rather than delegating so that all four laws share one geometry, and
    so the equivalence is something a test asserts rather than something assumed.
    Unlike its siblings it does *not* clip at zero, because the stock law does not.
    """

    name = 'constant'

    def __init__(self, years=None, tf=None, k=None, **kwargs):
        self._init_accounting()
        self.k = k  # a-1, or None to take the model's calving_k
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

    def frontal_speed_components(self, h, d, w, yr):
        k = self._k(self.k0)
        return k * (self._tf_at(yr) / self.tf_ref) ** self.gamma * h, 0.


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

    def __init__(self, *args, k_c=None, lam=None, water_depth=None, alpha=None,
                 beta=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.A = ocean_param('calving_melt_A')
        self.B = ocean_param('calving_melt_B')
        # per law, so a sweep need not mutate cfg.PARAMS
        self.alpha = ocean_param('calving_melt_alpha') if alpha is None else alpha
        self.beta = ocean_param('calving_melt_beta') if beta is None else beta
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

    def frontal_speed_components(self, h, d, w, yr):
        return self._k(self.k_c) * h, self.lam * self.melt_rate(d, w, yr)


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

    def frontal_speed_components(self, h, d, w, yr):
        if self.open_water is None:
            raise InvalidWorkflowError('SeaIceModulated needs open_water_frac in '
                                       'ocean_data.nc')
        f = max(self._at(self.open_water, yr), 0.)
        melt = self.lam * self.melt_rate(d, w, yr)
        return self._k(self.k_c) * h, (f ** self.delta) * melt


LAWS = {'constant': ConstantK, 'tf_power': TFPower,
        'melt_calving': MeltPlusCalving, 'sea_ice': SeaIceModulated}


def partition_law_k(gdir, law, period=None):
    """Give a melt-bearing law the residual calving constant, keeping the total.

    ``k`` was fitted against an *observed* frontal ablation, so it already contains
    the calving that submarine melt drives. A melt term added on top of it counts
    that melt twice: measured on the twelve FIIC divides with `melt_calving` on the
    200-500 m band, the total came out at 13.7 Gt yr-1 against a calibrated 0.12.

    :func:`partition_calving_constant` solves ``k_c h + lam mdot = k h`` for ``k_c``,
    so the total is unchanged and only its split moves. The reference melt rate is
    the law's own, averaged over the reference period, at the prescribed front.

    Sets ``law.k_c`` in place and returns ``(k_c, melt_fraction)``.
    """
    from oggm.core.ocean_inversion import _setting, partition_calving_constant

    if not hasattr(law, 'melt_rate'):
        return None, 0.
    k_total = _setting(gdir, 'calving_k', _setting(gdir, 'inversion_calving_k'))
    thick = _setting(gdir, 'calving_front_thick')
    depth = _setting(gdir, 'terminus_water_depth')
    width = _setting(gdir, 'calving_front_width')
    if None in (k_total, thick, depth, width):
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) partitioning needs calving_k, calving_front_thick, '
            'terminus_water_depth and calving_front_width in the settings; run the '
            'calving inversion first.')

    period = period or ocean_param('ocean_tf_ref_period')
    y0, y1 = (float(y) for y in period)
    yrs = np.asarray(law.years, dtype=float)
    sel = (yrs >= y0) & (yrs < y1 + 1)
    if not sel.any():
        raise InvalidWorkflowError(
            f'({gdir.rgi_id}) the ocean record does not cover {y0:.0f}-{y1:.0f}, so '
            'the reference melt rate cannot be formed. Give the law an explicit '
            'k_c, or pick a period the product covers.')
    mdot = float(np.mean([law.melt_rate(depth, width, y) for y in yrs[sel]]))

    k_c, frac = partition_calving_constant(k_total, mdot, thick, lam=law.lam)
    law.k_c = k_c
    return k_c, frac


def ocean_calving_law(gdir, calving_law=None, band=None, ocean_filesuffix='',
                      tf_ref=None, partition=False, **law_kwargs):
    """Build a calving law for one glacier from its ocean_data file.

    Separate from the run task so a law can be built, inspected and tested without
    running the model.

    Parameters
    ----------
    partition : bool
        for a melt-bearing law, set ``k_c`` from :func:`partition_law_k` so the melt
        term is carved out of the calibrated total rather than added to it. Off by
        default, because it changes what a law means and every run table should say
        so explicitly.
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
    law = LAWS[calving_law](
        yrs, ds['thermal_forcing'].values[:, i],
        open_water=(ds['open_water_frac'].values if 'open_water_frac' in ds
                    else None),
        q_sg=(ds['subglacial_discharge'].values if 'subglacial_discharge' in ds
              else None),
        tf_ref=tf_ref, **law_kwargs)
    if partition:
        k_c, frac = partition_law_k(gdir, law)
        if k_c is not None:
            gdir.add_to_diagnostics('calving_k_residual', float(k_c))
            gdir.add_to_diagnostics('calving_melt_fraction_reference', float(frac))
    return law


def _band_names(ds):
    """Band names, whether they are a coordinate or the char variable we write."""
    if 'band_name' not in ds:
        return [str(b) for b in ds['band'].values]
    raw = np.asarray(ds['band_name'].values)
    if raw.ndim == 2:  # netCDF char array, if xarray did not join it for us
        raw = [b''.join(r) for r in raw]
    return [(r.decode() if isinstance(r, bytes) else str(r)).strip('\x00')
            for r in raw]


def frontal_ablation_components(model):
    """``(calving, submarine melt)`` of a finished run, m3, from its model object.

    The counterpart of reading the two variables back out of the diagnostics file,
    for when the model is still in hand.
    """
    law = getattr(model, 'calving_law', None)
    if not hasattr(law, 'components_m3'):
        # the stock law, or any callable that is not one of ours: all of it is
        # frontal ablation and none of it is attributable to melt
        return float(getattr(model, 'calving_m3_since_y0', 0.)), 0.
    return law.components_m3(model)


def write_frontal_components(gdir, law, model=None, output_filesuffix=''):
    """Write the melt/calving split of a finished run to its own file on disk.

    Goes to ``frontal_ablation_diagnostics{output_filesuffix}.nc`` on the time axis of
    the run's ``model_diagnostics`` file, which is left untouched: ``calving_m3`` there
    stays the authority, and the two series here are a share of it, so they sum to
    it at every step rather than to a second, slightly different total.
    """
    import xarray as xr

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
        f = np.where(cum_total > 0, cum_melt / np.where(cum_total > 0, cum_total, 1.),
                     0.)
        f_t = np.interp(ds['time'].values.astype(float), yrs, f,
                        left=f[0], right=f[-1])

    out = xr.Dataset(coords={c: ds[c] for c in ds.coords})
    out['calving_m3'] = ds['calving_m3']
    out['submarine_melt_m3'] = ('time', cum * f_t)
    out['submarine_melt_m3'].attrs = {
        'description': 'Accumulated submarine melt part of the frontal ablation',
        'unit': 'm 3'}
    out['calving_only_m3'] = ('time', cum * (1 - f_t))
    out['calving_only_m3'].attrs = {
        'description': 'Accumulated iceberg calving part of the frontal ablation',
        'unit': 'm 3'}
    out.attrs.update({'rgi_id': gdir.rgi_id,
                      'cenlon': float(gdir.cenlon), 'cenlat': float(gdir.cenlat),
                      'calving_law': law.name,
                      'submarine_melt_fraction': frac,
                      'partition': 'modelled, not observed'})
    out.to_netcdf(gdir.get_filepath('frontal_ablation_diagnostics',
                                    filesuffix=output_filesuffix, delete=True))
    return calving, melt


def compile_frontal_components(gdirs, input_filesuffix='', path=True):
    """Stack the per-glacier split into one ``(time, rgi_id)`` file, map-ready.

    The counterpart of :func:`oggm.utils.compile_run_output` for the two variables
    that function refuses. Carries each glacier's lon/lat so the result can be
    mapped without the glacier directories.
    """
    import xarray as xr

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
            path = os.path.join(cfg.PATHS['working_dir'],
                                f'frontal_ablation_compiled{input_filesuffix}.nc')
        out.to_netcdf(path)
    return out


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

    model = run_from_climate_data(gdir, calving_law=law,
                                  climate_filename=climate_filename,
                                  climate_input_filesuffix=climate_input_filesuffix,
                                  output_filesuffix=output_filesuffix, **kwargs)
    write_frontal_components(gdir, law, model=model,
                             output_filesuffix=output_filesuffix)
    return model
