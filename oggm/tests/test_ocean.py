"""Tests for the ocean-forced frontal ablation extension.

The two that matter most are the ones that make the four laws a nested family with
the stock model at its root: `test_tf_power_reduces_to_stock_law` and
`test_delta_zero_recovers_melt_calving`. Every comparison the paper makes is then a
comparison inside one model rather than between four.
"""
import pickle

import numpy as np
import pytest
import xarray as xr

from oggm import cfg, utils
from oggm.core.flowline import k_calving_law
from oggm.core.ocean_calving import (ConstantK, MeltPlusCalving, SeaIceModulated,
                                     TFPower)
from oggm.core.ocean_params import DEFAULTS, init_ocean_params, ocean_param
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.shop.ocean import (LAMBDA1, LAMBDA2, LAMBDA3, freezing_point,
                             open_water_fraction, thermal_forcing_bands)

pytestmark = pytest.mark.test_env("models_dynamics")


class FakeFlowline:
    """The three arrays a calving law reads, and nothing else."""

    def __init__(self, thick=200., surface_h=100., width=1000.):
        self.thick = np.array([thick])
        self.surface_h = np.array([surface_h])
        self.widths_m = np.array([width])


class FakeModel:
    def __init__(self, calving_k=0.6, water_level=0., yr=2005.):
        self.calving_k = calving_k / cfg.SEC_IN_YEAR  # as the real models store it
        self.water_level = water_level
        self.yr = yr


@pytest.fixture(autouse=True)
def ocean_params():
    cfg.initialize_minimal()
    init_ocean_params(reset=True)
    yield


@pytest.fixture
def state():
    return FakeModel(), FakeFlowline(), 0


@pytest.fixture
def years():
    return np.arange(2000, 2022, 1 / 12)


# --- the data side -------------------------------------------------------------

def test_freezing_point_matches_slater_form():
    s, z = 34.8, 300.
    assert freezing_point(s, z) == pytest.approx(LAMBDA1 * s + LAMBDA2 - LAMBDA3 * z)
    # colder with depth, and colder with salinity
    assert freezing_point(34.8, 500.) < freezing_point(34.8, 100.)
    assert freezing_point(35.5, 300.) < freezing_point(33.0, 300.)


def test_open_water_fraction_is_bounded():
    owf = open_water_fraction([0., 0.05, 0.15, 0.9, 1.])
    assert owf[0] == 1.
    assert owf[-1] == 0.
    assert np.all((owf >= 0) & (owf <= 1))
    assert np.all(np.diff(owf) <= 0)


def test_terminus_band_follows_bathymetry():
    """A measured terminus depth replaces the band's default bottom."""
    z = np.arange(5., 705., 10.)
    t = np.full((12, len(z)), 1.5)
    s = np.full_like(t, 34.8)
    names, tops, bots, _, tf, _, _ = thermal_forcing_bands(
        t, s, z, DEFAULTS['ocean_depth_bands'], DEFAULTS['ocean_band_weighting'],
        terminus_depth=60.)
    assert names[0] == 'terminus'
    assert bots[0] == 60.
    assert tf.shape == (12, 3)


def test_empty_band_raises():
    """The 200-500 m band at a site with 60 m of water must not return nan."""
    z = np.arange(5., 65., 10.)  # nothing below 60 m
    t = np.full((12, len(z)), 1.5)
    with pytest.raises(InvalidWorkflowError, match='no ocean levels'):
        thermal_forcing_bands(t, np.full_like(t, 34.8), z,
                              DEFAULTS['ocean_depth_bands'],
                              DEFAULTS['ocean_band_weighting'])


def test_band_weighting_differs():
    """Depth weighting and uniform weighting are not the same number."""
    z = np.concatenate([np.arange(5., 100., 5.), np.arange(100., 705., 50.)])
    t = np.tile(np.linspace(-1.5, 2.5, len(z)), (12, 1))
    s = np.full_like(t, 34.8)
    bands = [('moller', 0., 700.)]
    _, _, _, _, tf_u, _, _ = thermal_forcing_bands(t, s, z, bands,
                                                   {'moller': 'uniform'})
    _, _, _, _, tf_d, _, _ = thermal_forcing_bands(t, s, z, bands,
                                                   {'moller': 'depth_weighted'})
    assert not np.isclose(tf_u[0, 0], tf_d[0, 0])


def test_unknown_weighting_raises():
    z = np.arange(5., 705., 10.)
    t = np.full((12, len(z)), 1.5)
    with pytest.raises(InvalidParamsError, match='weighting'):
        thermal_forcing_bands(t, np.full_like(t, 34.8), z, [('b', 0., 700.)],
                              {'b': 'inverse_distance'})


# --- the laws ------------------------------------------------------------------

def test_constant_k_is_the_stock_law(state):
    model, fl, i = state
    assert ConstantK()(model, fl, i) == k_calving_law(model, fl, i)
    # and an explicit k in a-1 must be converted, not used raw
    assert ConstantK(k=0.6)(model, fl, i) == pytest.approx(k_calving_law(model, fl, i))


def test_tf_power_reduces_to_stock_law(state, years):
    """TF == TF_ref for all t  =>  identical to the stock law. The nesting check."""
    model, fl, i = state
    law = TFPower(years, np.full_like(years, 1.5), tf_ref=1.5, gamma=1.18)
    assert law(model, fl, i) == pytest.approx(k_calving_law(model, fl, i), rel=1e-12)


def test_units_are_m3_per_second(state, years):
    """k in a-1, d, h, w in m: the flux must be m3 s-1, not m3 a-1."""
    model, fl, i = state
    q = ConstantK(k=0.6)(model, fl, i)
    h, w = 200., 1000.
    d = h - (100. - 0.)
    assert q == pytest.approx(0.6 / cfg.SEC_IN_YEAR * d * h * w)
    assert q * cfg.SEC_IN_YEAR == pytest.approx(0.6 * d * h * w)


def test_negative_tf_never_nan(state, years):
    """A freezing-point ocean gives zero, not a NaN from a fractional power."""
    model, fl, i = state
    law = TFPower(years, np.full_like(years, -1.8), tf_ref=1.5, gamma=1.18)
    q = law(model, fl, i)
    assert np.isfinite(q)
    assert q == 0.


def test_tf_ref_must_be_positive(years):
    with pytest.raises(InvalidParamsError, match='TF_ref'):
        TFPower(years, np.full_like(years, 0.05), tf_ref=0.)
    with pytest.raises(InvalidParamsError, match='TF_ref'):
        TFPower(years, np.full_like(years, -0.2))  # mean is negative


def test_monotone_in_tf(state, years):
    """gamma > 0 => the flux is non-decreasing in TF. The physical contract."""
    model, fl, i = state
    q = [TFPower(years, np.full_like(years, tf), tf_ref=1.5, gamma=1.18)(model, fl, i)
         for tf in [0., 0.5, 1.0, 1.5, 2.0, 3.0]]
    assert np.all(np.diff(q) >= 0)


def test_gamma_zero_is_the_control(state, years):
    model, fl, i = state
    law = TFPower(years, np.full_like(years, 4.2), tf_ref=1.5, gamma=0.)
    assert law(model, fl, i) == pytest.approx(k_calving_law(model, fl, i))


def test_no_extrapolation_past_the_forcing(state):
    """Running past the end of the forcing freezes it instead of trending off."""
    yrs = np.array([2000., 2001., 2002.])
    law = TFPower(yrs, np.array([1.0, 2.0, 3.0]), tf_ref=1.5)
    assert law._at(law.tf, 1990.) == 1.0
    assert law._at(law.tf, 2050.) == 3.0


def test_melt_calving_reduces_to_rignot_b_limit(state, years):
    """With no subglacial discharge the melt term is exactly B * TF**beta."""
    model, fl, i = state
    tf = 1.8
    law = MeltPlusCalving(years, np.full_like(years, tf), tf_ref=1.5, k_c=0.6)
    expected = (ocean_param('calving_melt_B') * tf ** ocean_param('calving_melt_beta')
                / 86400.)
    assert law.melt_rate(100., 1000., 2005.) == pytest.approx(expected)


def test_melt_calving_adds_to_the_control(state, years):
    """Law (iii) with k_c = k is the control plus a strictly positive melt term."""
    model, fl, i = state
    law = MeltPlusCalving(years, np.full_like(years, 1.5), tf_ref=1.5, k_c=0.6)
    assert law(model, fl, i) > k_calving_law(model, fl, i)


def test_delta_zero_recovers_melt_calving(state, years):
    """SeaIceModulated(delta=0) == MeltPlusCalving, exactly. The nesting check."""
    model, fl, i = state
    tf = np.full_like(years, 1.8)
    ow = np.linspace(0.1, 0.9, len(years))
    a = MeltPlusCalving(years, tf, open_water=ow, tf_ref=1.5, k_c=0.6)
    b = SeaIceModulated(years, tf, open_water=ow, tf_ref=1.5, k_ice=0.6, delta=0.)
    assert b(model, fl, i) == pytest.approx(a(model, fl, i), rel=1e-12)


def test_sea_ice_needs_open_water(state, years):
    model, fl, i = state
    law = SeaIceModulated(years, np.full_like(years, 1.8), tf_ref=1.5, k_ice=0.6)
    with pytest.raises(InvalidWorkflowError, match='open_water'):
        law(model, fl, i)


def test_sea_ice_can_fall_while_tf_rises(state, years):
    """The whole point of law (iv): a falling open-water fraction can outweigh a
    rising thermal forcing, which law (ii) cannot do with gamma > 0."""
    model, fl, i = state
    tf = np.linspace(1.0, 2.0, len(years))
    ow = np.linspace(0.9, 0.1, len(years))
    law = SeaIceModulated(years, tf, open_water=ow, tf_ref=1.5, k_ice=0., delta=2.)
    early = law(FakeModel(yr=2001.), fl, i)
    late = law(FakeModel(yr=2020.), fl, i)
    assert late < early

    rising = TFPower(years, tf, tf_ref=1.5, k0=0.6, gamma=1.18)
    assert rising(FakeModel(yr=2020.), fl, i) > rising(FakeModel(yr=2001.), fl, i)


def test_no_calving_above_water(years):
    """A front sitting above the water level gives zero, not a negative flux."""
    fl = FakeFlowline(thick=50., surface_h=100.)  # d = 50 - 100 < 0
    law = TFPower(years, np.full_like(years, 1.5), tf_ref=1.5)
    assert law(FakeModel(), fl, 0) == 0.


def test_laws_are_picklable(state, years):
    """execute_entity_task pickles kwargs to workers; someone will pass a law."""
    model, fl, i = state
    for law in (ConstantK(k=0.6),
                TFPower(years, np.full_like(years, 1.5), tf_ref=1.5),
                MeltPlusCalving(years, np.full_like(years, 1.5), tf_ref=1.5, k_c=0.6),
                SeaIceModulated(years, np.full_like(years, 1.5),
                                open_water=np.full_like(years, 0.5),
                                tf_ref=1.5, k_ice=0.6)):
        again = pickle.loads(pickle.dumps(law))
        assert again(model, fl, i) == law(model, fl, i)


def test_ocean_param_falls_back_to_defaults():
    del cfg.PARAMS['calving_tf_exponent']
    assert ocean_param('calving_tf_exponent') == DEFAULTS['calving_tf_exponent']
    with pytest.raises(InvalidParamsError):
        ocean_param('not_an_ocean_param')
