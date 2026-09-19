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


# --- the file round trip -------------------------------------------------------

class FakeGdir:
    """Only the GlacierDirectory API the ocean file writer and reader touch."""

    rgi_id = 'RGI60-05.10315'
    is_tidewater = True
    cenlon, cenlat = -17.0, 81.3

    def __init__(self, path):
        self.dir = path
        self.settings = {'task_timeout': 0}

    def get_filepath(self, name, filesuffix='', delete=False):
        fp = self.dir / (cfg.BASENAMES[name][0].replace('.nc', f'{filesuffix}.nc'))
        if delete and fp.exists():
            fp.unlink()
        return str(fp)

    def has_file(self, name, filesuffix=''):
        from pathlib import Path
        return Path(self.get_filepath(name, filesuffix=filesuffix)).exists()

    def add_to_diagnostics(self, key, value):
        pass

    # entity_task needs these three to run a task against this stand-in.
    def get_task_status(self, name):
        return None

    def log(self, name, task_time=None, err=None):
        pass

    def get_filepath_dir(self):
        return str(self.dir)


@pytest.fixture
def ocean_file(tmp_path):
    """An ocean_data.nc with a rising thermal forcing and a shrinking ice season."""
    from oggm.shop.ocean import _write_ocean_file
    import pandas as pd

    gdir = FakeGdir(tmp_path)
    time = pd.date_range('2000-01-01', '2021-12-01', freq='MS')
    n = len(time)
    tf = np.stack([np.linspace(0.2, 0.6, n),
                   np.linspace(1.0, 2.0, n),
                   np.linspace(0.5, 1.2, n)], axis=1)
    _write_ocean_file(gdir, time.values, ['terminus', 'ismip6', 'moller'],
                      [0., 200., 0.], [60., 500., 700.],
                      ['uniform', 'uniform', 'depth_weighted'],
                      tf, tf + 1.0, np.full_like(tf, 34.8),
                      siconc=np.linspace(0.9, 0.2, n),
                      open_water=open_water_fraction(np.linspace(0.9, 0.2, n)),
                      lon=-17.0, lat=81.3, terminus_depth=60.,
                      source='test', ocean_model='fake')
    return gdir


def test_ocean_file_round_trip(ocean_file):
    with xr.open_dataset(ocean_file.get_filepath('ocean_data')) as ds:
        assert ds.sizes['band'] == 3
        assert ds.sizes['time'] == 264
        assert list(ds['band_top'].values) == [0., 200., 0.]
        assert ds.attrs['yr_0'] == 2000 and ds.attrs['yr_1'] == 2021
        assert ds.attrs['ref_bathymetry_m'] == 60.
        assert ds.attrs['tf_method'] == 'linear_lambda_jenkins2011'
        assert str(ds['time'].values[0])[:7] == '2000-01'

    from oggm.core.ocean_calving import _band_names
    with xr.open_dataset(ocean_file.get_filepath('ocean_data')) as ds:
        assert _band_names(ds) == ['terminus', 'ismip6', 'moller']


def test_law_from_file_reads_the_named_band(ocean_file, state):
    from oggm.core.ocean_calving import ocean_calving_law
    model, fl, i = state
    q = {}
    for band in ('terminus', 'ismip6'):
        law = ocean_calving_law(ocean_file, calving_law='tf_power', band=band,
                                tf_ref=1.0, k0=0.6)
        q[band] = law(model, fl, i)
    # the ismip6 band is the warmer one in this file, so it must calve more
    assert q['ismip6'] > q['terminus']

    with pytest.raises(InvalidParamsError, match='band'):
        ocean_calving_law(ocean_file, calving_law='tf_power', band='not_a_band')


def test_law_from_file_is_time_varying(ocean_file, state):
    from oggm.core.ocean_calving import ocean_calving_law
    _, fl, i = state
    law = ocean_calving_law(ocean_file, calving_law='tf_power', band='terminus',
                            tf_ref=0.2, k0=0.6)
    assert law(FakeModel(yr=2020.5), fl, i) > law(FakeModel(yr=2000.5), fl, i)


def test_missing_ocean_file_raises(tmp_path, state):
    from oggm.core.ocean_calving import ocean_calving_law
    with pytest.raises(InvalidWorkflowError, match='process_ocean_data'):
        ocean_calving_law(FakeGdir(tmp_path), calving_law='tf_power')


# --- inside a real flowline model ----------------------------------------------

pytest.importorskip('geopandas')
pytest.importorskip('salem')


def _marine_model(calving_law=None, years=2500):
    """A Bassis & Ultee tidewater flowline, run with our law attached."""
    from oggm.core.flowline import FluxBasedModel
    from oggm.core.massbalance import ScalarMassBalance
    from oggm.tests.funcs import bu_tidewater_bed

    model = FluxBasedModel(bu_tidewater_bed(), mb_model=ScalarMassBalance(),
                           is_tidewater=True, do_kcalving=True,
                           calving_use_limiter=True, flux_gate=0.06,
                           calving_k=0.2, water_level=0.,
                           **({} if calving_law is None
                              else {'calving_law': calving_law}))
    ds = model.run_until_and_store(years)
    # The Bassis & Ultee front only reaches the water after ~1500 years. Without
    # this the comparisons below would all pass on two runs that never calved.
    assert model.calving_m3_since_y0 > 0
    return model, ds


@pytest.mark.slow
def test_mass_is_conserved_with_an_ocean_law():
    """Volume plus cumulative calving must equal what came through the flux gate."""
    yrs = np.arange(0, 2501, 1.)
    law = TFPower(yrs, np.full_like(yrs, 1.5), tf_ref=1.5, gamma=1.18)
    model, ds = _marine_model(calving_law=law)
    np.testing.assert_allclose(model.volume_m3 + model.calving_m3_since_y0,
                               model.flux_gate_m3_since_y0, rtol=1e-6)
    np.testing.assert_allclose(ds.calving_m3[-1], model.calving_m3_since_y0)


@pytest.mark.slow
def test_constant_law_reproduces_the_stock_run():
    """The control must be the stock model, run for run, not just call for call."""
    stock, ds_stock = _marine_model()
    ours, ds_ours = _marine_model(calving_law=ConstantK())
    np.testing.assert_allclose(ds_ours.volume_m3, ds_stock.volume_m3)
    np.testing.assert_allclose(ds_ours.calving_m3, ds_stock.calving_m3)


@pytest.mark.slow
def test_tf_power_reduces_to_the_control_in_a_run():
    """TF == TF_ref over the whole run is the stock model, including the geometry."""
    yrs = np.arange(0, 2501, 1.)
    law = TFPower(yrs, np.full_like(yrs, 1.5), tf_ref=1.5, gamma=1.18)
    stock, ds_stock = _marine_model()
    ours, ds_ours = _marine_model(calving_law=law)
    np.testing.assert_allclose(ds_ours.calving_m3, ds_stock.calving_m3, rtol=1e-8)


@pytest.mark.slow
def test_warming_ocean_calves_more():
    """A thermal forcing rising through the run must raise cumulative calving."""
    yrs = np.arange(0, 2501, 1.)
    warming = TFPower(yrs, np.linspace(1.5, 4.0, len(yrs)), tf_ref=1.5, gamma=1.18)
    steady = TFPower(yrs, np.full_like(yrs, 1.5), tf_ref=1.5, gamma=1.18)
    m_warm, ds_warm = _marine_model(calving_law=warming)
    m_flat, ds_flat = _marine_model(calving_law=steady)
    assert float(ds_warm.calving_m3[-1]) > float(ds_flat.calving_m3[-1])
    np.testing.assert_allclose(m_warm.volume_m3 + m_warm.calving_m3_since_y0,
                               m_warm.flux_gate_m3_since_y0, rtol=1e-6)


# --- the extraction side -------------------------------------------------------

def _destine_file(path, y0=1990, y1=1994, lon=343.0, months=None, bands=None,
                  nan_below=None, siconc_len=None):
    """A DestinE per-site extraction as extract_ocean_footprint.py writes it."""
    import pandas as pd

    time = months if months is not None else pd.date_range(
        f'{y0}-01-01', f'{y1}-12-01', freq='MS')
    depth = np.array([5., 20., 45., 80., 150., 250., 400., 600.])
    n = len(time)
    thetao = 274.0 + np.linspace(0, 0.5, n)[:, None] + np.zeros((n, depth.size))
    so = np.full((n, depth.size), 34.6)
    if nan_below is not None:
        thetao[:, depth > nan_below] = np.nan
    ds = xr.Dataset(
        {'thetao': (('time', 'depth'), thetao), 'so': (('time', 'depth'), so),
         'siconc': ('time', np.full(siconc_len or n, 0.5))},
        coords={'time': time, 'depth': depth, 'lon': lon, 'lat': 81.4},
    )
    ds.attrs.update(bands=bands if bands is not None else 'terminus:0:60 ismip6:200:500',
                    terminus_depth_m=60.0, n_cells=1, extraction='test footprint',
                    model='ICON')
    ds.to_netcdf(path)
    return path


@pytest.fixture
def destine_file(tmp_path):
    return _destine_file(tmp_path / 'ocean_forcing.nc')


def test_nan_in_a_band_raises():
    """A band whose levels are below the sea floor must not average to a nan column."""
    z = np.array([10., 50., 300.])
    thetao = np.array([[1., 1., np.nan]])
    so = np.full((1, 3), 34.8)
    with pytest.raises(InvalidWorkflowError, match='non-finite'):
        thermal_forcing_bands(thetao, so, z, [('ismip6', 200., 500.)],
                              {'ismip6': 'uniform'})


def test_destine_reader_writes_the_ocean_file(tmp_path, destine_file):
    from oggm.shop.ocean import process_destine_ocean_data

    gdir = FakeGdir(tmp_path)
    process_destine_ocean_data(gdir, fpath=str(destine_file))

    with xr.open_dataset(gdir.get_filepath('ocean_data')) as ds:
        assert ds.sizes['time'] == 60
        assert [b''.join(r).decode().strip() for r in ds['band_name'].values] == [
            'terminus', 'ismip6']
        assert np.isfinite(ds.thermal_forcing).all()
        assert ds.ref_pix_lon == pytest.approx(-17.0)   # 343 E, normalised
        assert ds.ref_bathymetry_m == pytest.approx(60.0)
        assert ds.ocean_model == 'ICON'


def test_destine_reader_takes_bands_from_the_file(tmp_path):
    """A band the footprint could not build is not in the file and is not requested."""
    from oggm.shop.ocean import process_destine_ocean_data

    fpath = _destine_file(tmp_path / 'shallow.nc', bands='terminus:0:60')
    gdir = FakeGdir(tmp_path)
    process_destine_ocean_data(gdir, fpath=str(fpath))

    with xr.open_dataset(gdir.get_filepath('ocean_data')) as ds:
        assert ds.sizes['band'] == 1


def test_destine_reader_trims_to_whole_years(tmp_path):
    import pandas as pd

    from oggm.shop.ocean import process_destine_ocean_data

    time = pd.date_range('1990-07-01', '1993-04-01', freq='MS')
    fpath = _destine_file(tmp_path / 'partial.nc', months=time)
    gdir = FakeGdir(tmp_path)
    process_destine_ocean_data(gdir, fpath=str(fpath))

    with xr.open_dataset(gdir.get_filepath('ocean_data')) as ds:
        assert ds.sizes['time'] == 24        # 1991 and 1992 only
        assert ds.yr_0 == 1991 and ds.yr_1 == 1992


def test_destine_reader_rejects_non_finite_input(tmp_path):
    from oggm.shop.ocean import process_destine_ocean_data

    fpath = _destine_file(tmp_path / 'wet.nc', nan_below=100.)
    with pytest.raises(InvalidWorkflowError, match='non-finite'):
        process_destine_ocean_data(FakeGdir(tmp_path), fpath=str(fpath))


def test_short_siconc_raises(tmp_path):
    """Written into an unlimited dimension, a short series reads back as a fill value."""
    import pandas as pd

    from oggm.shop.ocean import process_ocean_data

    time = pd.date_range('2000-01-01', '2001-12-01', freq='MS')
    depth = np.array([10., 50., 300.])
    coords = {'time': time, 'depth': depth, 'lon': -17.0, 'lat': 81.4}
    thetao = xr.DataArray(np.full((24, 3), 275.), dims=('time', 'depth'), coords=coords)
    so = xr.DataArray(np.full((24, 3), 34.6), dims=('time', 'depth'), coords=coords)
    siconc = xr.DataArray(np.full(20, 0.5), dims=('time',),
                          coords={'time': time[:20]})
    with pytest.raises(InvalidParamsError, match='siconc'):
        process_ocean_data(FakeGdir(tmp_path), thetao=thetao, so=so, siconc=siconc,
                           depth_bands=[('terminus', 0., 60.)])


def test_destine_reader_rejects_kwargs_it_builds(tmp_path, destine_file):
    from oggm.shop.ocean import process_destine_ocean_data

    with pytest.raises(InvalidParamsError, match='thetao'):
        process_destine_ocean_data(FakeGdir(tmp_path), fpath=str(destine_file),
                                   thetao='something')


def test_destine_reader_skips_land_terminating(tmp_path, destine_file):
    from oggm.shop.ocean import process_destine_ocean_data

    gdir = FakeGdir(tmp_path)
    gdir.is_tidewater = False
    process_destine_ocean_data(gdir, fpath=str(destine_file))
    assert not gdir.has_file('ocean_data')


def test_destine_reader_needs_a_path(tmp_path):
    from oggm.shop.ocean import process_destine_ocean_data

    cfg.PATHS.pop('destine_ocean_file', None)
    with pytest.raises(InvalidParamsError, match='destine_ocean_file'):
        process_destine_ocean_data(FakeGdir(tmp_path))
