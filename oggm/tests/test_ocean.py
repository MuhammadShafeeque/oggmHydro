"""Tests for the ocean-forced frontal ablation extension.

The two that matter most are the ones that make the four laws a nested family with
the stock model at its root: `test_tf_power_reduces_to_stock_law` and
`test_delta_zero_recovers_melt_calving`. Every comparison the paper makes is then a
comparison inside one model rather than between four.
"""
import os
import pickle

import numpy as np
import pytest
import shapely.geometry as shpg
import xarray as xr

from oggm import cfg, utils
from oggm.core.bedmachine_flowline import (bed_extension_statistics,
                                           bedmachine_calving_extension,
                                           bedmachine_terminus_bed,
                                           calving_front_width_check,
                                           calving_vs_bed_extension,
                                           extension_slice,
                                           sample_gridded_on_line)
from oggm.core.flowline import init_present_time_glacier, k_calving_law
from oggm.core.ocean_calving import (ConstantK, MeltPlusCalving, SeaIceModulated,
                                     TFPower, frontal_ablation_components,
                                     write_frontal_components)
from oggm.core.ocean_inversion import partition_calving_constant
from oggm.core.ocean_params import DEFAULTS, init_ocean_params, ocean_param
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError
from oggm.shop.bedmachine_bed import (BEDMACHINE_URLS, BEDMACHINE_VARS,
                                      DEFAULT_VERSION, bedmachine_bed_to_gdir,
                                      bedmachine_file)
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
        assert [b.decode().strip() for b in ds['band_name'].values] == ['terminus',
                                                                        'ismip6']
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


# --- the bed under and beyond the calving front --------------------------------

def _columbia(flowlines):
    """A real tidewater gdir, inverted, with a synthetic BedMachine beside it.

    Columbia rather than a dummy flowline because the two numbers this part of the
    code exists to produce -- the inversion's `calving_front_width` and the width
    `init_present_time_glacier` puts on the extension -- only exist after a real
    inversion. `clip_tidewater_border` is off because OGGM forces the grid of a
    tidewater glacier to a 10 pixel border while extending its flowline by
    `calving_line_extension * dx` pixels, so the extension leaves the grid at once.

    `flowlines` picks the geometry, and the two kinds live in separate working
    directories because they write the same files. `elev_bands` is what production
    runs on. `centerlines` is the only kind that carries a real `fl.line`, which is
    what `bedmachine_calving_extension` samples the gridded bed along.
    """
    import geopandas as gpd

    import oggm
    from oggm import tasks
    from oggm.core import gis, centerlines
    from oggm.core.ocean_params import init_ocean_params
    from oggm.tests.funcs import get_test_dir
    from oggm.utils import get_demo_file, mkdir

    testdir = os.path.join(get_test_dir(), f'tmp_bedmachine_flowline_{flowlines}')
    mkdir(testdir)

    cfg.initialize()
    init_ocean_params(reset=True)
    cfg.PATHS['working_dir'] = testdir
    cfg.PATHS['dem_file'] = get_demo_file('dem_Columbia.tif')
    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['border'] = 100
    cfg.PARAMS['clip_tidewater_border'] = False
    cfg.PARAMS['use_kcalving_for_inversion'] = True
    cfg.PARAMS['use_kcalving_for_run'] = True
    cfg.PARAMS['prcp_fac'] = 2.5
    cfg.PARAMS['baseline_climate'] = 'CRU'
    cfg.PARAMS['evolution_model'] = 'FluxBased'

    entity = gpd.read_file(get_demo_file('01_rgi60_Columbia.shp')).iloc[0]
    gdir = oggm.GlacierDirectory(entity)
    if not gdir.has_file('climate_historical'):
        gis.define_glacier_region(gdir)
        if flowlines == 'elev_bands':
            gis.simple_glacier_masks(gdir)
            centerlines.elevation_band_flowline(gdir)
            centerlines.fixed_dx_elevation_band_flowline(gdir)
        else:
            gis.glacier_masks(gdir)
            centerlines.compute_centerlines(gdir)
            centerlines.initialize_flowlines(gdir)
            centerlines.catchment_area(gdir)
            centerlines.catchment_intersections(gdir)
            centerlines.catchment_width_geom(gdir)
            centerlines.catchment_width_correction(gdir)
        centerlines.compute_downstream_line(gdir)
        if flowlines == 'centerlines':
            centerlines.compute_downstream_bedshape(gdir)
        tasks.process_dummy_cru_file(gdir, seed=0)
        tasks.mb_calibration_from_geodetic_mb(gdir)
        tasks.apparent_mb_from_any_mb(gdir)
        tasks.find_inversion_calving_from_any_mb(gdir)

    path = os.path.join(testdir, 'synthetic_bedmachine.nc')
    if not os.path.exists(path):
        write_synthetic_bedmachine(gdir, path)
    return gdir, path


@pytest.fixture
def columbia():
    """Production geometry: elevation-band flowlines, which carry no `line`."""
    return _columbia('elev_bands')


@pytest.fixture
def columbia_lines():
    """Geometrical flowlines, the only kind `bedmachine_calving_extension` accepts.

    Production does not hit that restriction: it runs on elevation bands and uses
    `bedmachine_terminus_bed`, which reads the mask instead of a line. The guard
    itself is pinned by `test_calving_extension_refuses_elevation_band_flowlines`.
    """
    return _columbia('centerlines')


def synthetic_bed(x, y):
    """A plane in the BedMachine projection, so a sampled value has a known answer."""
    return -80. + 3e-4 * (x - X_REF) - 1e-3 * (y - Y_REF)


X_REF, Y_REF = -3.2e6, 8.5e5


def write_synthetic_bedmachine(gdir, path, dx=150.):
    """A BedMachine-shaped file over this glacier: same names, same projection."""
    import pyproj

    proj = 'epsg:3413'
    x0, x1, y0, y1 = gdir.grid.extent_in_crs(proj)
    pad = 20e3
    x = np.arange(x0 - pad, x1 + pad, dx)
    y = np.arange(y1 + pad, y0 - pad, -dx)  # BedMachine's y descends
    xx, yy = np.meshgrid(x, y)
    bed = synthetic_bed(xx, yy)
    ds = xr.Dataset(
        {'bed': (('y', 'x'), bed.astype('f4'), {'units': 'meters'}),
         'errbed': (('y', 'x'), np.full(bed.shape, 42., dtype='f4'),
                    {'units': 'meters'}),
         'source': (('y', 'x'), np.where(xx > xx.mean(), 10, 2).astype('i2'),
                    {'flag_values': '2, 10',
                     'flag_meanings': 'mass_conservation multibeam'}),
         'thickness': (('y', 'x'), np.full(bed.shape, 500., dtype='f4'),
                       {'units': 'meters'}),
         'surface': (('y', 'x'), (bed + 500.).astype('f4'), {'units': 'meters'}),
         'mask': (('y', 'x'), np.full(bed.shape, 2, dtype='i2'), {})},
        coords={'x': x, 'y': y})
    ds.attrs['proj4'] = pyproj.CRS(proj).to_proj4()
    ds.to_netcdf(path)
    return path


def test_greenland_v6_is_the_default():
    """The stock shop module predates v6 (released 2025-12-11); this one does not."""
    assert DEFAULT_VERSION['05'] == '6'
    assert 'BedMachineGreenland-v6.nc' in BEDMACHINE_URLS[('05', '6')]
    assert 'BedMachineGreenland-v5.nc' in BEDMACHINE_URLS[('05', '5')]
    # Antarctica moved to v4 as well.
    assert DEFAULT_VERSION['19'] == '4'


def test_bedmachine_file_rejects_a_missing_local_file(tmp_path):
    with pytest.raises(InvalidParamsError):
        bedmachine_file(None, local_file=str(tmp_path / 'nope.nc'))


def test_bedmachine_bed_to_gdir_writes_bed_errbed_and_source(columbia):
    gdir, path = columbia
    bedmachine_bed_to_gdir(gdir, local_file=path)

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        for vn in BEDMACHINE_VARS:
            assert vn in ds
        # Categorical fields are mapped with nearest neighbour: an interpolated
        # source flag would be a number that means nothing.
        assert set(np.unique(ds['bedmachine_source'].data)) <= {2., 10.}
        np.testing.assert_allclose(ds['bedmachine_errbed'].data, 42., rtol=1e-5)
        # The bed is a plane in EPSG:3413, so the regridded values are the
        # analytic ones at the same points.
        xx, yy = np.meshgrid(np.arange(gdir.grid.nx), np.arange(gdir.grid.ny))
        x3413, y3413 = gdir.grid.ij_to_crs(xx, yy, crs='epsg:3413')
        np.testing.assert_allclose(ds['bedmachine_bed'].data,
                                   synthetic_bed(x3413, y3413), atol=1.)
        # and the thickness keeps the name and the masking of the stock task
        assert ds['bedmachine_ice_thickness'].long_name.startswith('Ice thickness')


def test_bedmachine_bed_to_gdir_rejects_unknown_variables(columbia):
    gdir, path = columbia
    with pytest.raises(InvalidParamsError):
        bedmachine_bed_to_gdir(gdir, local_file=path, add_vars=('bedrock',))


def test_extension_slice_finds_what_init_present_time_glacier_built(columbia):
    gdir, _ = columbia
    init_present_time_glacier(gdir)
    fl = gdir.read_pickle('model_flowlines')[-1]

    sl = extension_slice(gdir, fl)
    assert sl is not None
    n = gdir.settings['calving_line_extension']
    assert sl.stop - sl.start == n
    assert sl.stop == fl.nx
    # ice-free, rectangular, one width, and a linearly deepening bed
    assert np.all(fl.thick[sl] == 0)
    assert np.all(fl.is_rectangular[sl])
    assert len(np.unique(fl._w0_m[sl])) == 1
    steps = np.diff(fl.bed_h[sl])
    np.testing.assert_allclose(steps, steps[0])
    assert steps[0] < 0


def test_extension_slice_refuses_a_bed_it_did_not_build(columbia):
    """The guard against overwriting real ice, or an extension already replaced."""
    gdir, _ = columbia
    init_present_time_glacier(gdir)
    fl = gdir.read_pickle('model_flowlines')[-1]

    fl.bed_h[-5] += 10.
    assert extension_slice(gdir, fl) is None


def test_calving_extension_replaces_the_bed_and_keeps_the_synthetic(columbia_lines):
    gdir, path = columbia_lines
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)

    syn_before = gdir.read_pickle('model_flowlines')[-1].bed_h.copy()
    out = bedmachine_calving_extension(gdir)

    meas = gdir.read_pickle('model_flowlines')[-1]
    syn = gdir.read_pickle('model_flowlines', filesuffix='_synthetic')[-1]
    n = gdir.settings['calving_line_extension']

    # the synthetic copy is untouched, and everything upstream of the front is
    np.testing.assert_allclose(syn.bed_h, syn_before)
    np.testing.assert_allclose(meas.bed_h[:-n], syn.bed_h[:-n])
    assert not np.allclose(meas.bed_h[-n:], syn.bed_h[-n:])

    # the measured bed is the one in gridded_data, at the flowline's own points
    x, y = (np.asarray(c) for c in meas.line.coords.xy)
    x3413, y3413 = gdir.grid.ij_to_crs(x[-n:], y[-n:], crs='epsg:3413')
    np.testing.assert_allclose(meas.bed_h[-n:], synthetic_bed(x3413, y3413),
                               atol=2.)

    # the ice-free extension stays ice-free: only the ground under it moved
    assert np.all(meas.thick[-n:] == 0)
    np.testing.assert_allclose(meas.surface_h[-n:], meas.bed_h[-n:])
    # one entry per flowline the task edited: a tributary has no extension slice,
    # so on centerlines the edited one is the main flowline, not `fl_0`
    assert out, 'the task edited no flowline'
    edited = out[list(out)[-1]]
    assert edited['bed_extension_measured_mean'] != \
        edited['bed_extension_synthetic_mean']


def test_match_terminus_removes_the_step_at_the_junction(columbia_lines):
    """The inverted bed and the measured one need not meet, and a step is a spike."""
    gdir, path = columbia_lines
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)

    raw = bedmachine_calving_extension(gdir, match_terminus=False)
    bed_raw = gdir.read_pickle('model_flowlines')[-1].bed_h
    matched = bedmachine_calving_extension(gdir, match_terminus=True)
    bed_matched = gdir.read_pickle('model_flowlines')[-1].bed_h

    n = gdir.settings['calving_line_extension']
    # the edited flowline is the last key: on centerlines a tributary has no
    # extension slice, so it is not `fl_0`
    edited = list(raw)[-1]
    offset = raw[edited]['bed_terminus_offset']
    assert abs(offset) > 100.  # the inversion invented this water depth
    np.testing.assert_allclose(bed_matched[-n:], bed_raw[-n:] + offset)
    # matched, the first extension point continues the inverted bed
    assert abs(bed_matched[-n] - bed_matched[-n - 1]) < abs(bed_raw[-n] -
                                                            bed_raw[-n - 1])
    assert matched[edited]['bed_terminus_offset'] == offset


def test_calving_extension_reruns_from_the_synthetic_bed(columbia_lines):
    """Re-running with other options must not compound onto the first result."""
    gdir, path = columbia_lines
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)

    bedmachine_calving_extension(gdir, width_method='terminus')
    first = gdir.read_pickle('model_flowlines')[-1].bed_h.copy()
    bedmachine_calving_extension(gdir, width_method='mean5')
    second = gdir.read_pickle('model_flowlines')[-1]

    np.testing.assert_allclose(second.bed_h, first)
    n = gdir.settings['calving_line_extension']
    syn = gdir.read_pickle('model_flowlines', filesuffix='_synthetic')[-1]
    np.testing.assert_allclose(second._w0_m[-n:], syn._w0_m[-n:])


def test_calving_extension_raises_outside_the_grid(columbia):
    """OGGM clips a tidewater grid to 10 pixels; the extension is 60 pixels long."""
    gdir, path = columbia
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)

    fl = gdir.read_pickle('model_flowlines')[-1]
    with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
        nx = ds.sizes['x']
    assert np.max(fl.line.coords.xy[0]) < nx  # this fixture has room

    # the same sampling one grid width further out has none
    shifted = shpg.LineString(np.array(fl.line.coords) + [nx, 0])
    vals = sample_gridded_on_line(gdir, shifted, 'bedmachine_bed')
    assert np.all(np.isnan(vals))


def test_calving_extension_refuses_elevation_band_flowlines(columbia):
    """The guard that sends production to `bedmachine_terminus_bed` instead.

    Elevation-band flowlines carry `line=None`, and `Flowline` then invents a
    straight line along the first grid row. Sampling a gridded bed along that line
    would return values from somewhere else on the grid, silently. The `columbia`
    fixture builds exactly those flowlines, because that is what production runs on.
    """
    gdir, path = columbia
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)

    with pytest.raises(InvalidWorkflowError, match='no geometry'):
        bedmachine_calving_extension(gdir)


def test_calving_extension_refuses_a_land_terminating_glacier(columbia):
    gdir, path = columbia
    init_present_time_glacier(gdir)
    gdir.is_tidewater = False
    try:
        with pytest.raises(InvalidWorkflowError):
            bedmachine_calving_extension(gdir)
    finally:
        gdir.is_tidewater = True


def test_width_methods_pick_the_width_the_law_will_use(columbia_lines):
    gdir, path = columbia_lines
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)
    n = gdir.settings['calving_line_extension']
    w_inv = gdir.settings['calving_front_width']

    widths = {}
    for method in ('mean5', 'terminus', 'inversion'):
        out = bedmachine_calving_extension(gdir, width_method=method)
        widths[method] = gdir.read_pickle('model_flowlines')[-1]._w0_m[-n:]
        assert out[list(out)[-1]]['width_method'] == method

    np.testing.assert_allclose(widths['inversion'], w_inv)
    # the terminus cell is the inversion's own front, so those two agree exactly
    np.testing.assert_allclose(widths['terminus'], w_inv)
    # OGGM's five-cell mean does not. How far it differs is geometry-dependent:
    # on elevation bands it is several times the front width, on centerlines a few
    # per cent, so the test asserts that it is a different width and not how much.
    assert not np.allclose(widths['mean5'], w_inv)

    with pytest.raises(InvalidParamsError):
        bedmachine_calving_extension(gdir, width_method='mean10')


def test_calving_front_width_check_is_exact_at_the_front_and_not_beyond(columbia):
    """OGGM #875. Every law is proportional to this width."""
    gdir, path = columbia
    init_present_time_glacier(gdir)

    d = calving_front_width_check(gdir)
    # the run's width at the present terminus IS the inversion's, to machine
    # precision: the defect is latent, not immediate
    assert d['rel_diff_terminus'] < 1e-12
    # and it is the extension that breaks it, by a factor, not a few per cent
    assert d['rel_diff_extension'] > 1.
    assert d['passes'] is False

    with pytest.raises(InvalidWorkflowError):
        calving_front_width_check(gdir, raise_on_fail=True)


def test_calving_front_width_check_passes_once_the_bed_is_replaced(columbia_lines):
    gdir, path = columbia_lines
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)
    bedmachine_calving_extension(gdir, width_method='inversion')

    d = calving_front_width_check(gdir)
    assert d['passes'] is True
    assert d['rel_diff_extension'] < 1e-12


def test_bed_extension_statistics_reports_both_beds(columbia_lines):
    gdir, path = columbia_lines
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)
    bedmachine_calving_extension(gdir)

    d = bed_extension_statistics(gdir)
    assert d['n_extension'] == gdir.settings['calving_line_extension']
    assert d['extension_length_m'] == d['n_extension'] * d['dx_meter']
    # the synthetic bed deepens by construction; the measured one need not
    assert d['slope_synthetic'] > 0
    assert d['depth_mean_synthetic'] != d['depth_mean_measured']
    assert d['bed_rmse'] > 0


def _extension_bed_model(bed_profile, flux_gate=0.5, years=800, calving_k=0.6,
                         nx=60):
    """A marine flowline whose extension bed is prescribed, and nothing else.

    Everything upstream of the terminus is identical between calls, so a difference
    in `calving_m3` is the extension bed and only the extension bed. The flux gate
    has to be large enough to push the front past cell `nx`: while it stays in the
    inverted domain the extension is inert, which is itself the finding.
    """
    from oggm.core.flowline import FluxBasedModel, MixedBedFlowline
    from oggm.core.massbalance import ScalarMassBalance

    n_ext = len(bed_profile)
    dx_meter, map_dx = 400., 200.
    bed_h = np.concatenate([np.linspace(900., -60., nx), bed_profile])
    thick = np.zeros(nx + n_ext)
    thick[:nx] = np.linspace(20., 260., nx)
    widths_m = np.full(nx + n_ext, 1000.)
    fl = MixedBedFlowline(dx=dx_meter / map_dx, map_dx=map_dx,
                          surface_h=bed_h + thick, bed_h=bed_h,
                          section=widths_m * thick,
                          bed_shape=np.zeros(nx + n_ext),
                          is_trapezoid=np.ones(nx + n_ext, dtype=bool),
                          lambdas=np.zeros(nx + n_ext), widths_m=widths_m)
    model = FluxBasedModel([fl], mb_model=ScalarMassBalance(),
                           is_tidewater=True, do_kcalving=True,
                           calving_use_limiter=True, flux_gate=flux_gate,
                           calving_k=calving_k, water_level=0.)
    model.run_until(years)
    return model, int(np.nonzero(model.fls[0].thick > 0)[0][-1])


@pytest.mark.slow
def test_a_deepening_extension_calves_more_than_a_measured_one():
    """The first-order consequence: Q is proportional to d, and d is invented.

    OGGM's extension deepens at `calving_front_slope` for 30 cells. A measured bed
    that holds the terminus depth, or shoals onto a sill, is a different run: less
    frontal ablation, and a front that gets further.
    """
    n, dx, nx = 30, 400., 60
    deepening = np.linspace(-60., -60. - n * dx * 0.05, n)   # what OGGM builds
    flat = np.full(n, -60.)                                  # a measured shelf
    sill = np.linspace(-60., 5., n)                          # a measured sill

    m_deep, front_deep = _extension_bed_model(deepening)
    m_flat, front_flat = _extension_bed_model(flat)
    m_sill, front_sill = _extension_bed_model(sill)

    # all three have to have reached the extension, or this compares nothing
    for front in (front_deep, front_flat, front_sill):
        assert front > nx
    assert m_deep.calving_m3_since_y0 > 0

    # the invented bed calves more, and by a factor rather than a few per cent
    assert m_deep.calving_m3_since_y0 / m_flat.calving_m3_since_y0 > 1.5
    assert m_flat.calving_m3_since_y0 > m_sill.calving_m3_since_y0
    # and it holds the front back, and the glacier smaller, for the same reason
    assert front_deep < front_flat < front_sill
    assert m_deep.volume_m3 < m_flat.volume_m3 < m_sill.volume_m3


@pytest.mark.slow
def test_calving_vs_bed_extension_compares_two_runs_of_one_glacier(columbia):
    """Two runs differing only in the bed at and beyond the front.

    On the production geometry, elevation bands, and through the production task:
    `bedmachine_terminus_bed` reads the mask rather than a line, so it is the one
    the FIIC runs use. Columbia on centerlines blows the CFL limit on a tributary
    (max_u of order 1e7 m yr-1 at fl_id 21), which is a property of the demo glacier
    rather than of the bed code, so the comparison is made here where it is stable.
    """
    gdir, path = columbia
    bedmachine_bed_to_gdir(gdir, local_file=path)
    init_present_time_glacier(gdir)
    bedmachine_terminus_bed(gdir, water_depth=200.)

    d = calving_vs_bed_extension(gdir, ys=1950, ye=2000)
    assert d['calving_m3_synthetic'] > 0
    assert d['calving_m3_measured'] > 0
    assert np.isfinite(d['calving_ratio'])


# --- the melt/calving split ----------------------------------------------------
#
# The model adds the two summands and keeps one number, so the laws keep their own
# books. These tests hold that bookkeeping to three promises: the parts sum to the
# whole, the nesting survives at the component level, and only the laws that have a
# melt term report one.

def _laws(years):
    tf = np.full_like(years, 1.8)
    ow = np.full_like(years, 0.5)
    return [ConstantK(k=0.6),
            TFPower(years, tf, tf_ref=1.5, k0=0.6),
            MeltPlusCalving(years, tf, tf_ref=1.5, k_c=0.6),
            SeaIceModulated(years, tf, open_water=ow, tf_ref=1.5, k_ice=0.6)]


def test_components_sum_to_the_flux(state, years):
    """Whatever the law splits, the total it returns must not change."""
    model, fl, i = state
    for law in _laws(years):
        h, w = fl.thick[i], fl.widths_m[i]
        d = h - (fl.surface_h[i] - model.water_level)
        u_c, u_m = law.frontal_speed_components(h, d, w, model.yr)
        assert u_c + u_m == pytest.approx(law.frontal_speed(h, d, w, model.yr))
        assert (u_c + u_m) * d * w == pytest.approx(law(model, fl, i))


def test_only_the_melt_laws_report_melt(state, years):
    """The control and the power law put everything in the calving slot -- their
    constant already contains the melt, and pretending otherwise would be a claim."""
    model, fl, i = state
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    for law in _laws(years)[:2]:
        assert law.frontal_speed_components(h, d, w, model.yr)[1] == 0.
    for law in _laws(years)[2:]:
        assert law.frontal_speed_components(h, d, w, model.yr)[1] > 0.


def test_delta_zero_recovers_the_split_not_only_the_total(state, years):
    """The nesting property has to hold component by component, or the split is a
    different model rather than a report on the same one."""
    model, fl, i = state
    tf = np.full_like(years, 1.8)
    ow = np.linspace(0.1, 0.9, len(years))
    a = MeltPlusCalving(years, tf, open_water=ow, tf_ref=1.5, k_c=0.6)
    b = SeaIceModulated(years, tf, open_water=ow, tf_ref=1.5, k_ice=0.6, delta=0.)
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    assert (b.frontal_speed_components(h, d, w, 2005.) ==
            pytest.approx(a.frontal_speed_components(h, d, w, 2005.)))


def test_the_sea_ice_gate_moves_the_split(state, years):
    """Closing the ice season must move mass from the melt term to the calving term
    without changing what the calving term itself is."""
    model, fl, i = state
    tf = np.full_like(years, 1.8)
    h, w = fl.thick[i], fl.widths_m[i]
    d = h - (fl.surface_h[i] - model.water_level)
    open_sea = SeaIceModulated(years, tf, open_water=np.full_like(years, 0.9),
                               tf_ref=1.5, k_ice=0.6, delta=2.)
    icy = SeaIceModulated(years, tf, open_water=np.full_like(years, 0.1),
                          tf_ref=1.5, k_ice=0.6, delta=2.)
    c_open, m_open = open_sea.frontal_speed_components(h, d, w, 2005.)
    c_icy, m_icy = icy.frontal_speed_components(h, d, w, 2005.)
    assert m_icy < m_open
    assert c_icy == pytest.approx(c_open)


def test_a_bare_call_does_not_accumulate(state, years):
    """A law called without a model clock is an inspection, not a run."""
    model, fl, i = state
    law = MeltPlusCalving(years, np.full_like(years, 1.8), tf_ref=1.5, k_c=0.6)
    law(model, fl, i)
    assert law.frontal_ablation_m3 == 0.
    assert law.components_m3() == (0., 0.)


def test_accounting_survives_pickle(state, years):
    law = MeltPlusCalving(years, np.full_like(years, 1.8), tf_ref=1.5, k_c=0.6)
    law.frontal_ablation_m3, law.submarine_melt_m3 = 10., 3.
    again = pickle.loads(pickle.dumps(law))
    assert again.components_m3() == (7., 3.)


def test_partition_calving_constant_preserves_the_total():
    """k_c + the melt term must reproduce exactly what the calibrated k gave."""
    h, k = 200., 0.6
    mdot = 0.1 / 86400.  # 0.1 m per day, well inside the calibrated total
    k_c, frac = partition_calving_constant(k, mdot, h, lam=1.)
    assert k_c < k
    assert (k_c / cfg.SEC_IN_YEAR * h + mdot ==
            pytest.approx(k / cfg.SEC_IN_YEAR * h))
    assert frac == pytest.approx(mdot / (k / cfg.SEC_IN_YEAR * h))


def test_partition_calving_constant_refuses_to_go_negative():
    """Melt exceeding the observed total is a result to report, not a negative k."""
    k_c, frac = partition_calving_constant(0.01, 1.0, 200., lam=1.)
    assert (k_c, frac) == (0., 1.)
    with pytest.raises(InvalidParamsError, match='thickness'):
        partition_calving_constant(0.6, 1e-7, 0.)


@pytest.mark.slow
def test_the_split_sums_to_calving_m3_in_a_run():
    """The two components must close against the model's own counter, including the
    step that was still open when the run stopped."""
    yrs = np.arange(0, 2501, 1.)
    law = MeltPlusCalving(yrs, np.full_like(yrs, 1.8), tf_ref=1.5, k_c=0.2)
    model, ds = _marine_model(calving_law=law)
    calving, melt = frontal_ablation_components(model)
    assert melt > 0
    np.testing.assert_allclose(calving + melt, model.calving_m3_since_y0, rtol=1e-12)
    np.testing.assert_allclose(calving + melt, float(ds.calving_m3[-1]), rtol=1e-12)


@pytest.mark.slow
def test_the_control_reports_no_melt_in_a_run():
    law = ConstantK()
    model, ds = _marine_model(calving_law=law)
    calving, melt = frontal_ablation_components(model)
    assert melt == 0.
    np.testing.assert_allclose(calving, model.calving_m3_since_y0, rtol=1e-12)


@pytest.mark.slow
def test_the_split_series_is_monotone_and_closes_at_every_step():
    yrs = np.arange(0, 2501, 1.)
    law = MeltPlusCalving(yrs, np.full_like(yrs, 1.8), tf_ref=1.5, k_c=0.2)
    model, ds = _marine_model(calving_law=law)
    t, cum_total, cum_melt = law.component_series(model)
    assert len(t) > 100
    assert np.all(np.diff(cum_total) >= 0)
    assert np.all(np.diff(cum_melt) >= 0)
    assert np.all(cum_melt <= cum_total)


@pytest.mark.slow
def test_write_frontal_components_writes_a_mappable_sidecar(tmp_path):
    """The split must be on disk, a share of calving_m3 at every step, and must not
    touch model_diagnostics, which compile_run_output would then refuse."""
    from oggm.core.ocean_calving import compile_frontal_components

    yrs = np.arange(0, 2501, 1.)
    law = MeltPlusCalving(yrs, np.full_like(yrs, 1.8), tf_ref=1.5, k_c=0.2)
    model, ds = _marine_model(calving_law=law)

    gdir = FakeGdir(tmp_path)
    fp = gdir.get_filepath('model_diagnostics')
    ds.to_netcdf(fp)
    calving, melt = write_frontal_components(gdir, law, model=model)

    with xr.open_dataset(fp) as stock:
        assert 'submarine_melt_m3' not in stock
    with xr.open_dataset(gdir.get_filepath('frontal_ablation_diagnostics')) as out:
        np.testing.assert_allclose(out.submarine_melt_m3 + out.calving_only_m3,
                                   out.calving_m3, rtol=1e-12)
        assert float(out.submarine_melt_m3[-1]) == pytest.approx(melt, rel=1e-9)
        assert float(out.calving_only_m3[-1]) == pytest.approx(calving, rel=1e-9)
        assert np.all(out.submarine_melt_m3.values >= 0)
        assert out.attrs['calving_law'] == 'melt_calving'

    comp = compile_frontal_components([gdir], path=str(tmp_path / 'c.nc'))
    assert comp.sizes['rgi_id'] == 1
    assert float(comp.lon[0]) == FakeGdir.cenlon
    assert (tmp_path / 'c.nc').exists()


def test_write_frontal_components_without_a_file(tmp_path, state, years):
    """No diagnostics file is a missing output, not an error: the totals still go to
    the glacier's own diagnostics."""
    law = MeltPlusCalving(years, np.full_like(years, 1.8), tf_ref=1.5, k_c=0.6)
    assert write_frontal_components(FakeGdir(tmp_path), law) == (0., 0.)


def test_beta_is_a_law_parameter(state, years):
    """The low-discharge branch (beta = 1.61) has to be reachable per run, without
    touching the global parameters another glacier in the same worker reads."""
    tf = 1.8
    law = MeltPlusCalving(years, np.full_like(years, tf), tf_ref=1.5, k_c=0.6,
                          beta=1.61)
    assert law.beta == 1.61
    assert ocean_param('calving_melt_beta') == 1.18
    assert law.melt_rate(100., 1000., 2005.) == pytest.approx(
        ocean_param('calving_melt_B') * tf ** 1.61 / 86400.)
    ice = SeaIceModulated(years, np.full_like(years, tf), open_water=np.ones_like(years),
                          tf_ref=1.5, k_ice=0.6, beta=1.61)
    assert ice.beta == 1.61
