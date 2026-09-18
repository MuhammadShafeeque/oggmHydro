"""Ocean boundary conditions at the glacier terminus, for frontal ablation.

Writes one ``ocean_data.nc`` per glacier directory: thermal forcing in several depth
bands, sea-ice concentration and open-water fraction, and optionally subglacial
discharge. Deliberately mirrors :func:`oggm.shop.gcm_climate.process_gcm_data` so the
calendar contract, the longitude convention and the provenance attributes match.

Three bands live in one file rather than one band per file, so that comparing the
ISMIP6 200-500 m average against a band matched to the real terminus depth is a
selection rather than a re-extraction.
"""
import logging

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as ncDataset
from netCDF4 import date2num

from oggm import cfg
from oggm import entity_task
from oggm import utils
from oggm.core.ocean_params import ocean_param
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

log = logging.getLogger(__name__)

cfg.add_to_basenames(
    'ocean_data', 'ocean_data.nc',
    'Monthly ocean boundary conditions at the glacier terminus: thermal forcing '
    'in one or more depth bands, sea-ice concentration, and optionally '
    'subglacial discharge.')

# Jenkins (2011) Table 2, as used by Slater et al. (2020) Eq. (6).
LAMBDA1, LAMBDA2, LAMBDA3 = -5.73e-2, 8.32e-2, 7.61e-4  # degC psu-1, degC, degC m-1


def freezing_point(salinity, depth, teos10=False, lat=81.3):
    """Freezing point of sea water, degC. `depth` is positive downwards, in metres.

    The linear form is Slater et al. (2020) Eq. (6). Their z is negative downwards,
    so with depth positive the third term lowers the freezing point with depth.
    `teos10` switches to the GSW conservative-temperature freezing point, which is
    the convention Möller et al. (2024) used.
    """
    depth = np.asarray(depth, dtype=float)
    if teos10:
        import gsw
        p = gsw.p_from_z(-depth, lat)
        return gsw.CT_freezing(salinity, p, 0)
    return LAMBDA1 * np.asarray(salinity) + LAMBDA2 - LAMBDA3 * depth


def open_water_fraction(siconc, threshold=None):
    """Open-water fraction from sea-ice concentration, clipped to [0, 1].

    From monthly means this is an area proxy; from daily data it is the real
    fraction of time below `threshold`.
    """
    thr = ocean_param('ocean_open_water_threshold') if threshold is None else threshold
    return np.clip((thr - np.asarray(siconc, dtype=float)) / thr, 0, 1)


def thermal_forcing_bands(thetao, so, depth, depth_bands, band_weighting,
                          terminus_depth=None, teos10=False, lat=81.3):
    """Band-average thermal forcing, temperature and salinity.

    Parameters
    ----------
    thetao, so : ndarray
        (time, depth) potential temperature in degC and salinity in psu.
    depth : ndarray
        depth coordinate in metres, positive down.
    depth_bands : list of (name, top, bottom)
    band_weighting : dict
        band name -> 'uniform' or 'depth_weighted'.
    terminus_depth : float, optional
        replaces the bottom of the 'terminus' band, so the shallow band follows the
        real bathymetry instead of a global constant.

    Returns
    -------
    names, tops, bottoms, weightings, tf, thetao_b, so_b -- the last three with
    shape (time, band).
    """
    z = np.asarray(depth, dtype=float)
    names, tops, bots, weights = [], [], [], []
    tf_b, th_b, so_b = [], [], []

    for name, top, bot in depth_bands:
        if name == 'terminus' and terminus_depth is not None:
            bot = float(terminus_depth)
        sel = (z >= top) & (z <= bot)
        if sel.sum() == 0:
            # A band with no model levels is an error, not a nan column: silently
            # averaging over zero levels is how a 200-500 m thermal forcing reaches
            # a manuscript at a site with 60 m of water.
            raise InvalidWorkflowError(
                f'band {name} [{top}, {bot}] m contains no ocean levels; '
                f'available depths are {z.min():.0f}-{z.max():.0f} m')

        how = band_weighting.get(name, 'uniform')
        if how == 'depth_weighted':
            w = np.gradient(z)[sel]
            w = w / w.sum()
        elif how == 'uniform':
            w = np.full(int(sel.sum()), 1 / sel.sum())
        else:
            raise InvalidParamsError(f'unknown band weighting {how!r}')

        zc = z[sel]
        tf_z = thetao[:, sel] - freezing_point(so[:, sel], zc, teos10=teos10, lat=lat)
        tf_b.append(tf_z @ w)
        th_b.append(thetao[:, sel] @ w)
        so_b.append(so[:, sel] @ w)
        names.append(name)
        tops.append(float(top))
        bots.append(float(bot))
        weights.append(how)

    return (names, tops, bots, weights,
            np.array(tf_b).T, np.array(th_b).T, np.array(so_b).T)


@entity_task(log, writes=['ocean_data'])
def process_ocean_data(gdir, thetao=None, so=None, siconc=None,
                       subglacial_discharge=None,
                       depth_bands=None, band_weighting=None,
                       terminus_depth=None,
                       year_range=None,
                       apply_bias_correction=None, ref_ocean=None,
                       teos10=None, source='', ocean_model='',
                       extraction='', n_cells=None, search_radius_km=None,
                       terminus_depth_source='',
                       output_filesuffix=''):
    """Write the ocean boundary conditions for one glacier directory.

    Parameters
    ----------
    thetao, so : xarray.DataArray
        monthly sea-water potential temperature (K or degC) and salinity (psu),
        with a `depth` coordinate in metres positive down and scalar `lon`/`lat`,
        i.e. already reduced over the extraction footprint.
    siconc : xarray.DataArray, optional
        monthly sea-ice area fraction, 0-1, scalar in space.
    subglacial_discharge : xarray.DataArray, optional
        monthly subglacial discharge in m3 s-1. Normally None on a first pass and
        filled from a `run_with_hydro` pass afterwards.
    depth_bands : list of (name, top, bottom)
        defaults to `cfg.PARAMS['ocean_depth_bands']`.
    band_weighting : dict
        band name -> 'uniform' or 'depth_weighted'.
    terminus_depth : float, optional
        measured or inverted water depth at this terminus, metres, positive.
    year_range : tuple of str
        reference period for the bias correction and for TF_ref.
    apply_bias_correction : bool
        add a time- and depth-independent offset per band so the `year_range` mean
        matches `ref_ocean` (Slater et al. 2020 Eq. 5). Default False: the offset is
        computed and written to the file, but not applied.
    ref_ocean : xarray.Dataset, optional
        observational reference with a `thermal_forcing(time, band)` variable.
    teos10 : bool
        use the GSW freezing point rather than the linear form.
    source, ocean_model, extraction, terminus_depth_source : str
        provenance, written as global attributes.
    """
    if thetao is None:
        raise InvalidParamsError('process_ocean_data needs thetao')

    # Contracts, copied from process_gcm_data.
    months = thetao['time.month']
    if months[0] != 1:
        raise InvalidParamsError('We expect the files to start in January!')
    if months[-1] != 12:
        raise InvalidParamsError('We expect the files to end in December!')
    if np.abs(float(thetao['lon'])) > 180:
        raise InvalidParamsError('We expect the longitude coordinates to be '
                                 'within [-180, 180].')
    if len(thetao['time']) % 12:
        raise InvalidParamsError("Somehow we didn't get full years")
    if so is not None and not np.array_equal(so['time'].values,
                                             thetao['time'].values):
        raise InvalidParamsError('thetao and so must share a time axis.')

    if not gdir.is_tidewater:
        # Not an error: land-terminating divides simply get no ocean file, and every
        # consumer checks has_file('ocean_data') first.
        log.warning('(%s) not tidewater, no ocean data written', gdir.rgi_id)
        return

    teos10 = ocean_param('ocean_use_teos10') if teos10 is None else teos10
    year_range = year_range or ocean_param('ocean_tf_ref_period')
    if apply_bias_correction is None:
        apply_bias_correction = ocean_param('ocean_bias_correct')
    depth_bands = depth_bands or ocean_param('ocean_depth_bands')
    band_weighting = dict(band_weighting or ocean_param('ocean_band_weighting'))
    n_cells = ocean_param('ocean_n_cells') if n_cells is None else n_cells
    if search_radius_km is None:
        search_radius_km = ocean_param('ocean_search_radius_km')

    t = np.asarray(thetao.values, dtype=float)
    if np.nanmean(t) > 100:  # some archives ship potential temperature in K
        t = t - 273.15
    s = (np.asarray(so.values, dtype=float) if so is not None
         else np.full_like(t, 34.8))

    lat = float(thetao['lat'])
    names, tops, bots, weights, tf, th, sa = thermal_forcing_bands(
        t, s, thetao['depth'].values, depth_bands, band_weighting,
        terminus_depth=terminus_depth, teos10=teos10, lat=lat)

    offsets = np.zeros(len(names))
    if ref_ocean is not None:
        ref = ref_ocean.sel(time=slice(*year_range)).thermal_forcing.mean('time')
        mod = xr.DataArray(
            tf, dims=('time', 'band'),
            coords={'time': thetao['time'].values, 'band': names},
        ).sel(time=slice(*year_range)).mean('time')
        offsets = np.asarray(ref.values - mod.values, dtype=float)
        if apply_bias_correction:
            tf = tf + offsets

    sic = None if siconc is None else np.clip(np.asarray(siconc.values, float), 0, 1)
    owf = None if sic is None else open_water_fraction(sic)
    q_sg = (None if subglacial_discharge is None
            else np.asarray(subglacial_discharge.values, dtype=float))

    _write_ocean_file(gdir, thetao['time'].values, names, tops, bots, weights,
                      tf, th, sa, sic, owf, q_sg=q_sg,
                      lon=float(thetao['lon']), lat=lat,
                      terminus_depth=terminus_depth,
                      terminus_depth_source=terminus_depth_source,
                      source=source, ocean_model=ocean_model,
                      extraction=extraction, n_cells=n_cells,
                      search_radius_km=search_radius_km,
                      offsets=offsets, teos10=teos10,
                      bias_applied=bool(apply_bias_correction),
                      filesuffix=output_filesuffix)


def _write_ocean_file(gdir, time, names, tops, bots, weights, tf, th, sa,
                      siconc, open_water, q_sg=None, lon=None, lat=None,
                      terminus_depth=None, terminus_depth_source='',
                      source='', ocean_model='', extraction='', n_cells=None,
                      search_radius_km=None, offsets=None, teos10=False,
                      bias_applied=False, filesuffix=''):
    """Write ocean_data.nc, following write_monthly_climate_file's conventions."""
    time = np.asarray(time)
    y0 = int(str(time[0])[:4])
    y1 = int(str(time[-1])[:4])
    time_unit = ('days since 1801-01-01 00:00:00' if y0 > 1800
                 else 'days since 0001-01-01 00:00:00')

    fpath = gdir.get_filepath('ocean_data', filesuffix=filesuffix, delete=True)
    nchar = max(len(n) for n in list(names) + list(weights) + ['depth_weighted'])
    zlib = cfg.PARAMS['compress_climate_netcdf']

    with ncDataset(fpath, 'w', format='NETCDF4') as nc:
        nc.createDimension('time', None)
        nc.createDimension('band', len(names))
        nc.createDimension('nchar', nchar)

        nc.ref_pix_lon = lon
        nc.ref_pix_lat = lat
        nc.ref_pix_dis = utils.haversine(lon, lat,
                                         gdir.cenlon, gdir.cenlat)
        nc.ocean_source = source
        nc.ocean_model = ocean_model
        nc.extraction = extraction
        if n_cells is not None:
            nc.n_cells = int(n_cells)
        if search_radius_km is not None:
            nc.search_radius_km = float(search_radius_km)
        if terminus_depth is not None:
            nc.ref_bathymetry_m = float(terminus_depth)
        nc.ref_bathymetry_src = terminus_depth_source
        nc.tf_method = 'teos10_gsw' if teos10 else 'linear_lambda_jenkins2011'
        nc.tf_lambda1, nc.tf_lambda2, nc.tf_lambda3 = LAMBDA1, LAMBDA2, LAMBDA3
        nc.bias_correction_applied = str(bias_applied)
        nc.yr_0 = y0
        nc.yr_1 = y1
        nc.author = 'OGGM'
        nc.author_info = 'Open Global Glacier Model'

        v = nc.createVariable('time', 'i4', ('time',))
        v.units = time_unit
        v.calendar = 'standard'
        v[:] = date2num(_as_datetimes(time), time_unit, calendar='standard')

        for var, vals in (('band_name', names), ('band_weighting', weights)):
            v = nc.createVariable(var, 'S1', ('band', 'nchar'))
            v[:] = _pad(vals, nchar)

        for var, vals, unit in (('band_top', tops, 'm'),
                                ('band_bottom', bots, 'm')):
            v = nc.createVariable(var, 'f4', ('band',))
            v.units = unit
            v[:] = np.asarray(vals, dtype=float)

        if offsets is not None:
            v = nc.createVariable('bias_offset', 'f4', ('band',))
            v.units = 'degC'
            v.long_name = ('offset that would align the reference-period mean with '
                           'ref_ocean; added to thermal_forcing only if '
                           'bias_correction_applied is True')
            v[:] = np.asarray(offsets, dtype=float)

        v = nc.createVariable('thermal_forcing', 'f4', ('time', 'band'), zlib=zlib)
        v.units = 'degC'
        v.long_name = 'ocean thermal forcing, in situ minus freezing point'
        v[:] = tf

        for var, vals, unit, name in (
                ('thetao', th, 'degC', 'band-mean potential temperature'),
                ('so', sa, 'psu', 'band-mean practical salinity')):
            v = nc.createVariable(var, 'f4', ('time', 'band'), zlib=zlib)
            v.units = unit
            v.long_name = name
            v[:] = vals

        for var, vals, unit, name in (
                ('siconc', siconc, '1', 'sea ice area fraction'),
                ('open_water_frac', open_water, '1', 'open water fraction'),
                ('subglacial_discharge', q_sg, 'm3 s-1', 'subglacial discharge')):
            if vals is None:
                continue
            v = nc.createVariable(var, 'f4', ('time',), zlib=zlib)
            v.units = unit
            v.long_name = name
            v[:] = vals


def _pad(values, nchar):
    """Fixed-width character array, as netCDF4 wants it for a char variable."""
    out = np.zeros((len(values), nchar), dtype='S1')
    for i, val in enumerate(values):
        out[i, :len(val)] = np.array(list(val), dtype='S1')
    return out


def _as_datetimes(time):
    """Whatever time axis we were handed, as objects date2num accepts."""
    time = np.asarray(time)
    if np.issubdtype(time.dtype, np.datetime64):
        return pd.to_datetime(time).to_pydatetime()
    return list(time)
