"""Ocean boundary conditions at the glacier terminus, for frontal ablation.

Writes one ``ocean_data`` file per glacier directory: thermal forcing in one
or more depth bands, sea-ice concentration and open-water fraction, and
optionally subglacial discharge. The calendar and longitude conventions are
those of :func:`oggm.shop.gcm_climate.process_gcm_data`.
"""
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset as ncDataset
from netCDF4 import date2num

from oggm import cfg
from oggm import entity_task
from oggm import utils
from oggm.core.ocean_calving import band_names
from oggm.shop.gcm_climate import _get_xr_cftime_kwargs
from oggm.exceptions import InvalidParamsError, InvalidWorkflowError

log = logging.getLogger(__name__)

# Jenkins (2011), as used by Slater et al. (2020) Eq. (6):
# degC psu-1, degC, degC m-1
LAMBDA1, LAMBDA2, LAMBDA3 = -5.73e-2, 8.32e-2, 7.61e-4


def parse_depth_bands(specs):
    """Depth bands and their weighting from ``PARAMS['ocean_depth_bands']``.

    Parameters
    ----------
    specs : list of str
        each ``name:top:bottom[:weighting]``, depths in m, positive downwards

    Returns
    -------
    (depth_bands, band_weighting) : a list of (name, top, bottom) and a dict
        of name -> 'uniform' or 'depth_weighted'
    """
    bands, weighting = [], {}
    for spec in specs:
        parts = str(spec).strip().split(':')
        if len(parts) not in (3, 4):
            raise InvalidParamsError(f'depth band {spec!r} is not '
                                     'name:top:bottom[:weighting]')
        bands.append((parts[0], float(parts[1]), float(parts[2])))
        weighting[parts[0]] = parts[3] if len(parts) == 4 else 'uniform'
    return bands, weighting


def freezing_point(salinity, depth, teos10=False, lat=None):
    """Freezing point of sea water, degC.

    Parameters
    ----------
    salinity : array
        practical salinity, psu
    depth : array
        m, positive downwards
    teos10 : bool
        use the conservative-temperature freezing point of TEOS-10 (needs
        gsw) instead of the linear form of Slater et al. (2020), Eq. (6)
    lat : float
        the latitude, needed with ``teos10`` to convert depth to pressure
    """
    depth = np.asarray(depth, dtype=float)
    if teos10:
        import gsw
        if lat is None:
            raise InvalidParamsError('the TEOS-10 freezing point needs `lat`')
        p = gsw.p_from_z(-depth, lat)
        return gsw.CT_freezing(salinity, p, 0)
    return LAMBDA1 * np.asarray(salinity) + LAMBDA2 - LAMBDA3 * depth


def open_water_fraction(siconc, threshold=None):
    """Open-water fraction from sea-ice concentration, clipped to [0, 1].

    Parameters
    ----------
    siconc : array
        sea-ice area fraction, 0-1
    threshold : float, optional
        the concentration at and above which no water is open. Default:
        ``cfg.PARAMS['ocean_open_water_threshold']``.
    """
    if threshold is None:
        threshold = cfg.PARAMS['ocean_open_water_threshold']
    siconc = np.asarray(siconc, dtype=float)
    return np.clip((threshold - siconc) / threshold, 0, 1)


def thermal_forcing_bands(thetao, so, depth, depth_bands, band_weighting,
                          terminus_depth=None, teos10=False, lat=None):
    """Band-average thermal forcing, temperature and salinity.

    Parameters
    ----------
    thetao, so : ndarray
        (time, depth) potential temperature in degC and salinity in psu
    depth : ndarray
        depth coordinate in m, positive downwards
    depth_bands : list of (name, top, bottom)
        the bands, in m
    band_weighting : dict
        band name -> 'uniform' or 'depth_weighted'
    terminus_depth : float, optional
        replaces the bottom of the 'terminus' band
    teos10 : bool
        see :func:`freezing_point`
    lat : float, optional
        the latitude, needed with ``teos10``

    Returns
    -------
    names, tops, bottoms, weightings, tf, thetao_b, so_b : the last three with
        shape (time, band)
    """
    z = np.asarray(depth, dtype=float)
    names, tops, bots, weights = [], [], [], []
    tf_b, th_b, so_b = [], [], []

    for name, top, bot in depth_bands:
        if name == 'terminus' and terminus_depth is not None:
            bot = float(terminus_depth)
        sel = (z >= top) & (z <= bot)
        if sel.sum() == 0:
            raise InvalidWorkflowError(
                f'band {name} [{top}, {bot}] m contains no ocean levels; '
                f'available depths are {z.min():.0f}-{z.max():.0f} m')

        if not (np.isfinite(thetao[:, sel]).all()
                and np.isfinite(so[:, sel]).all()):
            # levels below the sea floor pass the check on the coordinate
            raise InvalidWorkflowError(
                f'band {name} [{top}, {bot}] m has non-finite values; land and '
                f'sub-bathymetry levels must be dropped before this call')

        how = band_weighting.get(name, 'uniform')
        if how == 'depth_weighted':
            w = np.gradient(z)[sel]
            w = w / w.sum()
        elif how == 'uniform':
            w = np.full(int(sel.sum()), 1 / sel.sum())
        else:
            raise InvalidParamsError(f'unknown band weighting {how!r}')

        zc = z[sel]
        tf_z = thetao[:, sel] - freezing_point(so[:, sel], zc, teos10=teos10,
                                               lat=lat)
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
    """Write the ocean boundary conditions of one glacier directory.

    Nothing is written for a glacier that is not tidewater.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    thetao, so : :py:class:`xarray.DataArray`
        monthly sea-water potential temperature (degC or K, told apart by
        magnitude) and salinity (psu), with a ``depth`` coordinate in m
        positive downwards and scalar ``lon`` and ``lat``, i.e. one water
        column, already reduced over the cells it stands for. Whole calendar
        years only.
    siconc : :py:class:`xarray.DataArray`, optional
        monthly sea-ice area fraction, 0-1, on the same time axis
    subglacial_discharge : :py:class:`xarray.DataArray`, optional
        monthly subglacial discharge in m3 s-1 (see also
        :func:`oggm.core.ocean_inversion.write_subglacial_discharge`)
    depth_bands : list of (name, top, bottom), optional
        default: ``gdir.settings['ocean_depth_bands']``
    band_weighting : dict, optional
        band name -> 'uniform' or 'depth_weighted'. Default: as given in
        ``gdir.settings['ocean_depth_bands']``.
    terminus_depth : float, optional
        water depth at the terminus, m; the bottom of the 'terminus' band
    year_range : tuple of two years, optional
        reference period of the bias correction. Default:
        ``gdir.settings['ocean_tf_ref_period']``.
    apply_bias_correction : bool, optional
        add one offset per band so that the ``year_range`` mean matches
        ``ref_ocean`` (Slater et al. 2020, Eq. 5). When False the offset is
        written to the file but not applied. Default:
        ``gdir.settings['ocean_bias_correct']``.
    ref_ocean : :py:class:`xarray.Dataset`, optional
        the reference, with a ``thermal_forcing(time, band)`` variable
    teos10 : bool, optional
        default: ``gdir.settings['ocean_use_teos10']``
    source, ocean_model, extraction, terminus_depth_source : str
        provenance, written as global attributes
    n_cells : int, optional
        the number of ocean cells the column was reduced over (provenance)
    search_radius_km : float, optional
        the radius they were searched in (provenance)
    output_filesuffix : str
        the filesuffix of the written file
    """
    if thetao is None or so is None:
        raise InvalidParamsError('process_ocean_data needs thetao and so')

    # The contracts of process_gcm_data
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
    if not np.array_equal(so['time'].values, thetao['time'].values):
        raise InvalidParamsError('thetao and so must share a time axis.')
    if siconc is not None and len(siconc['time']) != len(thetao['time']):
        # the time dimension is unlimited: a short siconc would be padded
        raise InvalidParamsError('siconc must share the thetao time axis.')

    if not gdir.is_tidewater:
        log.warning('(%s) not tidewater, no ocean data written', gdir.rgi_id)
        return

    if teos10 is None:
        teos10 = gdir.settings['ocean_use_teos10']
    year_range = year_range or gdir.settings['ocean_tf_ref_period']
    year_range = slice(str(year_range[0]), str(year_range[1]))
    if apply_bias_correction is None:
        apply_bias_correction = gdir.settings['ocean_bias_correct']
    def_bands, def_weighting = parse_depth_bands(
        gdir.settings['ocean_depth_bands'])
    depth_bands = depth_bands or def_bands
    band_weighting = dict(band_weighting or def_weighting)

    t = np.asarray(thetao.values, dtype=float)
    if np.nanmean(t) > 100:  # K: sea water is never that warm in degC
        t = t - 273.15
    s = np.asarray(so.values, dtype=float)

    lat = float(thetao['lat'])
    names, tops, bots, weights, tf, th, sa = thermal_forcing_bands(
        t, s, thetao['depth'].values, depth_bands, band_weighting,
        terminus_depth=terminus_depth, teos10=teos10, lat=lat)

    offsets = np.zeros(len(names))
    if ref_ocean is not None:
        ref = ref_ocean.sel(time=year_range).thermal_forcing.mean('time')
        # `band` is a bare dimension: align by name, never by index
        ref_names = band_names(ref_ocean) if 'band_name' in ref_ocean else None
        if ref_names is not None:
            missing = [n for n in names if n not in ref_names]
            if missing:
                raise InvalidParamsError(
                    f'ref_ocean has no band(s) {", ".join(missing)}; it '
                    f'carries {", ".join(ref_names)}')
            ref = ref.isel(band=[ref_names.index(n) for n in names])
        elif ref.sizes['band'] != len(names):
            raise InvalidParamsError(
                f'ref_ocean has {ref.sizes["band"]} bands against '
                f'{len(names)} here, and no band_name to align them by')
        if not np.isfinite(ref.values).all():
            raise InvalidWorkflowError('ref_ocean has no data in the '
                                       'reference period')
        mod = xr.DataArray(
            tf, dims=('time', 'band'),
            coords={'time': thetao['time'].values, 'band': names},
        ).sel(time=year_range).mean('time')
        offsets = np.asarray(ref.values - mod.values, dtype=float)
        if apply_bias_correction:
            tf = tf + offsets

    sic = None
    if siconc is not None:
        sic = np.clip(np.asarray(siconc.values, dtype=float), 0, 1)
    owf = None
    if sic is not None:
        owf = open_water_fraction(
            sic, threshold=gdir.settings['ocean_open_water_threshold'])
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


@entity_task(log, writes=['ocean_data'])
def process_destine_ocean_data(gdir, fpath=None, y0=None, y1=None,
                               terminus_depth=None, terminus_depth_source='',
                               thetao_var='thetao', so_var='so',
                               siconc_var='siconc', depth_var='depth',
                               output_filesuffix='', source=None,
                               ocean_model='', **kwargs):
    """Read one ocean water column from a file and write the ocean_data file.

    The file holds the monthly water column that stands for one glacier, as
    extracted from the Destination Earth Climate DT ocean or from any other
    ocean product: ``thetao(time, depth)`` and ``so(time, depth)``, optionally
    ``siconc(time)``, and scalar ``lon`` and ``lat``. Its global attributes
    ``bands`` (``name:top:bottom`` separated by spaces), ``extraction``,
    ``n_cells``, ``search_radius_km``, ``terminus_depth_m``, ``source`` and
    ``model`` are used when the matching argument is not given.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    fpath : str, optional
        path to the file. Default: ``cfg.PATHS['destine_ocean_file']``.
    y0, y1 : int, optional
        clip to these years
    terminus_depth : float, optional
        water depth at the terminus, m
    terminus_depth_source : str
        provenance of that depth
    thetao_var, so_var, siconc_var, depth_var : str
        the variable names in the file
    output_filesuffix : str
        the filesuffix of the written file
    source, ocean_model : str, optional
        provenance. Default: the file's ``source`` and ``model`` attributes.
    **kwargs
        any other argument of :func:`process_ocean_data`
    """
    for key in ('thetao', 'so', 'siconc', 'subglacial_discharge'):
        if key in kwargs:
            raise InvalidParamsError(f'{key} is built by this task, not '
                                     'passed to it')

    if not gdir.is_tidewater:
        log.warning('(%s) not tidewater, no ocean data written', gdir.rgi_id)
        return

    fpath = fpath or cfg.PATHS.get('destine_ocean_file')
    if not fpath:
        raise InvalidParamsError("Need to set cfg.PATHS['destine_ocean_file']")

    with xr.open_dataset(fpath, **_get_xr_cftime_kwargs()) as ds:
        ds = ds.squeeze(drop=True)
        if y0 is not None or y1 is not None:
            ds = ds.sel(time=slice(str(y0) if y0 else None,
                                   str(y1) if y1 else None))
        ds = _whole_years(ds)
        if not ds.sizes.get('time'):
            raise InvalidWorkflowError('no complete calendar year in the '
                                       'selection')

        thetao = ds[thetao_var].rename({depth_var: 'depth'})
        thetao = thetao.transpose('time', 'depth')
        so = ds[so_var].rename({depth_var: 'depth'}).transpose('time', 'depth')
        if float(thetao['depth'][0]) < 0:
            thetao = thetao.assign_coords(depth=-thetao['depth'])
            so = so.assign_coords(depth=thetao['depth'])
        siconc = ds[siconc_var] if siconc_var in ds else None

        lon = ((float(ds['lon']) + 180) % 360) - 180
        for da in (thetao, so):
            da.coords['lon'] = lon
            da.coords['lat'] = float(ds['lat'])

        for name, da in (('thetao', thetao), ('so', so)):
            if not np.isfinite(da.values).all():
                raise InvalidWorkflowError(
                    f'{name} carries non-finite values; land and '
                    'sub-bathymetry levels must be dropped by the extraction')
        if 'depth_bands' not in kwargs and ds.attrs.get('bands'):
            kwargs['depth_bands'], _ = parse_depth_bands(
                ds.attrs['bands'].split())
        for key in ('extraction', 'n_cells', 'search_radius_km'):
            if key not in kwargs and key in ds.attrs:
                kwargs[key] = ds.attrs[key]
        if terminus_depth is None and 'terminus_depth_m' in ds.attrs:
            terminus_depth = float(ds.attrs['terminus_depth_m'])

        source = source or ds.attrs.get('source') or os.path.basename(fpath)
        ocean_model = ocean_model or ds.attrs.get('model', '')

        process_ocean_data(gdir, thetao=thetao, so=so, siconc=siconc,
                           terminus_depth=terminus_depth,
                           terminus_depth_source=terminus_depth_source,
                           source=source, ocean_model=ocean_model,
                           output_filesuffix=output_filesuffix, **kwargs)


def _whole_years(ds):
    """Trim to complete January-December years."""
    years = ds['time.year'].values
    months = ds['time.month'].values
    keep = np.zeros(ds.sizes['time'], dtype=bool)
    for year in np.unique(years):
        sel = years == year
        if sel.sum() == 12 and months[sel][0] == 1 and months[sel][-1] == 12:
            keep |= sel
    return ds.isel(time=keep)


def _write_ocean_file(gdir, time, names, tops, bots, weights, tf, th, sa,
                      siconc, open_water, q_sg=None, lon=None, lat=None,
                      terminus_depth=None, terminus_depth_source='',
                      source='', ocean_model='', extraction='', n_cells=None,
                      search_radius_km=None, offsets=None, teos10=False,
                      bias_applied=False, filesuffix=''):
    """Write the ocean_data file, as write_monthly_climate_file does."""
    time = np.asarray(time)
    y0 = int(str(time[0])[:4])
    y1 = int(str(time[-1])[:4])
    time_unit = ('days since 1801-01-01 00:00:00' if y0 > 1800
                 else 'days since 0001-01-01 00:00:00')

    fpath = gdir.get_filepath('ocean_data', filesuffix=filesuffix, delete=True)
    nchar = max(len(n) for n in list(names) + list(weights)
                + ['depth_weighted'])
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
            v.long_name = ('offset that would align the reference-period mean '
                           'with ref_ocean; added to thermal_forcing only if '
                           'bias_correction_applied is True')
            v[:] = np.asarray(offsets, dtype=float)

        v = nc.createVariable('thermal_forcing', 'f4', ('time', 'band'),
                              zlib=zlib)
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
                ('subglacial_discharge', q_sg, 'm3 s-1',
                 'subglacial discharge')):
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
