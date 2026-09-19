"""BedMachine bed topography on the glacier grid.

:py:func:`oggm.shop.bedmachine.bedmachine_to_gdir` writes only ``thickness``. The
bed itself is what a marine terminus needs: the water depth every calving law is
proportional to, over the ground the front advances onto. This module writes ``bed``,
``errbed`` and ``source`` beside it, so that the uncertainty and the provenance of
each sampled point travel with the bed rather than being looked up by hand.

``source`` is categorical and is mapped with nearest-neighbour rather than linear
interpolation, so that "this point is multibeam bathymetry" survives regridding.

Three facts about the data, checked 2026-09-19 rather than assumed:

- **Greenland v6 exists** (released 2025-12-11, DOI 10.5067/6B6B225B8V2D), which the
  stock module predates; it is the default here. v5 remains selectable.
- The **NSIDC cloud** URLs above are the ones NASA CMR returns. The stock module's
  ``n5eil01u.ecs.nsidc.org`` host belongs to the retired on-premises distribution.
- Antarctica is now **v4**, not the v3 the stock module points at.

NSIDC needs Earthdata credentials (``oggm_netrc_credentials``). Where the file is
already on disk -- an HPC without outbound access, say -- pass ``local_file``, or set
``cfg.PARAMS['bedmachine_file']``.
"""
import logging
import os

import numpy as np
import pandas as pd
import xarray as xr

from oggm import cfg, entity_task, global_task, utils
from oggm.core.ocean_params import ocean_param
from oggm.exceptions import InvalidParamsError

log = logging.getLogger(__name__)

# From NASA CMR (short_name IDBMG4 / NSIDC-0756), not from the dataset landing pages.
BEDMACHINE_URLS = {
    ('05', '6'): ('https://data.nsidc.earthdatacloud.nasa.gov/'
                  'nsidc-cumulus-prod-protected/ICEBRIDGE/IDBMG4/6/1993/01/01/'
                  'BedMachineGreenland-v6.nc'),
    ('05', '5'): ('https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/IDBMG4.005/'
                  '1993.01.01/BedMachineGreenland-v5.nc'),
    ('19', '4'): ('https://data.nsidc.earthdatacloud.nasa.gov/'
                  'nsidc-cumulus-prod-protected/MEASURES/NSIDC-0756/4/1970/01/01/'
                  'NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc'),
    ('19', '3'): ('https://n5eil01u.ecs.nsidc.org/MEASURES/NSIDC-0756.003/'
                  '1970.01.01/BedMachineAntarctica-v3.nc'),
}

DEFAULT_VERSION = {'05': '6', '19': '4'}

# gridded_data name -> (BedMachine name, interpolation). Nearest for the two
# categorical fields; a linearly interpolated source flag is a meaningless number.
BEDMACHINE_VARS = {
    'bedmachine_bed': ('bed', 'linear'),
    'bedmachine_errbed': ('errbed', 'linear'),
    'bedmachine_source': ('source', 'nearest'),
    'bedmachine_ice_thickness': ('thickness', 'linear'),
    'bedmachine_surface': ('surface', 'linear'),
    'bedmachine_mask': ('mask', 'nearest'),
}

LONG_NAMES = {
    'bedmachine_bed': 'Bed topography from BedMachine',
    'bedmachine_errbed': 'Bed topography error from BedMachine',
    'bedmachine_source': 'Data source flag from BedMachine',
    'bedmachine_ice_thickness': 'Ice thickness from BedMachine',
    'bedmachine_surface': 'Ice surface elevation from BedMachine',
    'bedmachine_mask': 'Ice/ocean/land mask from BedMachine',
}


def bedmachine_file(gdir, version=None, local_file=None):
    """Path to the BedMachine file for this glacier, downloading it if needed."""

    if local_file is None:
        local_file = ocean_param('bedmachine_file')
    if local_file is not None:
        if not os.path.exists(local_file):
            raise InvalidParamsError(f'BedMachine file not found: {local_file}')
        return local_file

    region = gdir.rgi_region
    if region not in DEFAULT_VERSION:
        raise NotImplementedError('BedMachine data not available for this '
                                  f'region: {region}')
    version = str(version or ocean_param('bedmachine_version')
                  or DEFAULT_VERSION[region])
    try:
        url = BEDMACHINE_URLS[(region, version)]
    except KeyError:
        have = sorted(v for r, v in BEDMACHINE_URLS if r == region)
        raise InvalidParamsError(f'No BedMachine v{version} for region {region} '
                                 f'(have: {", ".join(have)})')
    return utils.download_with_authentication(url, 'urs.earthdata.nasa.gov')


def _open_bedmachine(path):
    """BedMachine as a salem-aware dataset, with its projection filled in."""
    ds = xr.open_dataset(path)
    proj = ds.attrs.get('proj4', None)
    if proj is None:
        # Both grids are the standard polar stereographic ones; v6 dropped the
        # global attribute the stock module reads.
        proj = 'epsg:3413' if float(ds.y[0]) > 0 else 'epsg:3031'
    ds.attrs['pyproj_srs'] = proj
    return ds, proj


@entity_task(log, writes=['gridded_data'])
def bedmachine_bed_to_gdir(gdir, version=None, local_file=None, add_vars=None):
    """Add the BedMachine bed, its error and its source flag to ``gridded_data``.

    A superset of :py:func:`oggm.shop.bedmachine.bedmachine_to_gdir`: the ice
    thickness is written under the same name, so the two are interchangeable.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    version : str
        BedMachine version ('6' or '5' for Greenland, '4' or '3' for Antarctica).
        Default: ``cfg.PARAMS['bedmachine_version']``, else the newest known.
    local_file : str
        read this file instead of downloading. Default:
        ``cfg.PARAMS['bedmachine_file']``.
    add_vars : sequence of str
        which ``gridded_data`` variables to write. Default: all of
        ``BEDMACHINE_VARS``.
    """

    if add_vars is None:
        add_vars = tuple(BEDMACHINE_VARS)
    unknown = set(add_vars) - set(BEDMACHINE_VARS)
    if unknown:
        raise InvalidParamsError(f'Unknown BedMachine variables: {sorted(unknown)}')

    path = bedmachine_file(gdir, version=version, local_file=local_file)

    out = {}
    attrs = {}
    with _open_bedmachine(path) as (ds, proj):
        x0, x1, y0, y1 = gdir.grid.extent_in_crs(proj)
        dsroi = ds.salem.subset(corners=((x0, y0), (x1, y1)), crs=proj, margin=10)
        for vn in add_vars:
            src, interp = BEDMACHINE_VARS[vn]
            if src not in dsroi:
                log.warning(f'({gdir.rgi_id}) BedMachine file has no {src!r}')
                continue
            data = dsroi[src].data.astype(np.float64)
            out[vn] = gdir.grid.map_gridded_data(data, grid=dsroi.salem.grid,
                                                 interp=interp)
            attrs[vn] = {k: v for k, v in dsroi[src].attrs.items()
                         if k in ('units', 'flag_values', 'flag_meanings',
                                  'valid_range', 'source', 'grid_mapping')}

    if 'bedmachine_ice_thickness' in out:
        # As the stock task: no ice means no thickness, not a zero.
        thick = np.asarray(out['bedmachine_ice_thickness'], dtype=np.float64)
        thick[thick <= 0] = np.nan
        out['bedmachine_ice_thickness'] = thick

    with utils.ncDataset(gdir.get_filepath('gridded_data'), 'a') as nc:
        for vn, data in out.items():
            if vn in nc.variables:
                v = nc.variables[vn]
            else:
                v = nc.createVariable(vn, 'f4', ('y', 'x'), zlib=True,
                                      fill_value=np.nan)
            v.units = attrs[vn].get('units', 'm')
            v.long_name = LONG_NAMES[vn]
            v.data_source = path
            v.bedmachine_version = str(version or ocean_param('bedmachine_version')
                                       or DEFAULT_VERSION.get(gdir.rgi_region, ''))
            for k in ('flag_values', 'flag_meanings'):
                if k in attrs[vn]:
                    setattr(v, k, attrs[vn][k])
            v[:] = np.asarray(data, dtype=np.float32)


@entity_task(log)
def bedmachine_bed_statistics(gdir):
    """Per-glacier summary of the BedMachine bed, its error and its sources."""

    d = {'rgi_id': gdir.rgi_id,
         'rgi_region': gdir.rgi_region,
         'rgi_area_km2': gdir.rgi_area_km2,
         'is_tidewater': gdir.is_tidewater,
         'bedmachine_bed_min': np.nan,
         'bedmachine_bed_median': np.nan,
         'bedmachine_frac_below_sl': np.nan,
         'bedmachine_errbed_mean': np.nan,
         'bedmachine_errbed_max': np.nan,
         'bedmachine_source_mode': np.nan,
         'bedmachine_frac_multibeam': np.nan}

    try:
        with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
            mask = ds['glacier_mask'].data.astype(bool)
            bed = ds['bedmachine_bed'].data[mask]
            bed = bed[np.isfinite(bed)]
            if bed.size:
                d['bedmachine_bed_min'] = float(np.min(bed))
                d['bedmachine_bed_median'] = float(np.median(bed))
                d['bedmachine_frac_below_sl'] = float(np.mean(bed < 0))
            if 'bedmachine_errbed' in ds:
                err = ds['bedmachine_errbed'].data[mask]
                err = err[np.isfinite(err)]
                if err.size:
                    d['bedmachine_errbed_mean'] = float(np.mean(err))
                    d['bedmachine_errbed_max'] = float(np.max(err))
            if 'bedmachine_source' in ds:
                src = ds['bedmachine_source'].data[mask]
                src = np.round(src[np.isfinite(src)])
                if src.size:
                    vals, counts = np.unique(src, return_counts=True)
                    d['bedmachine_source_mode'] = int(vals[np.argmax(counts)])
                    d['bedmachine_frac_multibeam'] = float(np.mean(src == 10))
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@global_task(log)
def compile_bedmachine_bed_statistics(gdirs, filesuffix='', path=True):
    """Gather :py:func:`bedmachine_bed_statistics` over a list of glaciers."""
    from oggm.workflow import execute_entity_task

    out = pd.DataFrame(execute_entity_task(bedmachine_bed_statistics,
                                           gdirs)).set_index('rgi_id')
    if path:
        if path is True:
            path = os.path.join(cfg.PATHS['working_dir'],
                                f'bedmachine_bed_statistics{filesuffix}.csv')
        out.to_csv(path)
    return out
