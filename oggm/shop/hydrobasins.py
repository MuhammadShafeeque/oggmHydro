"""HydroSHEDS basin delineation and river network data for OGGM.

Provides download helpers for HydroBASINS sub-basin polygons and
HydroRIVERS polylines, together with a spatial join utility that maps
OGGM glacier directories to their enclosing HydroBASINS sub-basins.

HydroSHEDS Reference
--------------------
Lehner, B., Verdin, K., Jarvis, A. (2008): New global hydrography derived
from spaceborne elevation data. Eos, Transactions, AGU, 89(10): 93-94.

HydroBASINS Reference
---------------------
Lehner, B., Grill G. (2013): Global river hydrography and network routing:
baseline data and new approaches to study the world's large river systems.
Hydrological Processes, 27(15): 2171-2186.

Data Access
-----------
HydroSHEDS data (HydroBASINS, HydroRIVERS) is freely available at:
    https://www.hydrosheds.org/products/hydrobasins
    https://www.hydrosheds.org/products/hydrorivers

Set ``cfg.PARAMS['hydrobasins_server']`` (or the env variable
``HYDROBASINS_SERVER``) to a local HTTP mirror to avoid re-downloading.

Alternatively set ``cfg.PARAMS['hydrobasins_local_dir']`` to a directory
that already contains the unzipped or zip shapefiles.
"""

# Built-ins
import logging
import os

# External
import numpy as np
import pandas as pd

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# OGGM internals
from oggm import cfg, utils

# Module logger
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HydroSHEDS default server and URL patterns
# ---------------------------------------------------------------------------
_DEFAULT_SERVER = 'https://data.hydrosheds.org/file/'

# HydroBASINS: one ZIP per (region, level)
# URL: <server>/HydroBASINS/standard/hybas_{region}_lev{NN}_v1c.zip
_HYBAS_PATH = 'HydroBASINS/standard/hybas_{region}_lev{level:02d}_v1c.zip'

# HydroRIVERS: one ZIP per region
# URL: <server>/HydroRIVERS/HydroRIVERS_v10_{region}_shp.zip
_HYDRIV_PATH = 'HydroRIVERS/HydroRIVERS_v10_{region}_shp.zip'

# Recognised two-letter region codes used by HydroSHEDS
_REGIONS = ('af', 'ar', 'as', 'au', 'eu', 'gr', 'na', 'sa', 'si')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hydrosheds_server():
    """Return the configured HydroSHEDS server base URL (trailing slash)."""
    server = cfg.PARAMS.get('hydrobasins_server', _DEFAULT_SERVER)
    if not server.endswith('/'):
        server += '/'
    return server


def _hydrobasins_local_dir():
    """Return local override directory for pre-downloaded shapefiles (or '')."""
    return cfg.PARAMS.get('hydrobasins_local_dir', '')


def _lon_to_region(lon_deg, lat_deg=0.0):
    """Map a (lon, lat) coordinate to the HydroSHEDS two-letter region code.

    HydroSHEDS covers the globe in continental regions:
    - af : Africa
    - ar : Arctic (north of ~60-67 N)
    - as : Asia (non-Arctic)
    - au : Australasia / Oceania
    - eu : Europe
    - gr : Greenland
    - na : North America
    - sa : South America
    - si : Siberia (Russia east of ~60 E)

    This lookup is approximate and suitable for pre-filtering; the download
    functions further narrow by spatial extent.

    Parameters
    ----------
    lon_deg, lat_deg : float
        Representative longitude and latitude for the area of interest.

    Returns
    -------
    str
        HydroSHEDS two-letter region code.
    """
    if lat_deg >= 60.0:
        if lon_deg < -10.0:
            return 'ar'  # Arctic North America
        elif lon_deg < 50.0:
            return 'ar'  # Arctic Europe / Greenland
        elif lon_deg < 180.0:
            return 'ar'  # Arctic Asia
    if lat_deg >= 25.0 and -15.0 <= lon_deg <= 55.0:
        return 'eu'
    if lat_deg >= 55.0 and 55.0 < lon_deg <= 180.0:
        return 'si'
    if -35.0 <= lat_deg <= 80.0 and -170.0 <= lon_deg <= -30.0:
        if lat_deg >= 15.0:
            return 'na'
        return 'sa'
    if -60.0 <= lat_deg <= 40.0 and -20.0 <= lon_deg <= 55.0:
        return 'af'
    if lat_deg >= -12.0 and 55.0 < lon_deg <= 180.0:
        return 'as'
    if lat_deg <= -10.0 and lon_deg >= 110.0:
        return 'au'
    # Default fallback — broadest region
    return 'as'


def _get_hydrobasins_file(region, level, local_dir=''):
    """Return a local path to the HydroBASINS shapefile for *region/level*.

    Attempts the following in order:
    1. ``local_dir`` override (pre-downloaded files).
    2. OGGM download cache via :func:`oggm.utils.file_downloader`.

    Parameters
    ----------
    region : str
        Two-letter HydroSHEDS region code (e.g. ``'as'``).
    level : int
        Pfaffstetter aggregation level 1–12.
    local_dir : str
        Directory to search for pre-downloaded ZIP / SHP files.

    Returns
    -------
    str
        Local path to the ZIP archive (may be passed to
        ``gpd.read_file('zip://' + path)``).
    """
    fname = f'hybas_{region}_lev{level:02d}_v1c.zip'
    if local_dir:
        candidate = os.path.join(local_dir, fname)
        if os.path.isfile(candidate):
            return candidate

    url = _hydrosheds_server() + _HYBAS_PATH.format(region=region,
                                                     level=level)
    local = utils.file_downloader(url)
    if local is None:
        raise FileNotFoundError(
            f'Could not download HydroBASINS level-{level} for region '
            f'"{region}" from:\n  {url}\n'
            'Set cfg.PARAMS["hydrobasins_local_dir"] to a directory '
            'containing pre-downloaded files, or '
            'cfg.PARAMS["hydrobasins_server"] to an accessible mirror.'
        )
    return local


def _get_hydrorivers_file(region, local_dir=''):
    """Return a local path to the HydroRIVERS shapefile for *region*.

    Parameters
    ----------
    region : str
        Two-letter HydroSHEDS region code.
    local_dir : str
        Directory to search for pre-downloaded ZIP files.

    Returns
    -------
    str
        Local path to the ZIP archive.
    """
    fname = f'HydroRIVERS_v10_{region}_shp.zip'
    if local_dir:
        candidate = os.path.join(local_dir, fname)
        if os.path.isfile(candidate):
            return candidate

    url = _hydrosheds_server() + _HYDRIV_PATH.format(region=region)
    local = utils.file_downloader(url)
    if local is None:
        raise FileNotFoundError(
            f'Could not download HydroRIVERS for region "{region}" from:\n'
            f'  {url}\n'
            'Set cfg.PARAMS["hydrobasins_local_dir"] to a directory '
            'containing pre-downloaded files.'
        )
    return local


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_hydrobasins(bbox, level=8, region=None):
    """Download and return HydroBASINS sub-basin polygons for a bounding box.

    Parameters
    ----------
    bbox : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in decimal degrees.
    level : int, optional
        Pfaffstetter aggregation level 1–12.  Level 8 (default) gives
        typical sub-basin areas of 10²–10³ km².
    region : str, optional
        Two-letter HydroSHEDS region code.  Inferred automatically from
        *bbox* if not provided.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        Sub-basin polygons clipped to *bbox* with columns:
        ``HYBAS_ID``, ``NEXT_DOWN``, ``NEXT_SINK``, ``MAIN_BAS``,
        ``DIST_SINK``, ``DIST_MAIN``, ``SUB_AREA``, ``UP_AREA``,
        ``PFAF_ID``, ``ENDO``, ``COAST``, ``ORDER``, ``SORT``,
        ``geometry``.

    Raises
    ------
    ImportError
        If *geopandas* is not installed.
    FileNotFoundError
        If the HydroBASINS archive cannot be found locally or downloaded.
    """
    if not HAS_GEOPANDAS:
        raise ImportError(
            'geopandas is required for get_hydrobasins(). '
            'Install it with: conda install geopandas'
        )

    lon_min, lat_min, lon_max, lat_max = bbox
    if region is None:
        lon_c = (lon_min + lon_max) / 2.0
        lat_c = (lat_min + lat_max) / 2.0
        region = _lon_to_region(lon_c, lat_c)

    local_dir = _hydrobasins_local_dir()
    zpath = _get_hydrobasins_file(region, level, local_dir=local_dir)

    gdf = gpd.read_file('zip://' + zpath)
    # Clip to bounding box (spatial subset)
    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()
    gdf = gdf.reset_index(drop=True)

    log.info('get_hydrobasins: %d sub-basins in bbox for region=%s level=%d',
             len(gdf), region, level)
    return gdf


def get_hydrorivers(bbox, region=None):
    """Download and return HydroRIVERS polylines for a bounding box.

    Parameters
    ----------
    bbox : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in decimal degrees.
    region : str, optional
        Two-letter HydroSHEDS region code.  Inferred from *bbox* if None.

    Returns
    -------
    :class:`geopandas.GeoDataFrame`
        River polylines clipped to *bbox* with columns:
        ``HYRIV_ID``, ``NEXT_DOWN``, ``MAIN_RIV``, ``LENGTH_KM``,
        ``UPLAND_SKM``, ``CATCH_SKM``, ``ORD_STRA`` (Strahler order),
        ``ORD_CLAS``, ``ORD_FLOW``, ``ENDORHEIC``, ``DIS_AV_CMS``,
        ``geometry``.

    Raises
    ------
    ImportError
        If *geopandas* is not installed.
    FileNotFoundError
        If the HydroRIVERS archive cannot be found locally or downloaded.
    """
    if not HAS_GEOPANDAS:
        raise ImportError(
            'geopandas is required for get_hydrorivers(). '
            'Install it with: conda install geopandas'
        )

    lon_min, lat_min, lon_max, lat_max = bbox
    if region is None:
        lon_c = (lon_min + lon_max) / 2.0
        lat_c = (lat_min + lat_max) / 2.0
        region = _lon_to_region(lon_c, lat_c)

    local_dir = _hydrobasins_local_dir()
    zpath = _get_hydrorivers_file(region, local_dir=local_dir)

    gdf = gpd.read_file('zip://' + zpath)
    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()
    gdf = gdf.reset_index(drop=True)

    log.info('get_hydrorivers: %d river reaches in bbox for region=%s',
             len(gdf), region)
    return gdf


def assign_glaciers_to_subbasins(gdirs, subbasins_gdf):
    """Spatial join: map each glacier directory to a HydroBASINS sub-basin.

    Each glacier's centroid (lon/lat from ``gdir.cenlat``, ``gdir.cenlon``)
    is matched to the enclosing HydroBASINS polygon.  Glaciers whose
    centroid falls outside all polygons are assigned ``HYBAS_ID = -1``.

    Parameters
    ----------
    gdirs : list of :class:`oggm.GlacierDirectory`
        Glacier directories to assign.
    subbasins_gdf : :class:`geopandas.GeoDataFrame`
        HydroBASINS polygons as returned by :func:`get_hydrobasins`.
        Must contain columns ``HYBAS_ID``, ``SUB_AREA``, ``UP_AREA``,
        and a ``geometry`` column in geographic (lon/lat) CRS.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per glacier with columns:
        ``rgi_id``, ``cenlon``, ``cenlat``, ``glacier_area_km2``,
        ``HYBAS_ID``, ``SUB_AREA_km2``, ``UP_AREA_km2``,
        ``glacierized_fraction``.

    Raises
    ------
    ImportError
        If *geopandas* is not installed.
    ValueError
        If *subbasins_gdf* is missing required columns.
    """
    if not HAS_GEOPANDAS:
        raise ImportError(
            'geopandas is required for assign_glaciers_to_subbasins(). '
            'Install it with: conda install geopandas'
        )

    required_cols = {'HYBAS_ID', 'SUB_AREA', 'UP_AREA', 'geometry'}
    missing = required_cols - set(subbasins_gdf.columns)
    if missing:
        raise ValueError(
            f'subbasins_gdf is missing required columns: {missing}. '
            'Provide a GeoDataFrame as returned by get_hydrobasins().'
        )

    # Build GeoDataFrame of glacier centroids
    records = []
    for gdir in gdirs:
        records.append({
            'rgi_id': gdir.rgi_id,
            'cenlon': gdir.cenlon,
            'cenlat': gdir.cenlat,
            'glacier_area_km2': gdir.rgi_area_km2,
        })
    pts_gdf = gpd.GeoDataFrame(
        records,
        geometry=[Point(r['cenlon'], r['cenlat']) for r in records],
        crs='EPSG:4326',
    )

    # Ensure subbasins are in the same CRS
    subs = subbasins_gdf[['HYBAS_ID', 'SUB_AREA', 'UP_AREA',
                           'geometry']].copy()
    if subs.crs is None:
        subs = subs.set_crs('EPSG:4326')
    else:
        subs = subs.to_crs('EPSG:4326')

    # Spatial join: each glacier centroid → enclosing sub-basin
    joined = gpd.sjoin(pts_gdf, subs, how='left', predicate='within')

    # Handle duplicates from overlapping geometries (keep first match)
    joined = joined[~joined.index.duplicated(keep='first')]

    # Build output DataFrame
    out = pd.DataFrame({
        'rgi_id': joined['rgi_id'].values,
        'cenlon': joined['cenlon'].values,
        'cenlat': joined['cenlat'].values,
        'glacier_area_km2': joined['glacier_area_km2'].values,
        'HYBAS_ID': joined['HYBAS_ID'].fillna(-1).astype(int).values,
        'SUB_AREA_km2': joined['SUB_AREA'].values,
        'UP_AREA_km2': joined['UP_AREA'].values,
    })

    # Compute glacierized fraction per sub-basin
    sub_glac = (out[out['HYBAS_ID'] > 0]
                .groupby('HYBAS_ID')['glacier_area_km2']
                .sum()
                .rename('total_glacier_area_km2'))
    out = out.merge(sub_glac, on='HYBAS_ID', how='left')
    out['total_glacier_area_km2'] = out['total_glacier_area_km2'].fillna(0.0)
    out['glacierized_fraction'] = np.where(
        out['SUB_AREA_km2'] > 0,
        out['total_glacier_area_km2'] / out['SUB_AREA_km2'],
        0.0,
    )
    out = out.drop(columns=['total_glacier_area_km2'])

    log.info('assign_glaciers_to_subbasins: %d glaciers → %d unique sub-basins',
             len(out), out['HYBAS_ID'].nunique())
    return out
