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
# Standard:            <server>/hydrobasins/standard/hybas_{region}_lev{NN}_v1c.zip
# Customized (lakes):  <server>/hydrobasins/customized_with_lakes/hybas_lake_{region}_lev{NN}_v1c.zip
# "Customized with lakes" is preferred for glacierized catchments because it
# incorporates lake boundaries into sub-basin delineations.
_HYBAS_PATH_STANDARD = 'hydrobasins/standard/hybas_{region}_lev{level:02d}_v1c.zip'
_HYBAS_PATH_LAKES    = 'hydrobasins/customized_with_lakes/hybas_lake_{region}_lev{level:02d}_v1c.zip'

# HydroRIVERS: one ZIP per region
# URL: <server>/hydrorivers/HydroRIVERS_v10_{region}_shp.zip
_HYDRIV_PATH = 'hydrorivers/HydroRIVERS_v10_{region}_shp.zip'

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


def _get_hydrobasins_file(region, level, local_dir='', use_lakes=True):
    """Return a geopandas-readable path for the HydroBASINS shapefile.

    Search order
    ------------
    1. *local_dir* — individual level ZIP  (``hybas_lake_{r}_lev{NN}_v1c.zip``)
    2. *local_dir* — combined all-levels ZIP (``hybas_lake_{r}_lev01-12_v1c.zip``)
       with the specific level's shapefile referenced via a GDAL
       ``/vsizip/`` path so geopandas reads only the requested layer.
    3. Download the individual level ZIP via :func:`oggm.utils.file_downloader`.

    Parameters
    ----------
    region : str
        Two-letter HydroSHEDS region code (e.g. ``'as'``).
    level : int
        Pfaffstetter aggregation level 1–12.
    local_dir : str
        Directory that may contain pre-downloaded ZIP files.
    use_lakes : bool
        ``True`` (default) → "customized with lakes" variant
        (``hybas_lake_*``).  ``False`` → standard variant (``hybas_*``).

    Returns
    -------
    str
        A path string that can be passed **directly** to
        :func:`geopandas.read_file`.  This is either:
        * ``'zip://' + local_path``  for a standalone level ZIP, or
        * ``'/vsizip/{combo_zip}/{shp_name}'``  for a combined ZIP.
    """
    import zipfile as _zipfile

    # ---- filename components ------------------------------------------------
    if use_lakes:
        shp_name     = f'hybas_lake_{region}_lev{level:02d}_v1c.shp'
        indiv_zip    = f'hybas_lake_{region}_lev{level:02d}_v1c.zip'
        combo_zip    = f'hybas_lake_{region}_lev01-12_v1c.zip'
        fb_shp_name  = f'hybas_{region}_lev{level:02d}_v1c.shp'
        fb_indiv_zip = f'hybas_{region}_lev{level:02d}_v1c.zip'
        fb_combo_zip = f'hybas_{region}_lev01-12_v1c.zip'
    else:
        shp_name     = f'hybas_{region}_lev{level:02d}_v1c.shp'
        indiv_zip    = f'hybas_{region}_lev{level:02d}_v1c.zip'
        combo_zip    = f'hybas_{region}_lev01-12_v1c.zip'
        fb_shp_name  = f'hybas_lake_{region}_lev{level:02d}_v1c.shp'
        fb_indiv_zip = f'hybas_lake_{region}_lev{level:02d}_v1c.zip'
        fb_combo_zip = f'hybas_lake_{region}_lev01-12_v1c.zip'

    if local_dir:
        # 1. Individual level ZIP (preferred — one shapefile per zip)
        for fname in (indiv_zip, fb_indiv_zip):
            candidate = os.path.join(local_dir, fname)
            if os.path.isfile(candidate):
                log.debug('_get_hydrobasins_file: using %s', candidate)
                return 'zip://' + candidate

        # 2. Combined all-levels ZIP → GDAL /vsizip/ with explicit layer
        for (combo, shp) in [(combo_zip, shp_name),
                              (fb_combo_zip, fb_shp_name)]:
            combo_path = os.path.join(local_dir, combo)
            if not os.path.isfile(combo_path):
                continue
            try:
                with _zipfile.ZipFile(combo_path) as zf:
                    basenames = {os.path.basename(n) for n in zf.namelist()}
            except Exception:
                continue
            # Pick whichever shapefile name is present
            hit = shp if shp in basenames else (fb_shp_name if fb_shp_name in basenames else None)
            if hit is not None:
                read_path = f'/vsizip/{combo_path}/{hit}'
                log.debug('_get_hydrobasins_file: using combined zip %s !%s',
                          combo_path, hit)
                return read_path

    # 3. Download the individual level ZIP
    url_path = (_HYBAS_PATH_LAKES if use_lakes else _HYBAS_PATH_STANDARD)
    url = _hydrosheds_server() + url_path.format(region=region, level=level)
    downloaded = utils.file_downloader(url)
    if downloaded is None:
        raise FileNotFoundError(
            f'Could not download HydroBASINS level-{level} for region '
            f'"{region}" from:\n  {url}\n'
            'Set cfg.PARAMS["hydrobasins_local_dir"] to a directory '
            'containing the pre-downloaded ZIP (individual level or combined '
            'lev01-12), or cfg.PARAMS["hydrobasins_server"] to a mirror.'
        )
    return 'zip://' + downloaded


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

def get_hydrobasins(bbox, level=8, region=None, use_lakes=True):
    """Download and return HydroBASINS sub-basin polygons for a bounding box.

    Parameters
    ----------
    bbox : tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in decimal degrees.
    level : int, optional
        Pfaffstetter aggregation level 1–12.  Level 8 (default) gives
        typical sub-basin areas of ~10–200 km², appropriate for attributing
        individual glaciers to their enclosing sub-basins.
    region : str, optional
        Two-letter HydroSHEDS region code.  Inferred automatically from
        *bbox* if not provided.
    use_lakes : bool, optional
        If ``True`` (default), use the HydroBASINS "customized with lakes"
        variant, which incorporates lake boundaries into sub-basin
        delineations.  Recommended for glacierized catchments where
        glacial lakes and reservoirs are present.  Set ``False`` to use the
        standard variant.

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
    # _get_hydrobasins_file returns a full geopandas-readable path:
    # 'zip://...' for standalone ZIPs, '/vsizip/...' for combined ZIPs.
    read_path = _get_hydrobasins_file(region, level, local_dir=local_dir,
                                      use_lakes=use_lakes)

    gdf = gpd.read_file(read_path)
    # Clip to bounding box (spatial subset)
    gdf = gdf.cx[lon_min:lon_max, lat_min:lat_max].copy()
    gdf = gdf.reset_index(drop=True)

    log.info('get_hydrobasins: %d sub-basins in bbox for region=%s level=%d '
             '(use_lakes=%s)', len(gdf), region, level, use_lakes)
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


def assign_glaciers_to_subbasins_polygon(rgi_gdf, subbasins_gdf):
    """Polygon-overlay glacier-to-subbasin assignment.

    Intersects RGI glacier outline polygons with HydroBASINS sub-basin
    polygons so that:

    * A glacier straddling two sub-basins contributes discharge to *both*,
      weighted by the fraction of its area in each sub-basin.
    * Per-sub-basin glacierized fraction is computed from actual polygon
      overlap area, not from centroid containment.

    Parameters
    ----------
    rgi_gdf : :class:`geopandas.GeoDataFrame`
        RGI glacier outlines with at minimum columns ``RGIId`` (or ``rgi_id``)
        and a polygon ``geometry`` column in any geographic CRS.  An ``Area``
        (km²) column is used if present; otherwise it is computed from the
        geometry.
    subbasins_gdf : :class:`geopandas.GeoDataFrame`
        HydroBASINS polygons as returned by :func:`get_hydrobasins`.
        Must contain ``HYBAS_ID``, ``SUB_AREA``, ``UP_AREA``, ``geometry``.

    Returns
    -------
    :class:`pandas.DataFrame`
        One row per *(glacier, sub-basin)* intersection pair.  Columns:

        ``rgi_id``
            RGI glacier identifier.
        ``HYBAS_ID``
            HydroBASINS sub-basin identifier (−1 for glaciers outside all
            sub-basins).
        ``SUB_AREA_km2``
            Total area of the sub-basin [km²].
        ``UP_AREA_km2``
            Upstream drainage area of the sub-basin [km²].
        ``area_in_subbasin_km2``
            Area of the glacier–sub-basin intersection [km²].
        ``area_fraction``
            Fraction of the glacier's total area that falls in this
            sub-basin [0–1].  Sums to 1.0 for a given ``rgi_id`` when all
            of its area is within the bbox sub-basins.
        ``glacier_area_km2``
            Total glacier area [km²].
        ``glacierized_fraction``
            ``area_in_subbasin_km2 / SUB_AREA_km2`` — fraction of the
            sub-basin covered by this glacier's portion.

    Notes
    -----
    Area calculations are performed in the World Equal-Area Cylindrical
    projection (EPSG:6933) to minimise distortion at mid-to-high latitudes.
    Glaciers whose centroid (or any part) falls outside all sub-basin
    polygons are assigned ``HYBAS_ID = −1`` with ``area_fraction = 1.0``.

    Raises
    ------
    ImportError
        If *geopandas* is not installed.
    ValueError
        If *rgi_gdf* or *subbasins_gdf* is missing required columns.
    """
    if not HAS_GEOPANDAS:
        raise ImportError(
            'geopandas is required for assign_glaciers_to_subbasins_polygon(). '
            'Install it with: conda install geopandas'
        )

    required_subs = {'HYBAS_ID', 'SUB_AREA', 'UP_AREA', 'geometry'}
    missing = required_subs - set(subbasins_gdf.columns)
    if missing:
        raise ValueError(
            f'subbasins_gdf is missing required columns: {missing}.'
        )

    # Normalise RGI column name
    rgi_col = 'RGIId' if 'RGIId' in rgi_gdf.columns else 'rgi_id'
    if rgi_col not in rgi_gdf.columns:
        raise ValueError(
            'rgi_gdf must contain an "RGIId" or "rgi_id" column.'
        )

    # Equal-area projection for robust area calculations
    _EA_CRS = 'EPSG:6933'

    # Prepare glaciers
    gl = rgi_gdf[[rgi_col, 'geometry']].copy().rename(
        columns={rgi_col: 'rgi_id'})
    if gl.crs is None:
        gl = gl.set_crs('EPSG:4326')
    else:
        gl = gl.to_crs('EPSG:4326')

    # Compute total glacier area from geometry (authoritative)
    gl_ea = gl.to_crs(_EA_CRS)
    gl['glacier_area_km2'] = gl_ea.geometry.area / 1e6

    # Prepare sub-basins
    subs = subbasins_gdf[['HYBAS_ID', 'SUB_AREA', 'UP_AREA',
                           'geometry']].copy()
    if subs.crs is None:
        subs = subs.set_crs('EPSG:4326')
    else:
        subs = subs.to_crs('EPSG:4326')

    # Polygon intersection
    intersected = gpd.overlay(gl, subs, how='intersection', keep_geom_type=False)

    # Drop degenerate geometries (slivers from shared boundaries)
    intersected = intersected[~intersected.geometry.is_empty].copy()

    # Compute intersection area in equal-area projection
    isct_ea = intersected.to_crs(_EA_CRS)
    intersected['area_in_subbasin_km2'] = isct_ea.geometry.area / 1e6

    # Drop tiny slivers (< 0.001 km² numerical noise from shared edges)
    intersected = intersected[
        intersected['area_in_subbasin_km2'] > 0.001].copy()

    # Compute area fraction for each glacier
    total_area_per_glacier = (intersected
                              .groupby('rgi_id')['area_in_subbasin_km2']
                              .sum())
    intersected['area_fraction'] = (
        intersected['area_in_subbasin_km2']
        / intersected['rgi_id'].map(total_area_per_glacier)
    )

    # Glacierized fraction of each sub-basin (this intersection piece only)
    intersected['glacierized_fraction'] = (
        intersected['area_in_subbasin_km2']
        / intersected['SUB_AREA'].clip(lower=0.001)
    )

    # Glaciers completely outside all sub-basins
    assigned_rgi = set(intersected['rgi_id'])
    all_rgi = set(gl['rgi_id'])
    outside = all_rgi - assigned_rgi
    if outside:
        outside_rows = []
        ga_map = dict(zip(gl['rgi_id'], gl['glacier_area_km2']))
        for rgi_id in outside:
            outside_rows.append({
                'rgi_id': rgi_id,
                'HYBAS_ID': -1,
                'SUB_AREA': 0.0,
                'UP_AREA': 0.0,
                'area_in_subbasin_km2': ga_map.get(rgi_id, 0.0),
                'area_fraction': 1.0,
                'glacier_area_km2': ga_map.get(rgi_id, 0.0),
                'glacierized_fraction': 0.0,
            })
        intersected = pd.concat(
            [intersected, pd.DataFrame(outside_rows)], ignore_index=True)

    out = intersected[['rgi_id', 'HYBAS_ID', 'SUB_AREA', 'UP_AREA',
                        'area_in_subbasin_km2', 'area_fraction',
                        'glacier_area_km2', 'glacierized_fraction']].copy()
    out = out.rename(columns={'SUB_AREA': 'SUB_AREA_km2',
                               'UP_AREA': 'UP_AREA_km2'})
    out = out.reset_index(drop=True)

    n_pairs = len(out[out['HYBAS_ID'] > 0])
    n_glaciers = out[out['HYBAS_ID'] > 0]['rgi_id'].nunique()
    n_outside = len(outside)
    log.info(
        'assign_glaciers_to_subbasins_polygon: %d glaciers → '
        '%d (glacier, subbasin) pairs across %d sub-basins; '
        '%d glaciers outside bbox',
        len(all_rgi), n_pairs, out[out['HYBAS_ID'] > 0]['HYBAS_ID'].nunique(),
        n_outside,
    )
    return out
