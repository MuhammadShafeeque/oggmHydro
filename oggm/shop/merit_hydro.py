"""MERIT-Hydro DEM and pre-computed flow grids for OGGM.

Provides download helpers for MERIT-Hydro 90 m hydrologically conditioned
DEM tiles and pre-computed flow-direction / flow-accumulation grids.

MERIT-Hydro Reference
---------------------
Yamazaki, D., Ikeshima, D., Sosa, J., Bates, P. D., Allen, G., &
Pavelsky, T. (2019). MERIT Hydro: A high-resolution global hydrography
map based on latest topography datasets. Water Resources Research, 55,
5053-5073. https://doi.org/10.1029/2019WR024873

Data Access
-----------
MERIT-Hydro tiles require registration at:
    http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/

After downloading tiles, place them in a local directory and pass its
path via ``local_tile_dir`` or set ``cfg.PARAMS['merit_hydro_tile_dir']``.
Alternatively, set ``cfg.PARAMS['merit_hydro_server']`` to a mirror that
does not require authentication (e.g. a local HTTP server).
"""
# Built-ins
import logging
import os
import tarfile

# External
import numpy as np

# OGGM internals
from oggm import cfg
from oggm import utils

# Module logger
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default tile server (placeholder — registration required)
# ---------------------------------------------------------------------------
_DEFAULT_SERVER = (
    'http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/'
)

# Tile sub-paths within the MERIT-Hydro distribution
_PRODUCT_PATHS = {
    'dem':  'dem_tif/{tile}.tar',   # elevation [m]
    'fdir': 'flwdir/{tile}.tar',    # D8 flow direction (ESRI encoding)
    'acc':  'uparea/{tile}.tar',    # upstream area [km²]
}

# Resolution label
_RESOLUTION = '3sec'  # 3 arc-second ≈ 90 m at equator


# ---------------------------------------------------------------------------
# Tile naming utilities
# ---------------------------------------------------------------------------

def _tile_name(lat_deg, lon_deg):
    """Return the MERIT-Hydro tile identifier for a given location.

    MERIT-Hydro uses 5-degree tiles named as ``n35e070`` (lat 35-40 N,
    lon 70-75 E).  Tile origin is the south-west corner of the tile.

    Parameters
    ----------
    lat_deg : float
        Latitude in decimal degrees (south-negative).
    lon_deg : float
        Longitude in decimal degrees (west-negative).

    Returns
    -------
    str
        Tile identifier, e.g. ``'n35e070'``.
    """
    # Floor to nearest 5-degree tile origin
    lat_orig = int(np.floor(lat_deg / 5.0) * 5)
    lon_orig = int(np.floor(lon_deg / 5.0) * 5)

    lat_str = ('n' if lat_orig >= 0 else 's') + f'{abs(lat_orig):02d}'
    lon_str = ('e' if lon_orig >= 0 else 'w') + f'{abs(lon_orig):03d}'
    return lat_str + lon_str


def _tiles_for_bbox(lon_min, lat_min, lon_max, lat_max):
    """Return the set of MERIT-Hydro tile names covering a bounding box.

    Parameters
    ----------
    lon_min, lat_min, lon_max, lat_max : float
        Bounding box in decimal degrees.

    Returns
    -------
    list of str
        Unique tile identifiers covering the box.
    """
    tiles = set()
    lat = lat_min
    while lat < lat_max + 5:
        lon = lon_min
        while lon < lon_max + 5:
            tiles.add(_tile_name(lat, lon))
            lon += 5
        lat += 5
    return sorted(tiles)


def _bbox_from_gdir(gdir):
    """Extract bounding box (lon_min, lat_min, lon_max, lat_max) from a gdir.

    Parameters
    ----------
    gdir : :class:`oggm.GlacierDirectory`

    Returns
    -------
    tuple of float
        ``(lon_min, lat_min, lon_max, lat_max)`` in WGS84 degrees.
    """
    try:
        import pyproj
        from shapely.geometry import box
        from shapely.ops import transform
    except ImportError:
        raise ImportError(
            'pyproj and shapely are required to extract the bounding box '
            'from a GlacierDirectory.  Install them via '
            '"conda install -c conda-forge pyproj shapely".'
        )

    grid = gdir.grid
    proj = pyproj.Proj(grid.proj.srs)
    to_latlon = pyproj.Transformer.from_proj(
        proj, pyproj.Proj('epsg:4326'), always_xy=True
    ).transform

    xs = [grid.corner_grid.x_coord,
          grid.corner_grid.x_coord + grid.dx * grid.nx]
    ys = [grid.corner_grid.y_coord,
          grid.corner_grid.y_coord + grid.dy * grid.ny]

    corners = [to_latlon(x, y) for x in xs for y in ys]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return min(lons), min(lats), max(lons), max(lats)


# ---------------------------------------------------------------------------
# Internal download helper
# ---------------------------------------------------------------------------

def _get_merit_tile_path(tile_name, product, local_tile_dir=None):
    """Return the local path to a MERIT-Hydro product tile, downloading it
    if necessary.

    Parameters
    ----------
    tile_name : str
        Tile identifier, e.g. ``'n35e070'``.
    product : str
        One of ``'dem'``, ``'fdir'``, ``'acc'``.
    local_tile_dir : str, optional
        Directory that already contains pre-downloaded tar files.
        If *None*, uses ``cfg.PARAMS.get('merit_hydro_tile_dir', '')``.

    Returns
    -------
    str
        Path to the extracted GeoTIFF file inside the OGGM download cache.

    Raises
    ------
    RuntimeError
        If the tile cannot be found locally and no accessible server is
        configured.
    """
    if product not in _PRODUCT_PATHS:
        raise ValueError(
            f'Unknown MERIT-Hydro product {product!r}. '
            f'Choose from {list(_PRODUCT_PATHS)}.'
        )

    rel_path = _PRODUCT_PATHS[product].format(tile=tile_name)
    tar_name = os.path.basename(rel_path)
    tif_name = tar_name.replace('.tar', '.tif')

    # 1. Check local tile directory first
    if local_tile_dir is None:
        local_tile_dir = cfg.PARAMS.get('merit_hydro_tile_dir', '')

    if local_tile_dir:
        local_tar = os.path.join(local_tile_dir, tar_name)
        if os.path.isfile(local_tar):
            return _extract_tar(local_tar, tif_name)
        # Also accept a pre-extracted TIF
        local_tif = os.path.join(local_tile_dir, tif_name)
        if os.path.isfile(local_tif):
            return local_tif

    # 2. Try configured (or default) server
    server = cfg.PARAMS.get('merit_hydro_server', _DEFAULT_SERVER).rstrip('/')
    url = f'{server}/{rel_path}'

    dl_cache = cfg.PATHS.get('dl_cache_dir', '')
    if not dl_cache:
        import tempfile
        dl_cache = tempfile.mkdtemp()

    cached_tar = os.path.join(dl_cache, 'merit_hydro', tar_name)
    os.makedirs(os.path.dirname(cached_tar), exist_ok=True)

    if not os.path.isfile(cached_tar):
        log.info('Downloading MERIT-Hydro tile %s (%s) from %s',
                 tile_name, product, url)
        try:
            utils.oggm_urlretrieve(url, cached_tar)
        except Exception as exc:
            raise RuntimeError(
                f'Could not download MERIT-Hydro tile from {url}.\n'
                'MERIT-Hydro requires free registration at:\n'
                '  http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/\n'
                'After downloading tiles, either:\n'
                '  (a) set cfg.PARAMS["merit_hydro_tile_dir"] to the '
                'directory containing the .tar files, or\n'
                '  (b) set cfg.PARAMS["merit_hydro_server"] to a '
                'mirror URL.\n'
                f'Original error: {exc}'
            ) from exc

    return _extract_tar(cached_tar, tif_name)


def _extract_tar(tar_path, tif_name):
    """Extract a GeoTIFF from a tar archive and return its path.

    Parameters
    ----------
    tar_path : str
        Path to the `.tar` archive.
    tif_name : str
        Name of the TIF file to extract (may be any member containing
        this string).

    Returns
    -------
    str
        Path to the extracted TIF file.
    """
    extract_dir = os.path.splitext(tar_path)[0]  # same name without .tar
    os.makedirs(extract_dir, exist_ok=True)

    out_tif = os.path.join(extract_dir, tif_name)
    if os.path.isfile(out_tif):
        return out_tif

    with tarfile.open(tar_path, 'r') as tf:
        members = tf.getnames()
        # Find the TIF member (case-insensitive partial match)
        match = next(
            (m for m in members if tif_name.lower() in m.lower()), None
        )
        if match is None:
            # Try any .tif member
            match = next(
                (m for m in members if m.lower().endswith('.tif')), None
            )
        if match is None:
            raise RuntimeError(
                f'Could not find a .tif file in {tar_path}. '
                f'Archive members: {members}'
            )
        tf.extract(match, path=extract_dir)
        extracted = os.path.join(extract_dir, os.path.basename(match))
        if extracted != out_tif:
            import shutil
            shutil.move(extracted, out_tif)

    return out_tif


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_merit_dem(gdir_or_bbox, local_tile_dir=None):
    """Download and return the MERIT-Hydro 90 m DEM for a glacier/bbox.

    Parameters
    ----------
    gdir_or_bbox : :class:`oggm.GlacierDirectory` or tuple
        Either a GlacierDirectory (bounding box extracted automatically) or
        a ``(lon_min, lat_min, lon_max, lat_max)`` tuple in WGS84 degrees.
    local_tile_dir : str, optional
        Directory containing pre-downloaded MERIT-Hydro tar files.

    Returns
    -------
    xr.DataArray
        Mosaicked and clipped DEM [m] on a regular WGS84 grid.  NaN for
        nodata.

    Notes
    -----
    Tiles are cached in ``cfg.PATHS['dl_cache_dir']/merit_hydro/``.
    """
    return _load_merit_product(gdir_or_bbox, 'dem', local_tile_dir)


def get_merit_fdir(gdir_or_bbox, local_tile_dir=None):
    """Download and return the pre-computed MERIT-Hydro D8 flow direction.

    The flow direction uses the ESRI encoding (powers of 2: E=1, SE=2,
    S=4, SW=8, W=16, NW=32, N=64, NE=128).

    Parameters
    ----------
    gdir_or_bbox : :class:`oggm.GlacierDirectory` or tuple
    local_tile_dir : str, optional

    Returns
    -------
    xr.DataArray
        D8 flow direction grid.  0 = sink or nodata.
    """
    return _load_merit_product(gdir_or_bbox, 'fdir', local_tile_dir)


def get_merit_acc(gdir_or_bbox, local_tile_dir=None):
    """Download and return the pre-computed MERIT-Hydro upstream area grid.

    Parameters
    ----------
    gdir_or_bbox : :class:`oggm.GlacierDirectory` or tuple
    local_tile_dir : str, optional

    Returns
    -------
    xr.DataArray
        Upstream area [km²] grid.
    """
    return _load_merit_product(gdir_or_bbox, 'acc', local_tile_dir)


def _load_merit_product(gdir_or_bbox, product, local_tile_dir=None):
    """Internal: load a MERIT-Hydro product for a region.

    Parameters
    ----------
    gdir_or_bbox : GlacierDirectory or (lon_min, lat_min, lon_max, lat_max)
    product : str
    local_tile_dir : str, optional

    Returns
    -------
    xr.DataArray
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError('xarray is required. Install via conda.')

    try:
        import rasterio
        from rasterio.merge import merge as rio_merge
        from rasterio.enums import Resampling as RioResampling
    except ImportError:
        raise ImportError('rasterio is required. Install via conda.')

    # --- Resolve bounding box ---
    if hasattr(gdir_or_bbox, 'grid'):
        bbox = _bbox_from_gdir(gdir_or_bbox)
    else:
        bbox = tuple(gdir_or_bbox)
        if len(bbox) != 4:
            raise ValueError(
                'bbox must be (lon_min, lat_min, lon_max, lat_max).'
            )

    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = _tiles_for_bbox(lon_min, lat_min, lon_max, lat_max)

    log.debug('MERIT-Hydro %s: tiles = %s for bbox %s',
              product, tiles, bbox)

    # --- Load and mosaic tiles ---
    tif_paths = [
        _get_merit_tile_path(t, product, local_tile_dir) for t in tiles
    ]

    datasets = [rasterio.open(p) for p in tif_paths]
    mosaic, transform = rio_merge(datasets)
    meta = datasets[0].meta.copy()
    for ds in datasets:
        ds.close()

    meta.update({
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': transform,
    })

    # Convert to xr.DataArray
    nodata = meta.get('nodata', -9999)
    arr = mosaic[0].astype(float)
    arr[arr == nodata] = np.nan

    # Build coordinate arrays from rasterio transform
    height, width = arr.shape
    import affine
    x0 = transform.c + transform.a * 0.5   # centre of first column
    y0 = transform.f + transform.e * 0.5   # centre of first row
    lons = x0 + np.arange(width) * transform.a
    lats = y0 + np.arange(height) * transform.e

    da = xr.DataArray(
        arr,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        attrs={
            'source': 'MERIT-Hydro',
            'product': product,
            'units': 'm' if product == 'dem' else (
                'km2' if product == 'acc' else 'ESRI-D8'),
            'tiles': ', '.join(tiles),
        },
    )
    # Clip to requested bbox
    da = da.sel(
        lon=slice(lon_min, lon_max),
        lat=slice(lat_max, lat_min),  # lat usually descending
    )
    return da
