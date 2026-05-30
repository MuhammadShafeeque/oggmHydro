"""Terrain routing engine for OGGM glacial hydrology.

Provides DEM-based flow direction, flow accumulation, stream
delineation, and stream network construction for basin-scale
discharge routing.

Phase 6: D8 flow direction, flow accumulation, slope/aspect,
         stream delineation, stream network (NetworkX graph),
         sub-basin assignment.
Phase 7: Muskingum-Cunge channel routing (see route_stream_network).

Notes
-----
All functions operate on 2-D NumPy arrays.  Geographic context
(CRS, resolution) is passed explicitly via *cellsize_m* so the
functions remain independent of any particular raster library.

The D8 encoding follows the ESRI convention (powers of 2):

    NW  N  NE      32  64  128
    W   .   E  →   16   0    1
    SW  S  SE       8   4    2

    Code 0 indicates a sink, boundary cell, or nodata.

Dependencies
------------
Required : numpy, scipy
Optional : pysheds  (accelerates D8/accumulation on large grids;
                     falls back to pure-NumPy implementation)
           networkx (required for build_stream_network and
                     route_stream_network)
"""
# Built-ins
import logging
import warnings

# External — required
import numpy as np
from scipy.ndimage import label as _nd_label

# External — optional
try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

try:
    from pysheds.grid import Grid as _PyshedsGrid
    _HAS_PYSHEDS = True
except ImportError:
    _HAS_PYSHEDS = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# D8 encoding constants (ESRI convention)
# ---------------------------------------------------------------------------
#: Mapping from D8 code → (row_offset, col_offset).
#: Row increases **downward** in raster convention.
D8_OFFSETS = {
    1:   ( 0,  1),   # E
    2:   ( 1,  1),   # SE
    4:   ( 1,  0),   # S
    8:   ( 1, -1),   # SW
    16:  ( 0, -1),   # W
    32:  (-1, -1),   # NW
    64:  (-1,  0),   # N
    128: (-1,  1),   # NE
}

_D8_CODES = np.array(sorted(D8_OFFSETS.keys()), dtype=np.int32)
_D8_DR = np.array([D8_OFFSETS[c][0] for c in sorted(D8_OFFSETS)],
                  dtype=np.int32)
_D8_DC = np.array([D8_OFFSETS[c][1] for c in sorted(D8_OFFSETS)],
                  dtype=np.int32)

# Reverse: from code to array index (code 2^k → index k)
_D8_CODE_TO_IDX = {c: i for i, c in enumerate(sorted(D8_OFFSETS))}

# Opposite direction (used to detect inflows)
_D8_OPPOSITE = {1: 16, 16: 1, 2: 32, 32: 2, 4: 64, 64: 4, 8: 128, 128: 8}


# ---------------------------------------------------------------------------
# Pit filling (simple priority-queue approach)
# ---------------------------------------------------------------------------

def fill_pits(dem_arr):
    """Fill single-cell pits in a DEM.

    A pit is a cell lower than all 8 neighbours.  This simple filler
    raises the pit cell to the minimum neighbour elevation.  For
    multi-cell depressions, call this iteratively or use pysheds.

    Parameters
    ----------
    dem_arr : np.ndarray, shape (nrows, ncols)
        Elevation in metres.  NaN = nodata.

    Returns
    -------
    filled : np.ndarray
        Pit-filled elevation array (same shape, same dtype).
    """
    filled = dem_arr.copy()
    nrows, ncols = dem_arr.shape

    changed = True
    while changed:
        changed = False
        for di in range(8):
            dr, dc = int(_D8_DR[di]), int(_D8_DC[di])
            # Shift arrays
            r0, r1 = max(0, -dr), min(nrows, nrows - dr)
            c0, c1 = max(0, -dc), min(ncols, ncols - dc)
            r0n, r1n = r0 + dr, r1 + dr
            c0n, c1n = c0 + dc, c1 + dc

            center = filled[r0:r1, c0:c1]
            neigh = filled[r0n:r1n, c0n:c1n]

            # Centre is a pit relative to this neighbour
            is_pit = (center < neigh) & ~np.isnan(center) & ~np.isnan(neigh)
            # But also lower than ALL neighbours? — approximate by checking
            # if raising to neigh makes centre ≥ the opposite neighbour.
            # Simple single-pass: raise pits found w.r.t. minimum neighbour.
            new_val = np.where(is_pit, neigh, center)
            if np.any(new_val != center):
                filled[r0:r1, c0:c1] = new_val
                changed = True

    return filled


# ---------------------------------------------------------------------------
# Phase 6 — D8 flow direction
# ---------------------------------------------------------------------------

def compute_flow_direction(dem_arr, cellsize_m, method='D8',
                           fill_pits_first=True):
    """Compute D8 flow direction from a DEM.

    Each interior cell is assigned the direction of steepest downslope
    descent among its 8 neighbours.  Diagonal neighbours use distance
    ``sqrt(2) * cellsize_m``; cardinal neighbours use ``cellsize_m``.

    Uses *pysheds* when available (handles complex depressions better);
    falls back to a vectorised pure-NumPy implementation otherwise.

    Parameters
    ----------
    dem_arr : np.ndarray, shape (nrows, ncols)
        Elevation in metres.  NaN for nodata / outside domain.
    cellsize_m : float
        Cell size in metres (assumed square).
    method : str
        Flow direction algorithm.  Only ``'D8'`` is currently supported.
    fill_pits_first : bool
        If True, apply a simple pit-fill before computing flow direction.
        Recommended for real DEMs; disable for synthetic test cases.

    Returns
    -------
    fdir : np.ndarray, shape (nrows, ncols), dtype int32
        D8 flow direction (ESRI encoding: 1, 2, 4, 8, 16, 32, 64, 128).
        0 = sink / nodata / boundary.

    Notes
    -----
    For large DEMs or complex depression topography, install *pysheds*::

        conda install -c conda-forge pysheds
    """
    if method != 'D8':
        raise ValueError(f'Only D8 is supported, got {method!r}')

    dem = np.asarray(dem_arr, dtype=float)

    if fill_pits_first:
        dem = fill_pits(dem)

    if _HAS_PYSHEDS:
        return _d8_pysheds(dem, cellsize_m)
    else:
        return _d8_numpy(dem, cellsize_m)


def _d8_numpy(dem, cellsize_m):
    """Vectorised pure-NumPy D8 flow direction."""
    nrows, ncols = dem.shape

    # Pad with +inf so boundary cells always have a lower neighbour inside
    pad = np.pad(dem, 1, mode='constant', constant_values=np.inf)

    best_slope = np.full((nrows, ncols), -np.inf)
    fdir = np.zeros((nrows, ncols), dtype=np.int32)

    for i in range(8):
        code = int(_D8_CODES[i])
        dr = int(_D8_DR[i])
        dc = int(_D8_DC[i])

        # Distance to this neighbour
        dist = (np.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0) * cellsize_m

        # Neighbour elevation in the padded array
        # Padded centre is at row = r+1, col = c+1
        neigh = pad[1 + dr: nrows + 1 + dr,
                    1 + dc: ncols + 1 + dc]

        slope = (dem - neigh) / dist  # positive = downhill

        mask = slope > best_slope
        best_slope = np.where(mask, slope, best_slope)
        fdir = np.where(mask, code, fdir)

    # Cells where no positive slope exists → sink (code 0)
    fdir = np.where(best_slope <= 0.0, 0, fdir)
    # Nodata cells
    fdir = np.where(np.isnan(dem), 0, fdir)
    return fdir.astype(np.int32)


def _d8_pysheds(dem, cellsize_m):
    """Compute D8 flow direction using pysheds.

    Returns the same ESRI-encoded int32 array as :func:`_d8_numpy`.
    """
    try:
        import affine
        from pysheds.grid import Grid
        from pysheds.view import Raster as PyshedsRaster
    except ImportError:
        return _d8_numpy(dem, cellsize_m)

    nrows, ncols = dem.shape
    aff = affine.Affine(cellsize_m, 0, 0, 0, -cellsize_m, nrows * cellsize_m)

    nodata = -9999.0
    dem_clean = np.where(np.isnan(dem), nodata, dem)

    grid = Grid()
    raster = PyshedsRaster(dem_clean, affine=aff, nodata=nodata,
                           crs='+proj=longlat +datum=WGS84')
    pit_filled = grid.fill_pits(raster)
    conditioned = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(conditioned)
    fdir_ps = grid.flowdir(inflated)

    # pysheds uses the same ESRI D8 encoding → cast directly
    fdir = np.asarray(fdir_ps, dtype=np.int32)
    fdir[np.isnan(dem)] = 0
    return fdir


# ---------------------------------------------------------------------------
# Phase 6 — Flow accumulation
# ---------------------------------------------------------------------------

def compute_flow_accumulation(fdir_arr):
    """Compute upstream cell count (flow accumulation) from D8 pointer grid.

    Uses a Kahn topological-sort so that each cell is visited only after
    all its upstream neighbours have been processed.

    Parameters
    ----------
    fdir_arr : np.ndarray, shape (nrows, ncols), dtype int-like
        D8 flow direction (ESRI encoding).  0 = sink / nodata.

    Returns
    -------
    acc : np.ndarray, shape (nrows, ncols), dtype int64
        Number of upstream cells including the cell itself (≥ 1 for
        cells with a valid flow direction, 1 for headwaters).
        Cells with ``fdir == 0`` receive the value 1 (self-only).
    """
    from collections import deque

    fdir = np.asarray(fdir_arr, dtype=np.int32)
    nrows, ncols = fdir.shape

    # Each cell counts itself
    acc = np.ones((nrows, ncols), dtype=np.int64)

    # Build receiver arrays and in-degree counter
    recv_r = np.full((nrows, ncols), -1, dtype=np.int32)
    recv_c = np.full((nrows, ncols), -1, dtype=np.int32)
    n_in = np.zeros((nrows, ncols), dtype=np.int32)

    for code, (dr, dc) in D8_OFFSETS.items():
        mask = fdir == code
        rows, cols = np.where(mask)
        rr = rows + dr
        cc = cols + dc
        # Only valid in-bounds receivers
        valid = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
        recv_r[rows[valid], cols[valid]] = rr[valid]
        recv_c[rows[valid], cols[valid]] = cc[valid]
        np.add.at(n_in, (rr[valid], cc[valid]), 1)

    # Initialise queue with all headwater / sink cells (n_in == 0)
    r0, c0 = np.where(n_in == 0)
    queue = deque(zip(r0.tolist(), c0.tolist()))

    while queue:
        r, c = queue.popleft()
        rr = int(recv_r[r, c])
        cc = int(recv_c[r, c])
        if rr >= 0:  # has a valid receiver
            acc[rr, cc] += acc[r, c]
            n_in[rr, cc] -= 1
            if n_in[rr, cc] == 0:
                queue.append((rr, cc))

    return acc


# ---------------------------------------------------------------------------
# Phase 6 — Slope and aspect
# ---------------------------------------------------------------------------

def compute_slope_aspect(dem_arr, cellsize_m):
    """Compute slope (degrees) and aspect (degrees from N) using
    Horn (1981) 3×3 finite differences (same algorithm as GDAL/ArcGIS).

    Parameters
    ----------
    dem_arr : np.ndarray, shape (nrows, ncols)
        Elevation in metres.  NaN = nodata.
    cellsize_m : float
        Cell size in metres.

    Returns
    -------
    slope_deg : np.ndarray, shape (nrows, ncols)
        Terrain slope in degrees [0, 90].  NaN at borders / nodata.
    aspect_deg : np.ndarray, shape (nrows, ncols)
        Aspect in degrees clockwise from North [0, 360).
        NaN at flat cells and borders.

    Notes
    -----
    Horn (1981) weights:

        a b c       dz/dx = ((c + 2f + i) - (a + 2d + g)) / (8 * cs)
        d e f       dz/dy = ((a + 2b + c) - (g + 2h + i)) / (8 * cs)
        g h i
    """
    dem = np.asarray(dem_arr, dtype=float)
    nrows, ncols = dem.shape

    slope_deg = np.full((nrows, ncols), np.nan)
    aspect_deg = np.full((nrows, ncols), np.nan)

    # Only compute for interior cells (skip 1-pixel border)
    a = dem[:-2, :-2]   # NW
    b = dem[:-2, 1:-1]  # N
    c = dem[:-2, 2:]    # NE
    d = dem[1:-1, :-2]  # W
    f = dem[1:-1, 2:]   # E
    g = dem[2:, :-2]    # SW
    h = dem[2:, 1:-1]   # S
    i_ = dem[2:, 2:]    # SE

    dzdx = ((c + 2 * f + i_) - (a + 2 * d + g)) / (8.0 * cellsize_m)
    dzdy = ((a + 2 * b + c) - (g + 2 * h + i_)) / (8.0 * cellsize_m)

    s = np.degrees(np.arctan(np.sqrt(dzdx ** 2 + dzdy ** 2)))
    a_rad = np.degrees(np.arctan2(-dzdy, dzdx))
    # Convert to compass bearing (0 = N, 90 = E, ...)
    asp = 90.0 - a_rad
    asp = np.where(asp < 0, asp + 360.0, asp)

    slope_deg[1:-1, 1:-1] = s
    aspect_deg[1:-1, 1:-1] = np.where(s < 1e-9, np.nan, asp)

    return slope_deg, aspect_deg


# ---------------------------------------------------------------------------
# Phase 6 — Stream delineation
# ---------------------------------------------------------------------------

def delineate_streams(acc_arr, threshold_cells=None,
                      threshold_km2=None, cellsize_m=None):
    """Delineate stream cells from a flow accumulation grid.

    Exactly one of *threshold_cells* or *threshold_km2* must be supplied.

    Parameters
    ----------
    acc_arr : np.ndarray, shape (nrows, ncols)
        Flow accumulation (cells upstream including self).
    threshold_cells : int, optional
        Minimum upstream cells to define a stream.
    threshold_km2 : float, optional
        Minimum upstream area [km²] to define a stream.  Requires
        *cellsize_m* to be provided.
    cellsize_m : float, optional
        Cell size in metres; required when *threshold_km2* is given.

    Returns
    -------
    streams : np.ndarray, shape (nrows, ncols), dtype bool
        True where accumulation ≥ threshold.
    """
    if threshold_cells is None and threshold_km2 is None:
        raise ValueError(
            'Provide either threshold_cells or threshold_km2.'
        )
    if threshold_km2 is not None:
        if cellsize_m is None:
            raise ValueError(
                'cellsize_m is required when threshold_km2 is given.'
            )
        cell_area_km2 = (cellsize_m / 1000.0) ** 2
        threshold_cells = int(np.ceil(threshold_km2 / cell_area_km2))

    return np.asarray(acc_arr) >= threshold_cells


# ---------------------------------------------------------------------------
# Phase 6 — Stream network (NetworkX)
# ---------------------------------------------------------------------------

def build_stream_network(streams_arr, fdir_arr, dem_arr,
                         cellsize_m, acc_arr=None):
    """Convert a stream grid into a directed NetworkX graph.

    Nodes represent **junction**, **headwater**, and **outlet** stream
    cells.  Directed edges represent stream reaches connecting pairs of
    nodes, with topological attributes.

    Parameters
    ----------
    streams_arr : np.ndarray, shape (nrows, ncols), dtype bool
        Stream mask (True = stream cell).
    fdir_arr : np.ndarray, shape (nrows, ncols), dtype int32
        D8 flow direction (ESRI encoding).
    dem_arr : np.ndarray, shape (nrows, ncols)
        Elevation in metres.
    cellsize_m : float
        Cell size in metres.
    acc_arr : np.ndarray, optional
        Flow accumulation grid; used to populate ``upstream_area_km2``
        node attribute.

    Returns
    -------
    G : nx.DiGraph
        Directed graph.

        Node attributes (r, c) → row/col index:
            - ``elevation_m``      : float, cell elevation [m]
            - ``upstream_area_km2``: float (if acc_arr given) [km²]
            - ``is_junction``      : bool
            - ``is_headwater``     : bool
            - ``is_outlet``        : bool

        Edge attributes (u, v) → upstream to downstream node:
            - ``length_m``         : float, reach length [m]
            - ``mean_elev_m``      : float, mean elevation along reach [m]
            - ``mean_slope``       : float, mean slope [m/m]
            - ``n_cells``          : int, number of cells in reach

    Raises
    ------
    ImportError
        If *networkx* is not installed.

    Notes
    -----
    Algorithm
        1. For each stream cell, count how many **stream** neighbours
           flow into it → inflow degree.
        2. Nodes = cells with inflow_degree != 1 **or** no downstream
           stream cell (outlets).
        3. For each headwater/junction node, trace downstream until the
           next node, accumulating reach attributes.
    """
    if not _HAS_NX:
        raise ImportError(
            'networkx is required for build_stream_network. '
            'Install via: conda install -c conda-forge networkx'
        )

    streams = np.asarray(streams_arr, dtype=bool)
    fdir = np.asarray(fdir_arr, dtype=np.int32)
    dem = np.asarray(dem_arr, dtype=float)
    nrows, ncols = streams.shape

    cell_area_km2 = (cellsize_m / 1000.0) ** 2

    # ------------------------------------------------------------------
    # 1.  Compute stream-inflow degree for each stream cell
    # ------------------------------------------------------------------
    stream_inflow = np.zeros((nrows, ncols), dtype=np.int32)

    for code, (dr, dc) in D8_OFFSETS.items():
        # Find stream cells that flow in direction (dr, dc)
        r0, c0 = np.where(streams & (fdir == code))
        rr, cc = r0 + dr, c0 + dc
        valid = (rr >= 0) & (rr < nrows) & (cc >= 0) & (cc < ncols)
        # The receiver cell is a stream cell receiving inflow
        np.add.at(stream_inflow,
                  (rr[valid & streams[rr[valid], cc[valid]]],
                   cc[valid & streams[rr[valid], cc[valid]]]),
                  1)

    # ------------------------------------------------------------------
    # 2.  Identify node cells: headwaters (in=0), junctions (in≥2),
    #     outlets (no downstream stream cell)
    # ------------------------------------------------------------------
    def _has_stream_receiver(r, c):
        code = int(fdir[r, c])
        if code == 0:
            return False
        dr, dc = D8_OFFSETS[code]
        rr, cc_ = r + dr, c + dc
        return (0 <= rr < nrows and 0 <= cc_ < ncols
                and streams[rr, cc_])

    is_headwater = streams & (stream_inflow == 0)
    is_junction = streams & (stream_inflow >= 2)
    # Outlets: stream cell that does not flow to another stream cell
    is_outlet = np.zeros((nrows, ncols), dtype=bool)
    all_stream_rows, all_stream_cols = np.where(streams)
    for r, c in zip(all_stream_rows.tolist(), all_stream_cols.tolist()):
        if not _has_stream_receiver(r, c):
            is_outlet[r, c] = True

    # Nodes = headwaters ∪ junctions ∪ outlets
    is_node = is_headwater | is_junction | is_outlet
    node_rows, node_cols = np.where(is_node)

    # Build node set as (r, c) tuples
    node_set = set(zip(node_rows.tolist(), node_cols.tolist()))

    # ------------------------------------------------------------------
    # 3.  Build NetworkX graph
    # ------------------------------------------------------------------
    G = nx.DiGraph()

    for r, c in node_set:
        attrs = {
            'elevation_m': float(dem[r, c]),
            'is_junction': bool(is_junction[r, c]),
            'is_headwater': bool(is_headwater[r, c]),
            'is_outlet': bool(is_outlet[r, c]),
        }
        if acc_arr is not None:
            attrs['upstream_area_km2'] = (
                float(acc_arr[r, c]) * cell_area_km2
            )
        G.add_node((r, c), **attrs)

    # Trace reaches from each headwater/junction node
    start_rows, start_cols = np.where(is_headwater | is_junction)

    for r0, c0 in zip(start_rows.tolist(), start_cols.tolist()):
        # Walk downstream until we hit a node or leave the stream
        r, c = r0, c0
        reach_cells = [(r, c)]
        length_m = 0.0
        elev_sum = float(dem[r, c])

        while True:
            code = int(fdir[r, c])
            if code == 0:
                break
            dr, dc = D8_OFFSETS[code]
            rr, cc = r + dr, c + dc

            if not (0 <= rr < nrows and 0 <= cc < ncols):
                break
            if not streams[rr, cc]:
                break

            # Step distance
            step = (np.sqrt(2.0) if (dr != 0 and dc != 0) else 1.0) * cellsize_m
            length_m += step
            r, c = rr, cc
            elev_sum += float(dem[r, c])
            reach_cells.append((r, c))

            if (r, c) in node_set:
                break

        if (r, c) != (r0, c0) and (r, c) in node_set:
            n_cells = len(reach_cells)
            mean_elev = elev_sum / n_cells
            dz = abs(float(dem[r0, c0]) - float(dem[r, c]))
            mean_slope = dz / length_m if length_m > 0 else 0.0
            G.add_edge(
                (r0, c0), (r, c),
                length_m=length_m,
                mean_elev_m=mean_elev,
                mean_slope=mean_slope,
                n_cells=n_cells,
            )

    log.debug(
        'Stream network: %d nodes, %d edges',
        G.number_of_nodes(), G.number_of_edges(),
    )
    return G


# ---------------------------------------------------------------------------
# Phase 6 — Sub-basin delineation
# ---------------------------------------------------------------------------

def assign_subbasins(fdir_arr, outlet_nodes):
    """Delineate sub-catchment for each outlet node.

    Each raster cell is assigned to the outlet that it ultimately drains
    to by following the D8 flow direction.

    Parameters
    ----------
    fdir_arr : np.ndarray, shape (nrows, ncols), dtype int32
        D8 flow direction (ESRI encoding).  0 = sink / nodata.
    outlet_nodes : list of (int, int)
        ``(row, col)`` pairs identifying outlet cells.  Each outlet gets
        its own sub-basin label.

    Returns
    -------
    subbasins : np.ndarray, shape (nrows, ncols), dtype int32
        Sub-basin label array.  Cell value = 1-based index into
        *outlet_nodes*.  0 = unassigned (cell does not drain to any
        listed outlet, or is nodata).
    """
    from collections import deque

    fdir = np.asarray(fdir_arr, dtype=np.int32)
    nrows, ncols = fdir.shape

    # Build the reverse graph: for each cell, find all upstream cells
    upstream = {(r, c): [] for r in range(nrows) for c in range(ncols)}

    for code, (dr, dc) in D8_OFFSETS.items():
        mask_rows, mask_cols = np.where(fdir == code)
        recv_r = mask_rows + dr
        recv_c = mask_cols + dc
        valid = (
            (recv_r >= 0) & (recv_r < nrows) &
            (recv_c >= 0) & (recv_c < ncols)
        )
        for r, c, rr, cc in zip(
            mask_rows[valid].tolist(), mask_cols[valid].tolist(),
            recv_r[valid].tolist(), recv_c[valid].tolist()
        ):
            upstream[(rr, cc)].append((r, c))

    subbasins = np.zeros((nrows, ncols), dtype=np.int32)
    for label_idx, (or_, oc) in enumerate(outlet_nodes, start=1):
        queue = deque([(or_, oc)])
        while queue:
            r, c = queue.popleft()
            if subbasins[r, c] == 0:
                subbasins[r, c] = label_idx
                for ur, uc in upstream.get((r, c), []):
                    if subbasins[ur, uc] == 0:
                        queue.append((ur, uc))

    return subbasins


# ---------------------------------------------------------------------------
# Phase 7 — Muskingum-Cunge channel routing
# ---------------------------------------------------------------------------

def muskingum_cunge_route(q_in, K_dt, X):
    """Route discharge through one reach via the Muskingum-Cunge method.

    Parameters
    ----------
    q_in : array-like, shape (N,)
        Inflow hydrograph [m³ s⁻¹] (lateral + upstream tributaries).
    K_dt : float
        Ratio K / Δt where K is the wave travel time [timesteps] and
        Δt = 1.  ``K_dt = L / (c_k * dt)`` with ``c_k`` = kinematic
        wave celerity.
    X : float
        Muskingum weighting factor [0, 0.5].
        X = 0 → maximum attenuation (pure diffusion wave);
        X = 0.5 → pure translation (kinematic wave).

    Returns
    -------
    q_out : np.ndarray, shape (N,)
        Routed outflow hydrograph [m³ s⁻¹].

    Notes
    -----
    Routing coefficients (dt = 1)::

        D   = K_dt * (1 - X) + 0.5
        C0  = (-K_dt * X + 0.5) / D
        C1  = ( K_dt * X + 0.5) / D
        C2  = ( K_dt * (1 - X) - 0.5) / D

    Stability requires ``2 * K_dt * X ≤ 1`` and ``2 * K_dt * (1-X) ≥ 1``.
    """
    q_in = np.asarray(q_in, dtype=float)

    if K_dt <= 0:
        raise ValueError(f'K_dt must be positive, got {K_dt}')
    if not 0.0 <= X <= 0.5:
        raise ValueError(f'X must be in [0, 0.5], got {X}')

    D = K_dt * (1.0 - X) + 0.5
    C0 = (-K_dt * X + 0.5) / D
    C1 = (K_dt * X + 0.5) / D
    C2 = (K_dt * (1.0 - X) - 0.5) / D

    q_out = np.empty_like(q_in)
    q_out[0] = q_in[0]  # steady-state IC
    for t in range(1, len(q_in)):
        q_out[t] = C0 * q_in[t] + C1 * q_in[t - 1] + C2 * q_out[t - 1]
        q_out[t] = max(q_out[t], 0.0)  # non-negativity

    return q_out


def route_stream_network(network_graph, lateral_inflow_dict,
                         dt_seconds, muskingum_X=0.25,
                         celerity_m_per_s=1.5):
    """Route discharge through the entire stream network.

    Processes reaches in topological order (upstream to downstream),
    accumulating flow at each junction.

    Parameters
    ----------
    network_graph : nx.DiGraph
        Stream network from :func:`build_stream_network`.
    lateral_inflow_dict : dict
        ``{(r, c): np.ndarray}`` — lateral (hillslope) inflow [m³ s⁻¹]
        entering at each **edge** or node.  Keyed by the *upstream* node
        of each reach.
    dt_seconds : float
        Routing timestep in seconds.
    muskingum_X : float
        Muskingum attenuation coefficient [0, 0.5].
    celerity_m_per_s : float
        Kinematic wave celerity [m s⁻¹] used to compute K = L / c.

    Returns
    -------
    q_outlet : np.ndarray
        Discharge time series at the basin outlet (the node with
        out-degree 0) [m³ s⁻¹].
    q_per_node : dict
        ``{(r, c): np.ndarray}`` — discharge leaving each node [m³ s⁻¹].

    Raises
    ------
    ImportError
        If *networkx* is not installed.
    ValueError
        If the graph has no outlet node.
    """
    if not _HAS_NX:
        raise ImportError(
            'networkx is required for route_stream_network.'
        )

    if network_graph.number_of_nodes() == 0:
        raise ValueError('Empty stream network graph.')

    # Topological order (upstream → downstream)
    topo_order = list(nx.topological_sort(network_graph))

    # Determine length of time series from lateral inflow
    n_ts = None
    for v in lateral_inflow_dict.values():
        n_ts = len(v)
        break
    if n_ts is None:
        raise ValueError('lateral_inflow_dict is empty.')

    q_per_node = {}

    for node in topo_order:
        # Sum inflows: lateral at this node + all upstream branches
        q_total = lateral_inflow_dict.get(node, np.zeros(n_ts))
        for pred in network_graph.predecessors(node):
            q_total = q_total + q_per_node.get(pred, np.zeros(n_ts))

        # Route through the outgoing reach (if any)
        successors = list(network_graph.successors(node))
        if successors:
            edge_data = network_graph.edges[node, successors[0]]
            length_m = edge_data.get('length_m', 1.0)
            K_seconds = length_m / max(celerity_m_per_s, 1e-9)
            K_dt = K_seconds / max(dt_seconds, 1e-9)
            q_routed = muskingum_cunge_route(q_total, K_dt=K_dt,
                                             X=muskingum_X)
        else:
            q_routed = q_total  # outlet node

        q_per_node[node] = q_routed

    # Find outlet(s)
    outlets = [n for n in network_graph.nodes()
               if network_graph.out_degree(n) == 0]
    if not outlets:
        raise ValueError('Stream network has no outlet node.')

    if len(outlets) > 1:
        warnings.warn(
            f'Stream network has {len(outlets)} outlets; returning the '
            'sum of all outlet discharges.',
            stacklevel=2,
        )

    q_outlet = sum(q_per_node.get(o, np.zeros(n_ts)) for o in outlets)

    return q_outlet, q_per_node
