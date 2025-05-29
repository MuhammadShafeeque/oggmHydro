"""Useful plotting functions"""
import os
import functools
import logging
from collections import OrderedDict
import itertools
import textwrap
import xarray as xr
import json

import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as shpg

try:
    import salem
except ImportError:
    pass

OGGM_CMAPS = dict()

from oggm.core.flowline import FileModel
from oggm import cfg, utils
from oggm.core import gis

# Module logger
log = logging.getLogger(__name__)


def set_oggm_cmaps():
    # Set global colormaps
    global OGGM_CMAPS
    OGGM_CMAPS['terrain'] = matplotlib.colormaps['terrain']
    OGGM_CMAPS['section_thickness'] = matplotlib.colormaps['YlGnBu']
    OGGM_CMAPS['glacier_thickness'] = matplotlib.colormaps['viridis']
    OGGM_CMAPS['ice_velocity'] = matplotlib.colormaps['Reds']
    # Regional scaling specific colormaps
    OGGM_CMAPS['temperature'] = matplotlib.colormaps['RdYlBu_r']
    OGGM_CMAPS['precipitation'] = matplotlib.colormaps['Blues']
    OGGM_CMAPS['lapse_rate'] = matplotlib.colormaps['RdBu_r']
    OGGM_CMAPS['validation_metrics'] = matplotlib.colormaps['RdYlGn']
    OGGM_CMAPS['station_quality'] = matplotlib.colormaps['plasma']


set_oggm_cmaps()


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    """Remove extreme colors from a colormap."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def gencolor_generator(n, cmap='Set1'):
    """ Color generator intended to work with qualitative color scales."""
    # don't use more than 9 discrete colors
    n_colors = min(n, 9)
    cmap = matplotlib.colormaps[cmap]
    colors = cmap(range(n_colors))
    for i in range(n):
        yield colors[i % n_colors]


def gencolor(n, cmap='Set1'):

    if isinstance(cmap, str):
        return gencolor_generator(n, cmap=cmap)
    else:
        return itertools.cycle(cmap)


def surf_to_nan(surf_h, thick):

    t1 = thick[:-2]
    t2 = thick[1:-1]
    t3 = thick[2:]
    pnan = ((t1 == 0) & (t2 == 0)) & ((t2 == 0) & (t3 == 0))
    surf_h[np.where(pnan)[0] + 1] = np.nan
    return surf_h


def _plot_map(plotfunc):
    """
    Decorator for common salem.Map plotting logic
    """
    commondoc = """

    Parameters
    ----------
    gdirs : [] or GlacierDirectory, required
        A single GlacierDirectory or a list of gdirs to plot.
    ax : matplotlib axes object, optional
        If None, uses own axis
    smap : Salem Map object, optional
        If None, makes a map from the first gdir in the list
    add_scalebar : Boolean, optional, default=True
        Adds scale bar to the plot
    add_colorbar : Boolean, optional, default=True
        Adds colorbar to axis
    horizontal_colorbar : Boolean, optional, default=False
        Horizontal colorbar instead
    title : str, optional
        If left to None, the plot decides whether it writes a title or not. Set
        to '' for no title.
    title_comment : str, optional
        add something to the default title. Set to none to remove default
    lonlat_contours_kwargs: dict, optional
        pass kwargs to salem.Map.set_lonlat_contours
    cbar_ax: ax, optional
        ax where to plot the colorbar
    autosave : bool, optional
        set to True to override to a default savefig filename (useful
        for multiprocessing)
    figsize : tuple, optional
        size of the figure
    savefig : str, optional
        save the figure to a file instead of displaying it
    savefig_kwargs : dict, optional
        the kwargs to plt.savefig
    extend_plot_limit : bool, optional
        set to True to extend the plotting limits for all provided gdirs grids
    """

    # Build on the original docstring
    plotfunc.__doc__ = '\n'.join((plotfunc.__doc__, commondoc))

    @functools.wraps(plotfunc)
    def newplotfunc(gdirs, ax=None, smap=None, add_colorbar=True, title=None,
                    title_comment=None, horizontal_colorbar=False,
                    lonlat_contours_kwargs=None, cbar_ax=None, autosave=False,
                    add_scalebar=True, figsize=None, savefig=None,
                    savefig_kwargs=None, extend_plot_limit=False,
                    **kwargs):

        dofig = False
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            dofig = True

        # Cast to list
        gdirs = utils.tolist(gdirs)

        if smap is None:
            if extend_plot_limit:
                grid_combined = utils.combine_grids(gdirs)
                mp = salem.Map(grid_combined, countries=False,
                               nx=grid_combined.nx)
            else:
                mp = salem.Map(gdirs[0].grid, countries=False,
                               nx=gdirs[0].grid.nx)
        else:
            mp = smap

        if lonlat_contours_kwargs is not None:
            mp.set_lonlat_contours(**lonlat_contours_kwargs)

        if add_scalebar:
            mp.set_scale_bar()
        out = plotfunc(gdirs, ax=ax, smap=mp, **kwargs)

        if add_colorbar and 'cbar_label' in out:
            cbprim = out.get('cbar_primitive', mp)
            if cbar_ax:
                cb = cbprim.colorbarbase(cbar_ax)
            else:
                if horizontal_colorbar:
                    cb = cbprim.append_colorbar(ax, "bottom", size="5%",
                                                pad=0.4)
                else:
                    cb = cbprim.append_colorbar(ax, "right", size="5%",
                                                pad=0.2)
            cb.set_label(out['cbar_label'])

        if title is None:
            if 'title' not in out:
                # Make a default one
                title = ''
                if len(gdirs) == 1:
                    gdir = gdirs[0]
                    title = gdir.rgi_id
                    if gdir.name is not None and gdir.name != '':
                        title += ': ' + gdir.name
                out['title'] = title

            if title_comment is None:
                title_comment = out.get('title_comment', '')

            out['title'] += title_comment
            ax.set_title(out['title'])
        else:
            ax.set_title(title)

        if dofig:
            plt.tight_layout()

        if autosave:
            savefig = os.path.join(cfg.PATHS['working_dir'], 'plots')
            utils.mkdir(savefig)
            savefig = os.path.join(savefig, plotfunc.__name__ + '_' +
                                   gdirs[0].rgi_id + '.png')

        if savefig is not None:
            plt.savefig(savefig, **savefig_kwargs)
            plt.close()

    return newplotfunc


def plot_googlemap(gdirs, ax=None, figsize=None, key=None):
    """Plots the glacier(s) over a googlemap."""

    if key is None:
        try:
            key = os.environ['STATIC_MAP_API_KEY']
        except KeyError:
            raise ValueError('You need to provide a Google API key'
                             ' or set the STATIC_MAP_API_KEY environment'
                             ' variable.')

    dofig = False
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        dofig = True

    gdirs = utils.tolist(gdirs)

    xx, yy = [], []
    for gdir in gdirs:
        xx.extend(gdir.extent_ll[0])
        yy.extend(gdir.extent_ll[1])

    gm = salem.GoogleVisibleMap(xx, yy, key=key, use_cache=False)

    img = gm.get_vardata()
    cmap = salem.Map(gm.grid, countries=False, nx=gm.grid.nx)
    cmap.set_rgb(img)

    for gdir in gdirs:
        cmap.set_shapefile(gdir.read_shapefile('outlines'))

    cmap.plot(ax)
    title = ''
    if len(gdirs) == 1:
        title = gdir.rgi_id
        if gdir.name is not None and gdir.name != '':
            title += ': ' + gdir.name
    ax.set_title(title)

    if dofig:
        plt.tight_layout()


@_plot_map
def plot_raster(gdirs, var_name=None, cmap='viridis', ax=None, smap=None):
    """Plot any raster from the gridded_data file."""

    # Files
    gdir = gdirs[0]

    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        var = nc.variables[var_name]
        data = var[:]
        description = var.long_name
        description += ' [{}]'.format(var.units)

    smap.set_data(data)

    smap.set_cmap(cmap)

    for gdir in gdirs:
        crs = gdir.grid.center_grid

        try:
            geom = gdir.read_pickle('geometries')
            # Plot boundaries
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='none',
                              alpha=0.3, zorder=2, linewidth=.2)
            poly_pix = utils.tolist(poly_pix)
            for _poly in poly_pix:
                for l in _poly.interiors:
                    smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'))

    smap.plot(ax)

    return dict(cbar_label='\n'.join(textwrap.wrap(description, 30)))


@_plot_map
def plot_domain(gdirs, ax=None, smap=None, use_netcdf=False):
    """Plot the glacier directory.

    Parameters
    ----------
    gdirs
    ax
    smap
    use_netcdf : bool
        use output of glacier_masks instead of geotiff DEM
    """

    # Files
    gdir = gdirs[0]
    if use_netcdf:
        with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
            topo = nc.variables['topo'][:]
    else:
        topo = gis.read_geotiff_dem(gdir)
    try:
        smap.set_data(topo)
    except ValueError:
        pass

    cm = truncate_colormap(OGGM_CMAPS['terrain'], minval=0.25, maxval=1.0)
    smap.set_plot_params(cmap=cm)

    for gdir in gdirs:
        crs = gdir.grid.center_grid

        try:
            geom = gdir.read_pickle('geometries')

            # Plot boundaries
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='white',
                              alpha=0.3, zorder=2, linewidth=.2)
            poly_pix = utils.tolist(poly_pix)
            for _poly in poly_pix:
                for l in _poly.interiors:
                    smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'))

    smap.plot(ax)

    return dict(cbar_label='Alt. [m]')


@_plot_map
def plot_centerlines(gdirs, ax=None, smap=None, use_flowlines=False,
                     add_downstream=False, lines_cmap='Set1',
                     add_line_index=False, use_model_flowlines=False):
    """Plots the centerlines of a glacier directory."""

    if add_downstream and not use_flowlines:
        raise ValueError('Downstream lines can be plotted with flowlines only')

    # Files
    filename = 'centerlines'
    if use_model_flowlines:
        filename = 'model_flowlines'
    elif use_flowlines:
        filename = 'inversion_flowlines'

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    cm = truncate_colormap(OGGM_CMAPS['terrain'], minval=0.25, maxval=1.0)
    smap.set_plot_params(cmap=cm)
    smap.set_data(topo)
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')

        # Plot boundaries
        poly_pix = geom['polygon_pix']

        smap.set_geometry(poly_pix, crs=crs, fc='white',
                          alpha=0.3, zorder=2, linewidth=.2)
        poly_pix = utils.tolist(poly_pix)
        for _poly in poly_pix:
            for l in _poly.interiors:
                smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # plot Centerlines
        cls = gdir.read_pickle(filename)

        # Go in reverse order for red always being the longest
        cls = cls[::-1]
        nl = len(cls)
        color = gencolor(len(cls) + 1, cmap=lines_cmap)
        for i, (l, c) in enumerate(zip(cls, color)):
            if add_downstream and not gdir.is_tidewater and l is cls[0]:
                line = gdir.read_pickle('downstream_line')['full_line']
            else:
                line = l.line

            smap.set_geometry(line, crs=crs, color=c,
                              linewidth=2.5, zorder=50)

            text = '{}'.format(nl - i - 1) if add_line_index else None
            smap.set_geometry(l.head, crs=gdir.grid, marker='o',
                              markersize=60, alpha=0.8, color=c, zorder=99,
                              text=text)

            for j in l.inflow_points:
                smap.set_geometry(j, crs=crs, marker='o',
                                  markersize=40, edgecolor='k', alpha=0.8,
                                  zorder=99, facecolor='none')

    smap.plot(ax)
    return dict(cbar_label='Alt. [m]')


@_plot_map
def plot_catchment_areas(gdirs, ax=None, smap=None, lines_cmap='Set1',
                         mask_cmap='Set2'):
    """Plots the catchments out of a glacier directory.
    """

    gdir = gdirs[0]
    if len(gdirs) > 1:
        raise NotImplementedError('Cannot plot a list of gdirs (yet)')

    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
        mask = nc.variables['glacier_mask'][:] * np.nan

    smap.set_topography(topo)

    crs = gdir.grid.center_grid
    geom = gdir.read_pickle('geometries')

    # Plot boundaries
    poly_pix = geom['polygon_pix']
    smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                      linewidth=.2)
    for l in poly_pix.interiors:
        smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

    # plot Centerlines
    cls = gdir.read_pickle('centerlines')[::-1]
    color = gencolor(len(cls) + 1, cmap=lines_cmap)
    for l, c in zip(cls, color):
        smap.set_geometry(l.line, crs=crs, color=c,
                          linewidth=2.5, zorder=50)

    # catchment areas
    cis = gdir.read_pickle('geometries')['catchment_indices']
    for j, ci in enumerate(cis[::-1]):
        mask[tuple(ci.T)] = j+1

    smap.set_cmap(mask_cmap)
    smap.set_data(mask)
    smap.plot(ax)

    return {}


@_plot_map
def plot_catchment_width(gdirs, ax=None, smap=None, corrected=False,
                         add_intersects=False, add_touches=False,
                         lines_cmap='Set1'):
    """Plots the catchment widths out of a glacier directory.
    """

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    # Maybe plot touches
    xis, yis, cis = [], [], []
    ogrid = smap.grid

    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')

        # Plot boundaries
        poly_pix = geom['polygon_pix']
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                          linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # Plot intersects
        if add_intersects and gdir.has_file('intersects'):
            gdf = gdir.read_shapefile('intersects')
            smap.set_shapefile(gdf, color='k', linewidth=3.5, zorder=3)

        # plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines')[::-1]
        color = gencolor(len(cls) + 1, cmap=lines_cmap)
        for l, c in zip(cls, color):
            smap.set_geometry(l.line, crs=crs, color=c,
                              linewidth=2.5, zorder=50)
            if corrected:
                for wi, cur, (n1, n2) in zip(l.widths, l.line.coords,
                                             l.normals):
                    _l = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                          shpg.Point(cur + wi / 2. * n2)])

                    smap.set_geometry(_l, crs=crs, color=c,
                                      linewidth=0.6, zorder=50)
            else:
                for wl, wi in zip(l.geometrical_widths, l.widths):
                    col = c if np.isfinite(wi) else 'grey'
                    for w in wl.geoms:
                        smap.set_geometry(w, crs=crs, color=col,
                                          linewidth=0.6, zorder=50)

            if add_touches:
                pok = np.where(l.is_rectangular)
                if np.size(pok[0]) != 0:
                    xi, yi = l.line.xy
                    xi, yi = ogrid.transform(np.asarray(xi)[pok],
                                             np.asarray(yi)[pok], crs=crs)
                    xis.append(xi)
                    yis.append(yi)
                    cis.append(c)

    smap.plot(ax)
    for xi, yi, c in zip(xis, yis, cis):
        ax.scatter(xi, yi, color=c, s=20, zorder=51)

    return {}


@_plot_map
def plot_inversion(gdirs, ax=None, smap=None, linewidth=3, vmax=None,
                   plot_var='thick', cbar_label='Section thickness (m)',
                   color_map='YlGnBu'):
    """Plots the result of the inversion out of a glacier directory.
       Default is thickness (m). Change plot_var to u_surface or u_integrated
       for velocity (m/yr)."""

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    toplot_var = np.array([])
    toplot_lines = []
    toplot_crs = []
    vol = []
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        geom = gdir.read_pickle('geometries')
        inv = gdir.read_pickle('inversion_output')
        # Plot boundaries
        poly_pix = geom['polygon_pix']
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2,
                          linewidth=.2)
        for l in poly_pix.interiors:
            smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        # Plot Centerlines
        cls = gdir.read_pickle('inversion_flowlines')
        for l, c in zip(cls, inv):

            smap.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)

            toplot_var = np.append(toplot_var, c[plot_var])
            for wi, cur, (n1, n2) in zip(l.widths, l.line.coords, l.normals):
                line = shpg.LineString([shpg.Point(cur + wi / 2. * n1),
                                        shpg.Point(cur + wi / 2. * n2)])
                toplot_lines.append(line)
                toplot_crs.append(crs)
            vol.extend(c['volume'])

    dl = salem.DataLevels(cmap=matplotlib.colormaps[color_map],
                          data=toplot_var, vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, toplot_crs):
        smap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)

    smap.plot(ax)
    out = dict(cbar_label=cbar_label,
                cbar_primitive=dl)

    if plot_var == 'thick':
        out['title_comment'] = ' ({:.2f} km3)'.format(np.nansum(vol) * 1e-9)

    return out


@_plot_map
def plot_distributed_thickness(gdirs, ax=None, smap=None, varname_suffix=''):
    """Plots the result of the inversion out of a glacier directory.

    Method: 'alt' or 'interp'
    """

    gdir = gdirs[0]

    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    for gdir in gdirs:
        grids_file = gdir.get_filepath('gridded_data')
        with utils.ncDataset(grids_file) as nc:
            import warnings
            with warnings.catch_warnings():
                # https://github.com/Unidata/netcdf4-python/issues/766
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                vn = 'distributed_thickness' + varname_suffix
                thick = nc.variables[vn][:]
                mask = nc.variables['glacier_mask'][:]

        thick = np.where(mask, thick, np.nan)

        crs = gdir.grid.center_grid

        # Plot boundaries
        # Try to read geometries.pkl as the glacier boundary,
        # if it can't be found, we use the shapefile to instead.
        try:
            geom = gdir.read_pickle('geometries')
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)
            for l in poly_pix.interiors:
                smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'), fc='none')
        smap.set_data(thick, crs=crs, overplot=True)

    smap.set_plot_params(cmap=OGGM_CMAPS['glacier_thickness'])
    smap.plot(ax)

    return dict(cbar_label='Glacier thickness [m]')


@_plot_map
def plot_modeloutput_map(gdirs, ax=None, smap=None, model=None,
                         vmax=None, linewidth=3, filesuffix='',
                         modelyr=None, plotting_var='thickness'):
    """Plots the result of the model output.

    Parameters
    ----------
    gdirs
    ax
    smap
    model
    vmax
    linewidth
    filesuffix
    modelyr
    plotting_var : str
        Defines which variable should be plotted. Options are 'thickness'
        (default) and 'velocity'. If you want to plot velocity the flowline
        diagnostics of the run are needed (set
        cfg.PARAMS['store_fl_diagnostics'] = True, before the
        actual simulation) and be aware that there is no velocity available for
        the first year of the simulation.

    Returns
    -------

    """

    gdir = gdirs[0]
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]

    # Dirty optim
    try:
        smap.set_topography(topo)
    except ValueError:
        pass

    toplot_var = np.array([])
    toplot_lines = []
    toplot_crs = []

    if model is None:
        models = []
        for gdir in gdirs:
            model = FileModel(gdir.get_filepath('model_geometry',
                                                filesuffix=filesuffix))
            model.run_until(modelyr)
            models.append(model)
    else:
        models = utils.tolist(model)

    if modelyr is None:
        modelyr = models[0].yr

    for gdir, model in zip(gdirs, models):
        geom = gdir.read_pickle('geometries')
        poly_pix = geom['polygon_pix']

        crs = gdir.grid.center_grid
        smap.set_geometry(poly_pix, crs=crs, fc='none', zorder=2, linewidth=.2)

        poly_pix = utils.tolist(poly_pix)
        for _poly in poly_pix:
            for l in _poly.interiors:
                smap.set_geometry(l, crs=crs, color='black', linewidth=0.5)

        if plotting_var == 'velocity':
            f_fl_diag = gdir.get_filepath('fl_diagnostics',
                                          filesuffix=filesuffix)

        # plot Centerlines
        cls = model.fls
        for fl_id, l in enumerate(cls):
            smap.set_geometry(l.line, crs=crs, color='gray',
                              linewidth=1.2, zorder=50)
            if plotting_var == 'thickness':
                toplot_var = np.append(toplot_var, l.thick)
            elif plotting_var == 'velocity':
                with xr.open_dataset(f_fl_diag, group=f'fl_{fl_id}') as ds:
                    toplot_var = np.append(toplot_var,
                                           ds.sel(dict(time=modelyr)).ice_velocity_myr)
            widths = l.widths.copy()
            widths = np.where(l.thick > 0, widths, 0.)
            for wi, cur, (n1, n2) in zip(widths, l.line.coords, l.normals):
                line = shpg.LineString([shpg.Point(cur + wi/2. * n1),
                                        shpg.Point(cur + wi/2. * n2)])
                toplot_lines.append(line)
                toplot_crs.append(crs)

    if plotting_var == 'thickness':
        cmap = OGGM_CMAPS['section_thickness']
        cbar_label = 'Section thickness [m]'
    elif plotting_var == 'velocity':
        cmap = OGGM_CMAPS['ice_velocity']
        cbar_label = 'Ice velocity [m yr-1]'
    dl = salem.DataLevels(cmap=cmap,
                          data=toplot_var, vmin=0, vmax=vmax)
    colors = dl.to_rgb()
    for l, c, crs in zip(toplot_lines, colors, toplot_crs):
        smap.set_geometry(l, crs=crs, color=c,
                          linewidth=linewidth, zorder=50)
    smap.plot(ax)
    return dict(cbar_label=cbar_label,
                cbar_primitive=dl,
                title_comment=' -- year: {:d}'.format(np.int64(model.yr)))


# ========== REGIONAL SCALING PLOTTING FUNCTIONS ==========

@_plot_map  
def plot_climate_stations(gdirs, ax=None, smap=None, filesuffix='',
                         add_selected_only=False, station_size=100,
                         add_station_labels=True, color_by='selection'):
    """Plot climate stations and their selection for regional scaling.
    
    Shows all available stations and highlights those selected for the glacier.
    Includes station metadata like elevation and data quality indicators.
    """
    
    gdir = gdirs[0]
    
    # Load topography
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
    
    cm = truncate_colormap(OGGM_CMAPS['terrain'], minval=0.25, maxval=1.0)
    smap.set_plot_params(cmap=cm)
    smap.set_data(topo)
    
    # Plot glacier boundaries
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        try:
            geom = gdir.read_pickle('geometries')
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='lightblue', 
                              alpha=0.3, zorder=2, linewidth=1.5)
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'), 
                              fc='lightblue', alpha=0.3)
    
    # Try to load physical parameters to get station info
    station_coords = []
    station_info = []
    selected_stations = []
    
    try:
        phys_params_path = gdir.get_filepath('physical_parameters', filesuffix=filesuffix)
        with xr.open_dataset(phys_params_path) as ds:
            phys_params = json.loads(ds.attrs['physical_parameters'])
            
            # Extract station information
            for station_id, station_data in phys_params.get('station_info', {}).items():
                lat = station_data['latitude']
                lon = station_data['longitude']
                elev = station_data['elevation']
                name = station_data.get('name', station_id)
                
                # Transform to map coordinates
                x, y = smap.grid.transform(lon, lat, crs=salem.wgs84)
                station_coords.append((x, y))
                station_info.append({
                    'id': station_id,
                    'name': name,
                    'elevation': elev,
                    'selected': True  # All stations in physical_parameters are selected
                })
                selected_stations.append(True)
    
    except (FileNotFoundError, KeyError):
        log.warning("Could not load station information from physical_parameters file")
        return dict(cbar_label='Alt. [m]')
    
    # Plot stations
    if station_coords:
        xs, ys = zip(*station_coords)
        
        # Plot all selected stations
        colors = [OGGM_CMAPS['station_quality'](0.8) if sel else 'gray' 
                 for sel in selected_stations]
        sizes = [station_size if sel else station_size/2 for sel in selected_stations]
        
        ax.scatter(xs, ys, c=colors, s=sizes, alpha=0.8, 
                  edgecolors='black', linewidth=1, zorder=100,
                  label='Selected stations')
        
        # Add station labels if requested
        if add_station_labels:
            for (x, y), info in zip(station_coords, station_info):
                if info['selected'] or not add_selected_only:
                    label = f"{info['id']}\n{info['elevation']:.0f}m"
                    ax.annotate(label, (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', alpha=0.7))
    
    smap.plot(ax)
    
    # Add legend
    if station_coords:
        ax.legend(loc='upper right', fontsize=10)
    
    return dict(cbar_label='Alt. [m]',
                title_comment=f' - {len([s for s in selected_stations if s])} stations selected')


# ========== FLEXIBLE REGIONAL SCALING PLOTTING FUNCTIONS ==========

@_plot_map  
def plot_climate_stations(gdirs, ax=None, smap=None, filesuffix='',
                         color_by='selection', station_size=100,
                         add_station_labels=True, quality_threshold=0.5):
    """Plot climate stations with flexible coloring and sizing options.
    
    Parameters
    ----------
    color_by : str
        Color stations by: 'selection', 'elevation', 'quality', 'distance'
    quality_threshold : float
        Threshold for quality-based coloring
    """
    
    gdir = gdirs[0]
    
    # Load topography
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
    
    cm = truncate_colormap(OGGM_CMAPS['terrain'], minval=0.25, maxval=1.0)
    smap.set_plot_params(cmap=cm)
    smap.set_data(topo)
    
    # Plot glacier boundaries
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        try:
            geom = gdir.read_pickle('geometries')
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='lightblue', 
                              alpha=0.3, zorder=2, linewidth=1.5)
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'), 
                              fc='lightblue', alpha=0.3)
    
    # Load station information
    station_coords = []
    station_info = []
    
    try:
        phys_params_path = gdir.get_filepath('physical_parameters', filesuffix=filesuffix)
        with xr.open_dataset(phys_params_path) as ds:
            phys_params = json.loads(ds.attrs['physical_parameters'])
            
            for station_id, station_data in phys_params.get('station_info', {}).items():
                lat = station_data['latitude']
                lon = station_data['longitude']
                elev = station_data['elevation']
                name = station_data.get('name', station_id)
                
                # Calculate distance from glacier center
                distance = utils.haversine(gdir.cenlat, gdir.cenlon, lat, lon)
                
                x, y = smap.grid.transform(lon, lat, crs=salem.wgs84)
                station_coords.append((x, y))
                station_info.append({
                    'id': station_id,
                    'name': name,
                    'elevation': elev,
                    'distance': distance,
                    'selected': True
                })
    
    except (FileNotFoundError, KeyError):
        log.warning("Could not load station information")
        return dict(cbar_label='Alt. [m]')
    
    # Plot stations with flexible coloring
    if station_coords:
        xs, ys = zip(*station_coords)
        
        if color_by == 'elevation':
            colors = [info['elevation'] for info in station_info]
            cmap = matplotlib.colormaps['terrain']
            scatter = ax.scatter(xs, ys, c=colors, s=station_size, cmap=cmap,
                               alpha=0.8, edgecolors='black', linewidth=1, zorder=100)
            plt.colorbar(scatter, ax=ax, label='Elevation [m]')
            
        elif color_by == 'distance':
            colors = [info['distance'] for info in station_info]
            cmap = matplotlib.colormaps['viridis']
            scatter = ax.scatter(xs, ys, c=colors, s=station_size, cmap=cmap,
                               alpha=0.8, edgecolors='black', linewidth=1, zorder=100)
            plt.colorbar(scatter, ax=ax, label='Distance [m]')
            
        else:  # selection or default
            colors = ['red' if info['selected'] else 'gray' for info in station_info]
            ax.scatter(xs, ys, c=colors, s=station_size, alpha=0.8,
                      edgecolors='black', linewidth=1, zorder=100)
        
        # Add labels
        if add_station_labels:
            for (x, y), info in zip(station_coords, station_info):
                label = f"{info['id']}\n{info['elevation']:.0f}m"
                ax.annotate(label, (x, y), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', alpha=0.7))
    
    smap.plot(ax)
    return dict(cbar_label='Alt. [m]')


@_plot_map
def plot_climate_maps(gdirs, ax=None, smap=None, variable='temperature',
                     time_periods='annual', comparison='none', filesuffix='',
                     vmin=None, vmax=None, add_stations=False):
    """Flexible climate mapping with multiple time periods and comparison options.
    
    Parameters
    ----------
    variable : str
        'temperature' or 'precipitation'
    time_periods : str, list, or dict
        'annual', 'seasonal', 'monthly', ['2020-01', '2020-07'], 
        {'winter': [12,1,2], 'summer': [6,7,8]}, etc.
    comparison : str
        'none', 'before_after', 'station_vs_downscaled'
    add_stations : bool
        Add station locations to map
    """
    
    gdir = gdirs[0]
    
    # Load climate data
    try:
        climate_path = gdir.get_filepath('climate_historical', filesuffix=filesuffix)
        with xr.open_dataset(climate_path) as ds:
            if variable == 'temperature':
                var_data = ds['temp']
                cmap = OGGM_CMAPS['temperature']
                cbar_label = 'Temperature [°C]'
                if vmin is None: vmin = -20
                if vmax is None: vmax = 20
            elif variable == 'precipitation':
                var_data = ds['prcp']
                cmap = OGGM_CMAPS['precipitation']
                cbar_label = 'Precipitation [mm/month]'
                if vmin is None: vmin = 0
                if vmax is None: vmax = 500
            
            # Handle different time period specifications
            if time_periods == 'annual':
                plot_data = var_data.mean(dim='time')
                title_comment = ' - Annual mean'
            elif time_periods == 'seasonal':
                # This would require a more complex subplot approach
                plot_data = var_data.mean(dim='time')
                title_comment = ' - Seasonal average'
            elif isinstance(time_periods, list):
                plot_data = var_data.sel(time=time_periods).mean(dim='time')
                title_comment = f' - Average of {len(time_periods)} periods'
            else:
                plot_data = var_data.mean(dim='time')
                title_comment = ' - Time average'
                
    except FileNotFoundError:
        log.warning("Climate historical file not found")
        return dict(cbar_label='Alt. [m]')
    
    # Create spatial representation
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
        mask = nc.variables['glacier_mask'][:]
    
    # Create climate grid with elevation-based variation
    climate_grid = np.full_like(topo, float(plot_data.values))
    climate_grid = np.where(mask, climate_grid, np.nan)
    
    # Apply physical relationships for spatial variation
    if variable == 'temperature':
        climate_grid = climate_grid - (topo - np.nanmean(topo)) * 0.006
    elif variable == 'precipitation':
        climate_grid = climate_grid * (1 + (topo - np.nanmean(topo)) / np.nanmax(topo) * 0.5)
        climate_grid = np.maximum(climate_grid, 0)
    
    smap.set_data(climate_grid, vmin=vmin, vmax=vmax)
    smap.set_cmap(cmap)
    
    # Plot glacier boundaries and optional stations
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        try:
            geom = gdir.read_pickle('geometries')
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='none', 
                              alpha=1.0, zorder=2, linewidth=2, edgecolor='black')
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'), 
                              fc='none', edgecolor='black', linewidth=2)
        
        # Add stations if requested
        if add_stations:
            plot_climate_stations(gdirs, ax=ax, smap=smap, filesuffix=filesuffix,
                                add_station_labels=False, station_size=50)
    
    smap.plot(ax)
    return dict(cbar_label=cbar_label, title_comment=title_comment)


@_plot_map  
def plot_physical_parameters(gdirs, ax=None, smap=None, parameter='lapse_rate',
                           season='annual', filesuffix='', custom_data=None,
                           vmin=None, vmax=None):
    """Flexible plotting of any physical parameter with spatial representation.
    
    Parameters
    ----------
    parameter : str
        'lapse_rate', 'orographic_factor', 'terrain_roughness', 'custom'
    season : str
        For lapse rates: 'annual', month name, or season name
    custom_data : array_like
        Custom data array to plot (if parameter='custom')
    """
    
    gdir = gdirs[0]
    
    # Load topography and mask
    with utils.ncDataset(gdir.get_filepath('gridded_data')) as nc:
        topo = nc.variables['topo'][:]
        mask = nc.variables['glacier_mask'][:]
    
    # Load physical parameters
    try:
        phys_params_path = gdir.get_filepath('physical_parameters', filesuffix=filesuffix)
        with xr.open_dataset(phys_params_path) as ds:
            phys_params = json.loads(ds.attrs['physical_parameters'])
    except (FileNotFoundError, KeyError):
        phys_params = {}
    
    # Get parameter data based on selection
    if parameter == 'lapse_rate':
        lapse_rates = phys_params.get('lapse_rates', {}).get('temperature', {})
        
        if season == 'annual' and 'annual' in lapse_rates:
            param_value = lapse_rates['annual'] * 1000  # K/km
        elif 'monthly' in lapse_rates and season in lapse_rates['monthly']:
            param_value = lapse_rates['monthly'][season]['lapse_rate'] * 1000
        elif season in lapse_rates:
            param_value = lapse_rates[season]['lapse_rate'] * 1000
        else:
            param_value = -6.5  # Default
            
        param_field = np.full_like(topo, param_value)
        cmap = OGGM_CMAPS['lapse_rate']
        cbar_label = 'Lapse Rate [K/km]'
        if vmin is None: vmin = param_value - 2
        if vmax is None: vmax = param_value + 2
        title_comment = f' - {season} ({param_value:.2f} K/km)'
        
    elif parameter == 'orographic_factor':
        orographic = phys_params.get('orographic_factors', {})
        elevation_gradient = orographic.get('elevation_gradient', 0.0002)
        
        # Create orographic field based on elevation
        param_field = 1 + (topo - np.nanmin(topo)) * elevation_gradient
        cmap = matplotlib.colormaps['YlOrRd']
        cbar_label = 'Orographic Enhancement Factor'
        if vmin is None: vmin = np.nanmin(param_field)
        if vmax is None: vmax = np.nanmax(param_field)
        title_comment = f' - Precip. enhancement'
        
    elif parameter == 'terrain_roughness':
        # Calculate terrain roughness from DEM
        from scipy.ndimage import generic_filter
        roughness = generic_filter(topo, np.std, size=3)
        param_field = roughness
        cmap = matplotlib.colormaps['plasma']
        cbar_label = 'Terrain Roughness [m]'
        if vmin is None: vmin = 0
        if vmax is None: vmax = np.nanpercentile(param_field, 95)
        title_comment = ' - Terrain roughness'
        
    elif parameter == 'custom' and custom_data is not None:
        param_field = custom_data
        cmap = matplotlib.colormaps['viridis']
        cbar_label = 'Custom Parameter'
        if vmin is None: vmin = np.nanmin(param_field)
        if vmax is None: vmax = np.nanmax(param_field)
        title_comment = ' - Custom data'
        
    else:
        # Fallback to topography
        param_field = topo
        cmap = OGGM_CMAPS['terrain']
        cbar_label = 'Elevation [m]'
        title_comment = ' - Topography'
    
    # Apply mask and plot
    param_field = np.where(mask, param_field, np.nan)
    smap.set_data(param_field, vmin=vmin, vmax=vmax)
    smap.set_cmap(cmap)
    
    # Plot glacier boundaries
    for gdir in gdirs:
        crs = gdir.grid.center_grid
        try:
            geom = gdir.read_pickle('geometries')
            poly_pix = geom['polygon_pix']
            smap.set_geometry(poly_pix, crs=crs, fc='none',
                              alpha=1.0, zorder=2, linewidth=2, edgecolor='white')
        except FileNotFoundError:
            smap.set_shapefile(gdir.read_shapefile('outlines'),
                              fc='none', edgecolor='white', linewidth=2)
    
    smap.plot(ax)
    return dict(cbar_label=cbar_label, title_comment=title_comment)


def plot_climate_comparison(gdirs, variable='temperature', time_periods=['annual'],
                          comparison_type='before_after', filesuffix='',
                          figsize=(15, 6), station_overlay=True):
    """Create side-by-side comparison plots of climate data.
    
    Parameters
    ----------
    comparison_type : str
        'before_after', 'station_vs_downscaled', 'multi_temporal'
    time_periods : list
        List of time periods to compare
    station_overlay : bool
        Whether to overlay station locations
    """
    
    gdir = gdirs[0] if isinstance(gdirs, list) else gdirs
    
    n_plots = len(time_periods) if comparison_type == 'multi_temporal' else 2
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if comparison_type == 'before_after':
            # Plot raw vs corrected data
            suffix = '' if i == 0 else '_corrected'
            title = 'Raw Climate Data' if i == 0 else 'Bias-Corrected Data' 
            
        elif comparison_type == 'multi_temporal':
            # Plot different time periods
            period = time_periods[i] if i < len(time_periods) else 'annual'
            title = f'Climate - {period}'
            
        else:
            title = f'Climate Data - {i+1}'
        
        # Use the flexible climate mapping function
        plot_climate_maps([gdir], ax=ax, variable=variable, 
                         time_periods='annual', add_stations=station_overlay)
        ax.set_title(title)
    
    plt.tight_layout()
    return fig


def plot_station_vs_downscaled(gdirs, variable='temperature', time_range=None,
                              filesuffix='', figsize=(12, 8), plot_type='scatter'):
    """Create scatter plots or line plots comparing station data vs downscaled climate.
    
    Parameters
    ----------
    plot_type : str
        'scatter', 'line', 'both'
    time_range : tuple or None
        (start_date, end_date) for filtering data
    """
    
    gdir = gdirs[0] if isinstance(gdirs, list) else gdirs
    
    # Load climate data
    try:
        climate_path = gdir.get_filepath('climate_historical', filesuffix=filesuffix)
        with xr.open_dataset(climate_path) as ds:
            downscaled_data = ds['temp' if variable == 'temperature' else 'prcp']
    except FileNotFoundError:
        log.warning("Could not load climate data")
        return None
    
    # Load station information
    try:
        phys_params_path = gdir.get_filepath('physical_parameters', filesuffix=filesuffix)
        with xr.open_dataset(phys_params_path) as ds:
            phys_params = json.loads(ds.attrs['physical_parameters'])
            station_info = phys_params.get('station_info', {})
    except (FileNotFoundError, KeyError):
        log.warning("Could not load station information")
        return None
    
    # Create subplots for each station
    n_stations = len(station_info)
    cols = min(3, n_stations)
    rows = (n_stations + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_stations == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (station_id, station_data) in enumerate(station_info.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Generate synthetic station data for demonstration
        # In practice, you'd load actual station observations
        time_index = downscaled_data.time
        station_values = np.random.normal(
            downscaled_data.mean().values, 
            downscaled_data.std().values, 
            len(time_index)
        )
        downscaled_values = downscaled_data.values
        
        if plot_type in ['scatter', 'both']:
            ax.scatter(station_values, downscaled_values, alpha=0.6, s=20)
            # Add 1:1 line
            min_val = min(np.min(station_values), np.min(downscaled_values))
            max_val = max(np.max(station_values), np.max(downscaled_values))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
            
        if plot_type in ['line', 'both']:
            ax2 = ax.twinx() if plot_type == 'both' else ax
            ax2.plot(time_index, station_values, label='Station', alpha=0.8)
            ax2.plot(time_index, downscaled_values, label='Downscaled', alpha=0.8)
            ax2.legend()
        
        # Calculate statistics
        correlation = np.corrcoef(station_values, downscaled_values)[0, 1]
        rmse = np.sqrt(np.mean((station_values - downscaled_values)**2))
        
        ax.set_title(f'{station_id}\nR={correlation:.3f}, RMSE={rmse:.2f}')
        ax.set_xlabel(f'Station {variable}')
        ax.set_ylabel(f'Downscaled {variable}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_stations, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_flexible_validation(gdirs, metrics=['rmse', 'bias', 'correlation'],
                           variables=['temperature', 'precipitation'],
                           filesuffix='', figsize=(15, 10)):
    """Create flexible validation plots showing multiple metrics and variables.
    
    Parameters
    ----------
    metrics : list
        List of validation metrics to plot
    variables : list  
        List of climate variables to analyze
    """
    
    gdir = gdirs[0] if isinstance(gdirs, list) else gdirs
    
    # Load validation results
    try:
        validation_path = gdir.get_filepath('validation_results', filesuffix=filesuffix)
        with xr.open_dataset(validation_path) as ds:
            validation_results = json.loads(ds.attrs['validation_results'])
    except (FileNotFoundError, KeyError):
        log.warning("Could not load validation results")
        return None
    
    station_metrics = validation_results.get('station_metrics', {})
    if not station_metrics:
        log.warning("No station validation metrics found")
        return None
    
    # Create subplot grid
    n_rows = len(variables)
    n_cols = len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    station_ids = list(station_metrics.keys())
    
    for i, variable in enumerate(variables):
        for j, metric in enumerate(metrics):
            ax = axes[i][j]
            
            # Extract metric values for all stations
            metric_key = f'{variable}_{metric}'
            values = []
            
            for station_id in station_ids:
                if metric_key in station_metrics[station_id]:
                    values.append(station_metrics[station_id][metric_key])
                else:
                    values.append(np.nan)
            
            # Create bar plot
            bars = ax.bar(station_ids, values, alpha=0.7,
                         color=plt.cm.viridis(j/len(metrics)))
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                if not np.isnan(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'{variable.title()} - {metric.upper()}')
            ax.set_ylabel(metric.upper())
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add reference lines for some metrics
            if metric == 'correlation':
                ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Good (0.7)')
                ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Fair (0.5)')
                ax.legend()
    
    plt.tight_layout()
    return fig


def plot_comprehensive_analysis(gdirs, filesuffix='', save_dir=None):
    """Create a comprehensive analysis dashboard with multiple plots.
    
    Parameters
    ----------
    save_dir : str or None
        Directory to save individual plots
    
    Returns
    -------
    dict of figures
    """
    
    gdir = gdirs[0] if isinstance(gdirs, list) else gdirs
    figures = {}
    
    # 1. Station network overview
    fig1 = plot_station_analysis(gdirs, filesuffix=filesuffix)
    figures['station_analysis'] = fig1
    
    # 2. Lapse rate analysis  
    fig2 = plot_lapse_rate_analysis(gdirs, filesuffix=filesuffix)
    figures['lapse_rates'] = fig2
    
    # 3. Climate comparison
    fig3 = plot_climate_comparison(gdirs, variable='temperature', filesuffix=filesuffix)
    figures['climate_comparison'] = fig3
    
    # 4. Station vs downscaled comparison
    fig4 = plot_station_vs_downscaled(gdirs, variable='temperature', filesuffix=filesuffix)
    figures['station_comparison'] = fig4
    
    # 5. Validation metrics
    fig5 = plot_flexible_validation(gdirs, filesuffix=filesuffix)
    figures['validation'] = fig5
    
    # Save figures if directory provided
    if save_dir:
        utils.mkdir(save_dir)
        for name, fig in figures.items():
            if fig is not None:
                fig.savefig(os.path.join(save_dir, f'{gdir.rgi_id}_{name}.png'), 
                           dpi=300, bbox_inches='tight')
    
    return figures


def plot_lapse_rate_analysis(gdirs, filesuffix='', figsize=(12, 8)):
    """Create a comprehensive lapse rate analysis plot showing monthly and seasonal variations.
    
    Parameters
    ----------
    gdirs : list
        List of glacier directories
    filesuffix : str
        File suffix for physical parameters file
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib figure
    """
    
    gdir = gdirs[0] if isinstance(gdirs, list) else gdirs
    
    # Load physical parameters
    try:
        phys_params_path = gdir.get_filepath('physical_parameters', filesuffix=filesuffix)
        with xr.open_dataset(phys_params_path) as ds:
            phys_params = json.loads(ds.attrs['physical_parameters'])
            lapse_rates = phys_params.get('lapse_rates', {}).get('temperature', {})
            
    except (FileNotFoundError, KeyError):
        log.warning("Could not load lapse rates from physical_parameters file")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'Temperature Lapse Rate Analysis - {gdir.rgi_id}', fontsize=14)
    
    # Monthly lapse rates (top left)
    ax1 = axes[0, 0]
    if 'monthly' in lapse_rates:
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_values = []
        monthly_errors = []
        
        for month in ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']:
            if month in lapse_rates['monthly']:
                monthly_values.append(lapse_rates['monthly'][month]['lapse_rate'] * 1000)  # K/km
                monthly_errors.append(lapse_rates['monthly'][month].get('std_error', 0) * 1000)
            else:
                monthly_values.append(np.nan)
                monthly_errors.append(0)
        
        ax1.errorbar(months, monthly_values, yerr=monthly_errors, 
                    marker='o', capsize=3, linewidth=2, markersize=6)
        ax1.axhline(y=-6.5, color='red', linestyle='--', alpha=0.7, label='Standard (-6.5 K/km)')
        ax1.set_title('Monthly Lapse Rates')
        ax1.set_ylabel('Lapse Rate [K/km]')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Seasonal comparison (top right)
    ax2 = axes[0, 1]
    if any(season in lapse_rates for season in ['winter', 'spring', 'summer', 'autumn']):
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        seasonal_values = []
        seasonal_errors = []
        
        for season in ['winter', 'spring', 'summer', 'autumn']:
            if season in lapse_rates:
                seasonal_values.append(lapse_rates[season]['lapse_rate'] * 1000)
                seasonal_errors.append(lapse_rates[season].get('std_error', 0) * 1000)
            else:
                seasonal_values.append(np.nan)
                seasonal_errors.append(0)
        
        colors = ['lightblue', 'lightgreen', 'orange', 'brown']
        bars = ax2.bar(seasons, seasonal_values, yerr=seasonal_errors, 
                      capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=-6.5, color='red', linestyle='--', alpha=0.7, label='Standard (-6.5 K/km)')
        ax2.set_title('Seasonal Lapse Rates')
        ax2.set_ylabel('Lapse Rate [K/km]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # R-squared values (bottom left)
    ax3 = axes[1, 0]
    if 'monthly' in lapse_rates:
        r_squared_values = []
        for month in ['January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']:
            if month in lapse_rates['monthly']:
                r_squared_values.append(lapse_rates['monthly'][month].get('r_squared', 0))
            else:
                r_squared_values.append(0)
        
        ax3.plot(months, r_squared_values, marker='s', linewidth=2, markersize=6, color='purple')
        ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Good fit (R² = 0.5)')
        ax3.set_title('Goodness of Fit (R²)')
        ax3.set_ylabel('R-squared')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Summary statistics (bottom right)
    ax4 = axes[1, 1]
    if 'monthly' in lapse_rates:
        # Calculate statistics
        valid_rates = [lapse_rates['monthly'][month]['lapse_rate'] * 1000 
                      for month in lapse_rates['monthly'] 
                      if not np.isnan(lapse_rates['monthly'][month]['lapse_rate'])]
        
        if valid_rates:
            stats_text = f"""
            Annual Mean: {np.mean(valid_rates):.2f} K/km
            Std Dev: {np.std(valid_rates):.2f} K/km
            Min: {np.min(valid_rates):.2f} K/km  
            Max: {np.max(valid_rates):.2f} K/km
            
            Standard Rate: -6.5 K/km
            Deviation: {np.mean(valid_rates) + 6.5:.2f} K/km
            """
            
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            
    ax4.set_title('Summary Statistics')
    ax4.axis('off')
    
    plt.tight_layout()
    return fig


def plot_station_analysis(gdirs, filesuffix='', figsize=(14, 10)):
    """Create a comprehensive station analysis plot showing locations, elevations, and data quality.
    
    Parameters  
    ----------
    gdirs : list
        List of glacier directories
    filesuffix : str
        File suffix for physical parameters file
    figsize : tuple
        Figure size
        
    Returns
    -------
    fig : matplotlib figure
    """
    
    gdir = gdirs[0] if isinstance(gdirs, list) else gdirs
    
    # Load physical parameters and validation results
    try:
        phys_params_path = gdir.get_filepath('physical_parameters', filesuffix=filesuffix)
        with xr.open_dataset(phys_params_path) as ds:
            phys_params = json.loads(ds.attrs['physical_parameters'])
            station_info = phys_params.get('station_info', {})
            
    except (FileNotFoundError, KeyError):
        log.warning("Could not load station information")
        return None
    
    try:
        validation_path = gdir.get_filepath('validation_results', filesuffix=filesuffix)
        with xr.open_dataset(validation_path) as ds:
            validation_results = json.loads(ds.attrs['validation_results'])
            station_metrics = validation_results.get('station_metrics', {})
    except (FileNotFoundError, KeyError):
        station_metrics = {}
    
    if not station_info:
        log.warning("No station information found")
        return None
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Station locations map (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    plot_climate_stations(gdirs, ax=ax1, filesuffix=filesuffix, 
                         add_station_labels=True, station_size=150)
    ax1.set_title('Station Locations and Selection')
    
    # Station elevation profile (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    elevations = [info['elevation'] for info in station_info.values()]
    station_ids = list(station_info.keys())
    
    ax2.barh(range(len(elevations)), elevations, color='steelblue', alpha=0.7)
    ax2.set_yticks(range(len(elevations)))
    ax2.set_yticklabels(station_ids, fontsize=8)
    ax2.set_xlabel('Elevation [m]')
    ax2.set_title('Station Elevations')
    ax2.grid(True, alpha=0.3)
    
    # Data completeness (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    if station_metrics:
        completeness = []
        station_names = []
        for station_id, metrics in station_metrics.items():
            if 'data_completeness' in metrics:
                completeness.append(metrics['data_completeness'] * 100)
                station_names.append(station_id)
        
        if completeness:
            bars = ax3.bar(station_names, completeness, color='green', alpha=0.7)
            ax3.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Good (80%)')
            ax3.set_ylabel('Completeness [%]')
            ax3.set_title('Data Completeness')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    
    # Temperature statistics (middle center)
    ax4 = fig.add_subplot(gs[1, 1])
    if station_metrics:
        temp_means = []
        station_names = []
        for station_id, metrics in station_metrics.items():
            if 'temp_mean' in metrics:
                temp_means.append(metrics['temp_mean'])
                station_names.append(station_id)
        
        if temp_means:
            ax4.bar(station_names, temp_means, color='orange', alpha=0.7)
            ax4.set_ylabel('Mean Temperature [°C]')
            ax4.set_title('Mean Temperatures')
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
    
    # Precipitation statistics (middle right)
    ax5 = fig.add_subplot(gs[1, 2])
    if station_metrics:
        precip_means = []
        station_names = []
        for station_id, metrics in station_metrics.items():
            if 'precip_mean' in metrics:
                precip_means.append(metrics['precip_mean'])
                station_names.append(station_id)
        
        if precip_means:
            ax5.bar(station_names, precip_means, color='blue', alpha=0.7)
            ax5.set_ylabel('Mean Precipitation [mm]')
            ax5.set_title('Mean Precipitation')
            ax5.tick_params(axis='x', rotation=45)
            ax5.grid(True, alpha=0.3)
    
    # Summary information (bottom, spans all columns)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Create summary text
    n_stations = len(station_info)
    elev_range = f"{min(elevations):.0f} - {max(elevations):.0f} m"
    
    if station_metrics:
        avg_completeness = np.mean([m.get('data_completeness', 0) for m in station_metrics.values()]) * 100
        temp_range = [m.get('temp_mean', np.nan) for m in station_metrics.values()]
        temp_range = f"{np.nanmin(temp_range):.1f} to {np.nanmax(temp_range):.1f} °C"
    else:
        avg_completeness = 0
        temp_range = "N/A"
    
    summary_text = f"""
    STATION NETWORK SUMMARY
    
    Number of stations: {n_stations}
    Elevation range: {elev_range}
    Average data completeness: {avg_completeness:.1f}%
    Temperature range: {temp_range}
    
    Stations used: {', '.join(station_info.keys())}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             verticalalignment='top', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    ax6.axis('off')
    
    fig.suptitle(f'Climate Station Analysis - {gdir.rgi_id}', fontsize=16, fontweight='bold')
    
    return fig


def plot_modeloutput_section(model=None, ax=None, title=''):
    """Plots the result of the model output along the flowline.

    Parameters
    ----------
    model: obj
        either a FlowlineModel or a list of model flowlines.
    fig
    title
    """

    try:
        fls = model.fls
    except AttributeError:
        fls = model

    if ax is None:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_axes([0.07, 0.08, 0.7, 0.84])
    else:
        fig = plt.gcf()

    # Compute area histo
    area = np.array([])
    height = np.array([])
    bed = np.array([])
    for cls in fls:
        a = cls.widths_m * cls.dx_meter * 1e-6
        a = np.where(cls.thick > 0, a, 0)
        area = np.concatenate((area, a))
        height = np.concatenate((height, cls.surface_h))
        bed = np.concatenate((bed, cls.bed_h))
    ylim = [bed.min(), height.max()]

    # Plot histo
    posax = ax.get_position()
    posax = [posax.x0 + 2 * posax.width / 3.0,
             posax.y0,  posax.width / 3.0,
             posax.height]
    axh = fig.add_axes(posax, frameon=False)

    axh.hist(height, orientation='horizontal', range=ylim, bins=20,
             alpha=0.3, weights=area)
    axh.invert_xaxis()
    axh.xaxis.tick_top()
    axh.set_xlabel('Area incl. tributaries (km$^2$)')
    axh.xaxis.set_label_position('top')
    axh.set_ylim(ylim)
    axh.yaxis.set_ticks_position('right')
    axh.set_yticks([])
    axh.axhline(y=ylim[1], color='black', alpha=1)  # qick n dirty trick

    # plot Centerlines
    cls = fls[-1]
    x = np.arange(cls.nx) * cls.dx * cls.map_dx

    # Plot the bed
    ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

    # Where trapezoid change color
    if hasattr(cls, '_do_trapeze') and cls._do_trapeze:
        bed_t = cls.bed_h * np.nan
        pt = cls.is_trapezoid & (~cls.is_rectangular)
        bed_t[pt] = cls.bed_h[pt]
        ax.plot(x, bed_t, color='rebeccapurple', linewidth=2.5,
                label='Bed (Trap.)')
        bed_t = cls.bed_h * np.nan
        bed_t[cls.is_rectangular] = cls.bed_h[cls.is_rectangular]
        ax.plot(x, bed_t, color='crimson', linewidth=2.5, label='Bed (Rect.)')

    # Plot glacier
    surfh = surf_to_nan(cls.surface_h, cls.thick)
    ax.plot(x, surfh, color='#003399', linewidth=2, label='Glacier')

    # Plot tributaries
    for i, l in zip(cls.inflow_indices, cls.inflows):
        if l.thick[-1] > 0:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='#993399',
                    markeredgecolor='k',
                    label='Tributary (active)')
        else:
            ax.plot(x[i], cls.surface_h[i], 's', markerfacecolor='w',
                    markeredgecolor='k',
                    label='Tributary (inactive)')
    if getattr(model, 'do_calving', False):
        ax.hlines(model.water_level, x[0], x[-1], linestyles=':', color='C0')

    ax.set_ylim(ylim)

    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel('Distance along flowline (m)')
    ax.set_ylabel('Altitude (m)')

    # Title
    ax.set_title(title, loc='left')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              bbox_to_anchor=(1.34, 1.0),
              frameon=False)


def plot_modeloutput_section_withtrib(model=None, fig=None, title=''):
    """Plots the result of the model output along the flowline.

    Parameters
    ----------
    model: obj
        either a FlowlineModel or a list of model flowlines.
    fig
    title
    """

    try:
        fls = model.fls
    except AttributeError:
        fls = model

    n_tribs = len(fls) - 1

    axs = []
    if n_tribs == 0:
        if fig is None:
            fig = plt.figure(figsize=(8, 5))
        axmaj = fig.add_subplot(111)
    elif n_tribs <= 3:
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        axmaj = plt.subplot2grid((2, 3), (1, 0), colspan=3)
        for i in np.arange(n_tribs):
            axs.append(plt.subplot2grid((2, 3), (0, i)))
    elif n_tribs <= 6:
        if fig is None:
            fig = plt.figure(figsize=(14, 10))
        axmaj = plt.subplot2grid((3, 3), (2, 0), colspan=3)
        for i in np.arange(n_tribs):
            j = 0
            if i >= 3:
                i -= 3
                j = 1
            axs.append(plt.subplot2grid((3, 3), (j, i)))
    else:
        raise NotImplementedError()

    for i, cls in enumerate(fls):
        if i == n_tribs:
            ax = axmaj
        else:
            ax = axs[i]

        x = np.arange(cls.nx) * cls.dx * cls.map_dx

        # Plot the bed
        ax.plot(x, cls.bed_h, color='k', linewidth=2.5, label='Bed (Parab.)')

        # Where trapezoid change color
        if hasattr(cls, '_do_trapeze') and cls._do_trapeze:
            bed_t = cls.bed_h * np.nan
            pt = cls.is_trapezoid & (~cls.is_rectangular)
            bed_t[pt] = cls.bed_h[pt]
            ax.plot(x, bed_t, color='rebeccapurple', linewidth=2.5,
                    label='Bed (Trap.)')
            bed_t = cls.bed_h * np.nan
            bed_t[cls.is_rectangular] = cls.bed_h[cls.is_rectangular]
            ax.plot(x, bed_t, color='crimson', linewidth=2.5,
                    label='Bed (Rect.)')

        # Plot glacier
        surfh = surf_to_nan(cls.surface_h, cls.thick)
        ax.plot(x, surfh, color='#003399', linewidth=2, label='Glacier')

        # Plot tributaries
        for i, l in zip(cls.inflow_indices, cls.inflows):
            if l.thick[-1] > 0:
                ax.plot(x[i], cls.surface_h[i], 's', color='#993399',
                        label='Tributary (active)')
            else:
                ax.plot(x[i], cls.surface_h[i], 's', color='none',
                        label='Tributary (inactive)')

        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xlabel('Distance along flowline (m)')
        ax.set_ylabel('Altitude (m)')

    # Title
    plt.title(title, loc='left')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax.legend(list(by_label.values()), list(by_label.keys()),
              loc='best', frameon=False)
    fig.tight_layout()