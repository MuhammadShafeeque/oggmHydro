"""Functions for Greenland peripheral glacier classifications.

This module provides functionality for classifying Greenland's peripheral
glaciers according to connectivity levels (CL0, CL1, CL2) and regional
groupings.
"""

# Built-ins
import logging
import os

# External libs
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.geometry as shpg
from shapely.ops import unary_union

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm import entity_task
from oggm.exceptions import InvalidParamsError

# Module logger
log = logging.getLogger(__name__)


# Define regions for Greenland
GREENLAND_REGIONS = {
    'North-East': ['05-02', '05-03'],
    'Central-East': ['05-04', '05-05'],
    'South-East': ['05-06', '05-07'],
    'South-West': ['05-08', '05-09'],
    'Central-West': ['05-10', '05-11'],
    'North-West': ['05-12', '05-13'],
    'North': ['05-01', '05-14', '05-15']
}


def classify_connectivity(rgi_id):
    """Classify a glacier by its connectivity level based on RGI ID.
    
    Parameters
    ----------
    rgi_id : str
        RGI ID of the glacier
        
    Returns
    -------
    str
        Connectivity level: 'CL0', 'CL1', or 'CL2'
    """
    # Extract connectivity level from RGI ID
    # For RGI v6, connectivity is encoded in O1Region
    if rgi_id.startswith('RGI60-05'):
        # Get the O1Region attribute which contains connectivity info
        # This assumes the connectivity info is available in the RGI ID
        # In practice, this might need to be loaded from RGI attributes
        
        # For demonstration purposes:
        # This is a placeholder - in reality, you would use actual RGI data
        # Check the 13th or 14th position in the RGI ID for connectivity
        # or use a lookup table
        
        # Example mapping from RGI ID to connectivity level
        connectivity_mapping = {
            '0': 'CL0',  # Completely detached
            '1': 'CL1',  # Dynamically decoupled
            '2': 'CL2'   # Dynamically connected
        }
        
        # Placeholder - in practice, use actual RGI data
        # This would be replaced with real logic
        conn_code = '0'  # Default to CL0
        
        return connectivity_mapping.get(conn_code, 'Unknown')
    else:
        raise InvalidParamsError(f"Not a Greenland glacier: {rgi_id}")


def identify_greenland_region(rgi_id):
    """Identify the Greenland region a glacier belongs to.
    
    Parameters
    ----------
    rgi_id : str
        RGI ID of the glacier
        
    Returns
    -------
    str
        Region name: 'North-East', 'Central-East', 'South-East', 
        'South-West', 'Central-West', 'North-West', or 'North'
    """
    if not rgi_id.startswith('RGI60-05'):
        raise InvalidParamsError(f"Not a Greenland glacier: {rgi_id}")
    
    # Extract RGI subregion (e.g., "05-01")
    subregion = rgi_id[6:11]
    
    # Find matching region
    for region, subregions in GREENLAND_REGIONS.items():
        if subregion in subregions:
            return region
    
    return "Unknown"


@entity_task(log)
def add_greenland_attributes(gdir):
    """Add Greenland-specific attributes to a glacier directory.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
        
    Returns
    -------
    dict
        Dictionary with added Greenland attributes
    """
    # Check if it's a Greenland glacier
    if not gdir.rgi_id.startswith('RGI60-05'):
        log.warning(f"{gdir.rgi_id} is not a Greenland glacier. Skipping.")
        return
    
    # Get connectivity level
    connectivity = classify_connectivity(gdir.rgi_id)
    
    # Get region
    region = identify_greenland_region(gdir.rgi_id)
    
    # Check if this is Flade Isblink Ice Cap
    is_fiic = gdir.rgi_id.startswith('RGI60-05.10315')
    
    # Determine if marine-terminating
    # In practice, this would be read from existing attributes
    # or determined from other data
    is_marine_terminating = getattr(gdir, 'is_tidewater', False)
    
    # Add attributes to glacier directory
    gdir.add_to_diagnostics('connectivity_level', connectivity)
    gdir.add_to_diagnostics('greenland_region', region)
    gdir.add_to_diagnostics('is_fiic', is_fiic)
    
    # Only consider CL0 and CL1 for analysis
    include_in_gpg_analysis = connectivity in ['CL0', 'CL1']
    gdir.add_to_diagnostics('include_in_gpg_analysis', include_in_gpg_analysis)
    
    # Return the added attributes
    return {
        'connectivity_level': connectivity,
        'greenland_region': region,
        'is_fiic': is_fiic,
        'is_marine_terminating': is_marine_terminating,
        'include_in_gpg_analysis': include_in_gpg_analysis
    }


@entity_task(log)
def parse_fiic_basins(gdir, fiic_shapefile=None):
    """Parse the Flade Isblink Ice Cap basins.
    
    This function processes the custom Flade Isblink Ice Cap basin
    subdivision to create individual drainage basins.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory
    fiic_shapefile : str, optional
        path to the shapefile with FIIC basin subdivision
        
    Returns
    -------
    list
        List of basin GDirs created
    """
    # Check if it's the Flade Isblink Ice Cap
    if not gdir.rgi_id.startswith('RGI60-05.10315'):
        log.warning(f"{gdir.rgi_id} is not Flade Isblink Ice Cap. Skipping.")
        return []
    
    # If no shapefile provided, try to find it in the OGGM data directory
    if fiic_shapefile is None:
        fiic_shapefile = os.path.join(cfg.PATHS['data_dir'], 
                                      'fiic_basins', 
                                      'fiic_basins.shp')
    
    # Check if file exists
    if not os.path.exists(fiic_shapefile):
        raise InvalidParamsError(f"FIIC basin shapefile not found: {fiic_shapefile}")
    
    # Read shapefile
    basins = gpd.read_file(fiic_shapefile)
    
    # Make sure it's in the same CRS as the glacier
    if basins.crs != gdir.grid.proj:
        basins = basins.to_crs(gdir.grid.proj)
    
    # Create new glacier directories for each basin
    basin_gdirs = []
    
    # Create new glaciers from the basins
    for idx, basin in basins.iterrows():
        # Create a new unique ID for this basin
        basin_id = f"{gdir.rgi_id}.{idx:02d}"
        
        # Create a new base_dir for this basin
        basin_dir = os.path.join(os.path.dirname(gdir.dir), basin_id)
        
        # Create a new GDir
        from oggm.utils import GlacierDirectory
        basin_gdir = GlacierDirectory(basin_id, base_dir=basin_dir)
        
        # Add the basin geometry
        basin_gdir.set_geometry(basin.geometry)
        
        # Copy relevant attributes from parent glacier
        basin_gdir.rgi_date = gdir.rgi_date
        basin_gdir.name = f"{gdir.name} - Basin {idx:02d}"
        
        # Add basin-specific attributes
        basin_gdir.is_fiic_basin = True
        basin_gdir.fiic_parent_id = gdir.rgi_id
        basin_gdir.basin_number = idx
        
        # Add to list
        basin_gdirs.append(basin_gdir)
    
    return basin_gdirs
