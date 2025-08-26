""" OGGM package.

Copyright: OGGM e.V. and OGGM Contributors

License: BSD-3-Clause
"""
# flake8: noqa
try:
    from importlib.metadata import version, PackageNotFoundError
    try:
        __version__ = version(__name__)
    except PackageNotFoundError:
        # package is not installed
        pass
    finally:
        del version, PackageNotFoundError
except ModuleNotFoundError:
    from pkg_resources import get_distribution, DistributionNotFound
    try:
        __version__ = get_distribution(__name__).version
    except DistributionNotFound:
        # package is not installed
        pass
    finally:
        del get_distribution, DistributionNotFound

try:
    from oggm.mpi import _init_oggm_mpi
    _init_oggm_mpi()
except ImportError:
    pass

# TODO: remove this when geopandas will behave a bit better
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module=r'.*geopandas')

# API
# Some decorators used by many
from oggm.utils import entity_task, global_task

# Classes
from oggm.utils import GlacierDirectory
from oggm.core.centerlines import Centerline
from oggm.core.flowline import Flowline
from oggm.core.flowline import FlowlineModel
from oggm.core.flowline import FileModel
from oggm.core.massbalance import MassBalanceModel
# GPG-Freshwater specific imports
from oggm.core.freshwater import compute_glacier_runoff, analyze_peak_water
from oggm.core.greenland import add_greenland_attributes, identify_greenland_region
from oggm.core.gpg_calibration import calibrate_inversion_from_geodetic_mb
