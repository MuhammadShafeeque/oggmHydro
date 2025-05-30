"""OGGM core modules.

This package contains the core functionality of OGGM, including both 
traditional glacier modeling and the new hydro-glaciological capabilities.

Traditional modules:
- climate: Climate data processing
- massbalance: Mass balance models 
- flowline: Flowline dynamics
- centerlines: Centerline computation
- gis: Geographic information system tools
- inversion: Ice thickness inversion
- dynamic_spinup: Dynamic model spinup
- sia2d: 2D shallow ice approximation

Hydro-glaciological modules (parallel approach):
- hydro_climate: Hydro-climate data processing and runoff computation
- hydro_massbalance: Multi-target calibration mass balance models
- hydro_validation: Hydro-glaciological validation framework
"""

# Traditional OGGM core modules
from oggm.core import (
    climate,
    massbalance, 
    flowline,
    centerlines,
    gis,
    inversion,
    dynamic_spinup,
    sia2d
)

# Hydro-glaciological modules (parallel approach)
from oggm.core import (
    hydro_climate,
    hydro_massbalance, 
    hydro_validation
)

# Also add regional_scaling
from oggm.core import regional_scaling

# Make key classes easily accessible
from oggm.core.massbalance import (
    MassBalanceModel,
    MonthlyTIModel,
    ScalarMassBalance,
    LinearMassBalance,
    ConstantMassBalance
)

# Make hydro classes easily accessible
from oggm.core.hydro_massbalance import (
    HydroMassBalance,
    MultiTargetCalibrator,
    DataConfiguration,
    hydro_mb_calibration
)

from oggm.core.hydro_climate import (
    HydroClimateIntegration,
    RunoffComputation,
    compute_glacier_runoff,
    calibrate_glacier_runoff,
    setup_hydro_climate_integration
)

from oggm.core.hydro_validation import (
    HydroValidationFramework,
    validate_hydro_mass_balance,
    validate_hydro_mass_balance_batch,
    compare_hydro_models
)

__all__ = [
    # Traditional modules
    'climate',
    'massbalance',
    'flowline', 
    'centerlines',
    'gis',
    'inversion',
    'dynamic_spinup',
    'sia2d',
    'regional_scaling',
    
    # Hydro modules
    'hydro_climate',
    'hydro_massbalance',
    'hydro_validation',
    
    # Traditional classes
    'MassBalanceModel',
    'MonthlyTIModel', 
    'ScalarMassBalance',
    'LinearMassBalance',
    'ConstantMassBalance',
    
    # Hydro classes
    'HydroMassBalance',
    'MultiTargetCalibrator',
    'DataConfiguration',
    'hydro_mb_calibration',
    'HydroClimateIntegration',
    'RunoffComputation',
    'compute_glacier_runoff',
    'calibrate_glacier_runoff',
    'setup_hydro_climate_integration',
    'HydroValidationFramework',
    'validate_hydro_mass_balance',
    'validate_hydro_mass_balance_batch',
    'compare_hydro_models'
]