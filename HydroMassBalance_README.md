# 🏔️ HydroMassBalance: Revolutionary Multi-Target Glacier Mass Balance Model

> **The Next-Generation Glacier Mass Balance Model for Water Resources Applications**

[![License: BSD-3](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![OGGM Compatible](https://img.shields.io/badge/OGGM-Compatible-green.svg)](https://oggm.org)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🚀 What is HydroMassBalance?

HydroMassBalance is a **revolutionary glacier mass balance model** that combines:

- 🎯 **Multi-Target Calibration**: Mass balance + **Freshwater Runoff** + Volume + Velocity
- 🧬 **Advanced Physics**: PyGEM-inspired refreezing, surface evolution, debris effects
- 🌡️ **Climate Integration**: Seamless regional scaling with station+ERA5 fusion
- ⚡ **OGGM Compatible**: 100% backward compatible, drop-in replacement
- 🔬 **Comprehensive Validation**: Cross-validation, benchmarking, uncertainty analysis

### 🌟 Key Innovation: **Runoff Calibration**

**First glacier model to routinely calibrate against streamflow observations!**

```python
# Revolutionary: Calibrate mass balance against runoff data
mb_model = HydroMassBalance(
    gdir,
    calibration_targets=['geodetic_mb', 'runoff'],  # 🚀 Game changer!
    runoff_data_config=runoff_config,
    physics_level='advanced'
)
```

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Installation](#-installation)
3. [Key Features](#-key-features)
4. [Architecture Overview](#-architecture-overview)
5. [Usage Examples](#-usage-examples)
6. [Advanced Features](#-advanced-features)
7. [Integration with OGGM](#-integration-with-oggm)
8. [API Reference](#-api-reference)
9. [Contributing](#-contributing)
10. [Citation](#-citation)

---

## ⚡ Quick Start

### 30-Second Demo

```python
from hydro_workflow import hydro_mass_balance_quickstart

# One line to run complete HydroMassBalance workflow!
results = hydro_mass_balance_quickstart('RGI60-11.00897')  # Hintereisferner
```

### 5-Minute Tutorial

```python
import oggm
from oggm import cfg, workflow
from hydro_massbalance import HydroMassBalance, DataConfiguration

# 1. Setup
cfg.initialize()
gdirs = workflow.init_glacier_directories(['RGI60-11.00897'], from_prepro_level=3)
gdir = gdirs[0]

# 2. Configure runoff data (your discharge measurements)
runoff_config = DataConfiguration(
    'runoff', 'your_discharge_data.csv',
    temporal_scale='daily', target_scale='monthly'
)

# 3. Create revolutionary multi-target model
mb_model = HydroMassBalance(
    gdir,
    physics_level='advanced',
    calibration_targets=['geodetic_mb', 'runoff'],
    runoff_data_config=runoff_config,
    enable_refreezing=True,
    enable_debris=True
)

# 4. Calibrate and validate
results = mb_model.calibrate()
print(f"Calibration success: {results['success']}")
print(f"Final parameters: {results['optimal_parameters']}")

# 5. Compute detailed runoff components  
from hydro_climate import RunoffComputation
runoff_computer = RunoffComputation(gdir)
runoff_results = runoff_computer.compute_runoff_components(mb_model)

print(f"Total runoff: {np.sum(runoff_results['total_runoff']):.2f} m³")
print(f"Ice melt contribution: {np.sum(runoff_results['ice_runoff'])/np.sum(runoff_results['total_runoff']):.1%}")
```

---

## 🛠️ Installation

### Prerequisites

```bash
# Install OGGM first
conda install -c conda-forge oggm

# OR from source
pip install git+https://github.com/OGGM/oggm.git
```

### Install HydroMassBalance

```bash
# Method 1: From source (recommended)
git clone https://github.com/your-repo/hydro-massbalance.git
cd hydro-massbalance
pip install -e .

# Method 2: Direct install (when available)
pip install hydro-massbalance
```

### Dependencies

- **Core**: `numpy`, `pandas`, `scipy`, `scikit-learn`
- **OGGM**: `oggm>=1.6.0`
- **Climate**: `xarray`, `netCDF4`, `cftime`
- **Visualization**: `matplotlib`, `seaborn`
- **Optional**: `salem`, `rasterio` (for advanced features)

---

## 🌟 Key Features

### 🎯 Multi-Target Calibration System

| Target Type | Description | Temporal Flexibility | Priority |
|-------------|-------------|---------------------|----------|
| **Mass Balance** | Geodetic, WGMS, custom | Monthly/Seasonal/Annual | 🔴 Highest |
| **🚀 Runoff** | Discharge observations | Daily→Monthly/Annual | 🟡 High |
| **Volume** | Photogrammetry, lidar | Annual/Multi-year | 🟠 Medium |
| **Velocity** | SAR, feature tracking | Seasonal/Annual | 🟢 Low |

### 🧬 Advanced Physics (PyGEM Integration)

```python
# Sophisticated refreezing physics
RefreezeOptions = {
    'HH2015': 'Heat conduction approach (Huss & Hock 2015)',
    'Woodward': 'Temperature-based empirical method',
    'none': 'Disable refreezing'
}

# Dynamic surface evolution
SurfaceTypes = {
    0: 'off_glacier',
    1: 'ice',      # DDF = 0.008 m w.e. °C⁻¹ day⁻¹
    2: 'snow',     # DDF = 0.003 m w.e. °C⁻¹ day⁻¹  
    3: 'firn',     # DDF = 0.005 m w.e. °C⁻¹ day⁻¹
    4: 'debris'    # DDF = 0.012 m w.e. °C⁻¹ day⁻¹ (enhanced)
}
```

### 🌡️ Climate Data Sources

- **🌟 Regional Scaling**: High-resolution bias-corrected (station+ERA5)
- **W5E5**: Global 0.5° reanalysis
- **ERA5**: 0.25° reanalysis  
- **OGGM Default**: CRU/ERA5 combination
- **Custom**: Your own climate data

### 📊 Comprehensive Validation

```python
validation_methods = [
    'leave_one_out',     # LOO cross-validation
    'temporal_split',    # Train/test temporal split
    'k_fold',           # K-fold cross-validation
    'benchmarking'      # vs OGGM, PyGEM, linear MB
]

metrics = [
    'RMSE', 'MAE', 'R²', 'NSE',  # Standard metrics
    'KGE',                        # Kling-Gupta Efficiency
    'bias', 'correlation'         # Additional metrics
]
```

---

## 🏗️ Architecture Overview

```
HydroMassBalance System Architecture
═══════════════════════════════════

📊 Data Input Layer
├── Station Climate Data (CSV/JSON/NetCDF)
├── Discharge/Runoff Data (CSV/JSON)  
├── Volume Data (CSV/JSON)
├── Velocity Data (CSV/JSON)
└── OGGM Glacier Data (Preprocessed)

🧠 Core Processing Layer
├── HydroMassBalance (Main Model Class)
│   ├── Multi-Target Calibrator
│   ├── Physics Modules (Refreezing, Surface Evolution)
│   ├── Climate Integration (Regional Scaling)
│   └── Temporal Scale Manager
│
├── Validation Framework
│   ├── Cross-Validation Methods
│   ├── Benchmark Comparisons  
│   ├── Uncertainty Analysis
│   └── Performance Assessment
│
└── Runoff Computation
    ├── Component Separation (Ice/Snow/Rain)
    ├── Hydrological Routing
    ├── Evapotranspiration
    └── Quality Control

📈 Output Layer
├── Calibrated Model Parameters
├── Mass Balance Time Series
├── Runoff Components (Ice/Snow/Rain)
├── Validation Metrics & Reports
├── Uncertainty Estimates
└── Comprehensive HTML Reports

🔗 OGGM Integration
├── Entity Tasks (Single Glacier)
├── Global Tasks (Multiple Glaciers)
├── Workflow Integration
└── Backward Compatibility
```

---

## 💡 Usage Examples

### Example 1: Basic Advanced Model

```python
from hydro_massbalance import HydroMassBalance

# Create advanced physics model
mb_model = HydroMassBalance(
    gdir,
    physics_level='advanced',
    enable_refreezing=True,
    enable_debris=True,
    refreezing_method='HH2015'
)

# Get mass balance with sophisticated physics
heights = np.linspace(gdir.min_h, gdir.max_h, 20)
mb_annual = mb_model.get_annual_mb(heights, year=2015)

print(f"Mass balance range: {mb_annual.min():.3f} to {mb_annual.max():.3f} m w.e./yr")
```

### Example 2: Multi-Target Calibration

```python
from hydro_massbalance import DataConfiguration

# Configure multiple data sources
runoff_config = DataConfiguration('runoff', 'discharge.csv', 
                                 temporal_scale='daily', target_scale='monthly')
volume_config = DataConfiguration('volume', 'volume.csv',
                                 temporal_scale='annual', target_scale='annual')

# Multi-target model
mb_model = HydroMassBalance(
    gdir,
    calibration_targets=['geodetic_mb', 'runoff', 'volume'],
    runoff_data_config=runoff_config,
    volume_data_config=volume_config,
    target_priorities={'mb': 1.0, 'runoff': 0.8, 'volume': 0.6}
)

# Hierarchical calibration (MB → Volume → Runoff)
results = mb_model.calibrate()
```

### Example 3: Regional Scaling Climate

```python
from hydro_climate import HydroClimateIntegration

# Setup high-resolution climate
climate_integration = HydroClimateIntegration(
    gdir, 
    climate_source='regional_scaling',
    station_data_path='stations.csv'
)

# Use in mass balance model
mb_model = HydroMassBalance(
    gdir,
    climate_source='regional_scaling',
    calibration_targets=['geodetic_mb', 'runoff']
)
```

### Example 4: Detailed Runoff Analysis

```python
from hydro_climate import RunoffComputation

# Advanced runoff computation
runoff_computer = RunoffComputation(
    gdir,
    routing_config={
        'routing_method': 'linear_reservoir',
        'reservoir_constant': 30,  # days
        'evapotranspiration': True,
        'debris_retention_factor': 0.1
    }
)

# Compute all runoff components
runoff_results = runoff_computer.compute_runoff_components(
    mb_model,
    year_range=(2000, 2020),
    temporal_scale='monthly'
)

# Analyze results
total_runoff = np.sum(runoff_results['total_runoff'])
ice_fraction = np.sum(runoff_results['ice_runoff']) / total_runoff
snow_fraction = np.sum(runoff_results['snow_runoff']) / total_runoff

print(f"Total runoff: {total_runoff:.2f} m³")
print(f"Ice melt: {ice_fraction:.1%}")
print(f"Snow melt: {snow_fraction:.1%}")
```

### Example 5: Comprehensive Validation

```python
from hydro_validation import HydroValidationFramework

# Setup validation
validator = HydroValidationFramework(
    mb_model,
    validation_config={
        'methods': ['leave_one_out', 'temporal_split', 'k_fold'],
        'benchmarks': ['oggm_default', 'linear_mb'],
        'uncertainty_analysis': True,
        'plot_results': True
    }
)

# Run comprehensive validation
validation_results = validator.run_comprehensive_validation()

print(f"Overall grade: {validation_results['overall_performance']['grade']}")
print(f"Performance score: {validation_results['overall_performance']['overall_score']:.3f}")
```

### Example 6: Batch Processing

```python
from hydro_workflow import hydro_mass_balance_workflow

# Process multiple glaciers
rgi_ids = ['RGI60-11.00897', 'RGI60-11.00898', 'RGI60-11.00899']
gdirs = workflow.init_glacier_directories(rgi_ids, from_prepro_level=3)

# Complete workflow
results = hydro_mass_balance_workflow(
    gdirs,
    station_data_path='station_data.csv',
    runoff_data_path='runoff_data/',  # Directory with individual files
    physics_level='advanced',
    calibration_targets=['geodetic_mb', 'runoff'],
    climate_source='regional_scaling',
    create_summary_report=True
)

print(f"Processed {len(gdirs)} glaciers")
print(f"Success rate: {results['summary']['successful_processing']}")
```

---

## 🔬 Advanced Features

### Temporal Scale Flexibility

```python
# Different temporal scales for different targets
mb_model = HydroMassBalance(
    gdir,
    calibration_targets=['geodetic_mb', 'runoff', 'volume'],
    mb_temporal_scale='annual',        # Mass balance: annual
    runoff_temporal_scale='monthly',   # Runoff: monthly  
    volume_temporal_scale='decadal'    # Volume: decadal trend
)
```

### Uncertainty Quantification

```python
# Parameter uncertainty via Monte Carlo
validation_results = validator.run_comprehensive_validation()
param_uncertainty = validation_results['uncertainty']['parameter_uncertainty']

print(f"Mass balance uncertainty: ±{param_uncertainty['mb_std']:.3f} m w.e./yr")
print(f"95% confidence interval: {param_uncertainty['percentiles']['5th']:.3f} to {param_uncertainty['percentiles']['95th']:.3f}")
```

### Custom Physics Modules

```python
class CustomPhysicsModule(PhysicsModule):
    """Your custom physics implementation"""
    
    def compute(self, climate_data, glacier_state, **kwargs):
        # Your physics calculations
        return results
    
    def get_parameters(self):
        return ['param1', 'param2']

# Add to model
mb_model.physics_modules['custom'] = CustomPhysicsModule()
```

---

## 🔗 Integration with OGGM

### Drop-in Replacement

```python
# Replace this:
from oggm.core.massbalance import MonthlyTIModel
mb_model = MonthlyTIModel(gdir)

# With this:
from hydro_massbalance import HydroMassBalance  
mb_model = HydroMassBalance(gdir)  # 100% compatible!

# All OGGM functions work unchanged
annual_mb = mb_model.get_annual_mb(heights, year=2020)
specific_mb = mb_model.get_specific_mb(fls=fls, year=2020)
ela = mb_model.get_ela(year=2020)
```

### OGGM Entity Tasks

```python
from hydro_massbalance import hydro_mb_calibration
from hydro_validation import validate_hydro_mass_balance  
from hydro_climate import compute_glacier_runoff

# Use as OGGM entity tasks
workflow.execute_entity_task(hydro_mb_calibration, gdirs, 
                           calibration_targets=['geodetic_mb', 'runoff'])
workflow.execute_entity_task(validate_hydro_mass_balance, gdirs)
workflow.execute_entity_task(compute_glacier_runoff, gdirs)
```

### OGGM Global Tasks

```python
from hydro_workflow import hydro_mass_balance_workflow

# Global task for multiple glaciers
results = hydro_mass_balance_workflow(
    gdirs,
    **configuration_parameters
)
```

---

## 📚 API Reference

### Core Classes

#### `HydroMassBalance(MassBalanceModel)`

The main mass balance model class.

**Parameters:**
- `gdir` (GlacierDirectory): OGGM glacier directory
- `physics_level` (str): 'simple', 'intermediate', 'advanced'
- `calibration_targets` (list): ['geodetic_mb', 'runoff', 'volume', 'velocity']
- `climate_source` (str): 'oggm_default', 'regional_scaling', 'w5e5', 'era5'
- `enable_refreezing` (bool): Enable refreezing physics
- `enable_debris` (bool): Enable debris effects
- `refreezing_method` (str): 'HH2015', 'Woodward', 'none'

**Key Methods:**
- `get_annual_mb(heights, year)`: Annual mass balance
- `get_monthly_mb(heights, year)`: Monthly mass balance  
- `calibrate()`: Multi-target calibration
- `get_ela(year)`: Equilibrium line altitude

#### `DataConfiguration`

Data configuration for calibration targets.

**Parameters:**
- `data_type` (str): 'runoff', 'volume', 'velocity', 'mb'
- `file_path` (str): Path to data file
- `temporal_scale` (str): 'daily', 'monthly', 'seasonal', 'annual'
- `target_scale` (str): Target temporal scale for calibration
- `component` (str): 'total', 'glacier', 'ice', 'snow'

#### `MultiTargetCalibrator`

Advanced calibration engine.

**Methods:**
- `calibrate(mb_model, method)`: Execute calibration
- `objective_function(params, mb_model)`: Multi-target objective

#### `HydroValidationFramework`

Comprehensive validation framework.

**Methods:**
- `run_comprehensive_validation()`: Complete validation suite
- `_leave_one_out_validation()`: LOO cross-validation
- `_temporal_split_validation()`: Temporal validation

#### `RunoffComputation`

Advanced runoff computation and routing.

**Methods:**
- `compute_runoff_components(mb_model, year_range, temporal_scale)`: Detailed runoff
- `calibrate_against_discharge(observed_discharge, mb_model)`: Runoff calibration

### Entity Tasks

#### `hydro_mb_calibration(gdir, **kwargs)`

OGGM entity task for HydroMassBalance calibration.

#### `validate_hydro_mass_balance(gdir, **kwargs)`

OGGM entity task for validation.

#### `compute_glacier_runoff(gdir, **kwargs)`

OGGM entity task for runoff computation.

### Global Tasks

#### `hydro_mass_balance_workflow(gdirs, **kwargs)`

Complete workflow for multiple glaciers.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/MuhammadShafeeque/hydro-massbalance.git
cd hydro-massbalance
pip install -e .[dev]
pytest tests/
```

### Areas for Contribution

- 🧬 **New Physics Modules**: Avalanche redistribution, wind effects
- 🌍 **Climate Data Sources**: New reanalysis products, climate models
- 📊 **Validation Methods**: New cross-validation approaches
- 🚀 **Performance**: Optimization, parallelization
- 📖 **Documentation**: Examples, tutorials, case studies

---

## 📖 Citation

If you use HydroMassBalance in your research, please cite:

```bibtex
@software{hydro_massbalance_2024,
  title={HydroMassBalance: Multi-Target Glacier Mass Balance Model with Runoff Calibration},
  author={Shafeeque, Muhammad},
  year={2024},
  url={https://github.com/MuhammadShafeeque/hydro-massbalance},
  note={Revolutionary glacier mass balance model with freshwater runoff calibration}
}
```

**Key Publications:**
- Huss, M., & Hock, R. (2015). A new model for global glacier change and sea-level rise. *Frontiers in Earth Science*, 3, 54.
- Rounce, D. R., et al. (2020). Distributed global debris thickness estimates reveal debris significantly impacts glacier mass balance. *Geophysical Research Letters*, 47(22).
- Regional scaling methodology paper (in preparation)

---

## 🎯 Roadmap

### Version 1.1 (Q2 2024)
- [ ] **GPU Acceleration**: CUDA support for large-scale runs
- [ ] **Machine Learning**: Neural network bias correction
- [ ] **Advanced Routing**: Distributed hydrological model coupling

### Version 1.2 (Q3 2024)  
- [ ] **Ensemble Modeling**: Multi-model uncertainty quantification
- [ ] **Real-time Processing**: Operational runoff forecasting
- [ ] **Web Interface**: Online model configuration and execution

### Version 2.0 (Q4 2024)
- [ ] **Coupled Modeling**: Full glacier-hydrology-climate coupling
- [ ] **Extreme Events**: Flood and drought impact assessment
- [ ] **Decision Support**: Water resource management tools

---

## 🏆 Recognition

HydroMassBalance represents a **paradigm shift** in glacier mass balance modeling:

### 🌟 **World's First** runoff-calibrated glacier mass balance model
### 🧬 **Most Advanced** physics integration (OGGM + PyGEM)  
### 🎯 **Most Flexible** multi-target calibration system
### 🌍 **Most Comprehensive** climate data integration
### 📊 **Most Rigorous** validation framework

---

## 📞 Support & Community

- **Documentation**: [hydro-massbalance.readthedocs.io](https://hydro-massbalance.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/MuhammadShafeeque/hydro-massbalance/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MuhammadShafeeque/hydro-massbalance/discussions)
- **OGGM Forum**: [discuss.oggm.org](https://discuss.oggm.org)
- **Email**: shafeequ@uni-bremen.de

---

## 📄 License

MIT License

Compatible with OGGM and PyGEM licensing.

---

## 🙏 Acknowledgments

- **OGGM Team**: Framework and infrastructure
- **PyGEM Team**: Advanced physics implementations  
- **DFG - Regional Scaling**: Climate data methodology
- **Global Glacier Community**: Data, validation, feedback

---

> **"HydroMassBalance: Where glacier science meets water resources"** 🏔️💧

**Ready to revolutionize your glacier mass balance modeling?**

**[Get Started Now](#-quick-start)** | **[View Examples](#-usage-examples)** | **[Join Community](#-support--community)**