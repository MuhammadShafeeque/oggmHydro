# OGGM-GPG-Freshwater

This branch of OGGM (Open Global Glacier Model) is specifically enhanced for the Greenland Peripheral Glaciers (GPG) freshwater research project. It builds upon the `jog23` branch, which provides improved marine-terminating glacier dynamics.

## Enhanced Features for Greenland's Peripheral Glaciers

This branch implements methodologies and enhancements for accurate modeling of Greenland's peripheral glaciers, with a focus on:

### 1. Marine-Terminating Glacier Support

- Enhanced parametrization for terminal cliffs using hydrostatic pressure balance
- Updated sliding velocity calculations accounting for water depth
- Frontal ablation parametrization based on Oerlemans and Nick (2005): Q_f = k × d × h × w
- Consistency between ice thickness inversion and dynamical modeling

### 2. Glacier-Specific Calibration

- Individual glacier calibration using geodetic mass balance data (Hugonnet et al., 2021)
- Incorporation of frontal ablation data, including volume changes below sea level (Kochtitzky et al., 2022)
- Improved temperature sensitivity calibration

### 3. Freshwater Runoff Analysis

- Comprehensive freshwater runoff calculation including:
  - On-glacier melt (ice, snow)
  - On-glacier liquid precipitation
  - Off-glacier snowmelt and liquid precipitation in deglaciated areas
- Peak water detection with 11-year rolling mean smoothing

### 4. Glacier Classification Support

- Support for connectivity levels (CL0, CL1, CL2) based on Rastner et al. (2012)
- Enhanced subdivision for large ice caps (e.g., Flade Isblink Ice Cap)
- Regional classification (North-East, Central-East, South-East, South-West, Central-West, North-West, North)

### 5. Climate Data Integration

- Support for ERA5 climate data for historical simulations
- CMIP6 data integration for future projections across SSP scenarios (SSP126, SSP245, SSP370, SSP585)
- Implementation of delta method for bias correction

## Data Requirements

- RGI 6.0 glacier outlines
- ArcticDEM and GIMP DEM topographic data
- ERA5 climate data (historical)
- CMIP6 GCM outputs (future projections)
- Geodetic mass balance observations (Hugonnet et al., 2021)
- Frontal ablation estimates (Kochtitzky et al., 2022)

## Usage Notes

This branch is specifically designed for Greenland peripheral glacier studies focusing on freshwater contributions to the ocean and sea level rise. The enhancements allow for detailed projections of glacier area, volume, mass loss, sea level rise, solid ice discharge, and freshwater runoff from 2020 to 2100.
