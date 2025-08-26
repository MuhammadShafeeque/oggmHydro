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

## Workflow Overview

![GPG-Freshwater Workflow](https://raw.githubusercontent.com/MuhammadShafeeque/oggmHydro/oggm-GPG-freshwater/docs/_static/gpg_freshwater_workflow.png)

*Figure: Workflow diagram showing the integration of model setup, climate data preprocessing, calibration, and future projections analysis. The workflow highlights the key components of the GPG-Freshwater methodology.*

## Key Equations

### Mass Balance Calculation

The monthly surface mass balance at a grid point $i$ with elevation $z$ is calculated as:

$$m_i(z) = f_P P_i^{solid}(z) - \mu \max(T_i^m(z), 0)$$

Where:
- $m_i(z)$ is monthly surface mass balance for grid point $i$ [mm w.e.]
- $f_P$ is precipitation factor (default: 1.6)
- $P_i^{solid}(z)$ is solid precipitation [mm w.e.]
- $\mu$ is air temperature sensitivity [mm w.e. K$^{-1}$]
- $T_i^m(z)$ is air temperature above the threshold for ice melt [K]

### Frontal Ablation for Marine-Terminating Glaciers

For marine-terminating glaciers, frontal ablation is determined using:

$$Q_f = k \times d \times h \times w$$

Where:
- $Q_f$ is the frontal ablation flux
- $k$ is the water-depth sensitivity parameter [yr$^{-1}$]
- $d$ is water depth at the terminus [m]
- $h$ is ice thickness at the terminus [m]
- $w$ is glacier width at the terminus [m]

### Temperature Sensitivity Calibration

The temperature sensitivity ($\mu$) is calibrated using:

$$\mu = \frac{f_P P_{solid} - (\Delta M_{awl} + C + f_{bwl} \Delta M_f) / A_{RGI}}{T_m}$$

Where:
- $\Delta M_{awl}$ is observed annual volume change above sea level [Gt/yr]
- $C$ is observed annual frontal ablation rate [Gt/yr]
- $\Delta M_f$ is observed annual volume retreat due to area changes in terminus region [Gt/yr]
- $f_{bwl}$ is fraction of $\Delta M_f$ occurring below waterline
- $A_{RGI}$ is glacier surface area [km$^2$]
- $T_m$ is annually accumulated air temperature above melt threshold [K]

### Total Freshwater Runoff

The total annual freshwater runoff from a glacier is calculated as:

$$TR = \sum GR_{i,s,r} + SR + RR$$

Where:
- $TR$ is total liquid freshwater runoff
- $GR_{i,s,r}$ denotes the sum of runoff from glacier ice ($GR_i$), snow ($GR_s$), and rain ($GR_r$)
- $SR$ is snowmelt off-glacier (in deglaciated areas)
- $RR$ is rain runoff off-glacier (in deglaciated areas)

## Usage Notes

This branch is specifically designed for Greenland peripheral glacier studies focusing on freshwater contributions to the ocean and sea level rise. The enhancements allow for detailed projections of glacier area, volume, mass loss, sea level rise, solid ice discharge, and freshwater runoff from 2020 to 2100.
