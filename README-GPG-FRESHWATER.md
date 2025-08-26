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

## References

- Maussion, F., Butenko, A., Champollion, N., Dusch, M., Eis, J., Fourteau, K., Gregor, P., Jarosch, A. H., Landmann, J., Oesterle, F., Recinos, B., Rothenpieler, T., Vlug, A., Wild, C. T., and Marzeion, B. (2019). The Open Global Glacier Model (OGGM) v1.1. Geoscientific Model Development, 12(3), 909–931. https://doi.org/10.5194/gmd-12-909-2019
- Malles, J.-H., Maussion, F., Jarosch, A. H., Kochtitzky, W., Hock, R., and Marzeion, B. (2023). Improved model calibration and representation of glacier and ice cap dynamics in the Open Global Glacier Model. Journal of Advances in Modeling Earth Systems. https://doi.org/10.1029/2022MS003292
- Hugonnet, R., McNabb, R., Berthier, E., Menounos, B., Nuth, C., Girod, L., Farinotti, D., Huss, M., Dussaillant, I., Brun, F., and Kääb, A. (2021). Accelerated global glacier mass loss in the early twenty-first century. Nature, 592(7856), 726–731. https://doi.org/10.1038/s41586-021-03436-z
- Kochtitzky, W., Copland, L., Huss, M., Hugonnet, R., Jiskoot, H., Wadham, J. L., Bjørk, A. A., van den Broeke, M. R., Fujita, K., Jakob, L., King, M. D., Lai, C.-Y., Lhermitte, S., Maussion, F., Schwikowski, M., Smith, B. E., and Wagnon, P. (2022). Seven decades of uninterrupted advance of Good Friday Glacier, Axel Heiberg Island, Arctic Canada. Journal of Glaciology, 68(271), 903–918. https://doi.org/10.1017/jog.2022.25
- Oerlemans, J., and Nick, F. M. (2005). A minimal model of a tidewater glacier. Annals of Glaciology, 42, 1–6. https://doi.org/10.3189/172756405781813023
- Rastner, P., Bolch, T., Mölg, N., Machguth, H., Le Bris, R., and Paul, F. (2012). The first complete inventory of the local glaciers and ice caps on Greenland. The Cryosphere, 6(6), 1483–1495. https://doi.org/10.5194/tc-6-1483-2012
- Hersbach, H., Bell, B., Berrisford, P., Hirahara, S., Horányi, A., Muñoz-Sabater, J., Nicolas, J., Peubey, C., Radu, R., Schepers, D., Simmons, A., Soci, C., Abdalla, S., Abellan, X., Balsamo, G., Bechtold, P., Biavati, G., Bidlot, J., Bonavita, M., ... Thépaut, J.-N. (2020). The ERA5 global reanalysis. Quarterly Journal of the Royal Meteorological Society, 146(730), 1999–2049. https://doi.org/10.1002/qj.3803
