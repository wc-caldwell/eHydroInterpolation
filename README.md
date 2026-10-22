# eHydro Interpolation

**Author:** Clay Caldwell  
**Contact:** wccaldwe@syr.edu  
**Organization:** Syracuse University / U.S. Army ERDC-CHL  
**Website:** [LinkedIn](https://www.linkedin.com/in/clay-caldwell-9530011a3/)

## Project Overview
* The USACE defaults to Triangulated Irregular Networks (TIN) for gap-filling of echosounder point clouds to generate bathymetric surfaces within the eHydro national database. TINs have been shown to have reduced accuracy when soundings are sparse, as with singlebeam echosounders.
* Kriging methods utilize the covariance between soundings for more accurate gap-filling.
* 30 sets of multibeam surveys have been manually resampled to match the structure of singlebeam surveys at various trackline and along-track sounding spacings to quantify the impact of survey structure on TIN and Kriging interpolation accuracy.
* RMSE, MAE, and 95th percentile of |Error| are used to report error and compare the two interpolation methods.
* Statistical tests are used to identify statistically significant difference between the two methods.

## Data Sources

Dataset contains soundings from multibeam exhosounders (MBES) collected from 30 different navigational channels throughout the United States. Each have been manually resampled to resemble the structure of singlebeam echosounder (SBES) surveys (i.e., spaced tracklines with soundings along them). Each of the 30 synthetic SBES datasets has also been sampled at different trackline spacings (e.g., 25, 50, 100, 200, 300, 400 and 500 ft) and different sounding spacings along the tracklines (e.g., 4, 8, 12, 16, 20, 50, and 100 ft). 

### Published Data Sources

| Name | Source | Description | Access Method | Data URL | Metadata URL | Details | Data Citation |
|------|--------|-------------|---------------|----------|--------------|---------|---------------|
| eHydro | U.S. Army Corps of Engineers | An enterprise geospatial database used to process and disseminate hydrographic survey data for coastal and inland navigation channels. This data is used to inform navigational charts for waterways across the contiguous U.S., Alaska, Hawaii, and American territories. | ArcGIS REST API or Online | [ArcGIS Web Dashboard](https://www.arcgis.com/apps/dashboards/4b8f2ba307684cf597617bf1b6d2f85d) | [ArcGIS REST Database](https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/eHydro_Survey_Data/FeatureServer/0) | Data typically provided in state plane coordinates (US Survey Foot) horizontal projection systems. Vertical projection system typically referenced to local mean lowest low water (MLLW). Point clouds and vectorized survey boundaries provided within an ESRI geodatabase. | USACE. (n.d.). eHydro survey data [Dataset]. ArcGIS REST Services Directory. Retrieved from https://services7.arcgis.com/n1YM8pTrFmm7L4hs/ArcGIS/rest/services/eHydro_Survey_Data/FeatureServer/ |

## License
GPL-3.0

## Notes:
Under development, with USACE ERDC/CHL CHETN to follow.