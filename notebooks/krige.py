import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt

from interpolators.chunked_kriging import chunkKrige

import numpy as np
import gstools as gs
import processing_help

import warnings
# remove future warning of spline mindist since we aren't even providing it anyways
warnings.filterwarnings('ignore', message='.*mindist parameter.*', category=FutureWarning)     

WORK_DIR = Path('/mnt/teamgroup/AutoDBM/Public')
MBES = WORK_DIR / 'Multibeam_Validation_Points_For_Distribution'
SBES = WORK_DIR / 'Synthetic_Singlebeam_Surveys_For_Distribution'
CRS = pd.read_csv(WORK_DIR / 'survey_CRS_list.csv')

mbes = list((WORK_DIR / 'Multibeam_Validation_Points_For_Distribution').glob('*.csv'))
sbes = list((WORK_DIR / 'Synthetic_Singlebeam_Surveys_For_Distribution').glob('*.csv'))
crs_data = WORK_DIR / 'survey_CRS_list.csv'

geojson = Path(os.getcwd()).parent / 'shapefiles' / 'National_Channel_Framework_-6742069550782745925.geojson'
ncf = gpd.read_file(geojson)

# Build a boundary catalog from SurveyJob layers, then match each survey to the best boundary by SurveyId
data_pairs = {}

def _norm(s):
    return "".join(ch for ch in str(s).upper() if ch.isalnum())

def _tokens(s):
    return {t for t in str(s).upper().replace("-", "_").split("_") if t}

# 1) Read all boundary candidates once
boundary_catalog = []
for gdb_file in WORK_DIR.glob("*/*.gdb"):
    try:
        sj = gpd.read_file(gdb_file, layer="SurveyJob")
    except Exception:
        continue

    if "SurveyId" not in sj.columns or sj.empty:
        continue

    # keep one record per SurveyId (first occurrence)
    for sid, grp in sj.groupby("SurveyId", dropna=True):
        boundary_catalog.append({
            "survey_id": str(sid),
            "survey_norm": _norm(sid),
            "survey_tokens": _tokens(sid),
            "gdb_file": gdb_file,
            "boundary": grp.copy()
        })

# 2) Match each survey from CRS table to best boundary
for _, items in pd.read_csv(crs_data).iterrows():
    survey = items["Survey"]

    sbes_match = [p for p in sbes if survey in str(p)]
    mbes_match = [p for p in mbes if survey in str(p)]
    if not sbes_match or not mbes_match:
        print(f"Missing SBES/MBES file for: {survey}")
        continue

    s_norm = _norm(survey)
    s_tokens = _tokens(survey)

    # exact normalized SurveyId match first
    exact = [c for c in boundary_catalog if c["survey_norm"] == s_norm]

    if exact:
        best = exact[0]
    else:
        # fallback: token overlap score
        scored = []
        for c in boundary_catalog:
            inter = len(s_tokens & c["survey_tokens"])
            denom = max(1, len(s_tokens))
            score = inter / denom
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1] if scored and scored[0][0] > 0 else None

    if best is None:
        print(f"No boundary found for: {survey}")
        continue

    boundary = best["boundary"]
    data_pairs[survey] = [
        sbes_match[0],
        mbes_match[0],
        boundary,
    ]

    # print(f"{survey} -> {best['survey_id']} ({best['gdb_file'].name})")

# establishing params for the surface generation step(s)
vario_models = {
    "Gaussian": gs.Gaussian,
    "Exponential": gs.Exponential,
    "Matern": gs.Matern,
    "Stable": gs.Stable,
    "Rational": gs.Rational,
    "Circular": gs.Circular,
    "Spherical": gs.Spherical,
    "SuperSpherical": gs.SuperSpherical,
    "JBessel": gs.JBessel,
}

# least square estimation scheme for automatic variogram fitting
# 'cressie' or 'matheron'. Cressie tends to be more robust to outliers
estimator = 'cressie'

# specify the horizontal spatial resolution of the resulting bathymetric surface. This is in ft
# 10ft is being used here to match the resolution of the CSAT inputs
grid_res = 10.0

# True if you want the outputs of the interpolation methods will be showcased in this notebook using Matplotlib
plots = False  

# Set number of processors to use based on the number of CPUs your computer has
# NOTE: this may exceed your system memory depending on how much RAM you have
# My workstation has 24 CPUs and 96 GB of RAM; I will use between 8 and 12 processors
processors = 20

num_pts = 10000

survey = list(data_pairs.keys())[int(sys.argv[1])]
print(survey)

test_pair = data_pairs[survey]
sbes_data = pd.read_csv(test_pair[0])
mbes_data = pd.read_csv(test_pair[1])
boundary = test_pair[2]
data_crs = boundary.crs

sbes_gdf = gpd.GeoDataFrame(
        sbes_data.copy(),
        geometry=gpd.points_from_xy(sbes_data["X_ft"], sbes_data["Y_ft"]),
        crs=data_crs
    )

mbes_gdf = gpd.GeoDataFrame(
        mbes_data.copy(),
        geometry=gpd.points_from_xy(mbes_data["X_ft"], mbes_data["Y_ft"]),
        crs=data_crs
    )

OUT_DIR = WORK_DIR / survey
os.makedirs(OUT_DIR, exist_ok=True)

point_spacings = [2, 4, 6, 8, 10, 25, 50]
trackline_spacings = [1, 2, 4, 8, 12, 16, 20]

num_transects = max(sbes_gdf['Transect_ID']) - min(sbes_gdf['Transect_ID'])  # how many transects are available in the singlebeam survey?

for itspace, tspace in enumerate(trackline_spacings):  # loop through the different transect spacings
    track_spacing = tspace * 25

    transects_to_use = list(np.arange(min(sbes_gdf['Transect_ID']), min(sbes_gdf['Transect_ID']) + num_transects + 1, tspace))  # transect indices that we will use to achieve the desired spacing

    # Downsample the synthetic singlebeam data to only contain the desired transect spacing.
    transect_id_temp = list(sbes_gdf['Transect_ID'])  # transect ID for each row of the singlebeam dataframe
    use_row = [(id in transects_to_use) for id in transect_id_temp]
    sbes_gdf_this_tspace = sbes_gdf[use_row].reset_index(drop=True)
    del (use_row, transect_id_temp)

    # Remove any points used to create this set of transects from the multibeam validation data.
    mbes_gdf_this_tspace = mbes_gdf[~mbes_gdf['Used_to_create_transect'].isin(transects_to_use)].reset_index(drop=True)

    for ipspace, pspace in enumerate(point_spacings):  # loop through the different point spacings
        pt_spacing = pspace

        along_length_temp = np.array(sbes_gdf_this_tspace['Length_along_transect_ft'])
        along_length_temp = np.round(along_length_temp)  # some of them are decimals??? I have no clue why Arc did that.
        use_row = [(np.mod(thislength, pspace) == 0) for thislength in along_length_temp]
        singlebeam_this_tspace_pspace = sbes_gdf_this_tspace[use_row].reset_index(drop=True)  # this is the final set of points that we will use to create the TIN.
        del (along_length_temp, use_row)

        if len(singlebeam_this_tspace_pspace.Elevation_ft) > num_pts:
            x, y, z, _, _, _ = processing_help.decimate_survey_points(singlebeam_this_tspace_pspace, tgt_num_pts=num_pts, reduction_method='median')

            if z.mean() > -1:
                z = z * -1
        else:
            x = singlebeam_this_tspace_pspace.X_ft
            y = singlebeam_this_tspace_pspace.Y_ft
            if singlebeam_this_tspace_pspace.Elevation_ft.mean() > -1:
                z = singlebeam_this_tspace_pspace.Elevation_ft * -1
            else:
                z = singlebeam_this_tspace_pspace.Elevation_ft

        
        isoKrige_AIC = chunkKrige(
            tgt_survey = survey,
            survey_path = OUT_DIR, 
            survey_boundary=boundary,
            vario_models=vario_models,
            x=x, y=y, z=z, 
            echosounder=f"SBES_{track_spacing}_{pt_spacing}",
            trend_polynomial=2,                                           
            vario_estimator = estimator,            
            grid_res=grid_res,
            detrend=False,
            plot_outputs=plots,
            n_jobs=processors
        )

        krige_path_AIC, _ = isoKrige_AIC.generate()