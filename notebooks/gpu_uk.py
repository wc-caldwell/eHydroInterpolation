import os
import sys
sys.path.insert(0, os.path.abspath('../src'))
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from shapely import contains_xy
import json
import numpy as np
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
import pygstat
from pygstat.utils.backend import CUPY_AVAILABLE, cupy_gpu_usable
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import processing_help
print(f"pygstat version : {pygstat.__version__}")
print(f"NumPy           : {np.__version__}")
print(f"pandas          : {pd.__version__}")
USE_GPU = cupy_gpu_usable()
if USE_GPU:
    import cupy as cp
    print(f"CuPy            : {cp.__version__}")
    try:
        name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        print(f"GPU             : {name}")
    except Exception:
        print("GPU             : available")
    print("Backend         : GPU (CuPy) — residual OK covariance + solve on GPU")
else:
    if CUPY_AVAILABLE:
        import cupy as cp
        print(f"CuPy            : {cp.__version__} (installed, but cannot run a kernel on this GPU/CUDA setup)")
    else:
        print("CuPy            : not installed  (pip install pygstat[gpu]  or  cupy-cuda12x)")
    print("Backend         : CPU (NumPy)")
print(f"use_gpu         : {USE_GPU}")

def grid_transform(x_grid, y_grid):
    """Affine transform for a regular grid whose x/y vectors are cell centers."""
    dx = float(np.mean(np.diff(x_grid)))
    dy = float(np.mean(np.abs(np.diff(y_grid))))
    return from_origin(x_grid.min() - dx / 2, y_grid.max() + dy / 2, dx, dy)

def write_cog(path, arr, x_grid, y_grid, crs, tags=None, nodata=-9999.0,
              dtype="float32", band_description=None):
    arr = np.asarray(arr)
    if y_grid[1] > y_grid[0]:          # rows run south→north; GeoTIFF expects north→south
        arr = np.flipud(arr)
    data = np.where(np.isfinite(arr), arr, nodata).astype(dtype)

    profile = dict(driver="GTiff", height=data.shape[0], width=data.shape[1],
                   count=1, dtype=dtype, crs=crs,
                   transform=grid_transform(x_grid, y_grid), nodata=nodata)

    with MemoryFile() as mem:
        with mem.open(**profile) as tmp:
            tmp.write(data, 1)
            if band_description:
                tmp.set_band_description(1, band_description)
            if tags:
                tmp.update_tags(**{k: str(v) for k, v in tags.items()})
        with mem.open() as tmp:
            rio_copy(tmp, str(path), driver="COG",
                     BLOCKSIZE=512, COMPRESS="DEFLATE", LEVEL=6,
                     PREDICTOR="YES",              # floating-point predictor for float data
                     OVERVIEW_RESAMPLING="AVERAGE",
                     BIGTIFF="IF_SAFER")

def predict_batched(model, pts, batch_size=5000):
    preds, variances = [], []
    pool = cp.get_default_memory_pool()
    for i in range(0, len(pts), batch_size):
        p, v = model.predict(pts[i:i + batch_size], return_variance=True)
        preds.append(cp.asnumpy(p))
        variances.append(cp.asnumpy(v))
        pool.free_all_blocks()  # release cached GPU memory between batches
    return np.concatenate(preds), np.concatenate(variances)

def find_best_variogram(
    coords,
    values,
    estimator="dowd",
    n_lags=15,
    criterion="AIC",
    verbose=True,
    use_gpu=USE_GPU,
):
    """
    Fit all standard variogram models and return the best one.

    Parameters
    ----------
    coords     : array-like, shape (n, 2)   — spatial coordinates
    values     : array-like, shape (n,)     — observed values
    estimator  : str — 'matheron', 'cressie', or 'dowd'
    n_lags     : int — number of lag classes
    criterion  : str — 'AIC', 'BIC', 'RSS', or 'R²'
    verbose    : bool — print ranking table
    use_gpu    : bool or 'auto' — empirical variogram backend (see §4.1)

    Returns
    -------
    best_name  : str              — name of the winning model
    best_v     : Variogram        — fitted Variogram object
    df_results : pd.DataFrame     — full ranking table
    """
    models = ["spherical", "exponential", "gaussian", "matern", "stable", "cubic"]
    n_params = {"spherical": 3, "exponential": 3, "gaussian": 3,
                "matern": 4, "stable": 4, "cubic": 3}

    fitted = {}
    records = []

    for model_name in models:
        try:
            v = pygstat.Variogram(
                coords=np.asarray(coords, dtype=float),
                values=np.asarray(values, dtype=float),
                model=model_name,
                estimator=estimator,
                n_lags=n_lags,
                use_gpu=use_gpu,
            )
            v.fit()

            obs  = v.experimental
            mask = ~np.isnan(obs)
            obs_v = obs[mask]
            pred  = v(v.lags)[mask]
            K = mask.sum()
            p = n_params[model_name]

            rss  = float(np.sum((pred - obs_v) ** 2))
            rmse = float(np.sqrt(rss / K)) if K > 0 else np.nan
            tss  = float(np.sum((obs_v - obs_v.mean()) ** 2))
            r2   = 1.0 - rss / tss if tss > 0 else np.nan
            aic  = K * np.log(rss / K) + 2 * p if rss > 0 else -np.inf
            bic  = K * np.log(rss / K) + p * np.log(K) if rss > 0 else -np.inf

            fitted[model_name] = v
            records.append({
                "Model": model_name.capitalize(),
                "Nugget": round(v.fitted_params[0], 5),
                "Sill":   round(v.fitted_params[1], 5),
                "Range":  round(v.fitted_params[2], 5),
                "RSS":    round(rss, 6),
                "RMSE":   round(rmse, 6),
                "R²":     round(r2, 5),
                "AIC":    round(aic, 4),
                "BIC":    round(bic, 4),
            })
        except Exception as exc:
            if verbose:
                print(f"  [SKIP] {model_name}: {exc}")

    df_results = pd.DataFrame(records).set_index("Model")

    ascending = criterion in ("AIC", "BIC", "RSS", "RMSE")
    df_results = df_results.sort_values(criterion, ascending=ascending)

    best_name_cap = df_results.index[0]
    best_name_key = best_name_cap.lower()
    best_v = fitted[best_name_key]

    if verbose:
        print(f"\nVariogram model ranking by {criterion}:")
        print(df_results[["Nugget", "Sill", "Range", "RSS", "R²", "AIC"]].to_string())
        print(f"\n>>> BEST MODEL: {best_name_cap} (by {criterion}) <<<")

    return best_name_cap, best_v, df_results, fitted

def fit_trend(coords, values, degree):
    """OLS polynomial drift. degree=0 is a constant mean."""
    
    n = len(values)
    if degree == 0:
        F = np.ones((n, 1))
        beta = np.array([float(np.mean(values))])
        fitted = np.full(n, beta[0])
        names = ["1"]
    else:
        poly = PolynomialFeatures(degree=degree, include_bias=True)
        F = poly.fit_transform(coords)
        names = list(poly.get_feature_names_out(["x", "y"]))
        lm = LinearRegression(fit_intercept=False)
        lm.fit(F, values)
        beta = lm.coef_.astype(float)
        fitted = lm.predict(F)
    resid = values - fitted
    p = F.shape[1]
    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((values - values.mean()) ** 2))
    r2 = 1.0 - rss / tss if tss > 0 else np.nan
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - p) if n > p else np.nan
    aic = n * np.log(rss / n) + 2 * p
    bic = n * np.log(rss / n) + p * np.log(n)
    rmse = float(np.sqrt(rss / n))
    return {
        "degree": degree, "p": p, "names": names, "beta": beta,
        "fitted": fitted, "resid": resid, "F": F,
        "R2": r2, "adj_R2": adj_r2, "AIC": aic, "BIC": bic, "RMSE": rmse,
    }

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

sbes_x = sbes_gdf.X_ft
sbes_y = sbes_gdf.Y_ft

if sbes_gdf.Elevation_ft.mean() > 0.:
    sbes_z = sbes_gdf.Elevation_ft * -1
else:
    sbes_z = sbes_gdf.Elevation_ft

mbes_x = mbes_gdf.X_ft
mbes_y = mbes_gdf.Y_ft
if mbes_gdf.Elevation_ft.mean() > 0.:
    mbes_z = mbes_gdf.Elevation_ft * -1
else:
    mbes_z = mbes_gdf.Elevation_ft

grid_res = 10. # 10 ft spatial resolution so the outputs match the resolution of CSAT inputs
num_pts = 10000

point_spacings = [2, 4, 6, 8, 10, 25, 50]
trackline_spacings = [1, 2, 4, 8, 12, 16, 20]

# point_spacings = [2, 10, 50]
# trackline_spacings = [1, 4, 12]

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
        pt_spacing = pspace * 2

        along_length_temp = np.array(sbes_gdf_this_tspace['Length_along_transect_ft'])
        along_length_temp = np.round(along_length_temp)  
        use_row = [(np.mod(thislength, pspace) == 0) for thislength in along_length_temp]
        singlebeam_this_tspace_pspace = sbes_gdf_this_tspace[use_row].reset_index(drop=True)  
        del (along_length_temp, use_row)

        if len(singlebeam_this_tspace_pspace.Elevation_ft) > num_pts:
            x, y, z, _, _, _ = processing_help.decimate_survey_points(singlebeam_this_tspace_pspace, tgt_num_pts=num_pts, reduction_method='median')
            krige_coords=np.column_stack([x, y])

            if z.mean() > -1:
                z = z * -1
        else:
            x = singlebeam_this_tspace_pspace.X_ft
            y = singlebeam_this_tspace_pspace.Y_ft
            krige_coords=np.column_stack([x, y])
            if singlebeam_this_tspace_pspace.Elevation_ft.mean() > -1:
                z = singlebeam_this_tspace_pspace.Elevation_ft * -1
            else:
                z = singlebeam_this_tspace_pspace.Elevation_ft

        print(f'Number of points = {len(z)}')

        x_min, x_max = float(sbes_x.min()), float(sbes_x.max())
        y_min, y_max = float(sbes_y.min()), float(sbes_y.max())

        # Build grid with consistent bounds (float64)
        nx = int(np.ceil((x_max - x_min) / grid_res)) + 1
        ny = int(np.ceil((y_max - y_min) / grid_res)) + 1
        x_grid = np.linspace(x_min, x_min + (nx - 1) * grid_res, nx, dtype=np.float64)
        y_grid = np.linspace(y_min, y_min + (ny - 1) * grid_res, ny, dtype=np.float64)

        XX, YY = np.meshgrid(x_grid, y_grid)

        # Mask of grid points inside the boundary
        polygon = boundary.geometry.iloc[0]  
        inside = contains_xy(polygon, XX, YY)

        # Only predict at the inside points
        pts = np.column_stack([XX[inside], YY[inside]])

        degrees = [0, 1, 2, 3]
        trends = {d: fit_trend(krige_coords, z, d) for d in degrees}

        trend_table = pd.DataFrame([
            {"degree": t["degree"], "n_coef": t["p"],
            "R2": round(t["R2"], 4), "adj_R2": round(t["adj_R2"], 4),
            "RMSE": round(t["RMSE"], 4), "AIC": round(t["AIC"], 3), "BIC": round(t["BIC"], 3)}
            for t in trends.values()
        ]).set_index("degree")

        best_degree = int(trend_table["AIC"].idxmin())

        t_best = trends[best_degree]
        if best_degree == 0:
            trend_vals = np.full(len(pts), t_best["beta"][0])
        else:
            poly_best = PolynomialFeatures(degree=best_degree, include_bias=True).fit(pts)
            trend_vals = poly_best.transform(pts) @ t_best["beta"]

        trend_grid = np.full(XX.shape, np.nan)
        trend_grid[inside] = trend_vals

        residuals = t_best['resid']

        best_name, vg_best, df_rank, fitted_models = find_best_variogram(
            coords=krige_coords,
            values=z,
            estimator="dowd",
            n_lags=30,
            criterion="AIC",
            verbose=False,
            use_gpu=USE_GPU,
        )

        uk = pygstat.UniversalKriging(vg_best, degree=best_degree, use_gpu=USE_GPU).fit(krige_coords, z)

        pred, var = predict_batched(uk, pts)

        pred_grid = np.full(XX.shape, np.nan)
        se_grid = np.full(XX.shape, np.nan)
        pred_grid[inside] = pred
        se_grid[inside] = np.sqrt(np.maximum(var, 0))

        crs = boundary.crs
        tags = {
            "fit_variogram": str(vg_best.model),
            "variogram_aic": vg_best.model,
            "variogram_model_ranking": json.dumps([{"Model": name, "AIC": float(row["AIC"])} for name, row in df_rank.iterrows()]),
            "trend_degree": best_degree,
            "trend_beta": json.dumps(np.asarray(t_best["beta"]).tolist()),
            "variance_note": "residual kriging variance; excludes trend uncertainty",
        }

        write_cog(OUT_DIR / f"isoUK_sbes_{track_spacing}_{pt_spacing}_pred.tif", pred_grid, x_grid, y_grid, crs,
                  tags=tags, band_description="Elevation (ft)")
        write_cog(OUT_DIR / f"isoUK_sbes_{track_spacing}_{pt_spacing}_se.tif", np.sqrt(np.maximum(se_grid, 0)), x_grid, y_grid, crs,
                  tags=tags, band_description="Kriging standard error (ft)")
        write_cog(OUT_DIR / f"isoUK_sbes_{track_spacing}_{pt_spacing}_trend.tif", trend_grid, x_grid, y_grid, crs,
                  tags={"trend_degree": best_degree}, band_description="Polynomial trend (ft)")