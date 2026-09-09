#!/usr/bin/env python
"""
Isotropic Kriging with AIC-based variogram model selection.
Uses Akaike Information Criterion to select the best variogram model.
"""

import warnings
import numpy as np
from pathlib import Path
import verde as vd
import geopandas as gpd
import matplotlib.pyplot as plt
import gstools as gs
from pykrige.ok import OrdinaryKriging
from shapely.vectorized import contains
import rasterio
from rasterio.transform import from_origin
import json
import gc
import multiprocessing
from joblib import Parallel, delayed
from scipy.spatial import Delaunay

class isoAutoKrige:
    """Generate Kriging bathymetric surfaces with AIC-based model selection."""
    
    __slots__ = [
        'surveyId',
        'survey_path',
        'survey_boundary',
        'echosounder', 
        'grid_res',
        'vario_models', 
        'vario_weights',
        'detrend', 
        'plot_outputs', 
        'coordinates',
        'bathymetry', 
        'max_lag', 
        'fit_model', 
        'fit_aic', 
        'chunked_kriging',
        'ranking', 
        'trendSurface', 
        'residuals', 
        'krige_model', 
        'x_grid',
        'y_grid', 
        'krige_pred', 
        'krige_var', 
        'transform', 
        'cv_metadata',
        'depth_tif_path', 
        'uncertainty_tif_path', 
        'meta', 
        'n_jobs', 
        'spline_damping',
        '_crs', '_fold_spacing', '_ill_conditioned', '_ill_conditioned_count', '_total_chunks',
        '_empirical_bins', '_empirical_gamma', '_vario_estimator'
    ]
    
    def __init__(self, surveyId, survey_path, survey_boundary, vario_models, x, y, z, echosounder,
                 spline_damping = 1e-1, vario_estimator = 'cressie', chunked_kriging = True,
                 grid_res=10.0, detrend=False, plot_outputs=False, n_jobs=-1):
        
        self.surveyId = surveyId
        self.survey_path = survey_path
        self.survey_boundary = survey_boundary
        self._crs = self.survey_boundary.crs  # Cache CRS
        self.echosounder = echosounder
        self.grid_res = grid_res
        self.vario_models = vario_models
        self.detrend = detrend
        self.plot_outputs = plot_outputs
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self._ill_conditioned = False
        self._ill_conditioned_count = 0
        self._total_chunks = 0
        self.spline_damping = spline_damping
        self._vario_estimator = vario_estimator
        self.chunked_kriging = chunked_kriging
        
        # Store as float32 to reduce memory by 50%
        self.coordinates = (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        self.bathymetry = np.asarray(z, dtype=np.float32)

        # Pre-compute range values
        x_range = float(self.coordinates[0].max() - self.coordinates[0].min())
        y_range = float(self.coordinates[1].max() - self.coordinates[1].min())
        self.max_lag = np.sqrt(x_range**2 + y_range**2) / 2

        self.fit_model = None
        self.fit_aic = None  # AIC score instead of R²
        self.ranking = None
        self.trendSurface = None
        self.residuals = None
        self.krige_model = None
        self.x_grid = None
        self.y_grid = None
        self.krige_pred = None
        self.krige_var = None
        self.transform = None
        self.cv_metadata = None
        self.depth_tif_path = None
        self.uncertainty_tif_path = None
        self._empirical_bins = None
        self._empirical_gamma = None

    def _spline_detrend(self):
        """Fit spline for detrending using Verde.Spline."""
        
        self.trendSurface = vd.Spline(damping = self.spline_damping)
        self.trendSurface.fit(self.coordinates, self.bathymetry)
        self.residuals = self.bathymetry - self.trendSurface.predict(self.coordinates)

    def _calculate_aic(self, model, bin_center, gamma):
        """
        Calculate AIC for a fitted variogram model.
        
        Parameters
        ----------
        model : gstools.CovModel
            Fitted variogram model
        bin_center : array
            Bin centers from empirical variogram
        gamma : array
            Empirical variogram values
        
        Returns
        -------
        float
            AIC score (lower is better)
        """
        # Get predicted values from the fitted model
        gamma_pred = model.variogram(bin_center)
        
        # Calculate residuals
        residuals = gamma - gamma_pred
        
        # Number of data points
        n = len(bin_center)

        wrss = np.sum(self.vario_weights * residuals**2)
        if wrss < 1e-10:
            wrss = 1e-10

        k = 3
        model_name = model.__class__.__name__.lower()
        if 'stable' in model_name:
            k = 4
        elif 'matern' in model_name:
            k = 4

        aic = n * np.log(wrss / n) + 2 * k
        if n / k < 40:
            aic = aic + (2 * k * (k + 1)) / (n - k - 1)

        return aic

    def _fit_variograms(self, max_evals=500000):
        """Fit variogram models and select best based on AIC."""
        bins = gs.variogram.standard_bins(
            pos=self.coordinates, dim=2, latlon=False,
            mesh_type="unstructured", max_dist=self.max_lag
        )
        
        data = self.residuals if self.detrend else self.bathymetry
        bin_center, gamma, counts = gs.vario_estimate(
            self.coordinates, data, bins, estimator=self._vario_estimator, latlon=False,     # cressie or matheron
            return_counts = True
        )

        self.vario_weights = counts / (bin_center**2 + 1e-6)
        
        # Store for plotting
        self._empirical_bins = bin_center
        self._empirical_gamma = gamma
        
        vario_scores = {}
        best_aic, best_model = np.inf, None
        
        for name, model_class in self.vario_models.items():
            model = model_class(dim=2)
            
            try:
                # Fit the model (we don't need R² anymore)
                model.fit_variogram(bin_center, gamma, max_eval=max_evals, weights=self.vario_weights)
                
                # Calculate AIC for this model
                aic = self._calculate_aic(model, bin_center, gamma)
                vario_scores[name] = aic
                
                # Lower AIC is better
                if aic < best_aic:
                    best_aic, best_model = aic, model
                    
            except Exception as e:
                print(f"Warning: Failed to fit {name} model: {e}")
                vario_scores[name] = np.inf
        
        self.fit_model = best_model
        self.fit_aic = best_aic
        # Rank by AIC (ascending - lower is better)
        self.ranking = sorted(vario_scores.items(), key=lambda x: x[1])

    def _process_single_chunk(self, iy, ix, chunk_size, overlap, nx, ny, n_neighbors, full_mask, trend_full=None):
        """Process a single chunk of the kriging grid (for parallel execution)."""
        iy_start = max(0, iy - overlap)
        ix_start = max(0, ix - overlap)
        iy_end = min(iy + chunk_size + overlap, ny)
        ix_end = min(ix + chunk_size + overlap, nx)

        x_chunk = self.x_grid[ix_start:ix_end]
        y_chunk = self.y_grid[iy_start:iy_end]

        chunk_mask = full_mask[iy_start:iy_end, ix_start:ix_end]
        
        if not np.any(chunk_mask):
            return None  # Skip empty chunks
        
        # Track ill-conditioning for this chunk
        ill_conditioned = False
        
        # Capture warnings during kriging execution
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            z_pred, ss = self.krige_model.execute(
                style="masked", xpoints=x_chunk, ypoints=y_chunk,
                mask=~chunk_mask, n_closest_points=n_neighbors, backend="loop"
            )

            # Check for ill-conditioning warnings
            for w in caught_warnings:
                msg = str(w.message).lower()
                if any(term in msg for term in 
                       ['singular', 'ill-conditioned', 'condition number', 
                        'poorly conditioned', 'ill conditioned']):
                    ill_conditioned = True
                    break
        
        if self.detrend and trend_full is not None:
            z = z_pred + trend_full[iy_start:iy_end, ix_start:ix_end]
        else:
            z = z_pred

        # Calculate extraction indices
        y_ext_start = iy - iy_start
        y_ext_end = y_ext_start + min(chunk_size, ny - iy)
        x_ext_start = ix - ix_start
        x_ext_end = x_ext_start + min(chunk_size, nx - ix)

        # Extract the core (non-overlapping) region
        # Use np.ma.filled to convert any masked entries to NaN explicitly,
        # rather than relying on a 0-sentinel downstream.
        z_slice = z[y_ext_start:y_ext_end, x_ext_start:x_ext_end]
        ss_slice = ss[y_ext_start:y_ext_end, x_ext_start:x_ext_end]
        if np.ma.is_masked(z_slice):
            z_core = np.ma.filled(z_slice, np.nan).astype(np.float32)
        else:
            z_core = np.asarray(z_slice, dtype=np.float32)
        if np.ma.is_masked(ss_slice):
            ss_core = np.ma.filled(ss_slice, np.nan).astype(np.float32)
        else:
            ss_core = np.asarray(ss_slice, dtype=np.float32)

        # Also return the core region of the polygon mask so the aggregator
        # only writes inside-polygon cells (prevents bleed-out artifacts).
        core_mask = np.asarray(
            chunk_mask[y_ext_start:y_ext_end, x_ext_start:x_ext_end]
        )

        return {
            'iy': iy,
            'ix': ix,
            'y_size': y_ext_end - y_ext_start,
            'x_size': x_ext_end - x_ext_start,
            'z': z_core,
            'ss': ss_core,
            'core_mask': core_mask,
            'ill_conditioned': ill_conditioned
        }

    def _krige_in_chunks(self, chunk_size=500, n_neighbors=100):    #, overlap=25
        """Chunked kriging with overlap for edge effects (parallelized)."""
        data = self.residuals if self.detrend else self.bathymetry

        self.krige_model = OrdinaryKriging(
            self.coordinates[0], self.coordinates[1], data,
            variogram_model=self.fit_model,
            coordinates_type = 'euclidean'
        )
        practical_range = self._practical_range(self.fit_model)
        overlap = int(np.ceil(2.0 * practical_range / self.grid_res))
        # overlap = int(np.ceil(2.0 * self.fit_model.len_scale / self.grid_res))
        # Pre-compute bounds
        x_min, x_max = float(self.coordinates[0].min()), float(self.coordinates[0].max())
        y_min, y_max = float(self.coordinates[1].min()), float(self.coordinates[1].max())

        # Build grid with consistent bounds (float32)
        nx = int(np.ceil((x_max - x_min) / self.grid_res)) + 1
        ny = int(np.ceil((y_max - y_min) / self.grid_res)) + 1
        self.x_grid = np.linspace(x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32)
        self.y_grid = np.linspace(y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32)

        self.transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res, self.grid_res
        )

        polygon = self.survey_boundary.geometry.iloc[0]
        nx, ny = len(self.x_grid), len(self.y_grid)

        z_final = np.full((ny, nx), np.nan, dtype=np.float32)
        ss_final = np.full((ny, nx), np.nan, dtype=np.float32)

        xx_full, yy_full = np.meshgrid(self.x_grid, self.y_grid)
        # xx_full, yy_full = np.meshgrid(self.x_grid, self.y_grid[::-1])
        polygon_mask = contains(polygon, xx_full, yy_full)

        delaunay_hull = Delaunay(np.column_stack(self.coordinates))

        grid_points = np.column_stack((xx_full.ravel(), yy_full.ravel()))
        delaunay_flat_mask = delaunay_hull.find_simplex(grid_points) >= 0
        delaunay_mask = delaunay_flat_mask.reshape(xx_full.shape)

        full_mask = delaunay_mask & polygon_mask

        # Pre-compute trend if detrending
        trend_full = None
        if self.detrend:
            trend_full = self.trendSurface.predict((xx_full, yy_full)).astype(np.float32)

        del xx_full, yy_full

        # Generate list of chunk coordinates
        chunk_jobs = []
        for iy in range(0, ny, chunk_size):
            for ix in range(0, nx, chunk_size):
                chunk_jobs.append((iy, ix))
        
        # Process chunks in parallel
        print(f"Processing {len(chunk_jobs)} chunks using {self.n_jobs} cores...")
        results = Parallel(n_jobs=self.n_jobs, verbose=5, backend='loky')(
            delayed(self._process_single_chunk)(
                iy, ix, chunk_size, overlap, nx, ny, n_neighbors, full_mask, trend_full
            ) for iy, ix in chunk_jobs
        )
        
        # Aggregate results
        ill_conditioned_chunks = 0
        total_chunks = 0
        
        for result in results:
            if result is None:
                continue
            
            total_chunks += 1
            if result['ill_conditioned']:
                ill_conditioned_chunks += 1
            
            iy = result['iy']
            ix = result['ix']
            y_size = result['y_size']
            x_size = result['x_size']
            core_mask = result['core_mask']

            # Only write predictions into cells inside the survey polygon.
            # Cells outside the polygon remain NaN (their initialized value),
            # which prevents the kriging surface from bleeding past the AOI.
            z_target = z_final[iy:iy + y_size, ix:ix + x_size]
            ss_target = ss_final[iy:iy + y_size, ix:ix + x_size]
            np.copyto(z_target, result['z'], where=core_mask)
            np.copyto(ss_target, result['ss'], where=core_mask)
            z_final[iy:iy + y_size, ix:ix + x_size] = z_target
            ss_final[iy:iy + y_size, ix:ix + x_size] = ss_target

        # Set flags for ill-conditioning 
        self._ill_conditioned = ill_conditioned_chunks > 0
        self._ill_conditioned_count = ill_conditioned_chunks
        self._total_chunks = total_chunks

        # Belt-and-suspenders: force NaN anywhere outside the polygon, in
        # case a chunk's local mask differed from the global mask at all.
        z_final[~full_mask] = np.nan
        ss_final[~full_mask] = np.nan

        if self.detrend:
            del trend_full
        del full_mask

        # Assign directly. Do NOT use `np.where(z_final == 0, np.nan, ...)`:
        # legitimate predictions of exactly 0 (common with detrended residuals)
        # would be wrongly nulled.
        self.krige_pred = z_final
        self.krige_var = ss_final

    def _krige(self):
        data = self.residuals if self.detrend else self.bathymetry

        self.krige_model = OrdinaryKriging(
            self.coordinates[0], self.coordinates[1], data,
            variogram_model=self.fit_model,
            coordinates_type = 'euclidean'
        )

        # Pre-compute bounds
        x_min, x_max = float(self.coordinates[0].min()), float(self.coordinates[0].max())
        y_min, y_max = float(self.coordinates[1].min()), float(self.coordinates[1].max())

        # Build grid with consistent bounds (float32)
        nx = int(np.ceil((x_max - x_min) / self.grid_res)) + 1
        ny = int(np.ceil((y_max - y_min) / self.grid_res)) + 1
        self.x_grid = np.linspace(x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32)
        self.y_grid = np.linspace(y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32)

        self.transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res, self.grid_res
        )

        polygon = self.survey_boundary.geometry.iloc[0]
        nx, ny = len(self.x_grid), len(self.y_grid)

        z_final = np.full((ny, nx), np.nan, dtype=np.float32)
        ss_final = np.full((ny, nx), np.nan, dtype=np.float32)

        xx_full, yy_full = np.meshgrid(self.x_grid, self.y_grid)
        # xx_full, yy_full = np.meshgrid(self.x_grid, self.y_grid[::-1])
        polygon_mask = contains(polygon, xx_full, yy_full)

        delaunay_hull = Delaunay(np.column_stack(self.coordinates))

        grid_points = np.column_stack((xx_full.ravel(), yy_full.ravel()))
        delaunay_flat_mask = delaunay_hull.find_simplex(grid_points) >= 0
        delaunay_mask = delaunay_flat_mask.reshape(xx_full.shape)

        full_mask = delaunay_mask & polygon_mask

        z_pred, ss_pred = self.krige_model.execute(
            style="masked", xpoints=self.x_grid, ypoints=self.y_grid,
            mask=~full_mask, backend="vectorized"
        )

        # PyKrige returns numpy.ma.MaskedArray under style="masked".
        # Convert masked entries to NaN explicitly so downstream math and
        # raster writes don't fall back to fill_value or 0.
        if np.ma.is_masked(z_pred):
            z_pred = np.ma.filled(z_pred, np.nan).astype(np.float32)
        else:
            z_pred = np.asarray(z_pred, dtype=np.float32)
        if np.ma.is_masked(ss_pred):
            ss_pred = np.ma.filled(ss_pred, np.nan).astype(np.float32)
        else:
            ss_pred = np.asarray(ss_pred, dtype=np.float32)

        if self.detrend:
            trend_full = self.trendSurface.predict((xx_full, yy_full)).astype(np.float32)
            z_final = z_pred + trend_full
        else:
            z_final = z_pred
        ss_final = ss_pred

        z_final[~full_mask] = np.nan
        ss_final[~full_mask] = np.nan

        self.krige_pred = z_final
        self.krige_var = ss_final

    def _practical_range(self, model, pct=0.95, r_max=None):
        """Distance at which the fitted variogram first reaches pct*sill.
        Model-agnostic: works regardless of rescale/len_scale semantics."""
        sill = model.var + model.nugget
        r_max = r_max or 20 * model.len_scale  # generous search bound
        r = np.linspace(0, r_max, 5000)
        gamma = model.variogram(r)
        idx = np.argmax(gamma >= pct * sill)
        if gamma[idx] < pct * sill:  # never actually reached it in range
            return r_max
        return r[idx]

    def _write_output(self, pred_out_path=None, var_out_path=None, nodata_value=-9999.0):
        """Write COG rasters with AIC metadata."""
        z_export = np.flipud(np.nan_to_num(self.krige_pred, nan=nodata_value)).astype(np.float32)
        ss_export = np.flipud(np.nan_to_num(self.krige_var, nan=nodata_value)).astype(np.float32)
        
        meta = {
            'driver': 'GTiff',
            'height': z_export.shape[0],
            'width': z_export.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': self._crs,
            'transform': self.transform,
            'nodata': nodata_value,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'deflate',
            'predictor': 3,
            'zlevel': 6
        }
        
        suffix = "_detrended" if self.detrend else ""
        if pred_out_path is None:
            self.depth_tif_path = self.survey_path / f"isoKrige_AIC_{self.echosounder}{suffix}_pred.tif"
        else:
            self.depth_tif_path = Path(pred_out_path)
        
        if var_out_path is None:
            self.uncertainty_tif_path = self.survey_path / f"isoKrige_AIC_{self.echosounder}{suffix}_var.tif"
        else:
            self.uncertainty_tif_path = Path(var_out_path)

        with rasterio.open(self.depth_tif_path, 'w', **meta) as dst:
            dst.write(z_export, 1)
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)

            tags = {}
            
            # Add variogram model with AIC score
            tags['fit_variogram'] = str(self.fit_model)
            tags['variogram_aic'] = str(self.fit_aic)
            tags['selection_method'] = 'AIC'
            
            # Add ranking of all models
            tags['model_ranking'] = json.dumps([
                {'model': name, 'aic': float(score)} 
                for name, score in self.ranking
            ])

            # Add ill-conditioning tags
            tags['kriging_ill_conditioned'] = str(self._ill_conditioned)
            tags['kriging_ill_conditioned_chunks'] = str(self._ill_conditioned_count)
            tags['kriging_total_chunks'] = str(self._total_chunks)

            dst.update_tags(**tags)

        # Write variance
        with rasterio.open(self.uncertainty_tif_path, 'w', **meta) as dst:
            dst.write(ss_export, 1)
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
        
        # Free export arrays
        del z_export, ss_export

    def generate(self, pred_out_path=None, var_out_path=None):
        """Run complete Kriging workflow with AIC-based model selection."""
        
        if self.detrend:
            self._spline_detrend()
        
        self._fit_variograms()

        if self.chunked_kriging == True:
            self._krige_in_chunks()
        else:
            self._krige()

        if self.plot_outputs == True:
            self.plot_predictions()
            self.plot_variance()
            self.plot_variogram()

        self._write_output(pred_out_path, var_out_path)
        
        # Aggressive cleanup after writing
        self.krige_pred = None
        self.krige_var = None
        self.x_grid = None
        self.y_grid = None
        self.krige_model = None
        self.trendSurface = None
        self.residuals = None

        gc.collect()
        
        krige_type = "Regression Kriging (AIC)" if self.detrend else "Ordinary Kriging (AIC)"
        print(f"Exported {krige_type}:")
        print(f"  Best model: {self.ranking[0][0]} (AIC={self.fit_aic:.2f})")
        print(f"  {self.depth_tif_path}")
        print(f"  {self.uncertainty_tif_path}")
        return self.depth_tif_path, self.uncertainty_tif_path

    def plot_predictions(self):
        """Quick visualization of predictions."""
        if self.krige_pred is None:
            raise ValueError("Prediction data has been cleared. Call plot_predictions() before generate() completes.")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [float(self.x_grid.min()), float(self.x_grid.max()), 
                  float(self.y_grid.min()), float(self.y_grid.max())]
        im = ax.imshow(self.krige_pred, origin="lower", extent=extent, cmap='terrain')
        plt.colorbar(im, ax=ax, label='Depth')
        ax.set(xlabel='Easting', ylabel='Northing', 
               title=f'Kriging Interpolation (AIC Selection)\nModel: {self.ranking[0][0]} (AIC={self.fit_aic:.2f})')
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.show()

    def plot_variance(self):
        """Quick visualization of variance."""
        if self.krige_var is None:
            raise ValueError("Variance data has been cleared. Call plot_variance() before generate() completes.")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [float(self.x_grid.min()), float(self.x_grid.max()), 
                  float(self.y_grid.min()), float(self.y_grid.max())]
        im = ax.imshow(self.krige_var, origin="lower", extent=extent, cmap='magma')
        plt.colorbar(im, ax=ax, label='Kriging Variance')
        ax.set(xlabel='Easting', ylabel='Northing',
               title=f'Kriging Uncertainty (AIC Selection)\nModel: {self.ranking[0][0]} (AIC={self.fit_aic:.2f})')
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.show()

    def plot_variogram(self):
        """Plot the best-fit variogram model with empirical data."""
        if self.fit_model is None:
            raise ValueError("No variogram model has been fitted yet. Call generate() first.")

        if not hasattr(self, '_empirical_bins') or not hasattr(self, '_empirical_gamma'):
            raise ValueError("Empirical variogram data not available. Call generate() first.")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

        # Plot empirical variogram
        ax.scatter(self._empirical_bins, self._empirical_gamma, 
                  c='blue', s=50, alpha=0.6, label='Empirical')

        # Generate smooth curve for fitted model
        x_model = np.linspace(0, self._empirical_bins.max(), 200)
        y_model = self.fit_model.variogram(x_model)

        ax.plot(x_model, y_model, 'r-', linewidth=2, 
               label=f'Fitted: {self.ranking[0][0]}')

        # Add horizontal line for sill (variance + nugget)
        sill = self.fit_model.var + self.fit_model.nugget
        ax.axhline(y=sill, color='green', linestyle='--', linewidth=1.5, 
                   alpha=0.7, label=f'Sill: {sill:.2f}')

        # Add horizontal line for nugget
        ax.axhline(y=self.fit_model.nugget, color='orange', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Nugget: {self.fit_model.nugget:.2f}')
        
        practical_range = self._practical_range(self.fit_model)
        ax.axvline(x=practical_range, color='purple', linestyle='--',
            linewidth=1.5, alpha=0.7, label=f'Range: {practical_range:.2f}')
        # Add vertical line for range (effective range)
        # ax.axvline(x=self.fit_model.len_scale, color='purple', linestyle='--', 
        #            linewidth=1.5, alpha=0.7, label=f'Range: {self.fit_model.len_scale:.2f}')

        # ax.text(0.98, 0.02, params_text, 
        #        transform=ax.transAxes,
        #        verticalalignment='bottom',
        #        horizontalalignment='right',
        #        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        #        fontsize=9)

        ax.set_xlabel('Lag Distance', fontsize=11)
        ax.set_ylabel('Semivariance', fontsize=11)
        ax.set_title(f'Best-Fit Variogram Model (AIC={self.fit_aic:.2f})', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()