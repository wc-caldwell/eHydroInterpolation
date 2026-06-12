#!/usr/bin/env python
"""
Optimized RBF interpolation with automatic parameter selection.
"""

import numpy as np
from pathlib import Path
import verde as vd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import json
import gc
import multiprocessing
from joblib import Parallel, delayed


class RBF:
    """Generate optimized RBF bathymetric surfaces with automatic parameter tuning."""

    __slots__ = [
        "surveyId",
        "survey_path",
        "grid_res",
        "plot_outputs",
        "coordinates",
        "bathymetry",
        "x_grid",
        "y_grid",
        "grid_z",
        "transform",
        "meta",
        "cv_metadata",
        "_crs",
        "_fold_spacing",
        "n_jobs",
        "best_params",
        "survey_boundary",
        "output_path",
    ]

    def __init__(
        self,
        surveyId,
        survey_path,
        x,
        y,
        z,
        survey_boundary,
        grid_res=10.0,
        plot_outputs=False,
        n_jobs=-1,
    ):
        self.surveyId = surveyId
        self.survey_path = survey_path
        self.survey_boundary = survey_boundary
        self._crs = self.survey_boundary.crs
        self.grid_res = grid_res
        self.plot_outputs = plot_outputs
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

        # Store as float32
        self.coordinates = (
            np.asarray(x, dtype=np.float32),
            np.asarray(y, dtype=np.float32),
        )
        self.bathymetry = np.asarray(z, dtype=np.float32)

        self.x_grid = None
        self.y_grid = None
        self.grid_z = None
        self.transform = None
        self.meta = None
        self.cv_metadata = None
        self._fold_spacing = None
        self.best_params = None
        self.output_path = None

    def _tune_rbf_parameters(self, n_folds=5, max_samples=5000):
        """
        Automatically tune RBF parameters using cross-validation.

        Parameters
        ----------
        n_folds : int
            Number of CV folds
        max_samples : int
            Maximum points for tuning (for speed)

        Returns
        -------
        dict : Best parameters (kernel, smoothing, neighbors)
        """
        print("Tuning RBF parameters...")

        x, y = self.coordinates
        z = self.bathymetry

        # Subsample if too many points
        if len(z) > max_samples:
            idx = np.random.choice(len(z), max_samples, replace=False)
            x_tune, y_tune, z_tune = x[idx], y[idx], z[idx]
        else:
            x_tune, y_tune, z_tune = x, y, z

        pts_tune = np.column_stack((x_tune, y_tune))

        # Parameter grid
        param_grid = {
            "kernel": ["thin_plate_spline", "cubic", "quintic"],  # Removed multiquadric
            "smoothing": [0.0, 0.001, 0.01],  # Much less smoothing
            "neighbors": [None, 50, 100],  # None = use all points (within reason)
        }

        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        best_rmse = np.inf
        best_params = None
        results = []

        for kernel in param_grid["kernel"]:
            for smoothing in param_grid["smoothing"]:
                for neighbors in param_grid["neighbors"]:
                    rmse_folds = []

                    try:
                        for train_idx, val_idx in kf.split(pts_tune):
                            # Train
                            rbf = RBFInterpolator(
                                pts_tune[train_idx],
                                z_tune[train_idx],
                                kernel=kernel,
                                smoothing=smoothing,
                                neighbors=neighbors,
                                degree=1,  # Linear polynomial trend
                            )

                            # Validate
                            z_pred = rbf(pts_tune[val_idx])
                            rmse = np.sqrt(np.mean((z_tune[val_idx] - z_pred) ** 2))
                            rmse_folds.append(rmse)

                        mean_rmse = np.mean(rmse_folds)
                        std_rmse = np.std(rmse_folds)

                        results.append(
                            {
                                "kernel": kernel,
                                "smoothing": smoothing,
                                "neighbors": neighbors,
                                "rmse_mean": mean_rmse,
                                "rmse_std": std_rmse,
                            }
                        )

                        print(
                            f"  {kernel:20s} smooth={smoothing:6.3f} neighbors={str(neighbors):5s}: "
                            f"RMSE={mean_rmse:.3f}±{std_rmse:.3f}"
                        )

                        if mean_rmse < best_rmse:
                            best_rmse = mean_rmse
                            best_params = {
                                "kernel": kernel,
                                "smoothing": smoothing,
                                "neighbors": neighbors,
                            }

                    except Exception as e:
                        print(
                            f"  {kernel:20s} smooth={smoothing:6.3f} neighbors={str(neighbors):5s}: FAILED ({e})"
                        )
                        continue

        print(f"\nBest parameters: {best_params}")
        print(f"Best CV RMSE: {best_rmse:.3f}ft\n")

        self.best_params = best_params
        return best_params

    def _interpolate_chunk(self, rbf, x_chunk, y_chunk):
        """Interpolate a single chunk."""
        xx, yy = np.meshgrid(x_chunk, y_chunk)
        grid_pts = np.column_stack((xx.ravel(), yy.ravel()))
        z_chunk = rbf(grid_pts).reshape(xx.shape).astype(np.float32)
        return z_chunk

    def _interpolate(self, chunk_size=500, auto_tune=True):
        """
        Perform RBF interpolation with parallel chunk processing.

        Parameters
        ----------
        chunk_size : int
            Size of processing chunks
        auto_tune : bool
            If True, automatically tune parameters. If False, use defaults.
        """
        x, y = self.coordinates

        # Auto-tune parameters if requested
        if auto_tune and self.best_params is None:
            self._tune_rbf_parameters()

        # Use tuned parameters or sensible defaults
        if self.best_params is not None:
            kernel = self.best_params["kernel"]
            smoothing = self.best_params["smoothing"]
            neighbors = self.best_params["neighbors"]
        else:
            kernel = "thin_plate_spline"  
            smoothing = 0.0  # No smoothing by default
            neighbors = (
                None  
            )

        print(
            f"Using RBF parameters: kernel={kernel}, smoothing={smoothing}, neighbors={neighbors}"
        )

        # Pre-compute bounds
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Cache fold spacing for CV
        max_dist = np.sqrt(x_range**2 + y_range**2) / 2
        self._fold_spacing = max_dist / 8

        # Build grid 
        nx = int(np.ceil(x_range / self.grid_res)) + 1
        ny = int(np.ceil(y_range / self.grid_res)) + 1
        self.x_grid = np.linspace(x_min, x_max, nx, dtype=np.float32)
        self.y_grid = np.linspace(y_min, y_max, ny, dtype=np.float32)

        # Build RBF interpolator
        pts = np.column_stack((x, y))
        rbf = RBFInterpolator(
            pts,
            self.bathymetry,
            kernel=kernel,
            smoothing=smoothing,
            neighbors=neighbors,
            degree=1,  # Include linear trend
        )

        # Initialize output grid
        self.grid_z = np.full((ny, nx), np.nan, dtype=np.float32)

        # Generate chunk jobs
        chunk_jobs = []
        for iy in range(0, ny, chunk_size):
            for ix in range(0, nx, chunk_size):
                iy_end = min(iy + chunk_size, ny)
                ix_end = min(ix + chunk_size, nx)
                x_chunk = self.x_grid[ix:ix_end]
                y_chunk = self.y_grid[::-1][iy:iy_end]
                chunk_jobs.append((iy, ix, iy_end, ix_end, x_chunk, y_chunk))

        # Process chunks in parallel
        print(f"Processing {len(chunk_jobs)} chunks using {self.n_jobs} cores...")
        results = Parallel(n_jobs=self.n_jobs, verbose=5, backend="loky")(
            delayed(self._interpolate_chunk)(rbf, x_chunk, y_chunk)
            for iy, ix, iy_end, ix_end, x_chunk, y_chunk in chunk_jobs
        )

        # Aggregate results
        for (iy, ix, iy_end, ix_end, _, _), z_chunk in zip(chunk_jobs, results):
            self.grid_z[iy:iy_end, ix:ix_end] = z_chunk

        # Free memory
        del pts, rbf

        self.transform = from_origin(x_min, y_max, self.grid_res, self.grid_res)

        # Apply boundary mask
        boundary_mask = geometry_mask(
            [self.survey_boundary.geometry.iloc[0]],
            out_shape=self.grid_z.shape,
            transform=self.transform,
            invert=True,
        )
        self.grid_z[~boundary_mask] = np.nan
        del boundary_mask

    def _prepare_metadata(self):
        """Prepare COG metadata."""
        self.meta = {
            "driver": "GTiff",
            "height": self.grid_z.shape[0],
            "width": self.grid_z.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": self._crs,
            "transform": self.transform,
            "nodata": -9999.0,
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "compress": "deflate",
            "predictor": 3,
            "zlevel": 6,
        }

    def _write_output(self, output_path=None):
        """Write COG raster with CV metadata."""
        if output_path is None:
            output_path = (
                self.survey_path
                / f"RBF_bathy.tif"
            )
        else:
            output_path = Path(output_path)

        self.output_path = output_path
        
        with rasterio.open(self.output_path, "w", **self.meta) as dst:
            dst.write(self.grid_z, 1)  # Already float32
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)

            if self.cv_metadata:
                summary = self.cv_metadata.get("summary", {})
                tags = {
                    f"cv_{k}": str(v)
                    for k, v in summary.items()
                    if isinstance(v, (int, float, str, type(None)))
                }
                tags["cv_results_json"] = json.dumps(summary, sort_keys=True)
                tags["cv_folds_json"] = json.dumps(
                    self.cv_metadata.get("folds", []), sort_keys=True
                )
                dst.update_tags(**tags)

        return self.output_path

    def generate(self, auto_tune=True):
        """
        Run complete RBF workflow.

        Parameters
        ----------
        output_path : str or Path, optional
            Output path for raster
        auto_tune : bool
            If True, automatically tune RBF parameters
        """
        self._interpolate(auto_tune=auto_tune)
        self._prepare_metadata()

        if self.plot_outputs:
            self.plot_output()

        out = self._write_output(output_path=self.output_path)

        # Cleanup
        self.grid_z = None
        self.x_grid = None
        self.y_grid = None

        gc.collect()

        print(f"Exported: {out}")
        return out

    def plot_output(self):
        """Quick visualization."""
        if self.grid_z is None:
            raise ValueError(
                "Grid data has been cleared. Call plot_output() before generate() completes."
            )
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [
            float(self.x_grid.min()),
            float(self.x_grid.max()),
            float(self.y_grid.min()),
            float(self.y_grid.max()),
        ]
        im = ax.imshow(self.grid_z, origin="upper", extent=extent, cmap="terrain")
        plt.colorbar(im, ax=ax, label="Depth")
        ax.set(xlabel="Easting", ylabel="Northing", title="RBF Interpolation")
        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.3)
        plt.show()
