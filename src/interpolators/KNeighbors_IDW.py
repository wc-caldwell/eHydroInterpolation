#!/usr/bin/env python
"""
IDW interpolation using scipy.spatial.cKDTree - optimized for speed and simplicity.
Uses decimated points for interpolation.
"""

import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import gc
import multiprocessing
from joblib import Parallel, delayed
import verde as vd
import geopandas as gpd


def _idw_chunk_worker(x_chunk, y_chunk, tree, z_data, n_neighbors, power):
    """
    Static function to interpolate a chunk using IDW.
    Defined at module level to avoid pickling issues with joblib.
    """
    xx, yy = np.meshgrid(x_chunk, y_chunk)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Query k nearest neighbors
    distances, indices = tree.query(grid_points, k=n_neighbors)

    # Handle coincident points (distance = 0)
    distances = np.maximum(distances, 1e-10)

    # Compute IDW weights: w = 1 / d^p
    weights = 1.0 / np.power(distances, power)
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_normalized = weights / weights_sum

    # Get neighbor values and compute weighted average
    neighbor_values = z_data[indices]
    interpolated = np.sum(weights_normalized * neighbor_values, axis=1)

    return interpolated.reshape(xx.shape).astype(np.float32)


class IDW:
    """Generate IDW bathymetric surfaces with optimized performance."""

    __slots__ = [
        "surveyId",
        "survey_path",
        "survey_boundary",
        "echosounder",
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
        "power",
        "n_neighbors",
        "n_jobs",
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
        echosounder,
        grid_res=10.0,
        power=2,
        n_neighbors=32,
        plot_outputs=False,
        n_jobs=-1,
    ):
        self.surveyId = surveyId
        self.survey_path = survey_path
        self.survey_boundary = survey_boundary
        self._crs = self.survey_boundary.crs
        self.echosounder = echosounder
        self.grid_res = grid_res

        self.power = power
        self.n_neighbors = n_neighbors
        self.plot_outputs = plot_outputs
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

        # Store as float32 to reduce memory by 50%
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
        self.output_path = None

    def _interpolate(self, chunk_size=500):
        """Perform IDW interpolation using scipy cKDTree."""
        x, y = self.coordinates

        # Pre-compute bounds
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Build grid with consistent bounds (float32)
        nx = int(np.ceil(x_range / self.grid_res)) + 1
        ny = int(np.ceil(y_range / self.grid_res)) + 1
        self.x_grid = np.linspace(
            x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32
        )
        self.y_grid = np.linspace(
            y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32
        )

        # Build KD-tree from data points
        print(f"Building KD-tree with {len(self.bathymetry)} points...")
        data_points = np.column_stack([x, y])
        tree = cKDTree(data_points)

        # Initialize output grid
        self.grid_z = np.full((ny, nx), np.nan, dtype=np.float32)

        # Generate chunk jobs
        chunk_jobs = []
        for iy in range(0, ny, chunk_size):
            for ix in range(0, nx, chunk_size):
                iy_end = min(iy + chunk_size, ny)
                ix_end = min(ix + chunk_size, nx)
                x_chunk = self.x_grid[ix:ix_end]
                y_chunk = self.y_grid[::-1][iy:iy_end]  # [::-1] for correct orientation
                chunk_jobs.append((iy, ix, iy_end, ix_end, x_chunk, y_chunk))

        # Process chunks in parallel using loky backend
        # The worker function is at module level so it can be pickled
        print(f"Processing {len(chunk_jobs)} chunks using {self.n_jobs} cores...")
        results = Parallel(n_jobs=self.n_jobs, verbose=5, backend="loky")(
            delayed(_idw_chunk_worker)(
                x_chunk, y_chunk, tree, self.bathymetry, self.n_neighbors, self.power
            )
            for iy, ix, iy_end, ix_end, x_chunk, y_chunk in chunk_jobs
        )

        # Aggregate results
        for (iy, ix, iy_end, ix_end, _, _), z_chunk in zip(chunk_jobs, results):
            self.grid_z[iy:iy_end, ix:ix_end] = z_chunk

        # Transform
        self.transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res,
            self.grid_res,
        )

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
        """Write COG raster."""
        if output_path is None:
            output_path = (
                self.survey_path / f"{self.echosounder}_IDW_bathy.tif"
            )
        else:
            output_path = Path(output_path)

        self.output_path = output_path

        with rasterio.open(output_path, "w", **self.meta) as dst:
            dst.write(self.grid_z, 1)
            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)

        return self.output_path

    def generate(self):
        """Run complete IDW workflow."""
        self._interpolate()
        self._prepare_metadata()
        if self.plot_outputs:
            self.plot_output()
        out = self._write_output(output_path=self.output_path)

        # Aggressive cleanup after writing
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
        ax.set(xlabel="Easting", ylabel="Northing", title="IDW Interpolation")
        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.3)
        plt.show()

    def cross_validate(
        self,
        points_gdf,
        boundary_gdf,
        z_column="z",
        n_splits=5,
        divisions=5,
        buffer_fraction=0.02,
        random_state=42,
        plot_folds=True,
    ):
        """
        Block spatial CV with edge-anchored training folds.
        Keeps the same blocking scheme as TIN/NN for direct comparison.
        """

        # Derive buffer and blocking params from data extent
        total_bounds = points_gdf.total_bounds
        total_extent = max(
            total_bounds[2] - total_bounds[0], total_bounds[3] - total_bounds[1]
        )
        buffer_distance = -(total_extent * buffer_fraction)

        # Split edge vs interior points
        inner_zone = boundary_gdf.geometry.union_all().buffer(buffer_distance)
        edge_mask = ~points_gdf.geometry.within(inner_zone)
        edge_points = points_gdf[edge_mask]
        inner_points = points_gdf[~edge_mask]

        # Block KFold on interior points — adapt divisions to ensure
        # at least n_splits non-empty folds
        inner_bounds = inner_points.total_bounds
        inner_extent = max(
            inner_bounds[2] - inner_bounds[0], inner_bounds[3] - inner_bounds[1]
        )

        inner_coords = np.column_stack(
            [inner_points.geometry.x.values, inner_points.geometry.y.values]
        )
        edge_coords = np.column_stack(
            [edge_points.geometry.x.values, edge_points.geometry.y.values]
        )
        edge_z = edge_points[z_column].values

        best_divisions = divisions
        found = False
        for trial_divisions in range(divisions, divisions * 3):
            spacing = inner_extent / trial_divisions
            kfold = vd.BlockKFold(
                spacing=spacing,
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
            try:
                splits = list(kfold.split(inner_coords))
            except ValueError:
                continue
            empty_folds = sum(
                1
                for train_idx, val_idx in splits
                if len(val_idx) == 0 or len(train_idx) == 0
            )
            if empty_folds == 0:
                best_divisions = trial_divisions
                found = True
                break

        if not found:
            print("  Warning: could not find a valid blocking scheme")
            self.cv_metadata = None
            return self.cv_metadata

        if best_divisions != divisions:
            print(
                f"  Adjusted divisions from {divisions} to {best_divisions} "
                f"to ensure {n_splits} non-empty folds"
            )

        spacing = inner_extent / best_divisions
        kfold = vd.BlockKFold(
            spacing=spacing,
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )

        # Build the same grid as _interpolate() for fold surface plots
        x_all = np.concatenate([edge_coords[:, 0], inner_coords[:, 0]])
        y_all = np.concatenate([edge_coords[:, 1], inner_coords[:, 1]])
        x_min, x_max = float(x_all.min()), float(x_all.max())
        y_min, y_max = float(y_all.min()), float(y_all.max())
        x_range = x_max - x_min
        y_range = y_max - y_min
        nx = int(np.ceil(x_range / self.grid_res)) + 1
        ny = int(np.ceil(y_range / self.grid_res)) + 1
        x_grid = np.linspace(
            x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32
        )
        y_grid = np.linspace(
            y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32
        )

        grid_transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res,
            self.grid_res,
        )
        boundary_mask = geometry_mask(
            [boundary_gdf.geometry.iloc[0]],
            out_shape=(ny, nx),
            transform=grid_transform,
            invert=True,
        )
        plot_extent = [x_min, x_max, y_min, y_max]

        fold_results = []
        for fold_i, (train_idx, val_idx) in enumerate(kfold.split(inner_coords)):
            if len(val_idx) == 0:
                print(f"  Fold {fold_i}: empty validation set, skipping")
                continue
            if len(train_idx) == 0:
                print(f"  Fold {fold_i}: empty training set, skipping")
                continue

            train_inner_coords = inner_coords[train_idx]
            train_inner_z = inner_points.iloc[train_idx][z_column].values
            val_coords = inner_coords[val_idx]
            val_z = inner_points.iloc[val_idx][z_column].values

            # Combine edge + inner training points
            train_x = np.concatenate([edge_coords[:, 0], train_inner_coords[:, 0]])
            train_y = np.concatenate([edge_coords[:, 1], train_inner_coords[:, 1]])
            train_z = np.concatenate([edge_z, train_inner_z])

            # Build KD-tree and predict at validation locations
            train_pts = np.column_stack((train_x, train_y))
            tree = cKDTree(train_pts)

            distances, indices = tree.query(val_coords, k=self.n_neighbors)
            distances = np.maximum(distances, 1e-10)
            weights = 1.0 / np.power(distances, self.power)
            weights_normalized = weights / weights.sum(axis=1, keepdims=True)
            pred_z = np.sum(weights_normalized * train_z[indices], axis=1)

            # Plot fold surface
            if plot_folds:
                xx, yy = np.meshgrid(x_grid, y_grid[::-1])
                grid_points = np.column_stack([xx.ravel(), yy.ravel()])

                dist_grid, idx_grid = tree.query(grid_points, k=self.n_neighbors)
                dist_grid = np.maximum(dist_grid, 1e-10)
                w_grid = 1.0 / np.power(dist_grid, self.power)
                w_grid_norm = w_grid / w_grid.sum(axis=1, keepdims=True)
                fold_grid = np.sum(w_grid_norm * train_z[idx_grid], axis=1)
                fold_grid = fold_grid.reshape(xx.shape).astype(np.float32)
                fold_grid[~boundary_mask] = np.nan

                fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)

                im = axes[0].imshow(
                    fold_grid, origin="upper", extent=plot_extent, cmap="terrain"
                )
                plt.colorbar(im, ax=axes[0], label="Depth", shrink=0.8)
                axes[0].set_title(f"Fold {fold_i} — IDW Interpolated Surface")
                axes[0].set(xlabel="Easting", ylabel="Northing")

                axes[1].scatter(
                    train_inner_coords[:, 0],
                    train_inner_coords[:, 1],
                    s=0.5,
                    c="blue",
                    alpha=0.3,
                    label=f"Train inner ({len(train_inner_coords)})",
                )
                axes[1].scatter(
                    edge_coords[:, 0],
                    edge_coords[:, 1],
                    s=0.5,
                    c="green",
                    alpha=0.3,
                    label=f"Edge ({len(edge_coords)})",
                )
                axes[1].scatter(
                    val_coords[:, 0],
                    val_coords[:, 1],
                    s=0.5,
                    c="red",
                    alpha=0.3,
                    label=f"Validation ({len(val_coords)})",
                )
                axes[1].set_title(f"Fold {fold_i} — Point Split")
                axes[1].set(xlabel="Easting", ylabel="Northing")
                axes[1].legend(loc="upper right", markerscale=10)
                axes[1].set_aspect("equal")

                plt.tight_layout()
                plt.show()

                del (
                    fold_grid,
                    xx,
                    yy,
                    grid_points,
                    dist_grid,
                    idx_grid,
                    w_grid,
                    w_grid_norm,
                )

            # Metrics
            residuals = pred_z - val_z
            fold_results.append(
                {
                    "fold": fold_i,
                    "n_train": len(train_z),
                    "n_val": len(val_z),
                    "n_valid": len(val_z),
                    "mae": float(np.mean(np.abs(residuals))),
                    "rmse": float(np.sqrt(np.mean(residuals**2))),
                    "me": float(np.mean(residuals)),
                    "std": float(np.std(residuals)),
                }
            )
            print(
                f"  Fold {fold_i}: RMSE={fold_results[-1]['rmse']:.3f}, "
                f"MAE={fold_results[-1]['mae']:.3f}, n_val={fold_results[-1]['n_val']}"
            )

            del train_pts, tree

        del boundary_mask

        if fold_results:
            summary = {
                "mean_rmse": float(np.mean([f["rmse"] for f in fold_results])),
                "mean_mae": float(np.mean([f["mae"] for f in fold_results])),
                "mean_me": float(np.mean([f["me"] for f in fold_results])),
                "std_rmse": float(np.std([f["rmse"] for f in fold_results])),
                "n_edge_points": len(edge_points),
                "n_inner_points": len(inner_points),
                "buffer_distance": float(buffer_distance),
                "block_spacing": float(spacing),
                "divisions_used": best_divisions,
            }
            self.cv_metadata = {"summary": summary, "folds": fold_results}
            print(
                f"\n  CV Summary: RMSE={summary['mean_rmse']:.3f} "
                f"± {summary['std_rmse']:.3f}"
            )
        else:
            print("  No valid folds produced.")
            self.cv_metadata = None

        return self.cv_metadata
