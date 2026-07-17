#!/usr/bin/env python
import numpy as np
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import json
import gc
import multiprocessing
from joblib import Parallel, delayed

# TODO: replace the survey_boundary argument/variable with an argument/variable that will just read in a geodataframe


class TIN:
    """Generate TIN bathymetric surfaces with optimized performance."""

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
        "output_path",
        "_crs",
        "_fold_spacing",
        "n_jobs",
        "max_edge_length",
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
        plot_outputs=False,
        n_jobs=-1,
        max_edge_length=None,
    ):
        self.surveyId = surveyId
        self.survey_path = survey_path
        self.survey_boundary = survey_boundary
        self._crs = self.survey_boundary.crs  # Cache CRS
        self.echosounder = echosounder
        self.grid_res = grid_res
        self.plot_outputs = plot_outputs
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()

        # Maximum allowed triangle edge length (same units as x/y).
        # Triangles longer than this are treated as bridging a data gap
        # and are excluded from the output, e.g. tspace * 25 * 1.5.
        # None disables the check (boundary-polygon masking only).
        self.max_edge_length = max_edge_length

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
        self._fold_spacing = None

    def _valid_simplices(self, tri, max_edge_length):
        """
        Boolean mask over tri.simplices: True where the triangle's longest
        edge is <= max_edge_length. Equivalent to matplotlib's
        create_TIN_mask, applied to a scipy.spatial.Delaunay object.
        """
        pts = tri.points  # (n_points, 2)
        simplices = tri.simplices  # (n_tri, 3) vertex indices

        p0 = pts[simplices[:, 0]]
        p1 = pts[simplices[:, 1]]
        p2 = pts[simplices[:, 2]]

        e0 = np.linalg.norm(p1 - p0, axis=1)
        e1 = np.linalg.norm(p2 - p1, axis=1)
        e2 = np.linalg.norm(p0 - p2, axis=1)

        longest_edge = np.maximum(np.maximum(e0, e1), e2)
        return longest_edge <= max_edge_length

    def _interpolate_chunk(self, interpolator, tri, valid_simplices, x_chunk, y_chunk):
        """Interpolate a single chunk of the grid (for parallel execution).

        If valid_simplices is provided, grid points that fall inside a
        triangle whose longest edge exceeds max_edge_length are set to NaN.
        """
        xx, yy = np.meshgrid(x_chunk, y_chunk)
        z_chunk = interpolator(xx, yy).astype(np.float32)

        if valid_simplices is not None:
            query_pts = np.column_stack((xx.ravel(), yy.ravel()))
            simplex_idx = tri.find_simplex(query_pts)  # -1 if outside hull

            z_flat = z_chunk.ravel()
            inside_hull = simplex_idx >= 0
            oversized = np.zeros(simplex_idx.shape, dtype=bool)
            oversized[inside_hull] = ~valid_simplices[simplex_idx[inside_hull]]

            z_flat[oversized] = np.nan
            z_chunk = z_flat.reshape(xx.shape)

        return z_chunk.astype(np.float32)

    def _interpolate(self, chunk_size=500):
        """Perform TIN interpolation with parallel chunk processing."""
        x, y = self.coordinates

        # Pre-compute bounds
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Cache fold spacing for CV
        max_dist = np.sqrt(x_range**2 + y_range**2) / 2
        self._fold_spacing = max_dist / 8

        # Build grid with consistent bounds (float32)
        nx = int(np.ceil(x_range / self.grid_res)) + 1
        ny = int(np.ceil(y_range / self.grid_res)) + 1
        self.x_grid = np.linspace(
            x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32
        )
        self.y_grid = np.linspace(
            y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32
        )

        # Build triangulation and interpolator once
        pts = np.column_stack((x, y))
        tri = Delaunay(pts)
        interpolator = LinearNDInterpolator(tri, self.bathymetry)

        # Compute per-triangle validity mask (edge-length check)
        valid_simplices = None
        if self.max_edge_length is not None:
            valid_simplices = self._valid_simplices(tri, self.max_edge_length)
            n_bad = int((~valid_simplices).sum())
            print(
                f"Edge-length mask: dropping {n_bad}/{len(valid_simplices)} "
                f"triangles exceeding {self.max_edge_length:.2f} (grid units)"
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
                y_chunk = self.y_grid[::-1][iy:iy_end]  # [::-1] for correct orientation
                chunk_jobs.append((iy, ix, iy_end, ix_end, x_chunk, y_chunk))

        # Process chunks in parallel
        print(f"Processing {len(chunk_jobs)} chunks using {self.n_jobs} cores...")
        results = Parallel(n_jobs=self.n_jobs, verbose=5, backend="loky")(
            delayed(self._interpolate_chunk)(
                interpolator, tri, valid_simplices, x_chunk, y_chunk
            )
            for iy, ix, iy_end, ix_end, x_chunk, y_chunk in chunk_jobs
        )

        # Aggregate results
        for (iy, ix, iy_end, ix_end, _, _), z_chunk in zip(chunk_jobs, results):
            self.grid_z[iy:iy_end, ix:ix_end] = z_chunk

        # Free triangulation
        del pts, tri, interpolator

        # Transform and mask
        self.transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res,
            self.grid_res,
        )

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
                / f"{self.echosounder}_TIN_bathy.tif"
            )
        else:
            output_path = Path(output_path)

        self.output_path = output_path

        with rasterio.open(output_path, "w", **self.meta) as dst:
            # dst.write(np.flipud(self.grid_z), 1)
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

        return output_path

    def generate(self, output_path=None):
        """Run complete TIN workflow."""
        self._interpolate()
        self._prepare_metadata()
        if self.plot_outputs:
            self.plot_output()
        out = self._write_output(output_path)

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
        im = ax.imshow(self.grid_z, origin="upper", extent=extent, cmap="viridis")
        plt.colorbar(im, ax=ax, label="Depth")
        ax.set(xlabel="Easting", ylabel="Northing", title="TIN Interpolation")
        ax.grid(color="white", linestyle="--", linewidth=0.5, alpha=0.3)
        plt.show()