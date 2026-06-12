#!/usr/bin/env python
import numpy as np
from pathlib import Path
import verde as vd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask
import json
from rasterio.mask import mask
from shapely.geometry import box
import gc
import multiprocessing
from joblib import Parallel, delayed

class NaturalNeighbor:
    """Generate Natural Neighbor bathymetric surfaces with optimized performance."""
    
    __slots__ = [
        "surveyId",
        "survey_path",
        "survey_boundary",
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
        self._crs = self.survey_boundary.crs  # Cache CRS
        self.grid_res = grid_res
        self.plot_outputs = plot_outputs
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        
        # Store as float32 to reduce memory by 50%
        self.coordinates = (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
        self.bathymetry = np.asarray(z, dtype=np.float32)
        
        self.x_grid = None
        self.y_grid = None
        self.grid_z = None
        self.transform = None
        self.meta = None
        self.cv_metadata = None
        self.output_path = None
        self._fold_spacing = None

    def _interpolate_chunk(self, interpolator, x_chunk, y_chunk):
        """Interpolate a single chunk of the grid (for parallel execution)."""
        xx, yy = np.meshgrid(x_chunk, y_chunk)
        z_chunk = interpolator(xx, yy).astype(np.float32)
        return z_chunk

    def _interpolate(self, chunk_size=500):
        """Perform Natural Neighbor (Clough-Tocher) interpolation with parallel chunk processing."""
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
        self.x_grid = np.linspace(x_min, x_min + (nx - 1) * self.grid_res, nx, dtype=np.float32)
        self.y_grid = np.linspace(y_min, y_min + (ny - 1) * self.grid_res, ny, dtype=np.float32)
        
        # Build interpolator once
        pts = np.column_stack((x, y))
        interpolator = CloughTocher2DInterpolator(pts, self.bathymetry)
        
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
        results = Parallel(n_jobs=self.n_jobs, verbose=5, backend='loky')(
            delayed(self._interpolate_chunk)(interpolator, x_chunk, y_chunk)
            for iy, ix, iy_end, ix_end, x_chunk, y_chunk in chunk_jobs
        )
        
        # Aggregate results
        for (iy, ix, iy_end, ix_end, _, _), z_chunk in zip(chunk_jobs, results):
            self.grid_z[iy:iy_end, ix:ix_end] = z_chunk
        
        # Free interpolator
        del pts, interpolator
        
        self.transform = from_origin(
            x_min - self.grid_res / 2,
            y_max + self.grid_res / 2,
            self.grid_res, self.grid_res
        )
        
        boundary_mask = geometry_mask(
            [self.survey_boundary.geometry.iloc[0]],
            out_shape=self.grid_z.shape,
            transform=self.transform,
            invert=True
        )
        self.grid_z[~boundary_mask] = np.nan
        del boundary_mask

    def _prepare_metadata(self):
        """Prepare COG metadata."""
        self.meta = {
            'driver': 'GTiff',
            'height': self.grid_z.shape[0],
            'width': self.grid_z.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': self._crs,
            'transform': self.transform,
            'nodata': -9999.0,
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'deflate',
            'predictor': 3,
            'zlevel': 6
        }

    def _write_output(self, output_path=None):
        """Write COG raster with CV metadata."""
        if output_path is None:
            output_path = self.survey_path / f"NaturalNeighbor_bathy.tif"
        else:
            output_path = Path(output_path)
        
        self.output_path = output_path

        with rasterio.open(output_path, 'w', **self.meta) as dst:
            # dst.write(np.flipud(self.grid_z), 1)
            dst.write(self.grid_z, 1)  # Already float32

            dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
        
        return output_path

    def generate(self, output_path=None):
        """Run complete Natural Neighbor workflow."""
        self._interpolate()
        self._prepare_metadata()
        if self.plot_outputs == True:
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
            raise ValueError("Grid data has been cleared. Call plot_output() before generate() completes.")
        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        extent = [float(self.x_grid.min()), float(self.x_grid.max()), 
                  float(self.y_grid.min()), float(self.y_grid.max())]
        im = ax.imshow(self.grid_z, origin="upper", extent=extent, cmap='terrain')
        plt.colorbar(im, ax=ax, label='Depth')
        ax.set(xlabel='Easting', ylabel='Northing', title='Natural Neighbor Interpolation')
        ax.grid(color='white', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.show()
