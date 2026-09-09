#!/usr/bin/env python

"""Stochastic Variational Gaussian Process model for generating bathymetric surfaces from hydrographic soundings (xyz)"""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from shapely import contains_xy
from scipy.spatial import Delaunay

import tqdm
import rasterio
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.constraints import Interval, GreaterThan
from gpytorch.priors import GammaPrior

class SVGP:
    """Generate a surface from an xyz point cloud using variational inference, using kernel covariance only."""

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
        "device",
        "trend",
        "residuals_norm"
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
        device,
        grid_res=10.0,
        plot_outputs=False,
    ):
        self.surveyId = surveyId
        self.survey_path = survey_path
        self.survey_boundary = survey_boundary
        self._crs = self.survey_boundary.crs  # Cache CRS
        self.echosounder = echosounder
        self.grid_res = grid_res
        self.plot_outputs = plot_outputs
        self.device = device

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

    def _detrend(self, reduce_spacing=50, spline_damping=1e-4):
        import verde as vd
        chain = vd.Chain([
            ("reduce", vd.BlockMean(spacing=reduce_spacing)),   
            ("spline", vd.Spline(damping=spline_damping)),
        ])
        xy_raw = np.column_stack([self.coordinates[0], self.coordinates[1]])

        self.trend = chain.fit((xy_raw[:, 0], xy_raw[:, 1]), self.bathymetry)
        trend_pred = chain.predict((xy_raw[:, 0], xy_raw[:, 1]))   # evaluate at full resolution
        residuals = self.bathymetry - trend_pred

        self.residuals_norm = (residuals - np.mean(residuals)) / np.std(residuals)

    def _prep_train_tensors(self):
        X_all = torch.as_tensor(xy_raw, dtype=torch.float32)
        y_all = torch.as_tensor(residuals_norm, dtype=torch.float32)

        x_mean = X_all.mean(0)
        x_std = X_all.std(0).clamp_min(1e-6)
        X_all_scaled = (X_all - x_mean) / x_std

    def _prediction_grid(self):
        xy_raw = np.column_stack([self.coordinates[0], self.coordinates[1]])

