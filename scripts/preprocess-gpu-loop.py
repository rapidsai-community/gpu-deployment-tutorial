# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import cupy as cp
import xarray as xr
import cupy_xarray  # noqa: F401  # registers the .cupy accessor


ds = xr.open_dataset("air.sig995.2025.nc")
T = ds["air"].cupy.as_cupy().data  # move the whole field to the GPU once
ntime, nlat, nlon = T.shape
T_flat = T.reshape(ntime, nlat * nlon)


def build_features_gpu_loop(matrix):
    ncells = matrix.shape[1]
    features = cp.empty((ncells, 4), dtype=cp.float64)
    for cell in range(ncells):
        column = matrix[:, cell]
        features[cell, 0] = column.mean()
        features[cell, 1] = column.std()
        features[cell, 2] = column.max() - column.min()
        features[cell, 3] = column.var()
    cp.cuda.Stream.null.synchronize()
    return features


start = time.perf_counter()
features = build_features_gpu_loop(T_flat)
elapsed = time.perf_counter() - start

print(f"grid cells: {T_flat.shape[1]:,}")
print(f"timesteps: {T_flat.shape[0]:,}")
print(f"per-cell GPU loop: {elapsed:.3f} seconds")
print("bottleneck: one tiny GPU launch per cell, thousands of times")
