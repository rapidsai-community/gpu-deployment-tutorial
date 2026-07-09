# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import xarray as xr


ds = xr.open_dataset("air.sig995.2025.nc")
T = ds["air"].to_numpy()
ntime, nlat, nlon = T.shape
T_flat = T.reshape(ntime, nlat * nlon)


def build_features_fixed(matrix):
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    ranges = matrix.max(axis=0) - matrix.min(axis=0)
    variances = matrix.var(axis=0)
    return np.column_stack((means, stds, ranges, variances))


start = time.perf_counter()
features = build_features_fixed(T_flat)
preprocess_seconds = time.perf_counter() - start

print(f"grid cells: {T_flat.shape[1]:,}")
print(f"timesteps: {T_flat.shape[0]:,}")
print(f"vectorized preprocessing: {preprocess_seconds:.3f} seconds")
print("fix: vectorized per-cell stats replace the Python loop")
