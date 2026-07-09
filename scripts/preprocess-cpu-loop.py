# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numpy as np
import xarray as xr


ds = xr.open_dataset("air.sig995.2025.nc")
T = ds["air"].to_numpy()  # shape (ntime, nlat, nlon)
ntime, nlat, nlon = T.shape
T_flat = T.reshape(ntime, nlat * nlon)  # (ntime, ncells)


def build_features_bad(matrix):
    ncells = matrix.shape[1]
    features = []
    for cell in range(ncells):
        column = matrix[:, cell]
        features.append(
            (
                column.mean(),
                column.std(),
                column.max() - column.min(),
                column.var(),
            )
        )
    return np.asarray(features, dtype=np.float64)


start = time.perf_counter()
features = build_features_bad(T_flat)
preprocess_seconds = time.perf_counter() - start

print(f"grid cells: {T_flat.shape[1]:,}")
print(f"timesteps: {T_flat.shape[0]:,}")
print(f"python preprocessing: {preprocess_seconds:.3f} seconds")
print("bottleneck: per-cell Python loop preprocessing dominates")
