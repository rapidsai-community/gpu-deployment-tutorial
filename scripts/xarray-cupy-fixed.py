# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import cupy as cp
import numpy as np
import xarray as xr
import cupy_xarray  # noqa: F401  # registers the .cupy accessor

ds = xr.open_dataset("air.sig995.2025.nc")

t0 = time.perf_counter()

# .cupy.as_cupy() moves the xarray DataArray's underlying data to the GPU
air = ds["air"].cupy.as_cupy()
T = air.data
ntime, nlat, nlon = T.shape

# Build X = anomalies (time x space) on the GPU
X = T.reshape(ntime, nlat * nlon)
X = X - X.mean(axis=0, keepdims=True)

# np.linalg.svd dispatches to CuPy via __array_function__ because X is a CuPy array
U, S, Vt = np.linalg.svd(X, full_matrices=False)

eof1 = Vt[0, :].reshape(nlat, nlon)
pc1 = U[:, 0] * S[0]
var_frac1 = (S[0] ** 2) / np.sum(S**2)

cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - t0

print(f"shape: {T.shape}")
print(f"fixed total seconds: {elapsed:.3f}")
print(f"first mode variance fraction: {float(var_frac1):.4f}")
print("fix: SVD stays on the GPU via __array_function__ dispatch to CuPy")
