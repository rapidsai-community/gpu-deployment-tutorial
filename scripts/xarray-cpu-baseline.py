# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Simplified EOF analysis for benchmarking. The full version uses a
# 1981 to 2010 climatology and latitude weighting. Here we use the
# 2025 time-mean and skip the weighting so the script isolates the
# SVD timing without the data-prep overhead.

import time

import numpy as np
import xarray as xr

ds = xr.open_dataset("air.sig995.2025.nc")

t0 = time.perf_counter()

air = ds["air"]
T = air.to_numpy()
ntime, nlat, nlon = T.shape

# Build X = anomalies (time x space)
X = T.reshape(ntime, nlat * nlon)
X = X - X.mean(axis=0, keepdims=True)

# EOF via SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)

eof1 = Vt[0, :].reshape(nlat, nlon)
pc1 = U[:, 0] * S[0]
var_frac1 = (S[0] ** 2) / np.sum(S**2)

elapsed = time.perf_counter() - t0
print(f"shape: {T.shape}")
print(f"matrix: {X.shape}")
print(f"cpu eof seconds: {elapsed:.3f}")
print(f"first mode variance fraction: {float(var_frac1):.4f}")
