#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTD-based stratification, neutral buoyancy depth, and vertical mixing
for a plume at depth.

- Loads CTD profile from results/ctd.zarr
- Computes density and Brunt-Vaisala frequency N^2
- Estimates neutral buoyancy depth for a chosen plume density
- Builds a simple Kz(z) profile and Kz at neutral depth
- Computes vertical spread sigma_z(t) from diffusion
"""

import numpy as np
import xarray as xr
import gsw  # pip install gsw

# ============================================================
# 1. Load CTD profile
# ============================================================
ds_ctd = xr.open_zarr("results/ctd.zarr")

# These should be 1D profiles; if they’re not, you may need to index them
lat_ctd = float(ds_ctd["lat"].values)
lon_ctd = float(ds_ctd["lon"].values)
depth   = ds_ctd["depth"].values       # 1D (nz,)
temp    = ds_ctd["temperature"].values # in-situ T (degC)
salt    = ds_ctd["salinity"].values    # practical salinity (PSU)

depth = np.asarray(depth).astype(float)
temp  = np.asarray(temp).astype(float)
salt  = np.asarray(salt).astype(float)

print("CTD profile length:", depth.size, "levels")
print("Depth range:", depth.min(), "to", depth.max(), "m")

# ============================================================
# 2. TEOS-10 conversions: pressure, SA, CT, density
# ============================================================
# depth is positive-down; gsw.p_from_z expects z negative below sea surface
p = gsw.p_from_z(-depth, lat_ctd)          # pressure (dbar), same size as depth

SA = gsw.SA_from_SP(salt, p, lon_ctd, lat_ctd)  # Absolute Salinity (g/kg)
CT = gsw.CT_from_t(SA, temp, p)                 # Conservative Temperature (degC)

rho = gsw.rho(SA, CT, p)   # in-situ density (kg/m^3), 1D

print("Density range:", rho.min(), "to", rho.max(), "kg/m^3")

# ============================================================
# 3. Brunt-Vaisala frequency N^2
# ============================================================
N2, p_mid = gsw.Nsquared(SA, CT, p, lat_ctd)  # N2 at midpoints
# Interpolate N2 back onto original depth grid
depth_mid = np.interp(p_mid, p, depth)   # depth at midpoints
N2_depth  = np.interp(depth, depth_mid, N2)  # N2 at each depth

# Clean up any NaNs / negatives (for simple parameterizations)
N2_depth = np.where(np.isfinite(N2_depth), N2_depth, np.nan)
print("N2 range (raw):", np.nanmin(N2_depth), "to", np.nanmax(N2_depth), "s^-2")

# ============================================================
# 4. Choose plume density and find neutral buoyancy depth
# ============================================================
# EXAMPLE CHOICE:
#   Here we choose a plume density slightly higher than surface,
#   but you should set this based on your mining plume scenario.
# For a first try, you can pick something between rho at 500–2000 m.

# Example: use density at some reference depth, e.g. ~800 m
ref_depth = 800.0  # m (edit as needed)
rho_ref = np.interp(ref_depth, depth, rho)
rho_plume = rho_ref + 0.1  # plume slightly denser than ambient at 800 m

print("Reference depth:", ref_depth, "m")
print("Ambient rho at ref depth:", rho_ref, "kg/m^3")
print("Assumed plume density rho_plume:", rho_plume, "kg/m^3")

# To find neutral depth: solve rho(depth) = rho_plume
# Ensure monotonicity / sorting for interpolation
mask = np.isfinite(rho) & np.isfinite(depth)
rho_valid   = rho[mask]
depth_valid = depth[mask]

# Sort by rho so np.interp behaves well even if depth is not strictly monotonic
sort_idx = np.argsort(rho_valid)
rho_sorted   = rho_valid[sort_idx]
depth_sorted = depth_valid[sort_idx]

# Clip rho_plume into range to avoid extrapolation warnings
rho_plume_clipped = np.clip(rho_plume, rho_sorted.min(), rho_sorted.max())
depth_neutral = np.interp(rho_plume_clipped, rho_sorted, depth_sorted)

print("Estimated neutral buoyancy depth:", depth_neutral, "m")

# ============================================================
# 5. Simple vertical diffusivity profile Kz(z)
# ============================================================
# This is a VERY simple parameterization just for the assignment:
#   - Where stratification is strong (large N2), Kz is small
#   - Where stratification is weak (small N2), Kz is larger
# Clip N2 to avoid zero/negative values
N2_safe = np.where(N2_depth > 1e-8, N2_depth, 1e-8)

# Base Kz scale (you can tune this)
Kz_base = 1e-5  # m^2/s, typical interior ocean
# Make Kz inversely related to sqrt(N2) (completely ad hoc but smooth)
Kz_depth = Kz_base * (N2_safe[0] / N2_safe)**0.5
Kz_depth = np.clip(Kz_depth, 1e-5, 1e-3)  # limit to a reasonable range

print("Kz range:", np.nanmin(Kz_depth), "to", np.nanmax(Kz_depth), "m^2/s")

# Kz at neutral depth
Kz_plume = np.interp(depth_neutral, depth, Kz_depth)
print("Estimated Kz at neutral depth:", Kz_plume, "m^2/s")

# ============================================================
# 6. Vertical spread sigma_z(t) for the plume around neutral depth
# ============================================================
# For pure diffusion: sigma_z(t) ≈ sqrt(2 Kz t)
t_hours = np.linspace(0, 72, 13)   # 0 to 72 h in 6 h steps
t_sec   = t_hours * 3600.0

sigma_z = np.sqrt(2.0 * Kz_plume * t_sec)   # m

print("\nApprox vertical spread (1-sigma) around neutral depth:")
for th, sz in zip(t_hours, sigma_z):
    print(f"  t = {th:4.1f} h  ->  sigma_z ≈ {sz:6.1f} m")

# Example: you can also get 2-sigma thickness ~ 4 * Kz * t**0.5, etc.
# For report, you might say:
#   "After 24 h, vertical spreading (1-sigma) is ~X m around Z_neutral."

#I have vertical velocity now
