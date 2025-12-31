#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plume model using bottom (deepest 50 m) flow from ADCP,
plus a vertical slice of ADCP velocities.

@author: jieranli
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ============================================================
# 1. Load ADCP data
# ============================================================
adcp = xr.open_zarr("results/adcp.zarr")
fil
lon = adcp["lon"].values       # expected shape: (trajectory, obs)
lat = adcp["lat"].values       # expected shape: (trajectory, obs)
z   = adcp["z"].values         # depth (nz,)
u   = adcp["U"].values         # shape: (nz, trajectory, obs)
v   = adcp["V"].values         # shape: (nz, trajectory, obs)

print("lon shape:", lon.shape)
print("lat shape:", lat.shape)
print("z shape:", z.shape)
print("u shape:", u.shape)
print("v shape:", v.shape)

# ============================================================
# 2. Select velocities in the DEEPEST 50 m and build COARSER grid
# ============================================================
depth_band = 50.0  # thickness of deepest layer (m)

z_arr = np.asarray(z)
z_min, z_max = z_arr.min(), z_arr.max()

# Handle both sign conventions (negative-down or positive-down)
if z_max <= 0:
    # depths are negative: 0, -10, -20, ...  (deepest = most negative)
    deepest_depth = z_min
    mask_depth = (z_arr >= deepest_depth) & (z_arr <= deepest_depth + depth_band)
else:
    # depths are positive: 0, 10, 20, ...  (deepest = largest)
    deepest_depth = z_max
    mask_depth = (z_arr <= deepest_depth) & (z_arr >= deepest_depth - depth_band)

print(f"Using deepest layer from ~{deepest_depth:.1f} m over a {depth_band:.1f} m band")

# Average U, V over that deepest band → bottom velocities (no depth dimension)
u_bottom = np.nanmean(u[mask_depth, :, :], axis=0)   # shape (trajectory, obs)
v_bottom = np.nanmean(v[mask_depth, :, :], axis=0)   # shape (trajectory, obs)

# Flatten for interpolation
lon_flat = lon.ravel()
lat_flat = lat.ravel()
u_flat   = u_bottom.ravel()
v_flat   = v_bottom.ravel()

# Remove NaNs
valid = np.isfinite(lon_flat) & np.isfinite(lat_flat) & np.isfinite(u_flat) & np.isfinite(v_flat)
lon_d, lat_d, u_d, v_d = lon_flat[valid], lat_flat[valid], u_flat[valid], v_flat[valid]

print("Number of valid bottom points:", lon_d.size)

# Domain bounds for grid
lon_min, lon_max = lon_d.min(), lon_d.max()
lat_min, lat_max = lat_d.min(), lat_d.max()

# ---- COARSE grid: ~100x100 cells ----
nx, ny = 100, 100
xi = np.linspace(lon_min, lon_max, nx)
yi = np.linspace(lat_min, lat_max, ny)
Xi, Yi = np.meshgrid(xi, yi)
ny, nx = Xi.shape
print(f"Grid size: nx={nx}, ny={ny}")

# Interpolate bottom u, v onto regular grid
Ui = griddata((lon_d, lat_d), u_d, (Xi, Yi), method="linear")
Vi = griddata((lon_d, lat_d), v_d, (Xi, Yi), method="linear")
Ui = np.nan_to_num(Ui, nan=0.0)
Vi = np.nan_to_num(Vi, nan=0.0)

# ============================================================
# 3. Set up advection–diffusion model (2D, depth-averaged / bottom-flow-driven)
# ============================================================
lat0_grid = float(np.mean(Yi))
dx_m = 111_000.0 * np.cos(np.deg2rad(lat0_grid)) * (Xi[0, 1] - Xi[0, 0])
dy_m = 111_000.0 * (Yi[1, 0] - Yi[0, 0])
print(f"dx ≈ {dx_m/1000:.2f} km, dy ≈ {dy_m/1000:.2f} km")

Kh = 10.0           # m^2/s (small diffusion)
dt = 10.0           # s
t_end = 24 * 31 * 3600  # ~31 days
n_steps = int(t_end / dt)
print(f"Running {n_steps} steps of {dt}s (~{t_end/3600:.1f} h)")

# ============================================================
# 4. Initial concentration field (2D)
# ============================================================
C = np.zeros((ny, nx))

# Mining site (given in lon/lat)
lon0 = 10.623317
lat0 = 73.097772

# nearest grid indices
i0 = np.argmin(np.abs(Xi[0, :] - lon0))
j0 = np.argmin(np.abs(Yi[:, 0] - lat0))
print("Source grid indices:", j0, i0)

# Gaussian blob (a few grid cells wide)
sigma_cells = 2.0
Y_idx, X_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
C0 = np.exp(-(((X_idx - i0)**2 + (Y_idx - j0)**2) / (2 * sigma_cells**2)))
C = C0 / C0.max()   # max concentration = 1

# ============================================================
# 5. Laplacian for diffusion
# ============================================================
def laplacian(F):
    lap = np.zeros_like(F)
    lap[1:-1, 1:-1] = (
        (F[1:-1, 2:] - 2*F[1:-1, 1:-1] + F[1:-1, :-2]) / dx_m**2 +
        (F[2:, 1:-1] - 2*F[1:-1, 1:-1] + F[:-2, 1:-1]) / dy_m**2
    )
    # zero-gradient boundaries
    lap[0, :]  = lap[1, :]
    lap[-1, :] = lap[-2, :]
    lap[:, 0]  = lap[:, 1]
    lap[:, -1] = lap[:, -2]
    return lap

# ============================================================
# 6. Time stepping: advection + diffusion
# ============================================================
snap_hours = [
    0.17, 1, 6, 24, 48, 72, 96, 120, 144, 168, 192, 216,
    240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480,
    504, 528, 552, 576, 600, 624, 648, 672, 696, 720
]  # in hours
snap_indices = [int(h*3600/dt) for h in snap_hours]
snapshots = {0: C.copy()}

for n in range(1, n_steps+1):

    Cx_plus  = np.roll(C, -1, axis=1)
    Cx_minus = np.roll(C,  1, axis=1)
    Cy_plus  = np.roll(C, -1, axis=0)
    Cy_minus = np.roll(C,  1, axis=0)

    dCdx = np.where(Ui > 0, (C - Cx_minus)/dx_m, (Cx_plus - C)/dx_m)
    dCdy = np.where(Vi > 0, (C - Cy_minus)/dy_m, (Cy_plus - C)/dy_m)

    adv_term  = -(Ui * dCdx + Vi * dCdy)
    diff_term = Kh * laplacian(C)

    C = C + dt * (adv_term + diff_term)
    C = np.clip(C, 0, 1)

    if n in snap_indices:
        snapshots[n] = C.copy()
        print(f"Snapshot step {n}, t={n*dt/3600:.2f} h, maxC={C.max():.3f}")

# ============================================================
# 7. TOP-DOWN PLOTS (plan view) – still 2D
# ============================================================
for n, Csnap in snapshots.items():
    hours = n * dt / 3600.0
    Cplot = Csnap / Csnap.max()  # normalise 0–1 so shape is visible

    plt.figure(figsize=(6, 5))
    pcm = plt.pcolormesh(Xi, Yi, Cplot, shading="auto",
                         cmap="Greens", vmin=0, vmax=1)
    plt.colorbar(pcm, label="Relative concentration")
    plt.contour(Xi, Yi, Cplot, levels=[0.1, 0.5], colors="k", linewidths=0.6)
    plt.plot(lon0, lat0, "ko", markersize=5, markeredgecolor="w", label="Source")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Plume concentration (plan view) after {hours:.2f} hours")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# 8. VERTICAL SLICE (ADCP vertical profile)
# ============================================================
"""
We cannot get a true vertical slice of the plume itself (model is 2D),
but we CAN plot a vertical slice of the ADCP velocities.

Here we plot speed |U,V| vs depth and profile index along trajectory 0.
"""

# pick first trajectory
traj_idx = 0

# U, V section: (nz, n_obs) along that trajectory
u_sec = u[:, traj_idx, :]    # shape (nz, n_obs)
v_sec = v[:, traj_idx, :]    # shape (nz, n_obs)
speed_sec = np.sqrt(u_sec**2 + v_sec**2)

nz, n_obs = speed_sec.shape
print("Vertical section shape (nz, n_obs):", speed_sec.shape)

# Create 2D grids for pcolormesh
obs_idx = np.arange(n_obs)  # profile index or time index
Obs, Z = np.meshgrid(obs_idx, z)

plt.figure(figsize=(8, 5))
pcm = plt.pcolormesh(Obs, Z, speed_sec, shading="auto", cmap="RdBu_r")
plt.gca().invert_yaxis()  # depth increasing downward
plt.colorbar(pcm, label="Speed |U,V| (m/s)")
plt.xlabel("Profile index along trajectory 0")
plt.ylabel("Depth (m)")
plt.title("ADCP vertical slice: speed vs depth along trajectory 0")
plt.tight_layout()
plt.show()
