#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 13:18:51 2025

@author: jieranli
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ============================================================
# 1. Load ADCP data
# ============================================================
adcp = xr.open_zarr("results/adcp.zarr")


lon = adcp["lon"].values       # (trajectory, obs)
lat = adcp["lat"].values
z   = adcp["z"].values         # depth (m, likely negative)
u   = adcp["U"].values         # eastward velocity (m/s)
v   = adcp["V"].values         # northward velocity (m/s)

# ============================================================
# 2. Select velocities around plume depth and build COARSER grid
# ============================================================

# mask for near-surface: z > -50 m  (adjust if needed)
surf_mask = z > -50

lon_d = lon[surf_mask]
lat_d = lat[surf_mask]
u_d   = u[surf_mask]
v_d   = v[surf_mask]
speed_s = np.sqrt(u_d**2 + v_d**2)


plume_depth = 800    # m
depth_band  = 50.0     # +/- around plume depth

mask_depth = np.abs(z + plume_depth) < depth_band

#lon_d = lon[mask_depth]
#lat_d = lat[mask_depth]
#u_d   = u[mask_depth]
#v_d   = v[mask_depth]

valid = np.isfinite(lon_d) & np.isfinite(lat_d) & np.isfinite(u_d) & np.isfinite(v_d)
lon_d, lat_d, u_d, v_d = lon_d[valid], lat_d[valid], u_d[valid], v_d[valid]

lon_min, lon_max = lon_d.min(), lon_d.max()
lat_min, lat_max = lat_d.min(), lat_d.max()

# ---- COARSE grid: ~100x100 cells ----
nx, ny = 100, 100
xi = np.linspace(lon_min, lon_max, nx)
yi = np.linspace(lat_min, lat_max, ny)
Xi, Yi = np.meshgrid(xi, yi)
ny, nx = Xi.shape
print(f"Grid size: nx={nx}, ny={ny}")

# Interpolate u, v onto regular grid
Ui = griddata((lon_d, lat_d), u_d, (Xi, Yi), method="linear")
Vi = griddata((lon_d, lat_d), v_d, (Xi, Yi), method="linear")
Ui = np.nan_to_num(Ui, nan=0.0)
Vi = np.nan_to_num(Vi, nan=0.0)

# ============================================================
# 3. Set up advection–diffusion model
# ============================================================
lat0 = float(np.mean(Yi))
dx_m = 111_000.0 * np.cos(np.deg2rad(lat0)) * (Xi[0,1] - Xi[0,0])
dy_m = 111_000.0 * (Yi[1,0] - Yi[0,0])
print(f"dx ≈ {dx_m/1000:.2f} km, dy ≈ {dy_m/1000:.2f} km")

Kh = 10.0           # m^2/s (small diffusion)
dt = 10.0          # s
t_end = 24 *31 * 3600  # 1 day
n_steps = int(t_end / dt)
print(f"Running {n_steps} steps of {dt}s (~{t_end/3600:.1f} h)")

# ============================================================
# 4. Initial concentration field
# ============================================================
C = np.zeros((ny, nx))

# mining site
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
    lap[1:-1,1:-1] = (
        (F[1:-1,2:] - 2*F[1:-1,1:-1] + F[1:-1,:-2]) / dx_m**2 +
        (F[2:,1:-1] - 2*F[1:-1,1:-1] + F[:-2,1:-1]) / dy_m**2
    )
    # zero-gradient boundaries
    lap[0,:]  = lap[1,:]
    lap[-1,:] = lap[-2,:]
    lap[:,0]  = lap[:,1]
    lap[:,-1] = lap[:,-2]
    return lap

# ============================================================
# 6. Time stepping: advection + diffusion
# ============================================================
snap_hours = [0.17, 1, 6, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288, 312, 336, 360, 384, 408, 432, 456, 480, 504, 528, 552, 576, 600, 624, 648, 672, 696, 720]     # 10 min, 1 h, 6 h, 24 h
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
# 7. Plot snapshots in green (dark=high -> light=low)
# ============================================================
for n, Csnap in snapshots.items():
    hours = n * dt / 3600.0
    Cplot = Csnap / Csnap.max()  # normalise 0–1 so shape is visible

    plt.figure(figsize=(6,5))
    pcm = plt.pcolormesh(Xi, Yi, Cplot, shading="auto",
                         cmap="Greens", vmin=0, vmax=1)
    plt.colorbar(pcm, label="Relative concentration")
    plt.contour(Xi, Yi, Cplot, levels=[0.1, 0.5], colors="k", linewidths=0.6)
    plt.plot(lon0, lat0, "ko", markersize=5, markeredgecolor="w", label="Source")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Plume concentration after {hours:.2f} hours")
    plt.legend()
    plt.tight_layout()
    plt.show()

# new plot
# Subsample grid for arrows (every 5th point)
# Ship trajectory (take any depth bin; lon/lat are same across depths at each time)
lon_track = adcp['lon'].isel(trajectory=0).values
lat_track = adcp['lat'].isel(trajectory=0).values
step = 5
Xq = Xi[::step, ::step]
Yq = Yi[::step, ::step]
Uq = Ui[::step, ::step]
Vq = Vi[::step, ::step]

for n, Csnap in snapshots.items():

    hours = n * dt / 3600.0
    Cplot = Csnap / Csnap.max()   # normalize so plume shape is visible

    plt.figure(figsize=(7,6))

    # ---- plume background (pcolormesh) ----
    pcm = plt.pcolormesh(
        Xi, Yi, Cplot,
        shading="auto",
        cmap="Greens",
        vmin=0, vmax=1
    )
    plt.colorbar(pcm, label="Relative concentration")

    # ---- flow arrows ----
    plt.quiver(
        Xq, Yq, Uq, Vq,
        color='k',
        scale=0.4,
        scale_units='xy',
        width=0.004,
        alpha=0.85
    )

    # ---- ship track ----
    plt.plot(lon_track, lat_track, color='0.5', linewidth=2, label='Ship trajectory')

    # ---- source marker ----
    plt.plot(lon0, lat0, "ko", markersize=5, markeredgecolor="w", label="Source")
    # ---- plume contours ----
    plt.contour(Xi, Yi, Cplot, levels=[0.1, 0.5], colors='red', linewidths=1.5)

    # ---- labels & formatting ----
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Combined plot: plume + flow + ship track ({hours:.2f} hrs)")
    plt.legend()
    plt.tight_layout()
    plt.show()

##g
import imageio
from matplotlib import pyplot as plt


frames = []  # store filenames of generated frames

for n, Csnap in snapshots.items():

    hours = n * dt / 3600.0
    Cplot = Csnap / Csnap.max()

    fig = plt.figure(figsize=(7,6))

    # ---- plume background ----
    pcm = plt.pcolormesh(Xi, Yi, Cplot, shading="auto",
                         cmap="Greens", vmin=0, vmax=1)
    plt.colorbar(pcm, label="Relative concentration")

    # ---- flow arrows ----
    plt.quiver(Xq, Yq, Uq, Vq, color='k', scale=0.4,
               scale_units='xy', width=0.004, alpha=0.85)

    # ---- ship track ----
    plt.plot(lon_track, lat_track, color='0.5', linewidth=2, label='Ship trajectory')

    # ---- source marker ----
    plt.plot(lon0, lat0, "ko", markersize=5, markeredgecolor="w", label="Source")

    # ---- plume contours ----
    plt.contour(Xi, Yi, Cplot, levels=[0.1, 0.5],
                colors='red', linewidths=1.5)

    # ---- labels ----
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Plume simulation linear({hours:.2f} hrs)")
    plt.legend()
    plt.tight_layout()

    # Save frame
    fname = f"frame_{n:04d}.png"
    plt.savefig(fname, dpi=120)
    frames.append(fname)

    plt.close(fig)

with imageio.get_writer("plume_animation_800_linear.gif", mode="I", duration=0.3) as writer:
    for f in frames:
        image = imageio.imread(f)
        writer.append_data(image)

