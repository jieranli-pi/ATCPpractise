#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:20:38 2025

@author: jieranli
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# --- Load ADCP ---
adcp = xr.open_zarr("results/adcp.zarr")
#print(adcp)

# --- Load CTD ---
ctd = xr.open_zarr("results/ctd.zarr")
#print(ctd)



### plot ctd at all stations
for i in range(0, 12):
    traj_id = i

    T = ctd['temperature'].isel(trajectory=traj_id)
    S = ctd['salinity'].isel(trajectory=traj_id)
    z_ctd = ctd['z'].isel(trajectory=traj_id)

    # Convert depth to negative values (0 at top, -downward)
    # If your z is already positive downward, we flip sign:
    depth = -z_ctd

    fig, ax1 = plt.subplots(figsize=(6, 8))

    # Temperature (left x-axis)
    ax1.plot(T, depth, color='tab:red', label='Temperature (°C)')
    ax1.set_xlabel("Temperature (°C)", color='tab:red')
    ax1.set_ylabel("Depth (m)")
    ax1.tick_params(axis='x', labelcolor='tab:red')

    # Now invert so that 0 is at the top visually
    ax1.invert_yaxis()

    # Salinity (right x-axis)
    ax2 = ax1.twiny()
    ax2.plot(S, depth, color='tab:blue', label='Salinity (psu)')
    ax2.set_xlabel("Salinity (psu)", color='tab:blue')
    ax2.tick_params(axis='x', labelcolor='tab:blue')

    plt.title(f"CTD Temperature & Salinity Profile — Trajectory {traj_id}")
    plt.tight_layout()
    plt.show()

###interpolate adcp data and plot surface velocity
# Convert to numpy arrays
lon = adcp['lon'].values      # (trajectory, obs)
lat = adcp['lat'].values
z   = adcp['z'].values
u   = adcp['U'].values
v   = adcp['V'].values

# mask for near-surface: z > -50 m  (adjust if needed)
surf_mask = z > -50

lon_s = lon[surf_mask]
lat_s = lat[surf_mask]
u_s   = u[surf_mask]
v_s   = v[surf_mask]
speed_s = np.sqrt(u_s**2 + v_s**2)

from scipy.interpolate import griddata

# Define grid spanning the region actually visited
lon_min, lon_max = lon_s.min(), lon_s.max()
lat_min, lat_max = lat_s.min(), lat_s.max()

xi = np.linspace(lon_min, lon_max, 60)
yi = np.linspace(lat_min, lat_max, 60)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolate u, v, speed onto grid
Ui = griddata((lon_s, lat_s), u_s, (Xi, Yi), method='cubic')
Vi = griddata((lon_s, lat_s), v_s, (Xi, Yi), method='cubic')
Si = np.sqrt(Ui**2 + Vi**2)

# Ship trajectory (take any depth bin; lon/lat are same across depths at each time)
lon_track = adcp['lon'].isel(trajectory=0).values
lat_track = adcp['lat'].isel(trajectory=0).values

# Subsample grid for arrows (every 5th point)
step = 5
Xq = Xi[::step, ::step]
Yq = Yi[::step, ::step]
Uq = Ui[::step, ::step]
Vq = Vi[::step, ::step]

plt.figure(figsize=(8,6))
pcm = plt.pcolormesh(Xi, Yi, Si, shading='auto', cmap='Blues')
plt.colorbar(pcm, label='Surface speed (m/s)')

# direction arrows on top
plt.quiver(
    Xq, Yq, Uq, Vq,
    color='k',
    scale=0.4,              # bigger arrows
    scale_units='xy',
    width=0.004,          # thicker arrows
    alpha=0.85
)

# ship track
plt.plot(lon_track, lat_track, 'r-', linewidth=2, label='Ship trajectory')

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Surface velocity field with direction + cruise track')
plt.legend()
plt.tight_layout()
plt.show()