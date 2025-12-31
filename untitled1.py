# ============================================================
# 3. Plot vertical slice (depth 0 at top, deep at bottom)
# ============================================================

# Ensure depth increases downward
# This forces z = 0 at top, max depth at bottom
Z_fixed = Z.copy()
if np.nanmean(Z_fixed[:5, :]) > np.nanmean(Z_fixed[-5:, :]):
    # Depth is upside-down → flip it
    Z_fixed = Z_fixed[::-1, :]
    speed_plot = speed[::-1, :]
else:
    speed_plot = speed

plt.figure(figsize=(10, 5))

pcm = plt.pcolormesh(
    X, Z_fixed, speed_plot,
    shading="auto",
    cmap="RdBu_r"
)

# Do NOT invert axis anymore — we fixed orientation ourselves
# plt.gca().invert_yaxis()   # REMOVE THIS

cbar = plt.colorbar(pcm)
cbar.set_label("Speed |U,V| (m/s)")

plt.xlabel("Distance along track (km)")
plt.ylabel("Depth (m)")
plt.title("ADCP Speed Profile")
plt.tight_layout()
plt.show()
