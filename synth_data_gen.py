import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Set up random initializer
rng = np.random.default_rng()

# Create inner ring
inner_angles = rng.uniform(0, 2 * np.pi, size=100)
inner_radii = rng.uniform(0, 1, size=100)

inner_ring_points = np.stack(
    [inner_radii * np.cos(inner_angles), inner_radii * np.sin(inner_angles)]
).transpose()

# Create outer ring
outer_angles = rng.uniform(0, 2 * np.pi, size=100)
outer_radii = rng.uniform(3, 4, size=100)

outer_ring_points = np.stack(
    [outer_radii * np.cos(outer_angles), outer_radii * np.sin(outer_angles)]
).transpose()

# Visualize data
F, ax = plt.subplots(1, 1)
ax.scatter(inner_ring_points[:, 0], inner_ring_points[:, 1], color="blue")
ax.scatter(outer_ring_points[:, 0], outer_ring_points[:, 1], color="orange")

F.savefig("data/synth_data_visualization.png")

# Set up dataset and save
inner_ring = xr.DataArray(
    inner_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
outer_ring = xr.DataArray(
    outer_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
ds = xr.Dataset({"inner_ring": inner_ring, "outer_ring": outer_ring})

ds.to_netcdf("data/two_ring_synth_dataset.nc")
