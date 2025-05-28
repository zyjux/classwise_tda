import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

##### Two ring dataset


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

F.savefig("/nfs/home/lverho/classwise_tda/figures/two_ring_data_visualization.png")

# Set up dataset and save
inner_ring = xr.DataArray(
    inner_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
outer_ring = xr.DataArray(
    outer_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
ds = xr.Dataset({"inner_ring": inner_ring, "outer_ring": outer_ring})

ds.to_netcdf("/nfs/home/lverho/classwise_tda/data/two_ring_synth_dataset.nc")


##### Two mixed rings dataset

# Set up random initializer
rng = np.random.default_rng()

# Create top ring
top_angles = rng.uniform(0, 2 * np.pi, size=100)
top_radii = rng.uniform(0.8, 1.2, size=100)

top_ring_points = np.stack(
    [top_radii * np.cos(top_angles), top_radii * np.sin(top_angles)]
).transpose()

# Create outer ring
bottom_angles = rng.uniform(0, 2 * np.pi, size=100)
bottom_radii = rng.uniform(0.8, 1.2, size=100)

bottom_ring_points = np.stack(
    [bottom_radii * np.cos(bottom_angles), bottom_radii * np.sin(bottom_angles)]
).transpose()

# Visualize data
F, ax = plt.subplots(1, 1)
ax.scatter(top_ring_points[:, 0], top_ring_points[:, 1], color="blue")
ax.scatter(bottom_ring_points[:, 0], bottom_ring_points[:, 1], color="orange")

F.savefig("/nfs/home/lverho/classwise_tda/figures/mixed_rings_data_visualization.png")

# Set up dataset and save
top_ring = xr.DataArray(
    top_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
bottom_ring = xr.DataArray(
    bottom_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
ds = xr.Dataset({"top_ring": top_ring, "bottom_ring": bottom_ring})

ds.to_netcdf("/nfs/home/lverho/classwise_tda/data/mixed_rings_synth_dataset.nc")


##### Two half rings dataset

# Set up random initializer
rng = np.random.default_rng()

# Create top ring
top_angles = rng.uniform(0, np.pi, size=100)
top_radii = rng.uniform(0.8, 1.2, size=100)

top_ring_points = np.stack(
    [top_radii * np.cos(top_angles), top_radii * np.sin(top_angles)]
).transpose()

# Create outer ring
bottom_angles = rng.uniform(np.pi, 2 * np.pi, size=100)
bottom_radii = rng.uniform(0.8, 1.2, size=100)

bottom_ring_points = np.stack(
    [bottom_radii * np.cos(bottom_angles), bottom_radii * np.sin(bottom_angles)]
).transpose()

# Visualize data
F, ax = plt.subplots(1, 1)
ax.scatter(top_ring_points[:, 0], top_ring_points[:, 1], color="blue")
ax.scatter(bottom_ring_points[:, 0], bottom_ring_points[:, 1], color="orange")

F.savefig("/nfs/home/lverho/classwise_tda/figures/half_rings_data_visualization.png")

# Set up dataset and save
top_ring = xr.DataArray(
    top_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
bottom_ring = xr.DataArray(
    bottom_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
ds = xr.Dataset({"top_ring": top_ring, "bottom_ring": bottom_ring})

ds.to_netcdf("/nfs/home/lverho/classwise_tda/data/half_rings_synth_dataset.nc")

##### One ring dataset

# Set up random initializer
rng = np.random.default_rng()

# Create top ring
top_angles = rng.uniform(0, 2 * np.pi, size=100)
top_radii = rng.uniform(0.8, 1.2, size=100)

top_ring_points = np.stack(
    [top_radii * np.cos(top_angles), top_radii * np.sin(top_angles)]
).transpose()

# Visualize data
F, ax = plt.subplots(1, 1)
ax.scatter(top_ring_points[:, 0], top_ring_points[:, 1], color="blue")

F.savefig("/nfs/home/lverho/classwise_tda/figures/single_ring_data_visualization.png")

# Set up dataset and save
ring = xr.DataArray(
    top_ring_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
ds = xr.Dataset({"ring": ring})

ds.to_netcdf("/nfs/home/lverho/classwise_tda/data/single_ring_synth_dataset.nc")

##### One disc dataset

# Set up random initializer
rng = np.random.default_rng()

# Create top ring
angles = rng.uniform(0, 2 * np.pi, size=100)
radii = rng.uniform(0.0, 1.2, size=100)

disc_points = np.stack([radii * np.cos(angles), radii * np.sin(angles)]).transpose()

# Visualize data
F, ax = plt.subplots(1, 1)
ax.scatter(disc_points[:, 0], disc_points[:, 1], color="blue")

F.savefig("/nfs/home/lverho/classwise_tda/figures/single_disc_data_visualization.png")

# Set up dataset and save
disc = xr.DataArray(
    disc_points, dims=("index", "coords"), coords={"coords": ["x", "y"]}
)
ds = xr.Dataset({"disc": disc})

ds.to_netcdf("/nfs/home/lverho/classwise_tda/data/single_disc_synth_dataset.nc")
