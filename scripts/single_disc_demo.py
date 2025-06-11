"""Demo script to compute P-landscapes for two mixed ring dataset"""

import xarray as xr

from classwise_tda import poset_landscapes, visualization

SAMPLING_RATIO = 1

ds = xr.open_dataset("~/classwise_tda/data/single_disc_synth_dataset.nc")

unified_points = ds["disc"]
unified_points = unified_points.isel({"index": slice(None, None, SAMPLING_RATIO)})
print(f"Number of points {unified_points.sizes['index']}")
A_slice = slice(0, None)
weights = {}
poset_graph, inclusion_graph = poset_landscapes.compute_classwise_landscape_poset(
    data_points=unified_points.values,
    class_slices={"A": A_slice},
    class_weights=weights,
    homology_coeff_field=2,
    return_inclusion_graph=True,
    path_landscape_resolution=1000,
)

landscape_array = poset_landscapes.create_poset_landscape_array(poset_graph)

F, ax = visualization.plot_all_landscapes(landscape_array, grid_layout=(1, 1))

F.savefig("/nfs/home/lverho/classwise_tda/figures/single_disc_landscape.png")
