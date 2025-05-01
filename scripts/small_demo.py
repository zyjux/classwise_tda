"""Demo script to compute P-landscapes for two half ring dataset"""

import xarray as xr

from classwise_tda import poset_landscapes, visualization

SAMPLING_RATIO = 5

ds = xr.open_dataset("~/classwise_tda/data/half_rings_synth_dataset.nc")

unified_points = xr.concat([ds["top_ring"], ds["bottom_ring"]], dim="index")
unified_points = unified_points.isel({"index": slice(None, None, SAMPLING_RATIO)})

unified_points = xr.DataArray(
    [
        [0.0, 0.0],
        [0.4, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.4, 1.0],
        [1.0, 1.0],
    ],
    dims=("index", "data"),
)
print(f"Number of points {unified_points.sizes['index']}")
A_slice = slice(None, 3)
B_slice = slice(3, None)
weights = {
    (("top",), ("top", "bottom")): 0.1,
    (("bottom",), ("top", "bottom")): 0.1,
}
poset_graph, inclusion_graph = poset_landscapes.compute_classwise_landscape_poset(
    data_points=unified_points.values,
    class_slices={"top": A_slice, "bottom": B_slice},
    class_weights=weights,
    homology_coeff_field=2,
    return_inclusion_graph=True,
    path_landscape_resolution=1000,
)

landscape_array = poset_landscapes.discretize_poset_graph_landscapes(poset_graph, 100)

F, ax = visualization.plot_all_landscapes(landscape_array, grid_layout=(3, 1))

F.savefig(
    "/nfs/home/lverho/classwise_tda/figures/two_half_ring_landscapes_weight_point1.png"
)
