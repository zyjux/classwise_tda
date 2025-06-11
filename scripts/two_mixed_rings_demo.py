"""Demo script to compute P-landscapes for two mixed ring dataset"""

import xarray as xr

from classwise_tda import poset_landscapes, visualization

SAMPLING_RATIO = 2

ds = xr.open_dataset("~/classwise_tda/data/mixed_rings_synth_dataset.nc")

unified_points = xr.concat([ds["top_ring"], ds["bottom_ring"]], dim="index")
unified_points = unified_points.isel({"index": slice(None, None, SAMPLING_RATIO)})
print(f"Number of points {unified_points.sizes['index']}")
A_slice = slice(0, int(ds["top_ring"].sizes["index"] / SAMPLING_RATIO))
B_slice = slice(
    int(ds["top_ring"].sizes["index"] / SAMPLING_RATIO),
    int(
        (ds["top_ring"].sizes["index"] + ds["bottom_ring"].sizes["index"])
        / SAMPLING_RATIO
    ),
)
weights = {
    (("A",), ("A", "B")): 0.5,
    (("B",), ("A", "B")): 0.5,
}
poset_graph, inclusion_graph = poset_landscapes.compute_classwise_landscape_poset(
    data_points=unified_points.values,
    class_slices={"A": A_slice, "B": B_slice},
    class_weights=weights,
    homology_coeff_field=2,
    return_inclusion_graph=True,
    path_landscape_resolution=1000,
)

landscape_array = poset_landscapes.create_poset_landscape_array(poset_graph, 100)

F, ax = visualization.plot_all_landscapes(landscape_array, grid_layout=(3, 1))

F.savefig(
    "/nfs/home/lverho/classwise_tda/figures/mixed_ring_landscapes_weight_point5.png"
)
