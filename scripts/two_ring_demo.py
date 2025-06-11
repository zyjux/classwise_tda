"""Demo script to compute P-landscapes for inner/outer ring dataset"""

import xarray as xr

from classwise_tda import poset_landscapes, visualization

SAMPLING_RATIO = 2

ds = xr.open_dataset("/nfs/home/lverho/classwise_tda/data/two_ring_synth_dataset.nc")

unified_points = xr.concat([ds["inner_ring"], ds["outer_ring"]], dim="index")
unified_points = unified_points.isel({"index": slice(None, None, SAMPLING_RATIO)})
print(f"Number of points {unified_points.sizes['index']}")
A_slice = slice(0, int(ds["inner_ring"].sizes["index"] / SAMPLING_RATIO))
B_slice = slice(
    int(ds["inner_ring"].sizes["index"] / SAMPLING_RATIO),
    int(
        (ds["inner_ring"].sizes["index"] + ds["outer_ring"].sizes["index"])
        / SAMPLING_RATIO
    ),
)
weights = {
    (("inner",), ("inner", "outer")): 0.5,
    (("outer",), ("inner", "outer")): 0.5,
}
poset_graph, inclusion_graph = poset_landscapes.compute_classwise_landscape_poset(
    data_points=unified_points.values,
    class_slices={"inner": A_slice, "outer": B_slice},
    class_weights=weights,
    homology_coeff_field=2,
    return_inclusion_graph=True,
    output_landscape_resolution=98,
    path_landscape_resolution=100000,
)

landscape_array = poset_landscapes.create_poset_landscape_array(poset_graph)

F, ax = visualization.plot_all_landscapes(landscape_array, grid_layout=(3, 1))

F.savefig("/nfs/home/lverho/classwise_tda/figures/two_ring_landscapes_weight_0.5.png")
