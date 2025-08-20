"""Demo script to compute P-landscapes for two half ring dataset"""

import xarray as xr

from classwise_tda import poset_landscapes, visualization

SAMPLING_RATIO = 4

ds = xr.open_dataset("~/classwise_tda/data/half_rings_synth_dataset.nc")

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
    (("top",), ("top", "bottom")): 0.5,
    (("bottom",), ("top", "bottom")): 0.5,
}
poset_graph, inclusion_graph = poset_landscapes.compute_classwise_landscape_poset(
    data_points=unified_points.values,
    class_slices={"top": A_slice, "bottom": B_slice},
    class_weights=weights,
    homology_coeff_field=2,
    return_inclusion_graph=True,
    output_landscape_resolution=98,
    path_landscape_resolution=1000,
    multiprocessing_workers=8,
)

landscape_array = poset_landscapes.create_poset_landscape_array(poset_graph)

F, ax = visualization.plot_all_landscapes(landscape_array, grid_layout=(3, 1))

F.savefig(
    "/nfs/home/lverho/classwise_tda/figures/two_half_ring_landscapes_weight_0.5.png"
)
