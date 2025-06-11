"""Demo script to compute P-landscapes for simple circle"""

import numpy as np
import xarray as xr

from classwise_tda import poset_landscapes, visualization

outer_ring = (
    2
    * np.array(
        [
            np.cos(np.linspace(0, 2 * np.pi, 18, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, 18, endpoint=False)),
        ]
    ).T
)
inner_ring = (
    0.5
    * np.array(
        [
            np.cos(np.linspace(0, 2 * np.pi, 3, endpoint=False)),
            np.sin(np.linspace(0, 2 * np.pi, 3, endpoint=False)),
        ]
    ).T
)

unified_points = xr.DataArray(
    np.concat([inner_ring, outer_ring], axis=0),
    dims=("index", "data"),
)
print(f"Number of points {unified_points.sizes['index']}")
A_slice = slice(None, 4)
B_slice = slice(4, None)
weights = {
    (("inner",), ("inner", "outer")): 0.1,
    (("outer",), ("inner", "outer")): 0.1,
}
poset_graph, inclusion_graph = poset_landscapes.compute_classwise_landscape_poset(
    data_points=unified_points.values,
    class_slices={"inner": A_slice, "outer": B_slice},
    class_weights=weights,
    homology_coeff_field=2,
    return_inclusion_graph=True,
    output_landscape_resolution=98,
    path_landscape_resolution=1000,
)

landscape_array = poset_landscapes.create_poset_landscape_array(poset_graph)

F, ax = visualization.plot_all_landscapes(landscape_array, grid_layout=(3, 1))

F.savefig("/nfs/home/lverho/classwise_tda/figures/small_circle_demo.png")
