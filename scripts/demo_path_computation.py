import gudhi
import matplotlib.pyplot as plt
import xarray as xr

from classwise_tda import pathwise_tda

# Open data
ds = xr.open_dataset("~/classwise_tda/data/half_rings_synth_dataset.nc")

class_A_name = "top_ring"
class_B_name = "bottom_ring"

# Create simplicial complexes
unified_points = xr.concat([ds[class_A_name], ds[class_B_name]], dim="index")
unified_points = unified_points.isel({"index": slice(None, None, 10)})
A_slice = slice(0, int(ds[class_A_name].sizes["index"] / 10))
B_slice = slice(
    int(ds[class_A_name].sizes["index"] / 10),
    int((ds[class_A_name].sizes["index"] + ds[class_B_name].sizes["index"]) / 10),
)
inclusion_graph = pathwise_tda.create_inclusion_graph([class_A_name, class_B_name])
inclusion_graph.edges[((class_A_name,), (class_A_name, class_B_name))]["weight"] = 0.0
inclusion_graph.edges[((class_B_name,), (class_A_name, class_B_name))]["weight"] = 0.0
inclusion_graph = pathwise_tda.add_classwise_complexes(
    inclusion_graph, unified_points.data, {class_A_name: A_slice, class_B_name: B_slice}
)
poset_graph = pathwise_tda.create_full_poset_graph(inclusion_graph)
A_complex = inclusion_graph.nodes[(class_A_name,)]["simplex"]
B_complex = inclusion_graph.nodes[(class_B_name,)]["simplex"]
union_complex = inclusion_graph.nodes[(class_A_name, class_B_name)]["simplex"]

# Compute PH
A_persistence = A_complex.persistence()
B_persistence = B_complex.persistence()
union_persistence = union_complex.persistence()

# Choose a value at which to union
alpha = 1.0
step_weight = 0.0

path_complex = pathwise_tda.step_func_path_complex(
    A_complex, union_complex, alpha, step_weight
)

# Compute new PH
path_persistence = path_complex.persistence()

F, ax = plt.subplots(1, 4, figsize=(16, 4))
gudhi.plot_persistence_diagram(persistence=A_persistence, axes=ax[0])
ax[0].set_title(class_A_name)
gudhi.plot_persistence_diagram(persistence=B_persistence, axes=ax[1])
ax[1].set_title(class_B_name)
gudhi.plot_persistence_diagram(persistence=union_persistence, axes=ax[2])
ax[2].set_title("Union")
gudhi.plot_persistence_diagram(persistence=path_persistence, axes=ax[3])
ax[3].set_title("Path")
F.savefig("half_rings_persistence.png")


all_landscapes = pathwise_tda.landscapes_for_all_paths(poset_graph, inclusion_graph)
