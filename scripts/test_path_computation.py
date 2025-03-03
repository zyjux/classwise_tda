import gudhi
import matplotlib.pyplot as plt
import xarray as xr

from classwise_tda import pathwise_tda

# Open data
ds = xr.open_dataset("data/two_ring_synth_dataset.nc")

# Create simplicial complexes
unified_points = xr.concat([ds["inner_ring"], ds["outer_ring"]], dim="index")
inner_slice = slice(0, ds["inner_ring"].sizes["index"])
outer_slice = slice(
    ds["inner_ring"].sizes["index"],
    ds["inner_ring"].sizes["index"] + ds["outer_ring"].sizes["index"],
)
complexes = pathwise_tda.create_classwise_complexes(
    unified_points.data, {"inner": inner_slice, "outer": outer_slice}
)
inner_complex = complexes["inner"]
outer_complex = complexes["outer"]
union_complex = complexes["inner_U_outer"]

# Compute PH
inner_persistence = inner_complex.persistence()
outer_persistence = outer_complex.persistence()
union_persistence = union_complex.persistence()

# Choose a value at which to union
alpha = 1.0

# Filtrations are generator returning a tuple ([vertices], filtration_value)
inner_filtration = inner_complex.get_filtration()
outer_filtration = outer_complex.get_filtration()

path_complex = pathwise_tda.step_func_path_complex(outer_complex, union_complex, alpha)

# Compute new PH
path_persistence = path_complex.persistence()

F, ax = plt.subplots(1, 4, figsize=(16, 5))
gudhi.plot_persistence_diagram(persistence=inner_persistence, axes=ax[0])
ax[0].set_title("Inner ring")
gudhi.plot_persistence_diagram(persistence=outer_persistence, axes=ax[1])
ax[1].set_title("Outer ring")
gudhi.plot_persistence_diagram(persistence=union_persistence, axes=ax[2])
ax[2].set_title("Union")
gudhi.plot_persistence_diagram(persistence=path_persistence, axes=ax[3])
ax[3].set_title("Path")
F.savefig("two_ring_persistence.png")
