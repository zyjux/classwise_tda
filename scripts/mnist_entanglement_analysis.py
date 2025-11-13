import pickle
from timeit import default_timer

import gudhi
import gudhi.representations as greps
import numpy as np
from scipy.io import loadmat

from classwise_tda import poset_landscapes, setup_utils

SAMPLES_PER_DIGIT = 50
NUM_DIGITS = 10
LANDSCAPE_OUTPUT_RESOLUTION = 98
MAX_COMPLEX_DIMENSION = 3
MAX_LANDSCAPE_DIM = 2
NUM_LANDSCAPES = 5

rng = np.random.default_rng()
DATA_DIR = "/nfs/home/lverho/classwise_tda/data/mnist/"

print("Loading data")
mnist_matcontents = loadmat(DATA_DIR + "emnist-mnist", spmatrix=False)

mnist_images = mnist_matcontents["dataset"]["train"][0][0]["images"][0][0]
mnist_labels = mnist_matcontents["dataset"]["train"][0][0]["labels"][0][0]

mnist_full_dict = [
    mnist_images[np.squeeze(mnist_labels == i), :] for i in range(NUM_DIGITS)
]
mnist_sample_list = [
    rng.choice(mnist_full_dict[i], SAMPLES_PER_DIGIT, replace=False)
    for i in range(NUM_DIGITS)
]
mnist_sample_ds = np.concat(mnist_sample_list, axis=0)
class_slices = {
    f"{i}": slice(i * SAMPLES_PER_DIGIT, (i + 1) * SAMPLES_PER_DIGIT)
    for i in range(NUM_DIGITS)
}

print("Computing class weights")
class_weights = setup_utils.compute_class_distances(
    mnist_sample_ds,
    class_slices,
    distance_function=setup_utils.hausdorff_distance_computation,
    distance_scale=0.1,
)

print("Starting classwise landscape computations")
pairwise_generalized_landscape_dict = {}
pairwise_unclassified_landscape_dict = {}
pairwise_mse = {}
for i in range(NUM_DIGITS):
    for j in range(i + 1, NUM_DIGITS):
        these_data_points = np.concat(
            [mnist_sample_list[i], mnist_sample_list[j]], axis=0
        )
        these_class_slices = {
            f"{i}": slice(0, SAMPLES_PER_DIGIT),
            f"{j}": slice(SAMPLES_PER_DIGIT, 2 * SAMPLES_PER_DIGIT),
        }
        i_to_union_tuple = ((f"{i}",), (f"{i}", f"{j}"))
        j_to_union_tuple = ((f"{j}",), (f"{i}", f"{j}"))
        these_class_weights = {
            i_to_union_tuple: class_weights[i_to_union_tuple],
            j_to_union_tuple: class_weights[j_to_union_tuple],
        }
        start_time = default_timer()
        pairwise_generalized_landscape_dict[(f"{i}", f"{j}")] = (
            poset_landscapes.compute_classwise_landscape_poset(
                these_data_points,
                these_class_slices,
                these_class_weights,
                complex_max_dim=MAX_COMPLEX_DIMENSION,
                landscape_max_dim=MAX_LANDSCAPE_DIM,
                homology_coeff_field=2,
                num_landscapes=NUM_LANDSCAPES,
                output_landscape_resolution=LANDSCAPE_OUTPUT_RESOLUTION,
                path_landscape_resolution=1000,
                multiprocessing_workers=15,
            )
        )
        print(
            f"Computed generalized landscape {i} vs {j} in {default_timer() - start_time:.3f} seconds"
        )

        start_time = default_timer()
        print("Starting to compute difference from non-classified landscape")
        unified_complex = gudhi.RipsComplex(
            points=these_data_points
        ).create_simplex_tree(max_dimension=MAX_COMPLEX_DIMENSION)
        max_filtration_value = list(unified_complex.get_filtration())[-1][-1]
        unified_complex.compute_persistence(homology_coeff_field=2)
        unified_diagrams = [
            unified_complex.persistence_intervals_in_dimension(i)
            for i in range(MAX_LANDSCAPE_DIM + 1)
        ]
        unified_diagrams[0][-1, 1] = max_filtration_value
        lscape = greps.Landscape(
            num_landscapes=NUM_LANDSCAPES,
            resolution=LANDSCAPE_OUTPUT_RESOLUTION,
            keep_endpoints=True,
        )
        lscape.fit(unified_diagrams)
        unified_landscapes = lscape.transform(unified_diagrams)
        unified_landscapes = unified_landscapes.reshape(
            (MAX_LANDSCAPE_DIM + 1, NUM_LANDSCAPES, LANDSCAPE_OUTPUT_RESOLUTION),
            order="C",
        )

        generalized_landscape_array = poset_landscapes.create_poset_landscape_array(
            pairwise_generalized_landscape_dict[(f"{i}", f"{j}")]
        )
        gen_grid = generalized_landscape_array["filt_vals"].values

        interpolated_unified_landscapes = np.empty(
            unified_landscapes.shape, dtype="float"
        )

        for hom_dim in range(MAX_LANDSCAPE_DIM + 1):
            for landscape_num in range(NUM_LANDSCAPES):
                interpolated_unified_landscapes[hom_dim, landscape_num, :] = np.interp(
                    gen_grid,
                    lscape.grid_,
                    unified_landscapes[hom_dim, landscape_num, :],
                )

        pairwise_unclassified_landscape_dict[(f"{i}", f"{j}")] = (
            interpolated_unified_landscapes
        )

        pairwise_mse[(f"{i}", f"{j}")] = (
            (
                generalized_landscape_array.sel({"union": f"{i} U {j}"}).values
                - interpolated_unified_landscapes
            )
            ** 2
        ).mean()
        print(
            f"Computed standard landscape and mse in {default_timer() - start_time:.3f} seconds"
        )

output_dict = {
    "generalized_landscapes": pairwise_generalized_landscape_dict,
    "unclassified_landscapes": pairwise_unclassified_landscape_dict,
    "pairwise_mse": pairwise_mse,
}


with open(DATA_DIR + "pairwise_comparison.pkl", "wb") as file:
    pickle.dump(output_dict, file)
