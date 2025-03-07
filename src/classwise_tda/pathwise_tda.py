"""Utilities to compute persistent homology along arbitrary paths"""

from itertools import chain, combinations
from typing import Optional

import gudhi
import numpy as np
from scipy.spatial.distance import cdist


def powerset(iterable) -> chain:
    """Helper function that returns powerset of input (without emptyset)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def create_classwise_complexes(
    data_points: np.ndarray,
    class_slices: dict[str, slice],
    max_dim: Optional[int] = None,
) -> dict[str, gudhi.SimplexTree]:
    """Creates filtered simplicial complexes for each class and combination of classes

    Arguments
    ----------
    data_points : numpy.ndarray
    (m, n) array of data points, where m is the number of points and n is the ambient
    dimension.

    class_slices : dictionary of the form str:slice
    Dictionary  where the key is the name of the class and the value is the slice
    indicating which points belong to that class. These slices do not need to be
    disjoint (i.e., a single point can belong to more than one class).

    max_dim : int or None
    Maximum dimension to which the simplicial complex should be expanded to. If None,
    defaults to n from data_points.

    Returns
    ----------
    dictionary of the form str:gudhi.SimplexTree
    Dictionary where the key indicates the class or union of classes and the value is
    the filtered simplicial complex for that class or union of classes.
    """

    # Set up
    if len(data_points.shape) != 2:
        raise ValueError("data_points must be an (m, n) array.")
    if max_dim is None:
        max_dim = data_points.shape[1]
    distance_matrix = cdist(data_points, data_points)
    # Compute data radius so we can ignore infinite-length edges
    data_radius = distance_matrix.max()
    class_combos = powerset(class_slices.keys())
    classwise_complexes = dict()

    # Iterate through combinations of classes
    for class_combo in class_combos:
        class_name = "_U_".join(class_combo)
        class_distances = distance_matrix.copy()
        missing_classes = [a for a in class_slices.keys() if a not in class_combo]
        for this_class in missing_classes:
            this_slice = class_slices[this_class]
            class_distances[this_slice, :] = np.inf
            class_distances[:, this_slice] = np.inf
        # We use max_filtration to ignore the inf-length edges in the distance matrix
        class_simplex = gudhi.SimplexTree.create_from_array(
            class_distances, max_filtration=data_radius
        )
        class_simplex.expansion(max_dim)  # type: ignore
        classwise_complexes[class_name] = class_simplex
    return classwise_complexes


def step_func_path_complex(
    base_complex: gudhi.SimplexTree,
    union_complex: gudhi.SimplexTree,
    alpha: float,
) -> gudhi.SimplexTree:
    """Computes filtered simplicial complex for path stepping at filtration value alpha

    Arguments
    ----------
    base_complex : gudhi.SimplexTree
    Filtered simplicial complex for the starting class.

    union_complex : gudhi.SimplexTree
    Filtered simplicial complex for the union of the starting class (base_complex) with
    some other class.

    alpha : float
    Filtration value at which to step from computing PH along base_complex to computing
    along union_complex.

    Returns
    ----------
    gudhi.SimplexTree
    Filtered simplicial complex for the path traveling along base_complex up to alpha
    and along union_complex after alpha. Note that only simplices from union_complex
    with finite filtration values will be inserted.
    """

    path_complex = base_complex.copy()
    for simplex, filt_val in union_complex.get_filtration():
        if filt_val < alpha:
            filt_val = alpha
        if np.isfinite(filt_val):
            _ = path_complex.insert(simplex, filt_val)

    return path_complex


def arbitrary_path(
    list_of_complexes: list[gudhi.SimplexTree], list_of_steps: list[float]
) -> gudhi.SimplexTree:
    """Computes filtered simplicial complex for an arbitrary path of complexes

    Arguments
    ---
    list_of_complexes : list of gudhi.SimplexTree
    List of filtered simplicial complexes to pass through in order.

    list_of_steps : list of floats
    List of filtration values at which to step between complexes. Must be increasing.

    Returns
    ---
    gudhi.SimplexTree : Filtered simplicial complex representing starting in
    list_of_complexes[0], stepping to list_of_complexes[1] at filtration value
    list_of_steps[0], and so forth until ending in list_of_complexes[-1].
    """

    if len(list_of_steps) != len(list_of_complexes) - 1:
        raise ValueError(
            "List of complexes must be 1 longer than list of steps: "
            f"got {len(list_of_complexes)} and {len(list_of_steps)}."
        )

    if list_of_steps != sorted(list_of_steps):
        raise ValueError("List of steps must be increasing.")

    path_complex = list_of_complexes[0]
    for i, alpha in enumerate(list_of_steps):
        path_complex = step_func_path_complex(
            path_complex, list_of_complexes[i + 1], alpha
        )
    return path_complex


def extract_filt_values_from_persistence(simplicial_complex: gudhi.SimplexTree):
    """Extract all unique filtration values that are a birth or death value"""
    max_dim = simplicial_complex.upper_bound_dimension()
    if not simplicial_complex._is_persistence_defined():
        simplicial_complex.compute_persistence()
    for i in range(max_dim + 1):
        try:
            birth_death_array = np.concatenate(
                [
                    birth_death_array,
                    simplicial_complex.persistence_intervals_in_dimension(i),
                ],
                axis=0,
            )
        except NameError:
            birth_death_array = simplicial_complex.persistence_intervals_in_dimension(i)
    return np.unique(birth_death_array)
