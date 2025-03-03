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


def step_func_path_complex(
    base_complex: gudhi.SimplexTree,
    union_complex: gudhi.SimplexTree,
    alpha: float,
    verbose: bool = False,
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

    verbose : bool
    Whether or not to print how many simplices were inserted in the union operation.
    Default False.

    Returns
    ----------
    gudhi.SimplexTree
    Filtered simplicial complex for the path traveling along base_complex up to alpha
    and along union_complex after alpha.
    """

    path_complex = base_complex.copy()
    simplices_inserted = 0
    for simplex, filt_val in union_complex.get_filtration():
        if filt_val < alpha:
            filt_val = alpha
        simplices_inserted += path_complex.insert(simplex, filt_val)

    if verbose:
        print(f"{simplices_inserted} simplices inserted")

    return path_complex


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
    if max_dim is None:
        max_dim = data_points.shape[1]
    distance_matrix = cdist(data_points, data_points)
    class_combos = powerset(class_slices.keys())
    classwise_complexes = dict()

    # Iterate through combinations of classes
    for class_combo in class_combos:
        class_name = "_U_".join(class_combo)
        class_distances = np.full(distance_matrix.shape, np.inf, dtype=float)
        for this_class in class_combo:
            this_slice = class_slices[this_class]
            class_distances[this_slice, this_slice] = distance_matrix[
                this_slice, this_slice
            ]
        class_simplex = gudhi.SimplexTree.create_from_array(class_distances)
        class_simplex.expansion(max_dim)
        classwise_complexes[class_name] = class_simplex
    return classwise_complexes
