"""Compute generalized persistence landscapes on poset persistence modules"""

import math
from operator import itemgetter
from typing import Any, Optional, Union

import gudhi
import gudhi.representations as greps
import networkx as nx
import numpy as np
import xarray as xr
from tqdm import tqdm

from . import setup_utils


def step_func_path_complex(
    base_complex: gudhi.SimplexTree,
    union_complex: gudhi.SimplexTree,
    alpha: float,
    step_weight: float,
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

    step_weight : float
    Additional value to add to filtration values in the union complex after the step.
    Can be interpreted as the "length" of the edge between the base complex and union
    complex.

    Returns
    ----------
    gudhi.SimplexTree
    Filtered simplicial complex for the path traveling along base_complex up to alpha
    and along union_complex after alpha. Note that only simplices from union_complex
    with finite filtration values will be inserted.
    """

    path_complex = base_complex.copy()
    for simplex, filt_val in path_complex.get_filtration():
        if filt_val > alpha:
            _ = path_complex.assign_filtration(simplex, filt_val + step_weight)
    for simplex, filt_val in union_complex.get_filtration():
        if filt_val < alpha:
            filt_val = alpha
        if np.isfinite(filt_val):
            _ = path_complex.insert(simplex, filt_val + step_weight)

    return path_complex


def arbitrary_path_complex(
    list_of_complexes: list[gudhi.SimplexTree],
    list_of_steps: list[float],
    list_of_step_weights: list[float],
) -> gudhi.SimplexTree:
    """Computes filtered simplicial complex for an arbitrary path of complexes

    Arguments
    ---
    list_of_complexes : list of gudhi.SimplexTree
    List of filtered simplicial complexes to pass through in order.

    list_of_steps : list of floats
    List of filtration values at which to step between complexes. Must be increasing.

    list_of_step_weights : list of floats
    List of step weights for the steps. Must have the same length as list_of_steps.

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

    if len(list_of_steps) != len(list_of_step_weights):
        raise ValueError("List of steps and list of step weights must have same length")

    if list_of_steps != sorted(list_of_steps):
        raise ValueError("List of steps must be increasing.")

    path_complex = list_of_complexes[0]
    for i, alpha in enumerate(list_of_steps):
        path_complex = step_func_path_complex(
            path_complex,
            list_of_complexes[i + 1],
            alpha + sum(list_of_step_weights[:i]),
            list_of_step_weights[i],
        )
    return path_complex


def compute_complex_from_graph_path(
    path: list[setup_utils.POSET_NODE_TYPE],
    poset_graph: nx.DiGraph,
    inclusion_graph: nx.DiGraph,
) -> tuple[gudhi.SimplexTree, list[float], list[float]]:
    """
    Creates path simplicial complex for a given path in poset graph

    Arguments
    ---
    path : list of nodes of the form (*"classes", filt_val) forming a path through the
    poset_graph.

    poset_graph : networkx.DiGraph
    Poset graph in which the path lies. Nodes are tuples of which the first n values are
    class(es) present in a given union and the final value is the filtration value.

    inclusion_graph : networkx.DiGraph
    Inclusion graph with simplicial complexes computed for each node.

    Returns
    ---
    gudhi.SimplexTree : Filtered simplicial complex corresponding to the given path
    through the poset graph.

    list of floats : List of filtration values (unadjusted) at which steps occur.

    list of floats : List of weights for the step edges.
    """
    # Check that the path is actually a path
    if not nx.is_simple_path(poset_graph, path):
        raise ValueError("Not a valid path in the poset graph")

    # Find list of edges crossing between classes
    list_of_step_edges = []
    for i in range(len(path) - 1):
        if path[i][:-1] != path[i + 1][:-1]:
            list_of_step_edges.append((path[i], path[i + 1]))

    # Extract filtration values from node names
    list_of_steps = [edge[0][-1] for edge in list_of_step_edges]
    # Use class portion of names to look up relevant simplices
    list_of_complexes = [
        inclusion_graph.nodes[edge[0][:-1]]["simplex"] for edge in list_of_step_edges
    ] + [inclusion_graph.nodes[path[-1][:-1]]["simplex"]]
    # Pull edge weights for edges crossing between classes
    list_of_step_weights = [
        poset_graph.edges[edge]["weight"] for edge in list_of_step_edges
    ]
    return (
        arbitrary_path_complex(list_of_complexes, list_of_steps, list_of_step_weights),
        list_of_steps,
        list_of_step_weights,
    )


def landscapes_for_all_paths(
    poset_graph: nx.DiGraph,
    inclusion_graph: nx.DiGraph,
    homology_coeff_field: int = 11,
    max_dim: int = -1,
    num_landscapes: int = 5,
    landscape_resolution: int = 100,
) -> dict[tuple[setup_utils.POSET_NODE_TYPE, ...], dict[str, Any]]:
    """Find all paths through poset graph and compute their one-dimensional landscapes

    Arguments
    ---
    poset_graph : networkx.DiGraph
    Directed graph containing the full poset structure. Must have weighted edges with
    the "weight" key.

    inclusion_graph : networkx.DiGraph
    Directed weighted graph containing the inclusion structure for classes. Each node
    must have the "simplex" attribute with the simplicial complex for that union of
    classes.

    homology_coeff_field : int
    Prime number indicating which finite field over which to compute homology. See the
    gudhi documentation on the SimplexTree.compute_persistence() method for more
    details. Default is 11.

    max_dim : int
    Maximum dimension of persistent homology to compute. If set to a negative number,
    will be computed from the data. To do so, in each complex in question, the highest
    dimensional simplex dimension will be found and the maximum dimension will be set to
    one less than the minimum of those dimensions -- i.e., it is one less than the
    largest dimension present in all complexes. Default is -1.

    num_landscapes : int
    How many landscape functions to compute. This corresponds to the maximal rank k
    considered in the lambda(k, x) definition of landscapes. Default is 5.

    landscape_resolution : int
    How many samples to take across the filtration value range, affecting how finely the
    landscape is discretized. Default is 100.

    Returns
    ---
    dictionary : Keys are tuples of nodes in the poset graph corresponding to a path
    through that graph. Nodes are tuples with the first n-1 elements being strings
    indicating which classes are used in the relevant union and the last element is the
    filtration value. Elements of the dictionary are dictionaries with two keys:
    "landscapes" and "grid". The "grid" key corresponds to a 1-D numpy array containing
    the x-values along which the landscapes are discretized.

    The "landscapes" key yields a (max_dim x num_landscapes x landscape_resolution)
    numpy array containing the landscapes for that path.
    """

    # Find minimal dimension across complexes to define max landscape dimension
    if max_dim < 0:
        max_dim = (
            min(
                [
                    inclusion_graph.nodes[node]["simplex"].dimension()
                    for node in inclusion_graph.nodes
                ]
            )
            - 1
        )
        max_dim = max([max_dim, 0])

    # Find all paths
    maximal_class = [
        node for node in inclusion_graph.nodes if inclusion_graph.out_degree(node) == 0
    ][0]
    final_node = (*maximal_class, np.inf)
    """
    # Uncomment this chunk to start paths in all classes, not just minimal ones
    sources = [
        (*node, next(inclusion_graph.nodes[node]["simplex"].get_filtration())[-1])
        for node in inclusion_graph.nodes
    ]
    """
    sources = [node for node in poset_graph.nodes if poset_graph.in_degree(node) == 0]
    path_set = list(
        nx.all_simple_paths(poset_graph.reverse(copy=False), final_node, sources)  # type: ignore
    )

    # Compute simplicial complexes and landscapes for all paths
    landscape_path_dict = dict()
    for path in tqdm(path_set):
        path_complex, list_of_steps, list_of_step_weights = (
            compute_complex_from_graph_path(path[::-1], poset_graph, inclusion_graph)
        )
        max_filtration_value = list(path_complex.get_filtration())[-1][-1]
        path_complex.compute_persistence(homology_coeff_field=homology_coeff_field)
        path_diagrams = [
            path_complex.persistence_intervals_in_dimension(i)
            for i in range(max_dim + 1)
        ]
        # Fix persistence for elements created right after a step
        for diagram in path_diagrams:
            for step_num in range(len(list_of_steps)):
                adjusted_end_step_filt_value = sum(
                    list_of_steps[: step_num + 1] + list_of_step_weights[: step_num + 1]
                )
                indices_to_be_corrected = np.nonzero(
                    np.isclose(diagram[:, 0], adjusted_end_step_filt_value)
                )
                np.subtract.at(
                    diagram[:, 0],
                    indices_to_be_corrected,
                    list_of_step_weights[step_num],
                )
        # Truncate infinite value at max filtration value
        path_diagrams[0][-1, 1] = max_filtration_value
        lscape = greps.Landscape(
            num_landscapes=num_landscapes,
            resolution=landscape_resolution,
            keep_endpoints=True,
        )
        lscape.fit(path_diagrams)
        landscapes = lscape.transform(path_diagrams)
        landscapes = landscapes.reshape(
            (max_dim + 1, num_landscapes, landscape_resolution), order="C"
        )
        landscape_path_dict[tuple(path[::-1])] = {
            "landscapes": landscapes,
            "grid": lscape.grid_,  # type: ignore
        }
    return landscape_path_dict


def find_node_landscape_value(
    node: setup_utils.POSET_NODE_TYPE,
    path_dict: dict[tuple[setup_utils.POSET_NODE_TYPE, ...], dict[str, np.ndarray]],
    poset_graph: nx.DiGraph,
) -> np.ndarray:
    """Compute poset landsacpe value at a particular node

    Arguments
    ---
    node : tuple
    A tuple of n-1 strings indicating classes and a float indicating the filtration
    value. This corresponds to a node in the poset graph.

    path_dict : dictionary
    Dictionary of landscapes computed along all paths in the poset graph. Each key is
    a tuple of poset graph nodes, and the values are dictionaries with the keys "grid"
    and "landscapes", each of which has a numpy array as a value.

    poset_graph : networkx.DiGraph
    Underlying weighted directed graph representing the poset.

    Returns
    ---
    numpy.ndarray : A (N x M) array, where N is the number of homological dimensions
    included in the landscape computation, and M is the number of landscapes requested.
    Values are the generalized landscape values at the requested node.
    """

    paths_through_node = {
        key: path_dict[key] for key in path_dict.keys() if node in key
    }

    if len(paths_through_node) == 0:
        raise RuntimeError("Node was not found on any paths.")

    path_landscape_values = []
    for path, path_landscape_dict in paths_through_node.items():
        # Find filtration value offset
        step_edges_up_to_node = []
        for i in range(len(path) - 1):
            if path[i][:-1] == node[:-1]:
                break
            if path[i][:-1] != path[i + 1][:-1]:
                step_edges_up_to_node.append((path[i], path[i + 1]))
        step_edge_weights = [
            poset_graph.edges[edge]["weight"] for edge in step_edges_up_to_node
        ]
        filt_val_offset = sum(step_edge_weights)

        x_grid = path_landscape_dict["grid"]
        landscapes = path_landscape_dict["landscapes"]
        node_landscapes = np.empty(landscapes.shape[:-1])

        for homology_dim in range(landscapes.shape[0]):
            for k in range(landscapes.shape[1]):
                node_landscapes[homology_dim, k] = np.interp(
                    node[-1] + filt_val_offset,
                    x_grid,
                    landscapes[homology_dim, k, :],
                )

        path_landscape_values.append(node_landscapes)

    return np.min(path_landscape_values, axis=0)


def add_landscape_values_to_poset_graph(
    poset_graph: nx.DiGraph,
    path_dict: dict[tuple[setup_utils.POSET_NODE_TYPE, ...], dict[str, np.ndarray]],
) -> nx.DiGraph:
    """Return graph with landscape values added to nodes

    Arguments
    ---
    poset_graph : networkx.DiGraph
    Directed weighted graph representing a poset persistence module.

    path_dict : networkx.DiGraph
    Dictionary with paths in poset_graph as keys and persistence landscape dictionaries
    as values.

    Returns
    ---
    networkx.DiGraph : A copy of poset_graph with the generalized landscape values added
    under the "landscape_vals" key at each node.
    """
    for node in poset_graph:
        poset_graph.nodes[node]["landscape_values"] = find_node_landscape_value(
            node, path_dict, poset_graph
        )
    return poset_graph


def compute_classwise_landscape_poset(
    data_points: np.ndarray,
    class_slices: dict[str, slice],
    class_weights: dict[tuple[tuple[str, ...], tuple[str, ...]], float],
    complex_max_dim: Optional[int] = None,
    landscape_max_dim: int = -1,
    homology_coeff_field: int = 11,
    num_landscapes: int = 5,
    path_landscape_resolution: int = 100,
    return_inclusion_graph: bool = False,
) -> Union[nx.DiGraph, tuple[nx.DiGraph, nx.DiGraph]]:
    """High-level routine that creates a poset graph with landscape values at every node

    Arguments
    ---
    data_points : numpy.ndarray
    (m, n) array of data points, where m is the number of points and n is the ambient
    dimension.

    class_slices : dictionary
    Dictionary  where the key is the name of the class and the value is the slice
    indicating which points belong to that class. These slices do not need to be
    disjoint (i.e., a single point can belong to more than one class). Must contain a
    key for each of the source nodes (i.e., single classes) in the inclusion_graph.

    class_weights : dictionary
    Dictionary with inclusion graph edge labels (2-tuples of nodes, which are themselves
    tuples of strings) as keys, and floats as values.

    complex_max_dim : int or None
    Maximum dimension to which the simplicial complex should be expanded to. If None,
    defaults to n from data_points. Default is None.

    landscape_max_dim : int
    Maximum dimension of persistent homology to compute. If set to a negative number,
    will be computed from the data. To do so, in each complex in question, the highest
    dimensional simplex dimension will be found and the maximum dimension will be set to
    one less than the minimum of those dimensions -- i.e., it is one less than the
    largest dimension present in all complexes. Default is -1.

    homology_coeff_field : int
    Prime number indicating which finite field over which to compute homology. See the
    gudhi documentation on the SimplexTree.compute_persistence() method for more
    details. Default is 11.

    num_landscapes : int
    How many landscape functions to compute. This corresponds to the maximal rank k
    considered in the lambda(k, x) definition of landscapes. Default is 5.

    path_landscape_resolution : int
    How many samples to take across the filtration value range, affecting how finely the
    landscape is discretized. Default is 100.

    return_inclusion_graph : bool
    Whether to return the inclusion graph as well as the full poset graph.

    Returns
    ---
    networkx.DiGraph : Weighted directed graph representing the poset persistence
    module. Nodes are tuples with the first n-1 entries being strings listing classes
    included in a given union and the last entry being the filtration value for that
    node. Every node has the "landscape_vals" key which contains poset persistence
    landscape values.

    OR

    tuple of two networkx.DiGraph : The first entry is as described above, and the
    second entry is a smaller directed graph representing the inclusion structure of the
    classes and their unions. Each node is a tuple of strings listing classes included
    in a given union. Every node has the "simplex" key which contains the filtered
    simplicial complex (a gudhi.SimplexTree) for that particular union.
    """

    inclusion_graph = setup_utils.create_inclusion_graph(
        tuple(class_slices.keys()), weights=class_weights
    )
    inclusion_graph = setup_utils.add_classwise_complexes(
        inclusion_graph, data_points, class_slices, max_dim=complex_max_dim
    )
    poset_graph = setup_utils.create_full_poset_graph(inclusion_graph)
    landscape_dict = landscapes_for_all_paths(
        poset_graph,
        inclusion_graph,
        homology_coeff_field=homology_coeff_field,
        max_dim=landscape_max_dim,
        num_landscapes=num_landscapes,
        landscape_resolution=path_landscape_resolution,
    )
    poset_graph = add_landscape_values_to_poset_graph(poset_graph, landscape_dict)
    if return_inclusion_graph:
        return poset_graph, inclusion_graph
    else:
        return poset_graph


def extract_landscape_and_filt_vals_from_union(
    poset_graph: nx.DiGraph, union: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Helper function to extract landscape values and corresponding filtration values

    Parameters
    ---
    poset_graph : networkx.DiGraph
    Poset graph representing poset persistence module. Must have landscape values added
    to nodes.

    union : tuple of strings
    Indicates all the classes present in the union in question.

    Returns
    ---
    tuple of numpy.ndarrays : The first array is a 1D array of length n, where n is the
    number of nodes in the path graph corresponding to the union specified. This array
    contains the filtration values in increasing order for those nodes. The second array
    is an (n x N x M) array containing landscape values where N is the number of
    homological dimensions computed and M is the number of landscape functions
    requested.
    """

    union_subgraph = nx.subgraph_view(
        poset_graph, filter_node=lambda node: node[:-1] == union
    )
    node_list = list(union_subgraph.nodes)

    if len(node_list) == 0:
        raise ValueError("Specified classes not present in graph")

    node_list.sort(key=itemgetter(-1))

    landscape_values = [
        union_subgraph.nodes[node]["landscape_values"] for node in node_list
    ]
    filtration_values = [node[-1] for node in node_list]

    return np.array(filtration_values), np.array(landscape_values)


def discretize_poset_graph_landscapes(
    poset_graph: nx.DiGraph, resolution: int
) -> xr.DataArray:
    """Create discretized array form of poset landscapes"""

    unions = list({node[:-1] for node in poset_graph.nodes})
    unions.sort()
    extracted_values = {
        union: extract_landscape_and_filt_vals_from_union(poset_graph, union)
        for union in unions
    }
    unshaped_filt_vals = np.concatenate(
        [np.ravel(values[0]) for values in extracted_values.values()]
    )
    min_filt_val = unshaped_filt_vals[np.isfinite(unshaped_filt_vals)].min()
    max_filt_val = unshaped_filt_vals[np.isfinite(unshaped_filt_vals)].max()

    grid = np.linspace(min_filt_val, max_filt_val, resolution)
    output_shape = (
        len(unions),
        extracted_values[unions[0]][1].shape[1],
        extracted_values[unions[0]][1].shape[2],
        resolution,
    )
    raw_interpolated_array = np.empty(output_shape, dtype=float)
    for i, union in enumerate(unions):
        for hom_dimension in range(output_shape[1]):
            for landscape_func in range(output_shape[2]):
                raw_interpolated_array[i, hom_dimension, landscape_func, :] = np.interp(
                    grid,
                    extracted_values[union][0],
                    extracted_values[union][1][:, hom_dimension, landscape_func],
                )

    unions_as_strings = [" U ".join(union) for union in unions]
    array = xr.DataArray(
        raw_interpolated_array,
        dims=("union", "homology_dimension", "landscape_func", "filt_vals"),
        coords={"union": unions_as_strings, "filt_vals": grid},
    )
    return array


def find_upper_and_lower_neighbors_in_sorted_list(
    value: float, sorted_list: list[float]
) -> Union[tuple[float, float], tuple[float]]:
    """Helper routine to find upper and lower neighbors for a given value in a list"""
    if value <= sorted_list[0]:
        return (sorted_list[0],)
    for i in range(1, len(sorted_list)):
        if sorted_list[i] > value:
            return (sorted_list[i - 1], sorted_list[i])
        elif sorted_list[i] == value:
            return (sorted_list[i],)
    return (sorted_list[-1],)


def compute_exact_landscape_value(
    filt_value: float,
    union: tuple[str, ...],
    poset_graph: nx.DiGraph,
    path_dict: dict,
    union_list_of_filt_values: Optional[list[float]] = None,
):
    """Compute generalized poset landscape value at any point in the poset"""
    if union_list_of_filt_values is None:
        union_list_of_filt_values = [
            node[-1] for node in poset_graph.nodes if node[:-1] == union
        ]
        union_list_of_filt_values = sorted(union_list_of_filt_values)
    if filt_value in union_list_of_filt_values:
        neighboring_nodes = {(*union, filt_value)}
    else:
        closest_filt_values = find_upper_and_lower_neighbors_in_sorted_list(
            filt_value, union_list_of_filt_values
        )
        neighboring_nodes = {(*union, value) for value in closest_filt_values}
    paths_through_point = {
        key: path_dict[key] for key in path_dict.keys() if neighboring_nodes <= set(key)
    }

    point_landscape_values = []
    for path, path_landscape_dict in paths_through_point.items():
        # Find filtration value offset
        step_edges_up_to_point = []
        for i in range(len(path) - 1):
            if path[i][:-1] == union:
                break
            if path[i][:-1] != path[i + 1][:-1]:
                step_edges_up_to_point.append((path[i], path[i + 1]))
        step_edge_weights = [
            poset_graph.edges[edge]["weight"] for edge in step_edges_up_to_point
        ]
        filt_val_offset = sum(step_edge_weights)

        x_grid = path_landscape_dict["grid"]
        landscapes = path_landscape_dict["landscapes"]
        point_landscapes = np.empty(landscapes.shape[:-1])

        for homology_dim in range(landscapes.shape[0]):
            for k in range(landscapes.shape[1]):
                point_landscapes[homology_dim, k] = np.interp(
                    filt_value + filt_val_offset,
                    x_grid,
                    landscapes[homology_dim, k, :],
                )

        point_landscape_values.append(point_landscapes)

    return np.min(point_landscape_values, axis=0)


def discretize_poset_landscapes_from_dict(
    poset_graph: nx.DiGraph, path_landscapes: dict, resolution: int
) -> xr.DataArray:
    unions = list({node[:-1] for node in poset_graph.nodes})
    unions.sort()
    extracted_values = {
        union: extract_landscape_and_filt_vals_from_union(poset_graph, union)
        for union in unions
    }
    unshaped_filt_vals = np.concatenate(
        [np.ravel(values[0]) for values in extracted_values.values()]
    )
    min_filt_val = unshaped_filt_vals[np.isfinite(unshaped_filt_vals)].min()
    max_filt_val = unshaped_filt_vals[np.isfinite(unshaped_filt_vals)].max()

    min_filt_val = unshaped_filt_vals[np.isfinite(unshaped_filt_vals)].min()
    max_filt_val = unshaped_filt_vals[np.isfinite(unshaped_filt_vals)].max()

    grid = np.linspace(min_filt_val, max_filt_val, resolution)
    output_shape = (
        len(unions),
        extracted_values[unions[0]][1].shape[1],
        extracted_values[unions[0]][1].shape[2],
        resolution,
    )
    raw_interpolated_array = np.empty(output_shape, dtype=float)

    for i, union in enumerate(unions):
        sorted_union_filt_values = sorted(extracted_values[union][0])
        for j, filt_val in enumerate(grid):
            raw_interpolated_array[i, :, :, j] = compute_exact_landscape_value(
                filt_val, union, poset_graph, path_landscapes, sorted_union_filt_values
            )

    unions_as_strings = [" U ".join(union) for union in unions]
    array = xr.DataArray(
        raw_interpolated_array,
        dims=("union", "homology_dimension", "landscape_func", "filt_vals"),
        coords={"union": unions_as_strings, "filt_vals": grid},
    )
    return array
