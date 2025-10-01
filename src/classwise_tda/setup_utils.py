"""Utilities to set up data structures"""

from collections.abc import Callable
from itertools import chain, combinations
from typing import Optional, Unpack

import gudhi
import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

POSET_NODE_TYPE = tuple[Unpack[tuple[str, ...]], float]


def powerset(iterable) -> chain:
    """Helper function that returns powerset of input (without emptyset)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


def directed_diameter_computation(
    distance_matrix: np.ndarray,
    class_slices: dict[str, slice],
    node_1: tuple[str, ...],
    node_2: tuple[str, ...],
) -> float:
    """Helper function that computes directed diameter of the union of two node sets"""
    class_distances = distance_matrix.copy()
    classes = list(class_slices.keys())
    node_1_missing_classes = [a for a in classes if a not in node_1]
    node_2_missing_classes = [a for a in classes if a not in node_2]
    for node_1_missing_class in node_1_missing_classes:
        class_distances[class_slices[node_1_missing_class], :] = np.inf
    for node_2_missing_class in node_2_missing_classes:
        class_distances[:, class_slices[node_2_missing_class]] = np.inf
    return np.max(class_distances[np.isfinite(class_distances)])


def union_diameter_computation(
    distance_matrix: np.ndarray,
    class_slices: dict[str, slice],
    node_1: tuple[str, ...],
    node_2: tuple[str, ...],
) -> float:
    """Helper function that computes the union diameter of the union of two node sets"""
    class_distances = distance_matrix.copy()
    classes = list(class_slices.keys())
    missing_classes = [a for a in classes if (a not in node_1 and a not in node_2)]
    for missing_class in missing_classes:
        class_distances[class_slices[missing_class], :] = np.inf
        class_distances[:, class_slices[missing_class]] = np.inf
    return np.max(class_distances[np.isfinite(class_distances)])


def compute_class_distances(
    data_points: np.ndarray,
    class_slices: dict[str, slice],
    distance_function: Callable[
        [np.ndarray, dict[str, slice], tuple[str, ...], tuple[str, ...]], float
    ] = union_diameter_computation,
) -> dict[tuple[tuple[str, ...], tuple[str, ...]], float]:
    """Function to compute diameter distances between each class combination"""
    classes = list(class_slices.keys())
    nodes = list(powerset(classes))
    distance_matrix = cdist(data_points, data_points)
    out_dict = {}
    for node_1 in nodes:
        for node_2 in nodes:
            if len(node_1) == (len(node_2) - 1) and set(node_1) <= set(node_2):
                out_dict[(node_1, node_2)] = distance_function(
                    distance_matrix, class_slices, node_1, node_2
                )
    return out_dict


def create_inclusion_graph(
    classes: tuple[str, ...] | list[str],
    weights: Optional[dict[tuple[tuple[str, ...], tuple[str, ...]], float]] = None,
) -> nx.DiGraph:
    """Function to compute all unions of given classes and their inclusion structure

    Arguments
    ---
    classes : tuple or list of strings
    List of classes to create inclusion structure for.

    weights : dict
    Dictionary with edge labels (2-tuples of nodes, which are themselves tuples of
    strings) as keys, and floats as values.

    Returns
    ---
    networkx.DiGraph wth tuples of strings as nodes :
    Weighted directed graph with nodes given by tuples of classes (with the order in
    each tuple derived from the ordering of classes in the input list), edges showing
    inclusion, and edge weights from the "weights" argument.
    """

    class_combos = powerset(classes)
    inclusion_graph = nx.DiGraph()

    for class_combo in class_combos:
        inclusion_graph.add_node(class_combo)
        for node in inclusion_graph.nodes:
            # If node is A U B U C, add edges from A U B, A U C, and B U C.
            if len(node) == (len(class_combo) - 1) and set(node) <= set(class_combo):
                if weights is not None:
                    inclusion_graph.add_edge(
                        node, class_combo, weight=weights[(node, class_combo)]
                    )
                else:
                    inclusion_graph.add_edge(node, class_combo)

    return inclusion_graph


def add_classwise_complexes(
    inclusion_graph: nx.DiGraph,
    data_points: np.ndarray,
    class_slices: dict[str, slice],
    max_dim: Optional[int] = None,
) -> nx.DiGraph:
    """Adds filtered simplicial complexes to inclusion graph

    Arguments
    ----------
    inclusion_graph : networkx.DiGraph
    Directed graph giving the inclusion structure for the set of classes being
    considered.

    data_points : numpy.ndarray
    (m, n) array of data points, where m is the number of points and n is the ambient
    dimension.

    class_slices : dictionary of the form str:slice
    Dictionary  where the key is the name of the class and the value is the slice
    indicating which points belong to that class. These slices do not need to be
    disjoint (i.e., a single point can belong to more than one class). Must contain a
    key for each of the source nodes (i.e., single classes) in the inclusion_graph.

    max_dim : int or None
    Maximum dimension to which the simplicial complex should be expanded to. If None,
    defaults to n from data_points.

    Returns
    ----------
    networkx.DiGraph with tuples of strings as nodes :
    Input inclusion graph with the addition that each node has a "simplex" key that
    contains the filtered simplicial complex for that class or union of classes.
    """

    # Check that our inputs are valid
    if len(data_points.shape) != 2:
        raise ValueError("data_points must be an (m, n) array.")
    graph_sources = {
        x[0] for x in inclusion_graph.nodes if inclusion_graph.in_degree(x) == 0
    }
    if graph_sources != set(class_slices.keys()):
        raise ValueError(
            "Classes must match between class_slices and the inclusion graph"
        )
    if max_dim is None:
        max_dim = data_points.shape[1]
    distance_matrix = cdist(data_points, data_points)
    # Compute data radius so we can ignore infinite-length edges
    data_radius = distance_matrix.max()

    # Iterate through combinations of classes
    for node in inclusion_graph.nodes:
        class_distances = distance_matrix.copy()
        missing_classes = [a for a in class_slices.keys() if a not in node]
        for this_class in missing_classes:
            this_slice = class_slices[this_class]
            class_distances[this_slice, :] = np.inf
            class_distances[:, this_slice] = np.inf
        # We use max_filtration to ignore the inf-length edges in the distance matrix
        class_simplex = gudhi.SimplexTree.create_from_array(
            class_distances, max_filtration=data_radius
        )
        class_simplex.expansion(max_dim)  # type: ignore
        inclusion_graph.nodes[node]["simplex"] = class_simplex
    return inclusion_graph


def create_full_poset_graph(
    inclusion_graph: nx.DiGraph, finite_nodes_per_union: int = 100
) -> nx.DiGraph:
    """Create full poset graph from inclusion graph

    Arguments
    ---
    inclusion_graph : networkx.DiGraph
    Inclusion graph showing the inclusion structure for all classes. Must have filtered
    simplicial complex for each node under the "simplex" key.

    finite_nodes_per_union : int
    How many nodes to add per ray. Excludes infinite filtration value nodes.

    Returns
    ---
    networkx.DiGraph : Full poset expanding the inclusion graph in the R direction.
    Nodes are tuples of strings and a float; the last element is always a float
    giving the filtration value for that node, while all the preceding elements are
    strings indicating which union of classes the node exists in. That is, the node
    tuple is the same as the node tuple from the inclusion graph with the filtration
    value appended.

    Nodes exist at nodes_per_union evenly spaced filtration values between the minimum
    and maximum filtration values across all unions, plus at -inf and +inf.
    """

    # Add filtration value information to nodes
    max_filt_value = -np.inf
    min_filt_value = np.inf
    for node in inclusion_graph.nodes:
        try:
            node_complex = list(inclusion_graph.nodes[node]["simplex"].get_filtration())
            max_filt_value = max(max_filt_value, node_complex[-1][-1])
            min_filt_value = min(min_filt_value, node_complex[0][-1])
        except KeyError as e:
            e.add_note("Inclusion graph must have its simplicial complexes added")
            raise

    poset_graph = nx.DiGraph()

    filtration_values = np.linspace(
        min_filt_value, max_filt_value, finite_nodes_per_union
    )
    filtration_values = np.concatenate([[-np.inf], filtration_values, [np.inf]])

    # Compute complete list of filtration values
    for node in inclusion_graph.nodes:
        poset_graph.add_nodes_from(
            [(*node, filt_val) for filt_val in filtration_values]
        )
        for i in range(filtration_values.shape[0] - 1):
            poset_graph.add_edge(
                (*node, filtration_values[i]),
                (*node, filtration_values[i + 1]),
                weight=filtration_values[i + 1] - filtration_values[i],
            )

    for inclusion_edge in inclusion_graph.edges:
        for filt_val in filtration_values:
            poset_graph.add_edge(
                (*inclusion_edge[0], filt_val),
                (*inclusion_edge[1], filt_val),
                weight=inclusion_graph.edges[inclusion_edge]["weight"],
            )

    return poset_graph
