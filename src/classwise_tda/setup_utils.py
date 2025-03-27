"""Utilities to set up data structures"""

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


def extract_filt_values_from_persistence(
    simplicial_complex: gudhi.SimplexTree,
) -> np.ndarray:
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


def create_full_poset_graph(inclusion_graph: nx.DiGraph) -> nx.DiGraph:
    """Create full poset graph from inclusion graph and PH information

    Arguments
    ---
    inclusion_graph : networkx.DiGraph
    Inclusion graph showing the inclusion structure for all classes. Must have had
    simplicial complexes already defined using add_classwise_complexes.

    Returns
    ---
    networkx.DiGraph : Full poset expanding the inclusion graph in the R direction.
    Nodes are tuples of strings and a float; the last element is always a float
    giving the filtration value for that node, while all the preceding elements are
    strings indicating which union of classes the node exists in. That is, the node
    tuple is the same as the node tuple from the inclusion graph with the filtration
    value appended.

    Nodes exist at every filtration value important to: the union of classes being
    considered and all predecessors and successors of that node in the inclusion graph.
    Edges are directed in increasing filtration values, and across classes and unions
    in the orientation from the inclusion graph.
    """

    # Add filtration value information to nodes
    for node in inclusion_graph.nodes:
        try:
            inclusion_graph.nodes[node]["filt_values"] = (
                extract_filt_values_from_persistence(
                    inclusion_graph.nodes[node]["simplex"]
                )
            )
        except KeyError as e:
            e.add_note("Inclusion graph must have its simplicial complexes added")
            raise

    poset_graph = nx.DiGraph()

    # Compute complete list of filtration values
    for node in inclusion_graph.nodes:
        inclusion_graph.nodes[node]["complete_filt_vals"] = np.unique(
            np.concatenate(
                [inclusion_graph.nodes[node]["filt_values"]]
                + [
                    inclusion_graph.nodes[predecessor]["filt_values"]
                    for predecessor in inclusion_graph.predecessors(node)
                ]
                + [
                    inclusion_graph.nodes[successor]["filt_values"]
                    for successor in inclusion_graph.successors(node)
                ]
            )
        )
        these_complete_filt_vals = inclusion_graph.nodes[node]["complete_filt_vals"]
        poset_graph.add_nodes_from(
            [(*node, filt_val) for filt_val in these_complete_filt_vals]
        )
        for i in range(these_complete_filt_vals.shape[0] - 1):
            poset_graph.add_edge(
                (*node, these_complete_filt_vals[i]),
                (*node, these_complete_filt_vals[i + 1]),
                weight=these_complete_filt_vals[i + 1] - these_complete_filt_vals[i],
            )

    for inclusion_edge in inclusion_graph.edges:
        these_filt_vals = np.unique(
            np.concatenate(
                [
                    inclusion_graph.nodes[inclusion_edge[0]]["filt_values"],
                    inclusion_graph.nodes[inclusion_edge[1]]["filt_values"],
                ]
            )
        )
        for filt_val in these_filt_vals:
            poset_graph.add_edge(
                (*inclusion_edge[0], filt_val),
                (*inclusion_edge[1], filt_val),
                weight=inclusion_graph.edges[inclusion_edge]["weight"],
            )

    return poset_graph
