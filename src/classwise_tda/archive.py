"""Module with unused code from previous versions of the algorithm"""


### From poset_landscapes
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


### From setup_utils
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
    midpoints = (birth_death_array[:, 1] + birth_death_array[:, 0]) / 2
    # midpoints = np.linspace(0, 2.4, 100)
    return np.unique(np.concatenate([np.ravel(birth_death_array), midpoints]))


def create_full_poset_graph_from_crit_points(inclusion_graph: nx.DiGraph) -> nx.DiGraph:
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
