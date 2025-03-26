import unittest

import gudhi
import networkx as nx
import numpy as np

from classwise_tda import poset_landscapes


def create_fake_class_A() -> gudhi.SimplexTree:
    """Helper function to create simplex tree for triangle at origin"""
    fake_result_complex = gudhi.SimplexTree()
    fake_result_complex.insert([0], 0.0)
    fake_result_complex.insert([1], 0.0)
    fake_result_complex.insert([2], 0.0)
    fake_result_complex.insert([0, 1], 1.0)
    fake_result_complex.insert([0, 2], 1.0)
    fake_result_complex.insert([1, 2], np.sqrt(2))
    fake_result_complex.insert([0, 1, 2], np.sqrt(2))
    return fake_result_complex


def create_fake_class_B() -> gudhi.SimplexTree:
    """Helper function to create simplex tree for triangle at (2, 1)"""
    fake_result_complex = gudhi.SimplexTree()
    fake_result_complex.insert([3], 0.0)
    fake_result_complex.insert([4], 0.0)
    fake_result_complex.insert([5], 0.0)
    fake_result_complex.insert([3, 4], 1.0)
    fake_result_complex.insert([3, 5], 1.0)
    fake_result_complex.insert([4, 5], np.sqrt(2))
    fake_result_complex.insert([3, 4, 5], np.sqrt(2))
    return fake_result_complex


def create_fake_union_class() -> gudhi.SimplexTree:
    """Helper function to create simplex tree for union of triangles"""
    fake_result_complex = gudhi.SimplexTree()
    fake_result_complex.insert([0], 0.0)
    fake_result_complex.insert([1], 0.0)
    fake_result_complex.insert([2], 0.0)
    fake_result_complex.insert([0, 1], 1.0)
    fake_result_complex.insert([0, 2], 1.0)
    fake_result_complex.insert([1, 2], np.sqrt(2))
    fake_result_complex.insert([3], 0.0)
    fake_result_complex.insert([4], 0.0)
    fake_result_complex.insert([5], 0.0)
    fake_result_complex.insert([3, 4], 1.0)
    fake_result_complex.insert([3, 5], 1.0)
    fake_result_complex.insert([4, 5], np.sqrt(2))
    fake_result_complex.insert([2, 4], 1.0)
    fake_result_complex.insert([1, 5], 1.0)
    fake_result_complex.insert([1, 4], 1.0)
    fake_result_complex.insert([0, 4], np.sqrt(2))
    fake_result_complex.insert([1, 3], np.sqrt(2))
    fake_result_complex.insert([0, 5], 2.0)
    fake_result_complex.insert([2, 3], 2.0)
    fake_result_complex.insert([0, 3], np.sqrt(5))
    fake_result_complex.insert([2, 5], np.sqrt(5))
    fake_result_complex.expansion(2)
    return fake_result_complex


def create_fake_poset_with_landscapes() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(("A", 0.0), landscape_values=np.zeros((2, 3), dtype=float))
    graph.add_node(("A", 1.0), landscape_values=np.ones((2, 3), dtype=float))
    graph.add_node(("A", np.inf), landscape_values=2 * np.ones((2, 3), dtype=float))
    graph.add_node(("A", "B", 0.0), landscape_values=np.zeros((2, 3), dtype=float))
    graph.add_node(("A", "B", 2.0), landscape_values=2 * np.ones((2, 3), dtype=float))
    graph.add_node(
        ("A", "B", np.inf), landscape_values=4 * np.ones((2, 3), dtype=float)
    )
    return graph


def create_fake_poset() -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_node(("A", 0.0))
    graph.add_node(("A", 1.0))
    graph.add_node(("A", np.inf))
    graph.add_node(("A", "B", 0.0))
    graph.add_node(("A", "B", 1.0))
    graph.add_node(("A", "B", np.inf))
    graph.add_weighted_edges_from(
        [
            (("A", 0.0), ("A", 1.0), 1.0),
            (("A", 1.0), ("A", np.inf), np.inf),
            (("A", "B", 0.0), ("A", "B", 1.0), 1.0),
            (("A", "B", 1.0), ("A", "B", np.inf), np.inf),
            (("A", 0.0), ("A", "B", 0.0), 1.0),
            (("A", 1.0), ("A", "B", 1.0), 1.0),
            (("A", np.inf), ("A", "B", np.inf), 1.0),
        ]
    )
    return graph


class Test_step_func_path_complex(unittest.TestCase):
    def test_InfiniteAlpha_ReturnBaseComplex(self):
        fake_base_complex = create_fake_class_A()
        fake_union_complex = create_fake_union_class()
        alpha = np.inf
        step_weight = 0.0
        result = poset_landscapes.step_func_path_complex(
            fake_base_complex, fake_union_complex, alpha, step_weight
        )
        self.assertListEqual(
            list(fake_base_complex.get_filtration()), list(result.get_filtration())
        )

    def test_Alpha0_ReturnUnionComplex(self):
        fake_base_complex = create_fake_class_A()
        fake_union_complex = create_fake_union_class()
        alpha = 0.0
        step_weight = 0.0
        result = poset_landscapes.step_func_path_complex(
            fake_base_complex, fake_union_complex, alpha, step_weight
        )
        self.assertListEqual(
            list(fake_union_complex.get_filtration()), list(result.get_filtration())
        )

    def test_Alpha1_UnionComplexWithBVerticesFilt1(self):
        fake_base_complex = create_fake_class_A()
        fake_union_complex = create_fake_union_class()
        alpha = 1.0
        step_weight = 0.0
        result = poset_landscapes.step_func_path_complex(
            fake_base_complex, fake_union_complex, alpha, step_weight
        )
        mock_result_complex = fake_union_complex.copy()
        mock_result_complex.assign_filtration([3], 1.0)
        mock_result_complex.assign_filtration([4], 1.0)
        mock_result_complex.assign_filtration([5], 1.0)
        self.assertListEqual(
            list(mock_result_complex.get_filtration()), list(result.get_filtration())
        )

    def test_StepWeight1_AddToAllFiltValsAfterStep(self):
        fake_complex_A = gudhi.SimplexTree()
        fake_complex_A.insert([0], 0.0)
        fake_complex_A.insert([1], 0.0)
        fake_complex_A.insert([2], 0.0)
        fake_complex_A.insert([0, 1], 1.0)
        fake_complex_A.insert([0, 2], 2.0)
        fake_complex_B = gudhi.SimplexTree()
        fake_complex_B.insert([3], 0.0)
        fake_complex_B.insert([4], 0.0)
        fake_complex_B.insert([3, 4], 2.0)
        alpha = 1.0
        step_weight = 1.0
        result = poset_landscapes.step_func_path_complex(
            fake_complex_A, fake_complex_B, alpha, step_weight
        )
        mock_result = gudhi.SimplexTree()
        mock_result.insert([0], 0.0)
        mock_result.insert([1], 0.0)
        mock_result.insert([2], 0.0)
        mock_result.insert([0, 1], 1.0)
        mock_result.insert([0, 2], 3.0)
        mock_result.insert([3], 2.0)
        mock_result.insert([4], 2.0)
        mock_result.insert([3, 4], 3.0)
        self.assertListEqual(
            list(result.get_filtration()), list(mock_result.get_filtration())
        )


class Test_arbitrary_path_complex(unittest.TestCase):
    def test_ListOfStepsNotIncreasing_RaiseValueError(self):
        """Test that an error is raised if steps are not in increasing order"""
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A, fake_complex_A, fake_complex_A]
        fake_list_of_steps = [1.0, 0.5]
        fake_list_of_weights = [0.0, 0.0]
        with self.assertRaises(ValueError):
            poset_landscapes.arbitrary_path_complex(
                fake_list_of_complexes, fake_list_of_steps, fake_list_of_weights
            )

    def test_TooManySteps_RaiseValueError(self):
        """Test that an error is raised if too many steps are provided"""
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A, fake_complex_A, fake_complex_A]
        fake_list_of_steps = [0.5, 1.0, 1.2]
        fake_list_of_weights = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            poset_landscapes.arbitrary_path_complex(
                fake_list_of_complexes, fake_list_of_steps, fake_list_of_weights
            )

    def test_ListOfWeightsShorterThanListOfSteps_RaiseValueError(self):
        """Test that an error is raised if too few weights are provided"""
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A, fake_complex_A, fake_complex_A]
        fake_list_of_steps = [0.5, 1.0]
        fake_list_of_weights = [0.0]
        with self.assertRaises(ValueError):
            poset_landscapes.arbitrary_path_complex(
                fake_list_of_complexes, fake_list_of_steps, fake_list_of_weights
            )

    def test_OneComplex_ReturnsSameComplex(self):
        """Test that if only a single complex is provided, that complex is returned"""
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A]
        fake_list_of_steps = []
        fake_list_of_weights = []
        result = poset_landscapes.arbitrary_path_complex(
            fake_list_of_complexes, fake_list_of_steps, fake_list_of_weights
        )
        self.assertListEqual(
            list(fake_complex_A.get_filtration()), list(result.get_filtration())
        )

    def test_TwoSteps_AddedSimplicesHaveAlphaVals(self):
        """Test that alpha values are added correctly"""
        fake_complex_1 = gudhi.SimplexTree()
        fake_complex_1.insert([0], 0.0)
        fake_complex_2 = gudhi.SimplexTree()
        fake_complex_2.insert([1], 0.0)
        fake_complex_3 = gudhi.SimplexTree()
        fake_complex_3.insert([2], 0.0)
        result = poset_landscapes.arbitrary_path_complex(
            [fake_complex_1, fake_complex_2, fake_complex_3], [1.0, 2.0], [0.0, 0.0]
        )
        mock_result = gudhi.SimplexTree()
        mock_result.insert([0], 0.0)
        mock_result.insert([1], 1.0)
        mock_result.insert([2], 2.0)
        self.assertListEqual(
            list(result.get_filtration()), list(mock_result.get_filtration())
        )

    def test_TwoSteps_AddAllWeights(self):
        """Test that weight values are added correctly"""
        fake_complex_1 = gudhi.SimplexTree()
        fake_complex_1.insert([0], 0.0)
        fake_complex_2 = gudhi.SimplexTree()
        fake_complex_2.insert([1], 0.0)
        fake_complex_3 = gudhi.SimplexTree()
        fake_complex_3.insert([2], 0.0)
        result = poset_landscapes.arbitrary_path_complex(
            [fake_complex_1, fake_complex_2, fake_complex_3], [0.0, 0.0], [0.5, 0.5]
        )
        mock_result = gudhi.SimplexTree()
        mock_result.insert([0], 0.0)
        mock_result.insert([1], 0.5)
        mock_result.insert([2], 1.0)
        self.assertListEqual(
            list(result.get_filtration()), list(mock_result.get_filtration())
        )


class Test_compute_complex_from_graph_path(unittest.TestCase):
    def test_PathInOneComplex_ReturnComplex(self):
        """Test that if the path lies in only one union, return the associated complex"""
        fake_poset = nx.DiGraph()
        fake_poset.add_node(("A", 0.0))
        fake_poset.add_node(("A", 1.0))
        fake_poset.add_node(("A", np.inf))
        fake_poset.add_weighted_edges_from(
            [(("A", 0.0), ("A", 1.0), 1.0), (("A", 1.0), ("A", np.inf), np.inf)]
        )
        fake_complex = gudhi.SimplexTree()
        fake_complex.insert([0], 0.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_complex)
        path = [("A", 0.0), ("A", 1.0), ("A", np.inf)]
        result = poset_landscapes.compute_complex_from_graph_path(
            path, fake_poset, fake_inclusion_graph
        )
        self.assertListEqual(
            list(result.get_filtration()), list(fake_complex.get_filtration())
        )

    def test_PathNotInGraph_RaiseValueError(self):
        """Test that function raises an error if path is not in the graph"""
        fake_poset = nx.DiGraph()
        fake_poset.add_node(("A", 0.0))
        fake_poset.add_node(("A", 1.0))
        fake_poset.add_node(("A", np.inf))
        fake_poset.add_weighted_edges_from(
            [(("A", 0.0), ("A", 1.0), 1.0), (("A", 1.0), ("A", np.inf), np.inf)]
        )
        fake_complex = gudhi.SimplexTree()
        fake_complex.insert([0], 0.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_complex)
        path = [("A", 0.0), ("A", np.inf)]
        with self.assertRaises(ValueError):
            _ = poset_landscapes.compute_complex_from_graph_path(
                path, fake_poset, fake_inclusion_graph
            )

    def test_PathAcrossUnionsNoWeights_ReturnAppropriateComplex(self):
        """Test that alpha values process correctly"""
        fake_poset = nx.DiGraph()
        fake_poset.add_node(("A", 0.0))
        fake_poset.add_node(("A", 1.0))
        fake_poset.add_node(("A", np.inf))
        fake_poset.add_weighted_edges_from(
            [(("A", 0.0), ("A", 1.0), 1.0), (("A", 1.0), ("A", np.inf), np.inf)]
        )
        fake_poset.add_node(("A", "B", 0.0))
        fake_poset.add_node(("A", "B", 1.0))
        fake_poset.add_node(("A", "B", np.inf))
        fake_poset.add_weighted_edges_from(
            [
                (("A", "B", 0.0), ("A", "B", 1.0), 1.0),
                (("A", "B", 1.0), ("A", "B", np.inf), np.inf),
            ]
        )
        fake_poset.add_weighted_edges_from(
            [
                (("A", 0.0), ("A", "B", 0.0), 0.0),
                (("A", 1.0), ("A", "B", 1.0), 0.0),
                (("A", np.inf), ("A", "B", np.inf), 0.0),
            ]
        )
        fake_complex_1 = gudhi.SimplexTree()
        fake_complex_1.insert([0], 0.0)
        fake_complex_2 = gudhi.SimplexTree()
        fake_complex_2.insert([1], 0.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_complex_1)
        fake_inclusion_graph.add_node(("A", "B"), simplex=fake_complex_2)
        fake_inclusion_graph.add_edge(("A",), ("A", "B"), weight=0.0)
        path = [("A", 0.0), ("A", 1.0), ("A", "B", 1.0), ("A", "B", np.inf)]
        result = poset_landscapes.compute_complex_from_graph_path(
            path, fake_poset, fake_inclusion_graph
        )
        mock_result = gudhi.SimplexTree()
        mock_result.insert([0], 0.0)
        mock_result.insert([1], 1.0)
        self.assertListEqual(
            list(result.get_filtration()), list(mock_result.get_filtration())
        )

    def test_PathAcrossUnionsWithWeights_ReturnAppropriateComplex(self):
        """Test edge weights process correctly"""
        fake_poset = nx.DiGraph()
        fake_poset.add_node(("A", 0.0))
        fake_poset.add_node(("A", 1.0))
        fake_poset.add_node(("A", np.inf))
        fake_poset.add_weighted_edges_from(
            [(("A", 0.0), ("A", 1.0), 1.0), (("A", 1.0), ("A", np.inf), np.inf)]
        )
        fake_poset.add_node(("A", "B", 0.0))
        fake_poset.add_node(("A", "B", 1.0))
        fake_poset.add_node(("A", "B", np.inf))
        fake_poset.add_weighted_edges_from(
            [
                (("A", "B", 0.0), ("A", "B", 1.0), 1.0),
                (("A", "B", 1.0), ("A", "B", np.inf), np.inf),
            ]
        )
        fake_poset.add_weighted_edges_from(
            [
                (("A", 0.0), ("A", "B", 0.0), 1.0),
                (("A", 1.0), ("A", "B", 1.0), 1.0),
                (("A", np.inf), ("A", "B", np.inf), 1.0),
            ]
        )
        fake_complex_1 = gudhi.SimplexTree()
        fake_complex_1.insert([0], 0.0)
        fake_complex_2 = gudhi.SimplexTree()
        fake_complex_2.insert([1], 0.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_complex_1)
        fake_inclusion_graph.add_node(("A", "B"), simplex=fake_complex_2)
        fake_inclusion_graph.add_edge(("A",), ("A", "B"), weight=1.0)
        path = [("A", 0.0), ("A", "B", 0.0), ("A", "B", 1.0), ("A", "B", np.inf)]
        result = poset_landscapes.compute_complex_from_graph_path(
            path, fake_poset, fake_inclusion_graph
        )
        mock_result = gudhi.SimplexTree()
        mock_result.insert([0], 0.0)
        mock_result.insert([1], 1.0)
        self.assertListEqual(
            list(result.get_filtration()), list(mock_result.get_filtration())
        )


class Test_landscapes_for_all_paths(unittest.TestCase):
    def test_GivenPoset_CorrectNumberOfPaths(self):
        """Test that the correct number of paths are found"""
        fake_poset = create_fake_poset()
        fake_inclusion_graph = nx.DiGraph()
        fake_complex_1 = gudhi.SimplexTree()
        fake_complex_1.insert([0], 0.0)
        fake_complex_1.insert([1], 0.0)
        fake_complex_1.insert([0, 1], 1.0)
        fake_complex_2 = gudhi.SimplexTree()
        fake_complex_2.insert([0], 0.0)
        fake_complex_2.insert([1], 0.0)
        fake_complex_2.insert([2], 0.0)
        fake_complex_2.insert([0, 1], 1.0)
        fake_complex_2.insert([0, 2], 2.0)
        fake_inclusion_graph.add_node(("A",), simplex=fake_complex_1)
        fake_inclusion_graph.add_node(("A", "B"), simplex=fake_complex_2)
        fake_inclusion_graph.add_edge(("A",), ("A", "B"), weight=1.0)
        result = poset_landscapes.landscapes_for_all_paths(
            fake_poset, fake_inclusion_graph, num_landscapes=2, landscape_resolution=7
        )
        self.assertEqual(len(result), 4)


class Test_extract_landscape_and_filt_vals_from_union(unittest.TestCase):
    def test_GivenGraphClassA_ExtractCorrectFiltVals(self):
        """Check filtration values for one-class union in fake poset graph"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        result = poset_landscapes.extract_landscape_and_filt_vals_from_union(
            fake_poset_graph, ("A",)
        )
        mock_result = np.array([0.0, 1.0, np.inf])
        np.testing.assert_allclose(result[0], mock_result)

    def test_GivenGraphUnionClass_ExtractCorrectFiltVals(self):
        """Check filtration values for two-class union in fake poset graph"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        result = poset_landscapes.extract_landscape_and_filt_vals_from_union(
            fake_poset_graph, ("A", "B")
        )
        mock_result = np.array([0.0, 2.0, np.inf])
        np.testing.assert_allclose(result[0], mock_result)

    def test_UnionClassNotInGraph_RaiseValueError(self):
        """Check that function raises an error if an invalid union is specified"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        with self.assertRaises(ValueError):
            _ = poset_landscapes.extract_landscape_and_filt_vals_from_union(
                fake_poset_graph, ("B",)
            )

    def test_GivenGraphClassA_ExtractCorrectLandscapeVals(self):
        """Test that function returns correct landscape values"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        result = poset_landscapes.extract_landscape_and_filt_vals_from_union(
            fake_poset_graph, ("A",)
        )
        mock_result = np.stack(
            [
                np.zeros((2, 3), dtype=float),
                np.ones((2, 3), dtype=float),
                2 * np.ones((2, 3), dtype=float),
            ],
            axis=0,
        )
        np.testing.assert_allclose(result[1], mock_result)


class Test_discretize_poset_graph_landscapes(unittest.TestCase):
    def test_GivenGraph_OutputIsCorrectShape(self):
        """Check that function returns an array of the correct shape"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        resolution = 10
        result = poset_landscapes.discretize_poset_graph_landscapes(
            fake_poset_graph, resolution
        )
        mock_shape = [2, 2, 3, 10]
        self.assertListEqual(list(result.shape), mock_shape)

    def test_GivenGraph_CorrectDiscretizationGrid(self):
        """Check that the discretization grid coordinates are correct"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        resolution = 5
        result = poset_landscapes.discretize_poset_graph_landscapes(
            fake_poset_graph, resolution
        )
        mock_grid = np.array([0, 0.5, 1.0, 1.5, 2.0])
        np.testing.assert_allclose(result["filt_vals"], mock_grid)

    def test_GivenGraph_CorrectUnionLabels(self):
        """Check that the union label strings are created and ordered correctly"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        resolution = 5
        result = poset_landscapes.discretize_poset_graph_landscapes(
            fake_poset_graph, resolution
        )
        mock_coords = np.array(["A", "A U B"])
        np.testing.assert_array_equal(result["union"], mock_coords)

    def test_GivenGraph_CorrectInterpolationValues(self):
        """Check that the interpolation is performed correctly"""
        fake_poset_graph = create_fake_poset_with_landscapes()
        resolution = 3
        result = poset_landscapes.discretize_poset_graph_landscapes(
            fake_poset_graph, resolution
        )
        mock_class_A_interpolation = np.stack(
            [np.zeros((2, 3), dtype=float), np.ones((2, 3)), np.ones((2, 3))], axis=-1
        )
        mock_class_AUB_interpolation = np.stack(
            [np.zeros((2, 3), dtype=float), np.ones((2, 3)), 2 * np.ones((2, 3))],
            axis=-1,
        )
        with self.subTest(union="A"):
            np.testing.assert_allclose(
                result.sel({"union": "A"}), mock_class_A_interpolation
            )
        with self.subTest(union="A U B"):
            np.testing.assert_allclose(
                result.sel({"union": "A U B"}), mock_class_AUB_interpolation
            )
