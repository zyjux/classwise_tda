import unittest

import gudhi
import networkx as nx
import numpy as np

from classwise_tda import setup_utils


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


class Test_powerset(unittest.TestCase):
    def test_OneEntryList_ListWithOneTuple(self):
        fake_input_list = ["test"]
        result = setup_utils.powerset(fake_input_list)
        self.assertEqual([("test",)], list(result))

    def test_TwoEntryList_ListIndividualAndBoth(self):
        fake_input_list = ["A", "B"]
        result = setup_utils.powerset(fake_input_list)
        fake_result = [("A",), ("B",), ("A", "B")]
        self.assertEqual(fake_result, list(result))


class Test_directed_diameter_computation(unittest.TestCase):
    def test_NoMissingClasses_MaxOfDistanceMatrix(self):
        fake_points_1 = np.zeros((3, 3))
        fake_points = (fake_points_1, fake_points_1)
        distance_matrix = np.ones((3, 3), dtype=np.float16)
        class_slices = {"A": slice(0, 3)}
        node_1 = ("A",)
        node_2 = ("A",)
        result = setup_utils.directed_diameter_computation(
            fake_points, distance_matrix, class_slices, node_1, node_2
        )
        self.assertEqual(result, 1.0)

    def test_SameMissingClass_MaxOfSymmetricRegion(self):
        fake_points_1 = np.zeros((3, 3))
        fake_points = (fake_points_1, fake_points_1)
        distance_matrix = np.array(range(9), dtype=np.float16).reshape((3, 3))
        class_slices = {"A": slice(0, 1), "B": slice(1, 3)}
        node_1 = ("A",)
        node_2 = ("A",)
        result = setup_utils.directed_diameter_computation(
            fake_points, distance_matrix, class_slices, node_1, node_2
        )
        self.assertEqual(result, 0.0)

    def test_DifferentMissingClasses_MaxOfAsymRegion(self):
        fake_points_1 = np.zeros((3, 3))
        fake_points = (fake_points_1, fake_points_1)
        distance_matrix = np.array(range(9), dtype=np.float16).reshape((3, 3))
        class_slices = {"A": slice(0, 1), "B": slice(1, 3)}
        node_1 = ("A",)
        node_2 = ("A", "B")
        result = setup_utils.directed_diameter_computation(
            fake_points, distance_matrix, class_slices, node_1, node_2
        )
        self.assertEqual(result, 2.0)


class Test_union_diameter_computation(unittest.TestCase):
    def test_NoMissingClasses_MaxOfDistanceMatrix(self):
        fake_points_1 = np.zeros((3, 3))
        fake_points = (fake_points_1, fake_points_1)
        distance_matrix = np.ones((3, 3), dtype=np.float16)
        class_slices = {"A": slice(0, 3)}
        node_1 = ("A",)
        node_2 = ("A",)
        result = setup_utils.union_diameter_computation(
            fake_points, distance_matrix, class_slices, node_1, node_2
        )
        self.assertEqual(result, 1.0)

    def test_SameMissingClass_MaxOfSymmetricRegion(self):
        fake_points_1 = np.zeros((3, 3))
        fake_points = (fake_points_1, fake_points_1)
        distance_matrix = np.array(range(9), dtype=np.float16).reshape((3, 3))
        class_slices = {"A": slice(0, 1), "B": slice(1, 3)}
        node_1 = ("A",)
        node_2 = ("A",)
        result = setup_utils.union_diameter_computation(
            fake_points, distance_matrix, class_slices, node_1, node_2
        )
        self.assertEqual(result, 0.0)

    def test_DifferentMissingClasses_MaxOfAsymRegion(self):
        fake_points_1 = np.zeros((3, 3))
        fake_points = (fake_points_1, fake_points_1)
        distance_matrix = np.array(range(9), dtype=np.float16).reshape((3, 3))
        class_slices = {"A": slice(0, 1), "B": slice(1, 3)}
        node_1 = ("A",)
        node_2 = ("A", "B")
        result = setup_utils.union_diameter_computation(
            fake_points, distance_matrix, class_slices, node_1, node_2
        )
        self.assertEqual(result, 8.0)


class Test_compute_class_distances(unittest.TestCase):
    def test_OneClass_ReturnEmptyDict(self):
        data_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        class_slices = {"A": slice(0, 4)}
        result = setup_utils.compute_class_distances(data_points, class_slices)
        self.assertDictEqual(result, {})

    def test_TwoClasses_TwoEntryDict(self):
        data_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        class_slices = {"A": slice(0, 2), "B": slice(2, 4)}
        result = setup_utils.compute_class_distances(data_points, class_slices)
        self.assertEqual(len(result), 2)

    def test_TwoClasses_TwoEntryDictSameWeights(self):
        data_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        class_slices = {"A": slice(0, 2), "B": slice(2, 4)}
        result = setup_utils.compute_class_distances(
            data_points, class_slices, distance_scale=1.0
        )
        with self.subTest("A -> A U B"):
            key = (("A",), ("A", "B"))
            self.assertAlmostEqual(result[key], np.sqrt(2))
        with self.subTest("B -> A U B"):
            key = (("B",), ("A", "B"))
            self.assertAlmostEqual(result[key], np.sqrt(2))

    def test_TwoClassesHausdorffDist_TwoEntryDictBoth1(self):
        data_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        class_slices = {"A": slice(0, 2), "B": slice(2, 4)}
        result = setup_utils.compute_class_distances(
            data_points,
            class_slices,
            distance_function=setup_utils.hausdorff_distance_computation,
            distance_scale=1.0,
        )
        with self.subTest("A -> A U B"):
            key = (("A",), ("A", "B"))
            self.assertAlmostEqual(result[key], 1.0)
        with self.subTest("B -> A U B"):
            key = (("B",), ("A", "B"))
            self.assertAlmostEqual(result[key], 1.0)


class Test_create_inclusion_graph(unittest.TestCase):
    def test_OneClass_ResultIsLen1(self):
        fake_classes = ("A",)
        result = setup_utils.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_nodes(), 1)

    def test_OneClass_NoEdges(self):
        fake_classes = ("A",)
        result = setup_utils.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_edges(), 0)

    def test_TwoClasses_ResultIsLen3(self):
        fake_classes = ("A", "B")
        result = setup_utils.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_nodes(), 3)

    def test_TwoClasses_TwoEdges(self):
        fake_classes = ("A", "B")
        result = setup_utils.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_edges(), 2)

    def test_TwoClasses_EdgeFromFirstClassToUnion(self):
        fake_classes = ("A", "B")
        result = setup_utils.create_inclusion_graph(fake_classes)
        edge_from_A_to_union = (("A",), ("A", "B"))
        self.assertIn(edge_from_A_to_union, result.edges)

    def test_TwoClasses_EdgeFromSecondClassToUnion(self):
        fake_classes = ("A", "B")
        result = setup_utils.create_inclusion_graph(fake_classes)
        edge_from_B_to_union = (("B",), ("A", "B"))
        self.assertIn(edge_from_B_to_union, result.edges)

    def test_ThreeClasses_SevenNodes(self):
        fake_classes = ("A", "B", "C")
        result = setup_utils.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_nodes(), 7)

    def test_ThreeClasses_SpecificEdgeList(self):
        fake_classes = ("A", "B", "C")
        result = setup_utils.create_inclusion_graph(fake_classes)
        mock_edge_set = {
            (("A",), ("A", "B")),
            (("B",), ("A", "B")),
            (("A",), ("A", "C")),
            (("C",), ("A", "C")),
            (("B",), ("B", "C")),
            (("C",), ("B", "C")),
            (("A", "B"), ("A", "B", "C")),
            (("A", "C"), ("A", "B", "C")),
            (("B", "C"), ("A", "B", "C")),
        }
        self.assertSetEqual(set(result.edges), mock_edge_set)

    def test_TwoClassesWithEdgeWeights_CorrectWeightsAdded(self):
        fake_classes = ("A", "B")
        fake_weights = {
            (("A",), ("A", "B")): 1.0,
            (("B",), ("A", "B")): 1.0,
        }
        result = setup_utils.create_inclusion_graph(fake_classes, weights=fake_weights)
        for edge in result.edges:
            with self.subTest(edge=edge):
                self.assertEqual(result.edges[edge]["weight"], 1.0)


class Test_add_classwise_complexes(unittest.TestCase):
    def test_OneClass_NodesContainSimplexTrees(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        fake_class_slices = {"A": slice(3)}
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",))
        result = setup_utils.add_classwise_complexes(
            fake_inclusion_graph, fake_data, fake_class_slices
        )
        self.assertIsInstance(result.nodes[("A",)]["simplex"], gudhi.SimplexTree)

    def test_OneClass_ComplexMatches(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        fake_class_slices = {"A": slice(3)}
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",))
        result = setup_utils.add_classwise_complexes(
            fake_inclusion_graph, fake_data, fake_class_slices
        )
        fake_result_complex = create_fake_class_A()
        self.assertListEqual(
            list(fake_result_complex.get_filtration()),
            list(result.nodes[("A",)]["simplex"].get_filtration()),
        )

    def test_TwoClasses_FirstComplexMatches(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ]
        )
        fake_class_slices = {"A": slice(3), "B": slice(3, 6)}
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_nodes_from(
            [
                ("A",),
                ("B",),
                ("A", "B"),
            ]
        )
        fake_inclusion_graph.add_edge(("A",), ("A", "B"))
        fake_inclusion_graph.add_edge(("B",), ("A", "B"))
        result = setup_utils.add_classwise_complexes(
            fake_inclusion_graph, fake_data, fake_class_slices
        )
        fake_result_complex = create_fake_class_A()
        self.assertListEqual(
            list(fake_result_complex.get_filtration()),
            list(result.nodes[("A",)]["simplex"].get_filtration()),
        )

    def test_TwoClasses_SecondComplexMatches(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ]
        )
        fake_class_slices = {"A": slice(3), "B": slice(3, 6)}
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_nodes_from(
            [
                ("A",),
                ("B",),
                ("A", "B"),
            ]
        )
        fake_inclusion_graph.add_edge(("A",), ("A", "B"))
        fake_inclusion_graph.add_edge(("B",), ("A", "B"))
        result = setup_utils.add_classwise_complexes(
            fake_inclusion_graph, fake_data, fake_class_slices
        )
        fake_result_complex = create_fake_class_B()
        self.assertListEqual(
            list(fake_result_complex.get_filtration()),
            list(result.nodes[("B",)]["simplex"].get_filtration()),
        )

    def test_TwoClasses_UnionComplexMatches(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
            ]
        )
        fake_class_slices = {"A": slice(3), "B": slice(3, 6)}
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_nodes_from(
            [
                ("A",),
                ("B",),
                ("A", "B"),
            ]
        )
        fake_inclusion_graph.add_edge(("A",), ("A", "B"))
        fake_inclusion_graph.add_edge(("B",), ("A", "B"))
        result = setup_utils.add_classwise_complexes(
            fake_inclusion_graph, fake_data, fake_class_slices
        )
        fake_result_complex = create_fake_union_class()
        self.assertListEqual(
            list(fake_result_complex.get_filtration()),
            list(result.nodes[("A", "B")]["simplex"].get_filtration()),
        )


class Test_create_full_poset_graph(unittest.TestCase):
    def test_NoSimplicialComplexes_RaisesKeyError(self):
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",))
        with self.assertRaises(KeyError):
            _ = setup_utils.create_full_poset_graph(fake_inclusion_graph)

    def test_Filts0to2SingleClass_ExpectedNodes(self):
        fake_simplex = gudhi.SimplexTree()
        fake_simplex.insert([0], 0.0)
        fake_simplex.insert([1], 2.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_simplex)
        nodes_per_union = 3
        result = setup_utils.create_full_poset_graph(
            fake_inclusion_graph, finite_nodes_per_union=nodes_per_union
        )
        mock_result_nodes = [
            ("A", -np.inf),
            ("A", 0.0),
            ("A", 1.0),
            ("A", 2.0),
            ("A", np.inf),
        ]
        self.assertListEqual(list(result.nodes), mock_result_nodes)

    def test_Filts0to2SingleClass_ExpectedEdges(self):
        fake_simplex = gudhi.SimplexTree()
        fake_simplex.insert([0], 0.0)
        fake_simplex.insert([1], 2.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_simplex)
        nodes_per_union = 3
        result = setup_utils.create_full_poset_graph(
            fake_inclusion_graph, finite_nodes_per_union=nodes_per_union
        )
        mock_result_edges = [
            (("A", -np.inf), ("A", 0.0)),
            (("A", 0.0), ("A", 1.0)),
            (("A", 1.0), ("A", 2.0)),
            (("A", 2.0), ("A", np.inf)),
        ]
        self.assertListEqual(list(result.edges), mock_result_edges)

    def test_Filts0to2SingleClass_EdgeWeightsCorrect(self):
        fake_simplex = gudhi.SimplexTree()
        fake_simplex.insert([0], 0.0)
        fake_simplex.insert([1], 2.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_simplex)
        nodes_per_union = 3
        result = setup_utils.create_full_poset_graph(
            fake_inclusion_graph, finite_nodes_per_union=nodes_per_union
        )
        with self.subTest():
            edge = (("A", -np.inf), ("A", 0.0))
            self.assertEqual(result.edges[edge]["weight"], np.inf)
        with self.subTest():
            edge = (("A", 0.0), ("A", 1.0))
            self.assertEqual(result.edges[edge]["weight"], 1.0)
        with self.subTest():
            edge = (("A", 1.0), ("A", 2.0))
            self.assertEqual(result.edges[edge]["weight"], 1.0)
        with self.subTest():
            edge = (("A", 2.0), ("A", np.inf))
            self.assertEqual(result.edges[edge]["weight"], np.inf)

    def test_Filts0to0SingleClass_OnlyOneFiniteNode(self):
        fake_simplex = gudhi.SimplexTree()
        fake_simplex.insert([0], 0.0)
        fake_simplex.insert([1], 0.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_simplex)
        nodes_per_union = 3
        result = setup_utils.create_full_poset_graph(
            fake_inclusion_graph, finite_nodes_per_union=nodes_per_union
        )
        mock_result_nodes = [
            ("A", -np.inf),
            ("A", 0.0),
            ("A", np.inf),
        ]
        self.assertListEqual(list(result.nodes), mock_result_nodes)

    def test_TwoClassesDifferentFiltMaxMins_ExpectedNodes(self):
        fake_simplex_A = gudhi.SimplexTree()
        fake_simplex_A.insert([0], 0.0)
        fake_simplex_A.insert([1], 2.0)
        fake_simplex_B = gudhi.SimplexTree()
        fake_simplex_B.insert([2], 1.0)
        fake_simplex_B.insert([3], 3.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_simplex_A)
        fake_inclusion_graph.add_node(("B",), simplex=fake_simplex_B)
        fake_inclusion_graph.add_edge(("A",), ("B",), weight=1.0)
        finite_nodes_per_union = 4
        result = setup_utils.create_full_poset_graph(
            fake_inclusion_graph, finite_nodes_per_union=finite_nodes_per_union
        )
        mock_result_nodes = [
            ("A", -np.inf),
            ("A", 0.0),
            ("A", 1.0),
            ("A", 2.0),
            ("A", 3.0),
            ("A", np.inf),
            ("B", -np.inf),
            ("B", 0.0),
            ("B", 1.0),
            ("B", 2.0),
            ("B", 3.0),
            ("B", np.inf),
        ]
        self.assertListEqual(list(result.nodes), mock_result_nodes)

    def test_TwoClassesDifferentFiltMaxMins_ExpectedEdges(self):
        fake_simplex_A = gudhi.SimplexTree()
        fake_simplex_A.insert([0], 0.0)
        fake_simplex_A.insert([1], 2.0)
        fake_simplex_B = gudhi.SimplexTree()
        fake_simplex_B.insert([2], 1.0)
        fake_simplex_B.insert([3], 3.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_simplex_A)
        fake_inclusion_graph.add_node(("B",), simplex=fake_simplex_B)
        fake_inclusion_graph.add_edge(("A",), ("B",), weight=1.0)
        finite_nodes_per_union = 4
        result = setup_utils.create_full_poset_graph(
            fake_inclusion_graph, finite_nodes_per_union=finite_nodes_per_union
        )
        mock_result_edges = {
            (("A", -np.inf), ("A", 0.0)),
            (("A", 0.0), ("A", 1.0)),
            (("A", 1.0), ("A", 2.0)),
            (("A", 2.0), ("A", 3.0)),
            (("A", 3.0), ("A", np.inf)),
            (("B", -np.inf), ("B", 0.0)),
            (("B", 0.0), ("B", 1.0)),
            (("B", 1.0), ("B", 2.0)),
            (("B", 2.0), ("B", 3.0)),
            (("B", 3.0), ("B", np.inf)),
            (("A", -np.inf), ("B", -np.inf)),
            (("A", 0.0), ("B", 0.0)),
            (("A", 1.0), ("B", 1.0)),
            (("A", 2.0), ("B", 2.0)),
            (("A", 3.0), ("B", 3.0)),
            (("A", np.inf), ("B", np.inf)),
        }
        self.assertSetEqual(set(result.edges), mock_result_edges)

    def test_TwoClassesDifferentFiltMaxMins_ExpectedEdgeWeights(self):
        fake_simplex_A = gudhi.SimplexTree()
        fake_simplex_A.insert([0], 0.0)
        fake_simplex_A.insert([1], 2.0)
        fake_simplex_B = gudhi.SimplexTree()
        fake_simplex_B.insert([2], 1.0)
        fake_simplex_B.insert([3], 3.0)
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",), simplex=fake_simplex_A)
        fake_inclusion_graph.add_node(("B",), simplex=fake_simplex_B)
        fake_inclusion_graph.add_edge(("A",), ("B",), weight=0.5)
        finite_nodes_per_union = 4
        result = setup_utils.create_full_poset_graph(
            fake_inclusion_graph, finite_nodes_per_union=finite_nodes_per_union
        )
        mock_result_edges = {
            (("A", -np.inf), ("A", 0.0)): np.inf,
            (("A", 0.0), ("A", 1.0)): 1.0,
            (("A", 1.0), ("A", 2.0)): 1.0,
            (("A", 2.0), ("A", 3.0)): 1.0,
            (("A", 3.0), ("A", np.inf)): np.inf,
            (("B", -np.inf), ("B", 0.0)): np.inf,
            (("B", 0.0), ("B", 1.0)): 1.0,
            (("B", 1.0), ("B", 2.0)): 1.0,
            (("B", 2.0), ("B", 3.0)): 1.0,
            (("B", 3.0), ("B", np.inf)): np.inf,
            (("A", -np.inf), ("B", -np.inf)): 0.5,
            (("A", 0.0), ("B", 0.0)): 0.5,
            (("A", 1.0), ("B", 1.0)): 0.5,
            (("A", 2.0), ("B", 2.0)): 0.5,
            (("A", 3.0), ("B", 3.0)): 0.5,
            (("A", np.inf), ("B", np.inf)): 0.5,
        }
        for key, value in mock_result_edges.items():
            with self.subTest(edge=key):
                self.assertEqual(result.edges[key]["weight"], value)
