import unittest

import gudhi
import networkx as nx
import numpy as np

from classwise_tda import pathwise_tda


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
        result = pathwise_tda.powerset(fake_input_list)
        self.assertEqual([("test",)], list(result))

    def test_TwoEntryList_ListIndividualAndBoth(self):
        fake_input_list = ["A", "B"]
        result = pathwise_tda.powerset(fake_input_list)
        fake_result = [("A",), ("B",), ("A", "B")]
        self.assertEqual(fake_result, list(result))


class Test_create_inclusion_graph(unittest.TestCase):
    def test_OneClass_ResultIsLen1(self):
        fake_classes = ("A",)
        result = pathwise_tda.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_nodes(), 1)

    def test_OneClass_NoEdges(self):
        fake_classes = ("A",)
        result = pathwise_tda.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_edges(), 0)

    def test_TwoClasses_ResultIsLen3(self):
        fake_classes = ("A", "B")
        result = pathwise_tda.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_nodes(), 3)

    def test_TwoClasses_TwoEdges(self):
        fake_classes = ("A", "B")
        result = pathwise_tda.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_edges(), 2)

    def test_TwoClasses_EdgeFromFirstClassToUnion(self):
        fake_classes = ("A", "B")
        result = pathwise_tda.create_inclusion_graph(fake_classes)
        edge_from_A_to_union = (("A",), ("A", "B"))
        self.assertIn(edge_from_A_to_union, result.edges)

    def test_TwoClasses_EdgeFromSecondClassToUnion(self):
        fake_classes = ("A", "B")
        result = pathwise_tda.create_inclusion_graph(fake_classes)
        edge_from_B_to_union = (("B",), ("A", "B"))
        self.assertIn(edge_from_B_to_union, result.edges)

    def test_ThreeClasses_SevenNodes(self):
        fake_classes = ("A", "B", "C")
        result = pathwise_tda.create_inclusion_graph(fake_classes)
        self.assertEqual(result.number_of_nodes(), 7)

    def test_ThreeClasses_SpecificEdgeList(self):
        fake_classes = ("A", "B", "C")
        result = pathwise_tda.create_inclusion_graph(fake_classes)
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


class Test_create_classwise_complexes(unittest.TestCase):
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
        result = pathwise_tda.add_classwise_complexes(
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
        result = pathwise_tda.add_classwise_complexes(
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
        result = pathwise_tda.add_classwise_complexes(
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
        result = pathwise_tda.add_classwise_complexes(
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
        result = pathwise_tda.add_classwise_complexes(
            fake_inclusion_graph, fake_data, fake_class_slices
        )
        fake_result_complex = create_fake_union_class()
        self.assertListEqual(
            list(fake_result_complex.get_filtration()),
            list(result.nodes[("A", "B")]["simplex"].get_filtration()),
        )


class Test_extract_filt_values_from_persistence(unittest.TestCase):
    def test_OneEdgeAtFilt2_ReturnZeroTwoInf(self):
        fake_complex = gudhi.SimplexTree()
        fake_complex.insert([0], 0.0)
        fake_complex.insert([1], 0.0)
        fake_complex.insert([0, 1], 2.0)
        result = pathwise_tda.extract_filt_values_from_persistence(fake_complex)
        np.testing.assert_allclose(result, np.array([0.0, 2.0, np.inf]))

    def test_OneEdgeAtTwoOneVertexAt1_ReturnZeroOneTwoInf(self):
        fake_complex = gudhi.SimplexTree()
        fake_complex.insert([0], 0.0)
        fake_complex.insert([1], 1.0)
        fake_complex.insert([0, 1], 2.0)
        result = pathwise_tda.extract_filt_values_from_persistence(fake_complex)
        np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0, np.inf]))

    def test_TriangleAt2EdgesAt1VerticesAt0_ReturnZeroOneTwoInf(self):
        fake_complex = gudhi.SimplexTree()
        fake_complex.insert([0], 0.0)
        fake_complex.insert([1], 0.0)
        fake_complex.insert([2], 0.0)
        fake_complex.insert([0, 1], 1.0)
        fake_complex.insert([0, 2], 1.0)
        fake_complex.insert([1, 2], 1.0)
        fake_complex.insert([0, 1, 2], 2.0)
        result = pathwise_tda.extract_filt_values_from_persistence(fake_complex)
        np.testing.assert_allclose(result, np.array([0.0, 1.0, 2.0, np.inf]))


class Test_create_full_poset_graph(unittest.TestCase):
    def test_NoSimplicialComplexes_RaisesKeyError(self):
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",))
        with self.assertRaises(KeyError):
            pathwise_tda.create_full_poset_graph(fake_inclusion_graph)

    def test_OneClassOneEdge_ProducesGivenPathGraphNodes(self):
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",))
        fake_inclusion_graph.nodes[("A",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0, 1], 2.0)
        result = pathwise_tda.create_full_poset_graph(fake_inclusion_graph)
        mock_result_nodes = [("A", 0.0), ("A", 2.0), ("A", np.inf)]
        self.assertListEqual(list(result.nodes), mock_result_nodes)

    def test_OneClassOneEdge_ProducesGivenPathGraphEdges(self):
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_node(("A",))
        fake_inclusion_graph.nodes[("A",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0, 1], 2.0)
        result = pathwise_tda.create_full_poset_graph(fake_inclusion_graph)
        mock_result_edges = [
            (("A", 0.0), ("A", 2.0)),
            (("A", 2.0), ("A", np.inf)),
        ]
        self.assertListEqual(list(result.edges), mock_result_edges)

    def test_TwoClassesAllDifferentFiltVals_ExpectedNodes(self):
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_nodes_from([("A",), ("B",), ("A", "B")])
        fake_inclusion_graph.add_weighted_edges_from(
            [
                (("A",), ("A", "B"), None),
                (("B",), ("A", "B"), None),
            ],
            weight="poset_weight",
        )
        fake_inclusion_graph.nodes[("A",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0, 1], 1.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([0, 1], 2.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([0, 1], 3.0)
        result = pathwise_tda.create_full_poset_graph(fake_inclusion_graph)
        mock_result_nodes = {
            ("A", 0.0),
            ("A", 1.0),
            ("A", 3.0),
            ("A", np.inf),
            ("B", 0.0),
            ("B", 2.0),
            ("B", 3.0),
            ("B", np.inf),
            ("A", "B", 0.0),
            ("A", "B", 1.0),
            ("A", "B", 2.0),
            ("A", "B", 3.0),
            ("A", "B", np.inf),
        }
        self.assertSetEqual(set(result.nodes), mock_result_nodes)

    def test_TwoClassesAllDifferentFiltVals_ExpectedEdges(self):
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_nodes_from([("A",), ("B",), ("A", "B")])
        fake_inclusion_graph.add_weighted_edges_from(
            [
                (("A",), ("A", "B"), None),
                (("B",), ("A", "B"), None),
            ],
            weight="poset_weight",
        )
        fake_inclusion_graph.nodes[("A",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0, 1], 1.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([0, 1], 2.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([0, 1], 3.0)
        result = pathwise_tda.create_full_poset_graph(fake_inclusion_graph)
        mock_result_edges = {
            (("A", 0.0), ("A", 1.0)),
            (("A", 1.0), ("A", 3.0)),
            (("A", 3.0), ("A", np.inf)),
            (("B", 0.0), ("B", 2.0)),
            (("B", 2.0), ("B", 3.0)),
            (("B", 3.0), ("B", np.inf)),
            (("A", "B", 0.0), ("A", "B", 1.0)),
            (("A", "B", 1.0), ("A", "B", 2.0)),
            (("A", "B", 2.0), ("A", "B", 3.0)),
            (("A", "B", 3.0), ("A", "B", np.inf)),
            (("A", 0.0), ("A", "B", 0.0)),
            (("A", 1.0), ("A", "B", 1.0)),
            (("A", 3.0), ("A", "B", 3.0)),
            (("A", np.inf), ("A", "B", np.inf)),
            (("B", 0.0), ("A", "B", 0.0)),
            (("B", 2.0), ("A", "B", 2.0)),
            (("B", 3.0), ("A", "B", 3.0)),
            (("B", np.inf), ("A", "B", np.inf)),
        }
        self.assertSetEqual(set(result.edges), mock_result_edges)

    def test_TwoClassesAllDifferentFiltVals_CorrectEdgeWeights(self):
        fake_inclusion_graph = nx.DiGraph()
        fake_inclusion_graph.add_nodes_from([("A",), ("B",), ("A", "B")])
        fake_inclusion_graph.add_weighted_edges_from(
            [
                (("A",), ("A", "B"), 0.5),
                (("B",), ("A", "B"), 1.5),
            ],
            weight="poset_weight",
        )
        fake_inclusion_graph.nodes[("A",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A",)]["simplex"].insert([0, 1], 1.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("B",)]["simplex"].insert([0, 1], 2.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"] = gudhi.SimplexTree()
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([0], 0.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([1], 0.0)
        fake_inclusion_graph.nodes[("A", "B")]["simplex"].insert([0, 1], 3.0)
        result = pathwise_tda.create_full_poset_graph(fake_inclusion_graph)
        mock_result_weights = {
            (("A", 0.0), ("A", 1.0)): {"weight": 1.0},
            (("A", 1.0), ("A", 3.0)): {"weight": 2.0},
            (("A", 3.0), ("A", np.inf)): {"weight": np.inf},
            (("B", 0.0), ("B", 2.0)): {"weight": 2.0},
            (("B", 2.0), ("B", 3.0)): {"weight": 1.0},
            (("B", 3.0), ("B", np.inf)): {"weight": np.inf},
            (("A", "B", 0.0), ("A", "B", 1.0)): {"weight": 1.0},
            (("A", "B", 1.0), ("A", "B", 2.0)): {"weight": 1.0},
            (("A", "B", 2.0), ("A", "B", 3.0)): {"weight": 1.0},
            (("A", "B", 3.0), ("A", "B", np.inf)): {"weight": np.inf},
            (("A", 0.0), ("A", "B", 0.0)): {"poset_weight": 0.5},
            (("A", 1.0), ("A", "B", 1.0)): {"poset_weight": 0.5},
            (("A", 3.0), ("A", "B", 3.0)): {"poset_weight": 0.5},
            (("A", np.inf), ("A", "B", np.inf)): {"poset_weight": 0.5},
            (("B", 0.0), ("A", "B", 0.0)): {"poset_weight": 1.5},
            (("B", 2.0), ("A", "B", 2.0)): {"poset_weight": 1.5},
            (("B", 3.0), ("A", "B", 3.0)): {"poset_weight": 1.5},
            (("B", np.inf), ("A", "B", np.inf)): {"poset_weight": 1.5},
        }
        self.assertDictEqual(dict(result.edges), mock_result_weights)


class Test_step_func_path_complex(unittest.TestCase):
    def test_InfiniteAlpha_ReturnBaseComplex(self):
        fake_base_complex = create_fake_class_A()
        fake_union_complex = create_fake_union_class()
        alpha = np.inf
        result = pathwise_tda.step_func_path_complex(
            fake_base_complex, fake_union_complex, alpha
        )
        self.assertListEqual(
            list(fake_base_complex.get_filtration()), list(result.get_filtration())
        )

    def test_Alpha0_ReturnUnionComplex(self):
        fake_base_complex = create_fake_class_A()
        fake_union_complex = create_fake_union_class()
        alpha = 0.0
        result = pathwise_tda.step_func_path_complex(
            fake_base_complex, fake_union_complex, alpha
        )
        self.assertListEqual(
            list(fake_union_complex.get_filtration()), list(result.get_filtration())
        )

    def test_Alpha1_UnionComplexWithBVerticesFilt1(self):
        fake_base_complex = create_fake_class_A()
        fake_union_complex = create_fake_union_class()
        alpha = 1.0
        result = pathwise_tda.step_func_path_complex(
            fake_base_complex, fake_union_complex, alpha
        )
        mock_result_complex = fake_union_complex.copy()
        mock_result_complex.assign_filtration([3], 1.0)
        mock_result_complex.assign_filtration([4], 1.0)
        mock_result_complex.assign_filtration([5], 1.0)
        self.assertListEqual(
            list(mock_result_complex.get_filtration()), list(result.get_filtration())
        )


class Test_arbitrary_path_complex(unittest.TestCase):
    def test_ListOfStepsNotIncreasing_RaiseValueError(self):
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A, fake_complex_A, fake_complex_A]
        fake_list_of_steps = [1.0, 0.5]
        with self.assertRaises(ValueError):
            pathwise_tda.arbitrary_path_complex(
                fake_list_of_complexes, fake_list_of_steps
            )

    def test_TooManySteps_RaiseValueError(self):
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A, fake_complex_A, fake_complex_A]
        fake_list_of_steps = [0.5, 1.0, 1.2]
        with self.assertRaises(ValueError):
            pathwise_tda.arbitrary_path_complex(
                fake_list_of_complexes, fake_list_of_steps
            )

    def test_OneComplex_ReturnsSameComplex(self):
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A]
        fake_list_of_steps = []
        result = pathwise_tda.arbitrary_path_complex(
            fake_list_of_complexes, fake_list_of_steps
        )
        self.assertListEqual(
            list(fake_complex_A.get_filtration()), list(result.get_filtration())
        )

    def test_TwoSteps_AddedSimplicesHaveAlphaVals(self):
        fake_complex_1 = gudhi.SimplexTree()
        fake_complex_1.insert([0], 0.0)
        fake_complex_2 = gudhi.SimplexTree()
        fake_complex_2.insert([1], 0.0)
        fake_complex_3 = gudhi.SimplexTree()
        fake_complex_3.insert([2], 0.0)
        result = pathwise_tda.arbitrary_path_complex(
            [fake_complex_1, fake_complex_2, fake_complex_3], [1.0, 2.0]
        )
        mock_result = gudhi.SimplexTree()
        mock_result.insert([0], 0.0)
        mock_result.insert([1], 1.0)
        mock_result.insert([2], 2.0)
        self.assertListEqual(
            list(result.get_filtration()), list(mock_result.get_filtration())
        )
