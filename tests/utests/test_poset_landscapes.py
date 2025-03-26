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


class Test_arbitrary_path_complex(unittest.TestCase):
    def test_ListOfStepsNotIncreasing_RaiseValueError(self):
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A, fake_complex_A, fake_complex_A]
        fake_list_of_steps = [1.0, 0.5]
        fake_list_of_weights = [0.0, 0.0]
        with self.assertRaises(ValueError):
            poset_landscapes.arbitrary_path_complex(
                fake_list_of_complexes, fake_list_of_steps, fake_list_of_weights
            )

    def test_TooManySteps_RaiseValueError(self):
        fake_complex_A = create_fake_class_A()
        fake_list_of_complexes = [fake_complex_A, fake_complex_A, fake_complex_A]
        fake_list_of_steps = [0.5, 1.0, 1.2]
        fake_list_of_weights = [0.0, 0.0, 0.0]
        with self.assertRaises(ValueError):
            poset_landscapes.arbitrary_path_complex(
                fake_list_of_complexes, fake_list_of_steps, fake_list_of_weights
            )

    def test_OneComplex_ReturnsSameComplex(self):
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
