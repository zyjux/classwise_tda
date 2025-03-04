import unittest

import gudhi
import numpy as np

from classwise_tda import pathwise_tda


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


class Test_create_classwise_complexes(unittest.TestCase):
    def test_OneClass_ResultIsLen1(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        fake_class_slices = {"A": slice(3)}
        result = pathwise_tda.create_classwise_complexes(fake_data, fake_class_slices)
        self.assertEqual(len(result), 1)

    def test_OneClass_DictOfSimplexTrees(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        fake_class_slices = {"A": slice(3)}
        result = pathwise_tda.create_classwise_complexes(fake_data, fake_class_slices)
        self.assertIsInstance(result["A"], gudhi.SimplexTree)

    def test_OneClass_ComplexMatches(self):
        fake_data = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        fake_class_slices = {"A": slice(3)}
        result = pathwise_tda.create_classwise_complexes(fake_data, fake_class_slices)
        fake_result_complex = gudhi.SimplexTree()
        fake_result_complex.insert([0], 0.0)
        fake_result_complex.insert([1], 0.0)
        fake_result_complex.insert([2], 0.0)
        fake_result_complex.insert([0, 1], 1.0)
        fake_result_complex.insert([0, 2], 1.0)
        fake_result_complex.insert([1, 2], np.sqrt(2))
        fake_result_complex.insert([0, 1, 2], np.sqrt(2))
        self.assertTrue(fake_result_complex.__eq__(result["A"]))

    def test_TwoClasses_ResultIsLen3(self):
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
        result = pathwise_tda.create_classwise_complexes(fake_data, fake_class_slices)
        self.assertEqual(len(result), 3)

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
        result = pathwise_tda.create_classwise_complexes(fake_data, fake_class_slices)
        fake_result_complex = gudhi.SimplexTree()
        fake_result_complex.insert([0], 0.0)
        fake_result_complex.insert([1], 0.0)
        fake_result_complex.insert([2], 0.0)
        fake_result_complex.insert([0, 1], 1.0)
        fake_result_complex.insert([0, 2], 1.0)
        fake_result_complex.insert([1, 2], np.sqrt(2))
        fake_result_complex.insert([0, 1, 2], np.sqrt(2))
        self.assertTrue(fake_result_complex.__eq__(result["A"]))

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
        result = pathwise_tda.create_classwise_complexes(fake_data, fake_class_slices)
        fake_result_complex = gudhi.SimplexTree()
        fake_result_complex.insert([3], 0.0)
        fake_result_complex.insert([4], 0.0)
        fake_result_complex.insert([5], 0.0)
        fake_result_complex.insert([3, 4], 1.0)
        fake_result_complex.insert([3, 5], 1.0)
        fake_result_complex.insert([4, 5], np.sqrt(2))
        fake_result_complex.insert([3, 4, 5], np.sqrt(2))
        self.assertTrue(fake_result_complex.__eq__(result["B"]))

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
        result = pathwise_tda.create_classwise_complexes(fake_data, fake_class_slices)
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
        self.assertTrue(fake_result_complex.__eq__(result["A_U_B"]))
