import unittest

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
