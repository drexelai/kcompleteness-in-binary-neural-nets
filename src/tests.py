import unittest

from main import calculate_model_architecture
from main import generate_search_space
from main import diagonal_search


class Test(unittest.TestCase):

    def test_calculate_model_architecture_1(self):
        self.assertEqual(calculate_model_architecture(48, 2), [
                         48, 24, 12, 6, 3], "Should be [48, 24, 12, 6, 3]")

    def test_calculate_model_architecture_2(self):
        self.assertEqual(calculate_model_architecture(40, 2), [
                         40, 20, 10, 5, 2], "Should be [40, 20, 10, 5, 2]")

    def test_generate_search_space_1(self):
        self.assertEqual(generate_search_space(-1, -1), [],
                         "Should return empty search space if max_x and max_y are less than 1")

    def test_generate_search_space_2(self):
        self.assertEqual(generate_search_space(-1, 1), [],
                         "Should return empty search space if max_x is less than 0")

    def test_generate_search_space_3(self):
        self.assertEqual(generate_search_space(-1, 1), [],
                         "Should return empty search space if max_y is less than 0")

    def test_generate_search_space_4(self):
        self.assertEqual(generate_search_space(5, 4), [
        ], "Should return empty search space max_x not a multiple of 2 and max_y is not a multiple of 8")

    def test_generate_search_space_5(self):
        self.assertEqual(generate_search_space(48, 64),[[(1, 8), (1, 16), (1, 24), (1, 32), (1, 40), (1, 48)], 
                                                        [(2, 8), (2, 16), (2, 24), (2, 32), (2, 40), (2, 48)], 
                                                        [(4, 8), (4, 16), (4, 24), (4, 32), (4, 40), (4, 48)], 
                                                        [(8, 8), (8, 16), (8, 24), (8, 32), (8, 40), (8, 48)], 
                                                        [(16, 8), (16, 16), (16, 24), (16, 32), (16, 40), (16, 48)], 
                                                        [(32, 8), (32, 16), (32, 24), (32, 32), (32, 40), (32, 48)]], 
                                                        "Should return collect search space")
    
    def test_diagonal_search_space_1(self):
        space = generate_search_space(48, 64)
        self.assertEqual(list(diagonal_search(space)), [[(1, 16), (2, 8)], [(1, 32), (2, 24), (4, 16), (8, 8)], 
                                                  [(1, 48), (2, 40), (4, 32), (8, 24), (16, 16), (32, 8)], 
                                                  [(4, 48), (8, 40), (16, 32), (32, 24)], [(16, 48), (32, 40)]], 
                                                  "Should return alternating diagonals in the search space matrix")


if __name__ == '__main__':
    unittest.main()
