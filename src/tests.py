import unittest

from main import calculate_model_architecture
from main import generate_search_space
from main import diagonal_search
from main import get
from main import generate_primary_diagonal
from main import generate_secondary_diagonal
from main import calculate_sparsity_score

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
        self.assertEqual(generate_search_space(48, 64), [[(1, 8), (1, 16), (1, 24), (1, 32), (1, 40), (1, 48)],
                                                         [(2, 8), (2, 16), (2, 24),
                                                          (2, 32), (2, 40), (2, 48)],
                                                         [(4, 8), (4, 16), (4, 24),
                                                          (4, 32), (4, 40), (4, 48)],
                                                         [(8, 8), (8, 16), (8, 24),
                                                          (8, 32), (8, 40), (8, 48)],
                                                         [(16, 8), (16, 16), (16, 24),
                                                          (16, 32), (16, 40), (16, 48)],
                                                         [(32, 8), (32, 16), (32, 24), (32, 32), (32, 40), (32, 48)]],
                         "Should return collect search space")

    def test_diagonal_search_space_1(self):
        search_space = generate_search_space(48, 64)
        self.assertEqual(list(diagonal_search(search_space)), [[(1, 16), (2, 8)], [(1, 32), (2, 24), (4, 16), (8, 8)],
                                                               [(1, 48), (2, 40), (4, 32),
                                                                (8, 24), (16, 16), (32, 8)],
                                                               [(4, 48), (8, 40), (16, 32), (32, 24)], [(16, 48), (32, 40)]],
                         "Should return alternating diagonals in the search space matrix")

    def test_get_1(self):
        search_space = generate_search_space(48, 64)
        self.assertEqual(get(search_space, 4, 3), (16, 32), "Should be equal")

    def test_get_2(self):
        search_space = generate_search_space(48, 64)
        self.assertEqual(get(search_space, 6, 6), None, "Should be None")

    def test_generate_primary_diagonal(self):
        search_space = generate_search_space(48, 64)
        correct_input_output_pairs = {(0, 0): {(0, 0), (3, 3), (5, 5), (4, 4), (2, 2), (1, 1)},
                                      (1, 1): {(0, 0), (3, 3), (5, 5), (4, 4), (2, 2), (1, 1)},
                                      (2, 2): {(0, 0), (3, 3), (5, 5), (4, 4), (2, 2), (1, 1)},
                                      (-1, -1): set(),
                                      (1, 2): {(1, 2), (0, 1), (4, 5), (2, 3), (3, 4)},
                                      (3, 5): {(0, 2), (1, 3), (2, 4), (3, 5)},
                                      (5, 4): {(5, 4), (3, 2), (2, 1), (4, 3), (1, 0)},
                                      (4, 5): {(0, 1), (1, 2), (4, 5), (2, 3), (3, 4)},
                                      (0, 4): {(1, 5), (0, 4)},
                                      (3, 2): {(5, 4), (3, 2), (2, 1), (4, 3), (1, 0)}}
        for key, expected in correct_input_output_pairs.items():
            actual = generate_primary_diagonal(search_space, *key)
            self.assertEqual(actual, expected,
                             "Expected: {}. Got: {}".format(expected, actual))

    def test_generate_secondary_diagonal(self):
        search_space = generate_search_space(48, 64)
        correct_input_output_pairs = {(0, 0): {(0, 0)}, 
                                      (1, 1): {(0, 2), (2, 0), (1, 1)}, 
                                      (2, 2): {(1, 3), (2, 2), (3, 1), (0, 4), (4, 0)}, 
                                      (-1, -1): set(), 
                                      (1, 2): {(1, 2), (3, 0), (0, 3), (2, 1)}, 
                                      (3, 5): {(4, 4), (5, 3), (3, 5)}, 
                                      (5, 4): {(4, 5), (5, 4)}, 
                                      (4, 5): {(4, 5), (5, 4)}, 
                                      (0, 4): {(1, 3), (3, 1), (0, 4), (2, 2), (4, 0)}, 
                                      (3, 2): {(3, 2), (1, 4), (0, 5), (2, 3), (5, 0), (4, 1)}}
        for key, expected in correct_input_output_pairs.items():
            actual = generate_secondary_diagonal(search_space, *key)
            self.assertEqual(actual, expected,
                             "Expected: {}. Got: {}".format(expected, actual))

    def test_calculate_sparsity_score(self):
        correct_input_output_pairs = {(0.1, 40, 10,  4): 0.625, 
                                      (0.1,  3, 10, 16): 0.0862, 
                                      (0.5, 28, 10,  4): 1.525, 
                                      (0.3,  2, 11,  8): 0.142}
        for key, expected in correct_input_output_pairs.items():
            actual = calculate_sparsity_score(*key)
            self.assertEqual(actual, expected,
                             "Expected: {}. Got: {}".format(expected, actual))

if __name__ == '__main__':
    unittest.main()
