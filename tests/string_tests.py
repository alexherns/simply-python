import unittest
from simply_python import string_algos

string_test_suite= [
        ('123', '1', [0]),
        ('123', '3', [2]),
        ('123', '4', []),
        ('1234', '23', [1]),
        ('1234', '1234', [0]),
        ('123412341', '1', [0, 4, 8])
        ]

class Naive_Tests(unittest.TestCase):
    
    def test_check_args(self):
        self.assertRaises(AssertionError, string_algos.check_args, '1', '12')
        self.assertRaises(AssertionError, string_algos.check_args, '12', 1)

    def test_naive_matching(self):
        for string_test_tuple in string_test_suite:
            self.assertEqual(string_algos.naive_matching(string_test_tuple[0],
                string_test_tuple[1]), string_test_tuple[2])

class RK_Tests(unittest.TestCase):

    def rk_matching(self):
        for string_test_tuple in string_test_suite:
            self.assertEqual(string_algos.rabin_karp_matching(string_test_tuple[0],
                string_test_tuple[1]), string_test_tuple[2])


if __name__ == '__main__':
    unittest.main()
