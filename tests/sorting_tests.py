from simply_python import sorting
import random

# Run with nosetests for badass-fast checking of all sorting methods

sorting_methods = [
    'quicksort',
    'insertion_sort',
    'selection_sort',
    'merge_sort',
    'bubble_sort',
    'shell_sort',
    'comb_sort',
    'counting_sort',
    'bucket_sort',
    'radix_sort',
    'heap_sort'
]

inplace_methods = [
    'qs_inplace'
]


def test_generator(tests=sorting_methods):
    for method_str in tests:
        function = getattr(sorting, method_str)
        for test_size in range(1, 1000, 100):
            test_list = [random.randint(0, test_size)
                         for _ in range(test_size)]
            yield check_stuff, function, test_list


def check_stuff(function, test_list):
    assert sorting.is_sorted(function(test_list))


def test_inplace_generator(tests=inplace_methods):
    for method_str in tests:
        function = getattr(sorting, method_str)
        for test_size in range(1, 1000, 100):
            test_list = [random.randint(0, test_size)
                         for _ in range(test_size)]
            yield inplace_check, function, test_list


def inplace_check(function, test_list):
    function(test_list)
    assert sorting.is_sorted(test_list)
