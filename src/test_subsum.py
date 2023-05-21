"""
Test functions for the 'subsum' module.
"""

from subsum import *
import numpy as np
from itertools import chain, combinations
from random import uniform


def test_metric():
    z1 = np.array([0, 0])
    z2 = np.array([1, 1])
    assert metric(z1, z2) == np.sqrt(2), 'Metric is Non-Euclidean'
    assert metric(z1, z1) < 1e-9, 'Point does not match itself'


def test_OPT():
    z1 = np.array([0, 0])
    z2 = np.array([1, 1])
    z3 = np.array([2, 2])
    assert np.array_equal(OPT(z1, z2, z3), z2), 'Should return closest point'


def test_complex2intarray():
    z1 = complex2intarray(0.117 + 1.223j, 1e-3)
    z2 = np.array([117, 1223])
    assert np.array_equal(z1, z2), 'Conversion is not working'


def powerset(iterable):
    """
    Obtain power set of an iterable.

    The term 'power set' is used here in the set-theoretic sense.
    """

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def test_powerset():
    assert list(powerset([1, 2])) == [(), (1,), (2,), (1, 2)]


def exhaustive(powers: complex, S: complex) -> tuple[list[complex], float]:
    """
    Solve subset-sum problem using brute force.

    Be careful with the size of 'powers' as this algorithm runs in time O(2^n).
    It should be used for testing purposes only.
    """

    # Initialize
    best_subset = None
    closest_distance = np.inf

    def complex2array(z: complex) -> np.ndarray:
        return np.array([np.real(z), np.imag(z)])

    # Search exhaustively for best subset
    for subset in powerset(powers):
        total = sum(subset)
        # Discard subsets that are too large
        if np.real(total) > np.real(S) or np.imag(total) > np.imag(S):
            continue
        # Update best subset
        else:
            d = metric(complex2array(total), complex2array(S))
            if d < closest_distance:
                best_subset = subset
                closest_distance = d

    return best_subset, closest_distance


def test_subsetsum():
    """
    Test the solution that uses dynamic programming.
    """

    powers = [uniform(0, 0.2) + 1j*uniform(0, 0.2) for i in range(10)]

    T = Template_set()
    for S in powers:
        T.add_template(S)

    for precision in [1e-2, 1e-9]:
        for S in [1+1j, 0+1j, 1, 0, 0.4 + 0.4j, 0.2 + 0.2j]:
            print(f'Disaggregating load {S}...')

            # Convert to numpy arrays
            powers_array = [np.array([z.real, z.imag]) for z in powers]
            S_array = np.array([S.real, S.imag])

            # Compare methods
            dist_os = T.optimal_sum(S, precision)[1]
            dist_ex = exhaustive(powers, S)[1]

            # Assert that the methods agree
            if precision == 1e-2:
                dist_ss = subsetsum(powers_array, S_array, precision)[0]
                assert abs(dist_os - dist_ss) < 1e-6, \
                    'Robust method is inconsistent with direct DP'
            elif precision == 1e-9:
                assert abs(dist_os - dist_ex) < 1e-6, \
                    'DP method is inconsistent with exhaustive search'


if __name__ == '__main__':

    test_metric()
    test_OPT()
    test_complex2intarray()
    test_powerset()
    test_subsetsum()

    print("Module 'subsum' passed all tests!")
