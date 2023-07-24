"""
Test functions for the "utils" module.
"""

from utils import *
import numpy as np
import matplotlib.pyplot as plt


def test_Timer():
    # Test Timer with name
    with Timer("Test Timer"):
        time.sleep(0.1)

    # Test Timer without name
    with Timer():
        time.sleep(0.1)

    # Test Timer with name and nested Timer
    with Timer("Test Timer"):
        time.sleep(0.1)
        with Timer("Nested Timer"):
            time.sleep(0.1)


def test_pol2rect():
    """
    Test conversion from polar to rectangular coordinates.
    """

    z1 = pol2rect(1, 0)
    z2 = 1 + 0j
    assert np.isclose(z1, z2), "Conversion is not working"


def test_rect2pol():
    """
    Test conversion from rectangular to polar coordinates.
    """

    mag, degrees = rect2pol(1 + 1j)
    assert np.isclose(mag, np.sqrt(2)), "Conversion is not working"
    assert np.isclose(degrees, 45), "Conversion is not working"


def test_var2mho():
    """
    Test conversion from reactive power to susceptance.
    """

    assert np.isclose(var2mho(9, 300), 1e-4), "Conversion is not working"


def test_change_base():
    assert np.isclose(
        change_base(quantity=2, base_MVA_old=200, base_MVA_new=100, type="S"), 4
    ), "Power conversion is not working"

    assert np.isclose(
        change_base(quantity=1, base_MVA_old=200, base_MVA_new=100, type="Z"), 0.5
    ), "Impedance conversion is not working"


def test_cvxopt_solve_qp():
    """
    Test solution of quadratic programs.
    """

    P = np.array([[1, 0], [0, 0]])
    q = np.array([[3], [4]])
    G = np.array([[-1, 0], [0, -1], [-1, -3], [2, 5], [3, 4]])
    h = np.array([[0], [0], [-15], [100], [80]])

    x = cvxopt_solve_qp(P=P, q=q, G=G, h=h)

    assert np.isclose(x[0], 7.13e-7), "cvxopt_solve_qp is not working"
    assert np.isclose(x[1], 5.00e0), "cvxopt_solve_qp is not working"


def test_distance():
    x1 = np.array([1, 1])
    x2 = np.array([2, 2])
    assert np.isclose(distance(x1, x2), np.sqrt(2)), "distance is not working"


def test_get_deviation():
    # Deviation of 180 degrees
    x1 = (1, 1)
    x2 = (2, 2)
    x3 = (1, 1 + 1e-9)  # almost the same as x1, to avoid null distance

    assert np.isclose(
        get_deviation(x1, x2, x3), 180
    ), "get_deviation is not working"

    # Deviation of 135 degrees
    x1 = (0, 0)
    x2 = (1, 0)
    x3 = (0, 1)

    assert np.isclose(
        get_deviation(x1, x2, x3), 135
    ), "get_deviation is not working"

    # Deviation of 90 degrees
    x1 = (0, 0)
    x2 = (1, 0)
    x3 = (1, 1)

    assert np.isclose(
        get_deviation(x1, x2, x3), 90
    ), "get_deviation is not working"

    # Deviation of 45 degrees
    x1 = (0, 0)
    x2 = (1, 0)
    x3 = (2, 1)

    assert np.isclose(
        get_deviation(x1, x2, x3), 45
    ), "get_deviation is not working"

    # Deviation of 0 degrees
    x1 = (0, 0)
    x2 = (1, 0)
    x3 = (2, 0)

    assert np.isclose(
        get_deviation(x1, x2, x3), 0
    ), "get_deviation is not working"


def test_remove_duplicates():
    x = [0, 3, 4, 4, 5]
    y = list(map(lambda xi: xi**2 + 1, x))

    x, y = remove_duplicates(x, y)

    assert x == [0, 3, 4, 5], "remove_duplicates is not working"
    assert y == [1, 10, 17, 26], "remove_duplicates is not working"

    # Introduce duplicates at the end
    x = [0, 3, 4, 5, 5]
    y = list(map(lambda xi: xi**2 + 1, x))

    x, y = remove_duplicates(x, y)

    assert x == [0, 3, 4, 5], "remove_duplicates is not working"
    assert y == [1, 10, 17, 26], "remove_duplicates is not working"

    # Introduce duplicates at the beginning
    x = [0, 0, 3, 4, 5]
    y = list(map(lambda xi: xi**2 + 1, x))

    x, y = remove_duplicates(x, y)

    assert x == [0, 3, 4, 5], "remove_duplicates is not working"
    assert y == [1, 10, 17, 26], "remove_duplicates is not working"


def test_introduce_fixed_points():
    x = [0, 3, 4, 5]
    y = list(map(lambda xi: 2 * xi + 1, x))

    x, y = introduce_fixed_points(x=x, y=y, step=0.5)

    desired_x = np.arange(0, 5.5, 0.5)
    desired_y = np.array(list(map(lambda xi: 2 * xi + 1, desired_x)))

    assert np.allclose(x, desired_x), "introduce_fixed_points is not working"
    assert np.allclose(y, desired_y), "introduce_fixed_points is not working"


def test_reduce():
    """
    Test reduction of a time series.
    """

    t = np.linspace(0, 4 * np.pi, 10_000)
    y = np.sin(t)

    t_reduced, y_reduced = reduce(x=t, y=y, step=0.1)

    plt.plot(t, y, label="Original")
    plt.plot(
        t_reduced, y_reduced, label="Reduced", marker="x", linestyle="None"
    )
    plt.legend()
    plt.show()

    looks_nice = input("Does the plot look correct? (y/n) ")
    did_reduce = input(
        f"Reduced to {len(t_reduced)/len(t)*100:.2f} %. OK? (y/n) "
    )

    assert looks_nice == "y", "reduce is not working"
    assert did_reduce == "y", "reduce is not working"


if __name__ == "__main__":
    test_Timer()
    test_pol2rect()
    test_rect2pol()
    test_var2mho()
    test_change_base()
    test_cvxopt_solve_qp()
    test_distance()
    test_get_deviation()
    test_remove_duplicates()
    test_introduce_fixed_points()
    test_reduce()

    print("Module 'utils' passed all tests!")
