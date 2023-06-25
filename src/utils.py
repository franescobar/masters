"""
Miscellaneous functions for timing code, converting units, and cleaning data.
"""

import numpy as np
import time  # for timing code
import cvxopt  # for solving quadratic programs
from collections.abc import Sequence  # for type checking


class Timer:
    """
    A context manager for timing code.
    """

    def __init__(self, name: str = None) -> None:
        # Set name/description of the code being timed
        self.name = name

    def __enter__(self) -> None:
        # When the context is entered, simply record the current time
        self.tstart = time.time()

    def __exit__(self, type, value, traceback) -> None:
        # When the context is exited, print the elapsed time
        if self.name:
            seconds = time.time() - self.tstart
            minutes = seconds / 60
            print(f"{self.name} took {seconds:.3f} s ({minutes:.2f} min)")
        else:
            print(f"Elapsed: {time.time() - self.tstart:.3f} s")


def pol2rect(mag: float, degrees: float) -> complex:
    """
    Return complex number defined by its magnitude and its angle in degrees.
    """

    return mag * np.exp(1j * np.deg2rad(degrees))


def rect2pol(z: complex) -> tuple[float, float]:
    """
    Return polar form (magnitude, angle in degrees) of complex number.
    """

    return np.abs(z), np.rad2deg(np.angle(z))


def var2mho(Mvar_3P: float, kV_LL: float) -> float:
    """
    Convert reactive power in Mvar to susceptance in mho.
    """

    return Mvar_3P / kV_LL**2


def change_base(
    quantity: complex, Sb_old: float, Sb_new: float, type: str = "Z"
) -> complex:
    """
    Convert quantity (impedance or power) to another base.
    """

    if type == "Z":
        return quantity * Sb_new / Sb_old
    elif type == "Y":
        return quantity * Sb_old / Sb_new
    elif type == "S":
        return quantity * Sb_old / Sb_new
    else:
        raise ValueError("type must be either 'Z' or 'S'")


def cvxopt_solve_qp(
    P: np.ndarray,
    q: np.ndarray,
    G: np.ndarray = None,
    h: np.ndarray = None,
    A: np.ndarray = None,
    b: np.ndarray = None,
) -> np.ndarray:
    """
    Using cvxopt, solve a quadratic program in the canonical form:

        minimize 0.5 * x.T @ P @ x + q.T @ x

        subject to G @ x <= h ,
                   A @ x == b .

    """

    # Cast matrices to floats
    P = P.astype(float)
    q = q.astype(float)
    if G is not None:
        G = G.astype(float)
        h = h.astype(float)
    if A is not None:
        A = A.astype(float)
        b = b.astype(float)

    # Make sure that P is symmetric
    P = 0.5 * (P + P.T)

    # Specify quadratic-programming problem
    args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])

    # Set up solver
    cvxopt.solvers.options["show_progress"] = False
    sol = cvxopt.solvers.qp(*args)

    # Handle case when no optimal solution was found
    if "optimal" not in sol["status"]:
        raise RuntimeError("cvxopt found no optimal solution")

    return np.array(sol["x"]).reshape((P.shape[1],))


def distance(x1: Sequence[float], x2: Sequence[float]) -> float:
    """
    Return distance between 2D points x1 and x2.
    """

    return np.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2)


def get_deviation(
    x1: Sequence[float], x2: Sequence[float], x3: Sequence[float]
) -> float:
    """
    Return angle (deviation in degrees) between vectors x1->x2 and x2->x3.
    """

    # Get distances
    a = distance(x1, x2)
    b = distance(x2, x3)
    c = distance(x3, x1)

    if a == 0 or b == 0 or c == 0:
        deviation = 0
        print("One distance is null. Please check data.")
    else:
        # Apply cosine's law
        cosine = (a**2 + b**2 - c**2) / (2 * a * b)
        # Correct rounding errors
        if cosine < -1:
            cosine = -1
        elif cosine > 1:
            cosine = 1
        # Get deviation
        angle_rad = np.arccos(cosine)
        deviation = 180 - angle_rad * 180 / np.pi

    return deviation


def remove_duplicates(
    x: Sequence[float], y: Sequence[float], tol: float = 1e-9
) -> tuple[list[float], list[float]]:
    """
    Remove duplicates from the x-y series up to a certain tolerance.
    """

    # Initialize new series
    new_x = [x[0]]
    new_y = [y[0]]

    # For each point in the series
    for i in range(len(x)):
        # Get previous and current points
        prev_x = new_x[-1]
        prev_y = new_y[-1]
        current_x = x[i]
        current_y = y[i]
        # If they are different, add them to the new series
        if abs(prev_x - current_x) > tol or abs(prev_y - current_y) > tol:
            new_x.append(current_x)
            new_y.append(current_y)

    return new_x, new_y


def reduce_series(
    x: Sequence[float],
    y: Sequence[float],
    deviation_tol: float,
    xtol: float,
    ytol: float,
) -> tuple[list[float], list[float], float]:
    """
    Reduce points in time series but keep resemblance to original data.

    The algorithm works as follows:

        1. Normalize data, i.e. map to unit square (maybe not centered at the
        origin)
        2. Initialize
        3. For each intermediate point (not including the extreme values)
            3.1. Get point and its neighbors
            3.2. Get deviation
            3.3. If tolerance is violated and series changed considerably
                3.3.1. Save values and restart
        4. Save last values
        5. Compute integral

    The integral is the area between the original and the reduced series. It
    is used as a measure of how much the reduced series resembles the original
    one.
    """

    x, y = remove_duplicates(x, y)

    x = np.array(x)
    y = np.array(y)

    # Normalize data, i.e. map to unit square (maybe not centered at the
    # origin)
    xrange = np.abs(np.max(x) - np.min(x))
    yrange = np.abs(np.max(y) - np.min(y))
    x_norm = x / xrange
    y_norm = y / yrange

    # Initialize. Notice that the new values must be saved in lists, and not in
    # arrays, because the number of points in the reduced series is not known.
    total_deviation = 0
    indices = [0]
    x_new = [x_norm[0]]
    y_new = [y_norm[0]]

    # For each intermediate point (not including the extreme values)
    for i in np.arange(1, len(x) - 1, 1):
        # Get point and its neighbors
        prev = [x_norm[i - 1], y_norm[i - 1]]
        current = [x_norm[i], y_norm[i]]
        next = [x_norm[i + 1], y_norm[i + 1]]

        # The deviation formed by the three points is added to the total in
        # absolute value, because the direction of the turn doesn't matter.
        total_deviation += abs(get_deviation(prev, current, next))

        # If tolerance is violated and series changed considerably
        if total_deviation > deviation_tol and (
            abs(x_norm[i] - x_new[-1]) > xtol
            or abs(y_norm[i] - y_new[-1]) > ytol
        ):
            # Save values and restart
            indices.append(i)
            x_new.append(x_norm[i])
            y_new.append(y_norm[i])
            total_deviation = 0

    # The last values will always be included
    indices.append(len(x) - 1)
    x_new.append(x_norm[-1])
    y_new.append(y_norm[-1])

    # Compute integral
    diffs = [
        yi - np.interp(xi, x_new, y_new) for xi, yi in zip(x_norm, y_norm)
    ]
    f = np.abs(np.array(diffs))
    integral = np.trapz(f, x=x_norm)

    # Convert back to arrays
    x = np.array([x[i] for i in indices])
    y = np.array([y[i] for i in indices])

    return x, y, integral


def introduce_fixed_points(
    x: Sequence[float], y: Sequence[float], step: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Introduce points with fixed x-coordinate in the data series.
    """

    # Create list with points
    new_points = [xi for xi in np.arange(0, x[-1], step) if xi > x[0]]

    # Introduce them in x by appending and then sorting
    new_x = np.array(sorted(set(list(x) + new_points)))

    # Introduce them in y using linear interpolation
    new_y = np.array([np.interp(xi, x, y) for xi in new_x])

    return new_x, new_y


def reduce(
    x: Sequence[float],
    y: Sequence[float],
    step: float = 1,
    integral_tol: float = 1e-3,
    max_samples: int = 1000,
    max_iters: int = 20,
    initial_tol: float = 10,
    xtol: float = 1e-3,
    ytol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce points in time series while ensuring a small IAE.

    The function iterates reducing the deviation tolerance until the integral
    of the absolute error is small (or the series are not being reduced at
    all).
    """

    # For each iteration below a certain limit
    for i in range(0, max_iters):
        # Define a hard-coded speed
        speed = 1

        # Reduce the deviation tolerance as the number of iterations increases
        deviation_tol = initial_tol / (speed * i + 1)

        # Try to reduce the series
        x_reduced, y_reduced, integral = reduce_series(
            x, y, deviation_tol=deviation_tol, xtol=xtol, ytol=ytol
        )

        # Stop if the integral is small enough or if the number of samples is
        # too large
        if integral < integral_tol or len(x_reduced) >= max_samples:
            break

    else:
        print("I had trouble ensuring a small IAE. Check data.")

    return introduce_fixed_points(x_reduced, y_reduced, step)
