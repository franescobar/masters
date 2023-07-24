"""
Utilities for solving the subset-sum problem, required for load disaggregation.
"""

import numpy as np


# The following functions solve the subset-sum problem. No reference is made to
# load disaggregation, which is simply a specific application. All functions
# assume that the 2D vectors (or complex numbers) are two-dimensional numpy
# arrays (instead of, say, tuples).


def metric(V1: np.ndarray, V2: np.ndarray) -> float:
    """
    Measure a (generic) distance between 2D vectors V1 and V2.
    """

    return np.linalg.norm(V1 - V2)


def best_vector(
    target: np.ndarray, V1: np.ndarray, V2: np.ndarray
) -> np.ndarray:
    """
    Return the vector closest to 'target' according to metric().

    If one vector exceeds 'target' in either dimension, return the other one.
    Otherwise, return the vector closest to 'target' according to metric().
    """

    if V1[0] > target[0] or V1[1] > target[1]:
        return V2
    elif V2[0] > target[0] or V2[1] > target[1]:
        return V1
    else:
        return V1 if metric(target, V1) <= metric(target, V2) else V2


def subsetsum(
    powers: list[np.ndarray],
    S: np.ndarray,
    precision: float = 1e-2,
    return_test_values: bool = False,
) -> tuple[float, list[int], np.ndarray]:
    """
    Find subset of 'powers' whose sum is closest to, but smaller than, 'S'.

    The function returns the distance between the sum of the subset and 'S',
    the indices of the elements of 'powers' that belong to the subset, and
    the sum of the subset.

    Interesting fact, found on April 14th, 2022: the algorithm returns the
    wrong indices if all elements of 'powers' are the same. This is solved by
    removing elements from 'solution' as soon as I reported their index.
    """

    # Correct small errors to that they are 0
    def round_small_zeros(x: np.ndarray) -> np.ndarray:
        tolerance = 1e-6
        if x[0] < - tolerance or x[1] < - tolerance:
            raise ValueError("Power is negative beyond tolerance.")

        first_entry = 0 if x[0] < 0 else x[0]
        second_entry = 0 if x[1] < 0 else x[1]

        return np.array([first_entry, second_entry])

    powers = [round_small_zeros(power) for power in powers]
    S = round_small_zeros(S)

    # Scale both the powers and the target vector to integers
    divisor = precision * np.linalg.norm(S) if np.linalg.norm(S) else precision
    original_powers = powers
    original_S = S
    powers = [np.floor(power / divisor) for power in powers]
    S = np.floor(S / divisor)

    # The inputs are converted to numpy arrays because all other variables are
    # numpy arrays anyway.
    X = np.array(powers, dtype=np.int32)
    T = np.array(S, dtype=np.int32)

    # Initialize OPT and POINT functions
    OPT = -1 * np.ones([X.shape[0] + 1, T[0] + 1, T[1] + 1, 2], dtype=np.int32)
    POINT = -1 * np.ones(
        [X.shape[0] + 1, T[0] + 1, T[1] + 1, 3], dtype=np.int32
    )

    # Implement dynamic programming solution
    for j in range(X.shape[0] + 1):
        for tx in range(T[0] + 1):
            for ty in range(T[1] + 1):
                # Compute vectors
                t = np.array([tx, ty])
                x_j = X[j - 1]
                x_jx = x_j[0]
                x_jy = x_j[1]
                # Solve subproblem using recursion
                if j == 0 or [tx, ty] == [0, 0]:
                    OPT[j][tx][ty] = np.array([0, 0])
                else:
                    if x_jx > tx or x_jy > ty:
                        OPT[j][tx][ty] = OPT[j - 1][tx][ty]
                    else:
                        option_1 = OPT[j - 1][tx][ty]
                        option_2 = x_j + OPT[j - 1][tx - x_jx][ty - x_jy]
                        OPT[j][tx][ty] = best_vector(t, option_1, option_2)
                        # Save pointer to previous subproblem
                        if np.array_equal(OPT[j][tx][ty], option_1):
                            POINT[j][tx][ty] = np.array([j - 1, tx, ty])
                        else:
                            POINT[j][tx][ty] = np.array(
                                [j - 1, tx - x_jx, ty - x_jy]
                            )

    # To find the optimal subset, traverse POINT from corner opposite to origin
    current_j = X.shape[0]
    current_tx = T[0]
    current_ty = T[1]
    solution = []

    # While not at the origin
    while [current_j, current_tx, current_ty] != [0, 0, 0]:
        # Read pointer saved at the current position
        next_j = int(POINT[current_j][current_tx][current_ty][0])
        next_tx = int(POINT[current_j][current_tx][current_ty][1])
        next_ty = int(POINT[current_j][current_tx][current_ty][2])
        # If next position was unchanged
        if [next_tx, next_ty] == [-1, -1]:
            # Break the loop if current_j is zero
            if current_j == 0:
                break
            # Otherwise, update it
            else:
                current_j -= 1
        # If next position was indeed changed
        else:
            # And it points to a subproblem with a different target vector
            if [next_tx, next_ty] != [current_tx, current_ty]:
                # Then x_j does belong to the solution
                x_j = X[current_j - 1]
                solution.append(x_j)
            # In any case, update j, tx, and try to move to the next position
            current_j = next_j
            current_tx = next_tx
            current_ty = next_ty

    # Fetch approximate solution
    solution = [list(x) for x in solution]

    # This rather convoluted fetching of the indices makes sure that the
    # algorithm doesn't break when all (or several) elements in 'powers' are
    # equal.
    indices = []
    for i in range(X.shape[0]):
        if len(solution) > 0 and list(X[i]) in solution:
            indices.append(i)
            solution.remove(list(X[i]))

    # Having found the indices, compute the best sum and its distance to S
    best_sum = sum(original_powers[i] for i in indices)
    closest_distance = metric(best_sum, original_S)

    if return_test_values:
        best_sum = sum(powers[i] for i in indices)
        closest_distance = metric(best_sum, S)
        return closest_distance * divisor, indices, best_sum * divisor
    else:
        return closest_distance, indices, best_sum


# The following functions and classes facilitate the use of robust_subsetsum()
# for load disaggregation. To facilitate using them from the power-flow module,
# these classes receive complex numbers as inputs, but convert them to arrays
# of integers before calling subsetsum().


def OPT(V1: np.ndarray, V2: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Return vector closest to S according to metric().
    """

    return V1 if metric(S, V1) <= metric(S, V2) else V2


def complex2intarray(S: complex, divisor: float) -> np.ndarray:
    """
    Convert complex number to 2D array with integer entries.
    """

    return np.array(
        [np.floor(np.real(S) / divisor), np.floor(np.imag(S) / divisor)]
    )


class Template:
    """
    A template is a network whose power consumption is a complex number.
    """

    def __init__(
        self, S: complex, system: "pf_static.StaticSystem" = None
    ) -> None:
        self.S = S
        self.S_array = None
        self.system = system


class Template_set:
    """
    A set of templates to choose from when disaggregating a load.
    """

    def __init__(self) -> None:
        self.S = 0
        self.templates = []

    def add_template(self, S: complex, system=None) -> None:
        """
        Add template to the template set.
        """

        T = Template(S, system)  # create template
        self.S += T.S  # update total power (complex)
        self.templates.append(T)  # add template to list

    def optimal_sum(
        self, S: complex, precision: float
    ) -> tuple[complex, float]:
        """
        Solve subset-sum problem for the template set and return solution.

        This method is only implemented to validate robust_subsetsum().

        For testing purposes, the method also returns the distance between S
        and the subset's power consumption.
        """

        # Scale target power and template powers so that they become integers
        divisor = (
            precision * np.linalg.norm(S) if np.linalg.norm(S) else precision
        )
        S_array = complex2intarray(S, divisor)
        for T in self.templates:
            T.S_array = complex2intarray(T.S, divisor)

        # Solve subset-sum problem recursively
        optimal_power = self.sigma_star(len(self.templates), S_array)

        return optimal_power, metric(optimal_power, S_array) * divisor

    def sigma_star(self, i: int, S: np.ndarray) -> np.ndarray:
        """
        Find consumption closest to S when only the first i templates are used.
        """

        # Treat base cases
        if i == 0:
            return 0
        elif S[0] == 0 and S[1] == 0:
            return 0
        # Treat non-trivial cases recursively
        else:
            # In Python, Ti has index i-1
            Ti = self.templates[i - 1]
            S1 = self.sigma_star(i - 1, S)
            S2 = Ti.S_array + self.sigma_star(i - 1, S - Ti.S_array)
            if S[0] < Ti.S_array[0] or S[1] < Ti.S_array[1]:
                return S1
            else:
                optimal_power = OPT(S1, S2, S)
                return optimal_power
