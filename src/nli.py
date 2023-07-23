"""
A module that implements the New LIVES Index (NLI) algorithm.
"""

# Modules from this repository
import control
import sim_interaction
import pf_dynamic
import records

# Modules from the standard library
from collections.abc import Sequence

# Other modules
import numpy as np

# The first few functions facilitate the computation of the NLI later on.


def average(x: Sequence[float], N: int, step: int = 1) -> float:
    """
    Return average of the last N elements of x.

    The average is computed by moving backwards with a step that is not
    necessarily 1. The function returns 0 is x is empty.
    """

    slice = x[-1 : -N * step - 1 : -step]

    return sum(slice) / min(N, len(slice))


def derivative(x: Sequence[float], dt: float) -> float:
    """
    Approximate the derivative of x(t) using a secant line from the left.

    If not enough points are given, the function returns None.
    """

    return None if len(x) < 2 else (x[-1] - x[-2]) / dt


def is_multiple(t: float, Ts: float, tol: float = 1e-6) -> bool:
    """
    Check if current time is a multiple of the sampling period within
    Return True if the current time t is a multiple of the sampling period Ts
    up to a tolerance tol.
    """

    return min([t % Ts, Ts - (t % Ts)]) < tol


def remained_constant(
    x: Sequence[float], tol: float = 1e-6
) -> tuple[bool, float]:
    """
    Check if x(t) is constant around its average within a tolerance tol.
    """

    if len(x) == 0:
        return False, 0

    avg = sum(x) / len(x)
    values_near_avg = [xi for xi in x if abs(xi - avg) < tol]

    return len(values_near_avg) == len(x), avg


class BoundaryBus:
    """
    A class used to store active power, conductance, averages, and NLI of each
    boundary bus. This should be a less error-prone method of remembering
    measurements than using global variables.

    Two features are important:

        - the constructor receives sample times as arguments, in order to avoid
          synchronization problems with the outer simulation loop;
        - after initializing each object, its NLI and its derivative should be
          fetched using the get_NLI method, which simply receives PMU
          measurements.
    """

    def __init__(
        self,
        name: str,
        delta_t: float,
        delta_T: float,
        tau_s: float,
        epsilon: float,
    ) -> None:
        """
        Initialize boundary bus.

        The sampling periods are left as arguments to avoid synchronization
        conflicts with the external simulation.
        """

        # Name (it never hurts)
        self.name = name
        # Sample periods
        self.delta_t = delta_t  # 20 ms
        self.delta_T = delta_T  # 7 s
        self.tau_s = tau_s  # 1 s
        # Sample numbers
        self.M = 350
        self.Md = 10
        # Threshold
        self.epsilon = epsilon
        # Lists for time series
        self.Gi = []
        self.Pi = []
        self.Gi_bar = []
        self.Pi_bar = []
        self.delta_Gi = []
        self.delta_Pi = []
        self.delta_Gi_bar = []
        self.delta_Pi_bar = []
        self.NLI = []
        self.NLI_bar = []
        self.constant_samples = 0

    def update_GP(self, voltage: complex, current: complex, tk: float) -> None:
        """
        Update measurements from conductance and active power.

        All values are saved with a unique timestamp.
        """

        G = np.real(current / voltage)
        P = np.real(voltage * np.conj(current))
        self.Gi.append((tk, G))
        self.Pi.append((tk, P))

    def update_NLI(self, tk: float, if_states: Sequence[float]) -> None:
        """
        Update NLI and its derivative given the phasor measurements of V and I.

        The NLI is saved with a unique timestamp.

        The NLI does not look exactly like the one from Costas' paper, but this
        is due to differences in the P and G curves that they used. Using their
        data, Fig. 15 from the paper could be reproduced perfectly.

        Although the if_states vector is made up of floats, these can be either
        0 or 1, depending on whether the field current is below (0) or above
        (1) the thermal limit of the field winding.
        """

        # Take average from eqs. 4 and 5
        Gi_bar = average([x[1] for x in self.Gi], self.M)
        Pi_bar = average([x[1] for x in self.Pi], self.M)

        # Save average from eqs. 4 and 5
        self.Gi_bar.append(Gi_bar)
        self.Pi_bar.append(Pi_bar)

        if tk >= self.delta_T:
            # Compute changes from eq. 6
            index_diff = int(self.delta_T // self.tau_s)
            den = self.Gi_bar[-1 - index_diff]
            delta_Gi = self.Gi_bar[-1] - den
            delta_Pi = self.Pi_bar[-1] - self.Pi_bar[-1 - index_diff]

            # Apply test from eq. 7
            if delta_Gi / den >= self.epsilon:
                # Record changes from eq. 6
                self.delta_Gi.append(delta_Gi)
                self.delta_Pi.append(delta_Pi)

                # Take average from eq. 8
                delta_Gi_bar = average(self.delta_Gi, self.Md)
                delta_Pi_bar = average(self.delta_Pi, self.Md)

                # Compute the NLI using eq. 9
                NLI = delta_Pi_bar / delta_Gi_bar
                self.NLI.append(NLI)

                # Apply additional filter (hardcoded as 10)
                NLI_bar = average(self.NLI, 10)
                self.NLI_bar.append((tk, NLI_bar))

            # Otherwise, if the condition is not met, append last value
            elif len(self.NLI_bar) > 0:
                self.NLI_bar.append((tk, self.NLI_bar[-1][1]))

            # If the NLI has remained constant, increase counter
            if (
                len(self.NLI_bar) > 1
                and abs(self.NLI_bar[-1][1] - self.NLI_bar[-2][1]) < 1e-4
            ):
                self.constant_samples += 1
            else:
                self.constant_samples = 0

            # If NLI remained stuck at a negative value for more than 24
            # samples
            if (
                self.constant_samples > 24
                and self.NLI_bar[-1][1] <= 0
                # The states of the field windings (0 or 1) are compared
                # against 0.5 for numerical robustness.
                and all(if_state < 0.5 for if_state in if_states)
            ):
                # Reset everything and mantain NLI at 0.1
                self.delta_Gi = []
                self.delta_Pi = []
                self.NLI = []
                self.NLI_bar.append((tk, 0.1))
                self.constant_samples = 0

    def get_NLI(self, tk: float) -> tuple[float, float]:
        """
        Return NLI and its derivative at time tk.

        By convention, the method returns 1 if no NLI measurements are
        available.
        """

        # Compute quantities
        NLI_bar = 1 if len(self.NLI_bar) == 0 else self.NLI_bar[-1][1]
        NLI_bar_prime = derivative([x[1] for x in self.NLI_bar], self.tau_s)

        return NLI_bar, NLI_bar_prime


class FieldCurrent(control.Detector):
    """
    Measure state of the field current (0 or 1) from all large generators.
    """

    type = "Field current"

    def __init__(self, machine_name: str) -> None:
        """
        Initialize the detector.

        Each detector of field currents is bound to a system to facilitate
        retrieving the measurements.
        """

        self.sys: pf_dynamic.System = None
        self.machine_name: str = machine_name
        self.t_last_measurement: float = -np.inf
        self.period: float = 20e-3
        self.history: list[tuple[float, float]] = []

    def get_required_observables(self) -> list[sim_interaction.Observable]:
        """
        Make sure that the observed machine is in the list of observables.
        """

        return [
            sim_interaction.Observable(
                observed_object=self.sys.get_generator(name=self.machine_name),
                # The observable name is not necessary as we only need that
                # SYNC <machine name> appears in the .dat file.
                obs_name=None,
            )
        ]

    def update_measurements(self, tk: float, indices_to_update: bool) -> None:
        """
        Update the measurements of the state of the field current.

        For other detectors, indices_to_update could be a sequence, perhaps
        even an array, of booleans. However, since the FieldCurrent detector
        only monitors a machine, it can be treated as a single boolean.
        """

        if indices_to_update:
            # Measure state of the field current. The following
            # will look at two things:
            # 1. if a machine is currently overexcited (zdead) and
            # 2. if a machine was over excited at some point in the
            #    past, to the point that the OEL acted (zswitch)
            # Then it will take the more pessimistic state (max)
            if_state = max(
                self.sys.ram.getObs(
                    ["EXC"], [self.machine_name], ["zdead"]
                )[0],
                self.sys.ram.getObs(
                    ["EXC"], [self.machine_name], ["zswitch"]
                )[0]
            )
            # Store those values
            self.history.append((tk, if_state))
            # Reset counter
            self.t_last_measurement = tk

    def get_reading(self):
        """
        Return actual
        """

        return self.history[-1][1] if len(self.history) > 0 else None


class NLI(control.Detector):
    """
    A class that implements the New LIVES Index (NLI) algorithm.
    """

    type = "NLI"

    def __init__(
        self,
        observed_corridor: tuple[str, list[str]],
        h: float,
        delta_T: float,
        tau_s: float,
        epsilon: float,
    ) -> None:
        """
        Initialize the NLI detector.

        Each detector of NLI is bound to a system to facilitate retrieving the
        measurements.
        """

        self.sys: pf_dynamic.System = None
        self.observed_corridor: tuple[str, list[str]] = observed_corridor
        self.boundary_bus = BoundaryBus(
            name=observed_corridor[0],
            delta_t=h,
            delta_T=delta_T,
            tau_s=tau_s,
            epsilon=epsilon,
        )
        # Since the NLI requires updating at two different periods
        # (specifically, 2) observable(s), t_last_measurement is an array of
        # booleans, as well as period.
        self.t_last_measurement: np.ndarray = -np.inf * np.ones(2)
        self.period: np.ndarray = np.array([h, tau_s])

    def get_required_observables(self) -> list[sim_interaction.Observable]:
        """
        Make sure to include involved buses and branches in the observables.
        """

        observables: list[sim_interaction.Observable] = []

        # Obtain buses
        boundary_bus: str = self.observed_corridor[0]
        sending_buses: list[str] = self.observed_corridor[1]

        # Add observables associated to buses
        for bus_name in [boundary_bus] + sending_buses:
            bus = self.sys.get_bus(name=bus_name)
            # The observable name is not necessary as we only need that
            # BUS <bus name> appears in the .dat file.
            observables.append(sim_interaction.Observable(bus, None))

        # Add observables associated to branches. The logic here is completely
        # analogous to the case of buses.
        branches: list[records.Branch] = []
        for sending_bus in sending_buses:
            sending_branches = self.sys.get_branches_between(
                bus_name_1=boundary_bus, bus_name_2=sending_bus
            )
            branches += sending_branches

        for branch in branches:
            # Again, the observable name is not necessary. See above.
            observables.append(sim_interaction.Observable(branch, None))

        return observables

    def measure_VI(self) -> tuple[complex, complex]:
        """
        Measure complex voltages and currents at the monitored bus.
        """

        # Get list with branch names
        boundary_bus = self.observed_corridor[0]
        sending_buses = self.observed_corridor[1]
        branches = [
            b
            for bus in sending_buses
            for b in self.sys.get_branches_between(
                bus_name_1=boundary_bus, bus_name_2=bus
            )
        ]
        branch_names = [b.name for b in branches]

        # Get measurements from RAMSES
        powers = self.sys.ram.getBranchPow(branch_names)

        # Compute P and Q
        P = 0
        Q = 0
        for i, branch in enumerate(branches):
            if boundary_bus == branch.from_bus.name:
                P -= powers[i][0]
                Q -= powers[i][1]
            elif boundary_bus == branch.to_bus.name:
                P -= powers[i][2]
                Q -= powers[i][3]

        # Compute voltage
        v_mag = self.sys.ram.getBusVolt([boundary_bus])[0]
        v_pha = self.sys.ram.getBusPha([boundary_bus])[0]

        # Compute current indirectly
        voltage = v_mag * np.exp(1j * v_pha)
        current = np.conj((P + 1j * Q) / voltage)

        return voltage, current

    def update_measurements(
        self, tk: float, indices_to_update: np.ndarray
    ) -> None:
        """
        Index 1 corresponds to fast measurements, index 2 to slow ones.
        """

        # Get normalized field currents
        if_states = [
            d.get_reading()
            for d in self.sys.detectors
            if isinstance(d, FieldCurrent)
            # if d.type == "Field current"
        ]

        if indices_to_update[0]:
            # Measure voltage and current
            voltage, current = self.measure_VI()
            # Update G and P
            self.boundary_bus.update_GP(
                voltage=voltage, current=current, tk=tk
            )
            # Restore timer
            self.t_last_measurement[0] = tk

        if indices_to_update[1]:
            # Update NLI
            self.boundary_bus.update_NLI(tk=tk, if_states=if_states)
            # Restore timer
            self.t_last_measurement[1] = tk

    def get_reading(self) -> float:
        """
        Get reading of the NLI now.
        """

        return self.boundary_bus.get_NLI(None)[0]
