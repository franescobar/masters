"""
    Detectors and controllers. These classes 'know' (->) each other according to

        System -> Controller -> Detector
          ^___________|____________|

    All Detectors have a method get_value(), whereas all Controllers have a
    method get_actions().

"""

# Modules from this repository
import utils
import records
import sim_interaction

# Modules from the standard library
import copy
import os
import math

# Other modules
import numpy as np
import tabulate


def average(x, N, step=1):
    """
    Return average of the last N elements of x, moving backwards with a step
    (possibly) other than 1. The function returns 0 is x is empty.
    """

    slice = x[-1 : -N * step - 1 : -step]

    return sum(slice) / min(N, len(slice))


def derivative(x, dt):
    """
    Approximate the derivative of x using a secant line from the left.

    If not enough points are given, the function returns zero.
    """

    return None if len(x) < 2 else (x[-1] - x[-2]) / dt


def is_multiple(t, Ts, tol=1e-6):
    """
    Return True if the current time t is a multiple of the sampling period Ts
    up to a tolerance tol.
    """

    return min([t % Ts, Ts - (t % Ts)]) < tol


def remained_constant(x, tol=1e-6):
    """
    Return (boolean, avg), where boolean indicates if x is a constant signal up
    to a tolerance tol and avg is the value it took (its average).
    """

    if len(x) == 0:
        return False, 0
    else:
        avg = sum(x) / len(x)
        values_near_avg = [xi for xi in x if abs(xi - avg) < tol]

        if len(values_near_avg) == len(x):
            boolean = True
        else:
            boolean = False

        return boolean, avg


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

    def __init__(self, name, delta_t, delta_T, tau_s, epsilon):
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

    def update_GP(self, V, I, tk):
        """
        Update measurements from conductance and active power. All values
        are saved with a unique timestamp.
        """

        G = np.real(I / V)
        P = np.real(V * np.conj(I))
        self.Gi.append((tk, G))
        self.Pi.append((tk, P))

    def update_NLI(self, tk, if_states):
        """
        Update NLI and its derivative given the phasor measurements of V and I.
        The NLI is saved with a unique timestamp.

        The NLI does not look exactly like the one from Costas' paper, but
        this is due to differences in the P and G curves that they used. Using
        their data, Fig. 15 from the paper could be reproduced perfectly.
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

                # print('These are the lists')
                # print(self.delta_Gi, self.delta_Pi, self.NLI)
                #
                # print('These are the averages')
                # print(self.delta_Gi_bar, self.delta_Pi_bar, self.NLI_bar)

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

            # If NLI remained stuck at a negative value for more than 24 samples
            if (
                self.constant_samples > 24
                and self.NLI_bar[-1][1] <= 0
                and all(if_state < 0.5 for if_state in if_states)
            ):
                # Reset everything and mantain NLI at 0.1
                self.delta_Gi = []
                self.delta_Pi = []
                self.NLI = []
                self.NLI_bar.append((tk, 0.1))
                self.constant_samples = 0

    def get_NLI(self, tk):
        """
        Return NLI and its derivative at time tk.

        By convention, the method returns 1 if no NLI measurements are
        available.
        """

        # # Update NLI
        # self.update_NLI(tk)

        # Compute quantities
        NLI_bar = 1 if len(self.NLI_bar) == 0 else self.NLI_bar[-1][1]
        NLI_bar_prime = derivative([x[1] for x in self.NLI_bar], self.tau_s)

        return NLI_bar, NLI_bar_prime


class Detector:
    """
    Common feature of detectors: maybe record their readings into an attribute.

    Important: they all 'know' the system they are connected to.
    """

    pass


class FieldCurrent(Detector):
    """
    Measure normalized if.
    """

    type = "Field current"

    def __init__(self, machine_name):
        self.sys = None
        self.machine_name = machine_name
        self.t_last_measurement = -np.inf
        self.period = 20e-3
        self.history = []

    def get_required_observables(self):
        return [
            sim_interaction.Observable(
                self.sys.get_generator(self.machine_name), None
            )
        ]

    def update_measurements(self, tk, indices_to_update):
        if indices_to_update:
            # Get maximum current
            exc = self.sys.get_generator(self.machine_name).exciter
            # Measure field current
            if_state = self.sys.ram.getObs(
                ["EXC"], [self.machine_name], ["zdead"]
            )[0]
            # Store those values
            self.history.append((tk, if_state))
            # Reset counter
            self.t_last_measurement = tk

    def get_reading(self):
        return self.history[-1][1] if len(self.history) > 0 else None


class NLI(Detector):
    type = "NLI"

    def __init__(self, observed_corridor, h, delta_T, tau_s, epsilon):
        self.sys = None
        self.observed_corridor = observed_corridor
        self.boundary_bus = BoundaryBus(
            observed_corridor[0], h, delta_T, tau_s, epsilon
        )
        self.t_last_measurement = -np.inf * np.ones(2)
        self.period = np.array([h, tau_s])

    def get_required_observables(self):
        observables = []

        # Obtain buses
        boundary_bus = self.observed_corridor[0]
        sending_buses = self.observed_corridor[1]

        # Add observables associated to buses
        for bus_name in [boundary_bus] + sending_buses:
            bus = self.sys.get_bus(bus_name)
            observables.append(sim_interaction.Observable(bus, None))

        # Add observables associated to branches
        branches = []
        for sending_bus in sending_buses:
            sending_branches = self.sys.get_branches_between(
                boundary_bus, sending_bus
            )
            branches += sending_branches

        for branch in branches:
            observables.append(sim_interaction.Observable(branch, None))

        return observables

    def measure_VI(self):
        """
        Implement function to measure stuff from RAMSES.
        """

        # Get list with branch names
        boundary_bus = self.observed_corridor[0]
        sending_buses = self.observed_corridor[1]
        branches = [
            b
            for bus in sending_buses
            for b in self.sys.get_branches_between(boundary_bus, bus)
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
        V = v_mag * np.exp(1j * v_pha)
        I = np.conj((P + 1j * Q) / V)

        return V, I

    def update_measurements(self, tk, indices_to_update):
        """
        Index 1 corresponds to fast measurements, index 2 to slow ones.
        """

        # Get normalized field currents
        if_states = [
            d.get_reading()
            for d in self.sys.detectors
            if d.type == "Field current"
        ]

        if indices_to_update[0]:
            # Measure voltage and current
            V, I = self.measure_VI()
            # Update G and P
            self.boundary_bus.update_GP(V, I, tk)
            # Restore timer
            self.t_last_measurement[0] = tk

        if indices_to_update[1]:
            # Update NLI
            self.boundary_bus.update_NLI(tk, if_states)
            # Restore timer
            self.t_last_measurement[1] = tk

    def get_reading(self):
        """
        Get current reading of the meter.
        """

        return self.boundary_bus.get_NLI(None)[0]


class LIVES(Detector):
    def __init__(self):
        pass

    def get_value(self):
        pass


class Controller:
    """
    All non-OLTCs controllers must have an attribute called
    overrides_OLTCs (boolean).
    """

    overrides_OLTCs = False
    type = None

    def get_actions(self):
        """
        Return a list of disturbances.

        All controllers must implement this method.
        """

        pass


class Pabon_controller(Controller):
    def __init__(self):
        pass

    def get_actions(self):
        pass


class Load_Shedder(Controller):
    pass


class MPC_controller(Controller):
    overrides_OLTCs = True

    def __init__(self):
        self.sys = None
        self.twin = None
        self.Np = None
        self.Nc = None
        self.period = None
        self.t_last_action = 0
        self.observed_corridors = []
        self.controlled_transformers = []
        self.B = 0
        self.T = 0
        self.current_iter = 0
        self.solutions = []
        self.dr = []
        self.dP = []
        self.dQ = []
        self.slacks = []

    def __str__(self):
        """
        Print relevant settings of the MPC controller.
        """

        data = [
            ["Observed corridors", self.observed_corridors],
            ["Controlled transformers", self.controlled_transformers],
            ["Np", self.Np],
            ["Nc", self.Nc],
            ["Period (s)", self.period],
            ["Lower NLI", self.NLI_lower.T[0]],
            ["Upper NLI", self.NLI_upper.T[0]],
            ["Lower VD (pu)", self.VD_lower.T[0]],
            ["Upper VD (pu)", self.VD_upper.T[0]],
            ["Lower VT (pu)", self.VT_lower.T[0]],
            ["Upper VT (pu)", self.VT_upper.T[0]],
            ["Weight on du (R1)", self.R1],
            ["Weight on u* deviations (R2)", self.R2],
            ["Weight on slacks (S)", self.S],
            ["Lower u", self.u_lower.T[0]],
            ["Upper u", self.u_upper.T[0]],
            ["Lower du", self.du_lower.T[0]],
            ["Upper du", self.du_upper.T[0]],
        ]

        table = tabulate.tabulate(data)

        return table

    def set_period(self, period):
        """
        Set controller's period (time between consecutive actions).
        """

        self.period = period

    def add_observed_corridor(self, boundary_bus, sending_buses):
        """
        Add corridor by specifying bus names.
        """

        self.observed_corridors.append((boundary_bus, sending_buses))
        self.B += 1

    def add_controlled_transformers(self, transformers):
        """
        Add named transformer.
        """

        self.controlled_transformers += transformers
        self.T += len(transformers)

    def set_horizons(self, Np, Nc):
        """
        Set prediction and control horizons.
        """

        self.Np = Np
        self.Nc = Nc

    @staticmethod
    def v_bound(bus, Np, iter, half_db, v_set=1.0):
        """
        Return two arrays of height Np with the voltage bounds of this bus.

        In set_bounds (see below), VT_fun can be replaced by any function (or
        lambda expression) that receives (bus, Np, iter) and behaves like
        this function.
        """

        v_min = (v_set - half_db) * np.ones([Np, 1])
        v_max = (v_set + half_db) * np.ones([Np, 1])

        return v_min, v_max

    @staticmethod
    def power_bound(bus, Nc, iter):
        """
        Return four arrays of height Nc with the power bounds of this bus.

        In set_bounds (see below), P_fun and Q_fun can be replaced by any
        function (or lambda expression) that receives (bus, Nc, iter) and
        behaves like this function.
        """

        p_min = -1e6 * np.ones([Nc, 1])
        p_max = -1.0 * p_min
        dp_min = 1.0 * p_min
        dp_max = 1.0 * p_max

        return p_min, p_max, dp_min, dp_max

    def set_bounds(
        self,
        # Fixed value
        NLI_min=0.1,
        # Return a tuple of arrays (min, max)
        VT_fun=lambda bus, Np, iter: MPC_controller.v_bound(
            bus, Np, iter, half_db=0.1
        ),
        VD_fun=lambda bus, Np, iter: MPC_controller.v_bound(
            bus, Np, iter, half_db=0.05
        ),
        # Return a tuple of arrays (min, max, min_delta, max_delta)
        P_fun=lambda bus, Nc, iter: MPC_controller.power_bound(bus, Nc, iter),
        Q_fun=lambda bus, Nc, iter: MPC_controller.power_bound(bus, Nc, iter),
    ):
        """
        Set bounds on u, the voltages, the NLI.

        These bounds may depend on the bus, the horizons, and the number of
        iterations that the controller has made. This implementation suffices
        for most experiments.
        """

        trafos = [
            self.sys.get_transformer(t) for t in self.controlled_transformers
        ]

        # Read limits on r and dr
        r_min = np.array([[t.OLTC.nmin] for t in trafos])
        r_max = np.array([[t.OLTC.nmax] for t in trafos])
        dr_min = np.array([[-t.OLTC.step] for t in trafos])
        dr_max = np.array([[t.OLTC.step] for t in trafos])

        # Build bounds on u and du
        r_bounds = [r_min, r_max, dr_min, dr_max]

        def get_u_bound(i, k):
            """
            Get array with i-th bound (min, max, min_delta, max_delta) at tk.
            """

            r_bound = r_bounds[i]
            P_bound = np.array(
                [
                    [
                        P_fun(t.get_LV_bus(), self.Nc, self.current_iter)[i][
                            k, 0
                        ]
                    ]
                    for t in trafos
                ]
            )
            Q_bound = np.array(
                [
                    [
                        Q_fun(t.get_LV_bus(), self.Nc, self.current_iter)[i][
                            k, 0
                        ]
                    ]
                    for t in trafos
                ]
            )

            return np.vstack([r_bound, P_bound, Q_bound])

        self.u_lower = np.vstack([get_u_bound(0, k) for k in range(self.Nc)])
        self.u_upper = np.vstack([get_u_bound(1, k) for k in range(self.Nc)])
        self.du_lower = np.vstack([get_u_bound(2, k) for k in range(self.Nc)])
        self.du_upper = np.vstack([get_u_bound(3, k) for k in range(self.Nc)])

        def get_VT_bound(i, k):
            """
            Get array with i-th bound (min, max) at tk.
            """

            return np.array(
                [
                    [
                        VT_fun(t.get_HV_bus(), self.Np, self.current_iter)[i][
                            k, 0
                        ]
                    ]
                    for t in trafos
                ]
            )

        # Build bounds on VT
        self.VT_lower = np.vstack([get_VT_bound(0, k) for k in range(self.Np)])
        self.VT_upper = np.vstack([get_VT_bound(1, k) for k in range(self.Np)])

        def get_VD_bound(i, k):
            """
            Get array with i-th bound (min, max) at tk.
            """

            return np.array(
                [
                    [
                        VD_fun(t.get_LV_bus(), self.Np, self.current_iter)[i][
                            k, 0
                        ]
                    ]
                    for t in trafos
                ]
            )

        # Build bounds on VD
        self.VD_lower = np.vstack([get_VD_bound(0, k) for k in range(self.Np)])
        self.VD_upper = np.vstack([get_VD_bound(1, k) for k in range(self.Np)])

        # Set limits on the NLI (stability)
        self.NLI_lower = NLI_min * np.ones([self.Np * self.B, 1])
        self.NLI_upper = 1e6 * self.NLI_lower

    @staticmethod
    def some_setpoint(bus_or_transformer, Nc, iter):
        """
        Return an array of height Nc with the bounds on the control variable.

        In define_setpoints (see below), r_fun, P_fun, and Q_fun can be replaced
        by any function (or lambda expression) that receives
        (bus_or_transformer, Nc, iter) and behaves like this function.
        """

        return np.zeros([Nc, 1])

    def define_setpoints(
        self,
        r_fun=lambda transformer, Nc, iter: MPC_controller.some_setpoint(
            transformer, Nc, iter
        ),
        P_fun=lambda bus, Nc, iter: MPC_controller.some_setpoint(
            bus, Nc, iter
        ),
        Q_fun=lambda bus, Nc, iter: MPC_controller.some_setpoint(
            bus, Nc, iter
        ),
    ):
        """
        Define setpoint of u.
        """

        trafos = [
            self.sys.get_transformer(t) for t in self.controlled_transformers
        ]

        def get_u_setpoints(k):
            """
            Get array with setpoints at time k.
            """

            r_setpoint = np.array(
                [[r_fun(t, self.Nc, self.current_iter)[k, 0]] for t in trafos]
            )
            P_setpoint = np.array(
                [
                    [P_fun(t.get_LV_bus(), self.Nc, self.current_iter)[k, 0]]
                    for t in trafos
                ]
            )
            Q_setpoint = np.array(
                [
                    [Q_fun(t.get_LV_bus(), self.Nc, self.current_iter)[k, 0]]
                    for t in trafos
                ]
            )

            return np.vstack([r_setpoint, P_setpoint, Q_setpoint])

        self.u_star = np.vstack([get_u_setpoints(k) for k in range(self.Nc)])

    @staticmethod
    def some_u_penalization(bus_or_transformer, Nc, iter):
        """
        Return array of height Nc with penalization factors.

        This method is used as a default for penalizations in P, Q, dP, and dQ.
        """

        return np.ones([Nc, 1])

    @staticmethod
    def some_slack_penalization(bus, Np, iter):
        """
        Return a tuple of penalizations for the slacks.

        The first element of the tuple penalizes the lower-bound violations,
        the second one penalizes the upper-bound violations.

        Because the penalization factors are kept constant across the horizon,
        the method returns a tuple of floats, not a tuple of arrays.
        """

        return 1.0, 1.0

    def set_weights(
        self,
        dr_fun=lambda transformer, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(transformer, Nc, iter),
        dP_fun=lambda bus, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(bus, Nc, iter),
        dQ_fun=lambda bus, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(bus, Nc, iter),
        r_devs_fun=lambda transformer, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(transformer, Nc, iter),
        P_devs_fun=lambda bus, Nc, iter: MPC_controller.some_u_penalization(
            bus, Nc, iter
        ),
        Q_devs_fun=lambda bus, Nc, iter: MPC_controller.some_u_penalization(
            bus, Nc, iter
        ),
        slacks_fun=lambda bus, Np, iter: MPC_controller.some_slack_penalization(
            bus, Np, iter
        ),
    ):
        """
        Define weight matrices.

        By default, neither r nor dr are penalized.
        """

        trafos = [
            self.sys.get_transformer(t) for t in self.controlled_transformers
        ]

        def get_du_penalizations(k):
            """
            Get array with penalization factors for du at time k.
            """

            dr_pen = np.array(
                [[dr_fun(t, self.Nc, self.current_iter)[k, 0]] for t in trafos]
            )

            dP_pen = np.array(
                [
                    [dP_fun(t.get_LV_bus(), self.Nc, self.current_iter)[k, 0]]
                    for t in trafos
                ]
            )

            dQ_pen = np.array(
                [
                    [dQ_fun(t.get_LV_bus(), self.Nc, self.current_iter)[k, 0]]
                    for t in trafos
                ]
            )

            return np.vstack([dr_pen, dP_pen, dQ_pen])

        factors = np.vstack([get_du_penalizations(k) for k in range(self.Nc)])
        # Transposing gets a row vector, indexing a 1-D array, np.diag converts
        # that 1-D array to a diagonal matrix
        self.R1 = np.diag(factors.T[0])

        def get_u_devs_penalizations(k):
            """
            Get array with penalization factors for u deviations at time k.
            """

            r_devs_pen = np.array(
                [
                    [r_devs_fun(t, self.Nc, self.current_iter)[k, 0]]
                    for t in trafos
                ]
            )

            P_devs_pen = np.array(
                [
                    [
                        P_devs_fun(t.get_LV_bus(), self.Nc, self.current_iter)[
                            k, 0
                        ]
                    ]
                    for t in trafos
                ]
            )

            Q_devs_pen = np.array(
                [
                    [
                        Q_devs_fun(t.get_LV_bus(), self.Nc, self.current_iter)[
                            k, 0
                        ]
                    ]
                    for t in trafos
                ]
            )

            return np.vstack([r_devs_pen, P_devs_pen, Q_devs_pen])

        factors = np.vstack(
            [get_u_devs_penalizations(k) for k in range(self.Nc)]
        )
        # For the following hack, see above in this same method
        self.R2 = np.diag(factors.T[0])

        # Build penalizations on the slacks
        VT_pen = np.array(
            [
                [slacks_fun(t.get_HV_bus(), self.Np, self.current_iter)[i]]
                for i in range(2)
                for t in trafos
            ]
        )

        VD_pen = np.array(
            [
                [slacks_fun(t.get_LV_bus(), self.Np, self.current_iter)[i]]
                for i in range(2)
                for t in trafos
            ]
        )

        factors = np.vstack([VT_pen, VD_pen])
        # For the following hack, see above in this same method
        self.S = np.diag(factors.T[0])

    def build_structural_matrices(self):
        """
        Build structural matrices, independent from measurements, bounds, etc.
        """

        # Build C1 and C2
        self.C1 = np.vstack([np.eye(3 * self.T) for i in range(self.Nc)])
        rows = []
        for row_no in range(self.Nc):
            cols = []
            for col_no in range(self.Nc):
                if col_no <= row_no:
                    cols.append(np.eye(3 * self.T))
                else:
                    cols.append(np.zeros([3 * self.T, 3 * self.T]))
            # Stack columns into rows
            rows.append(np.hstack(cols))
        # Stack rows into matrix
        self.C2 = np.vstack(rows)

        # Build F matrices
        self.F_N = np.vstack([np.eye(self.B) for i in range(self.Np)])
        self.F_VT = np.vstack([np.eye(self.T) for i in range(self.Np)])
        self.F_VD = np.vstack([np.eye(self.T) for i in range(self.Np)])

        # Build A matrices
        I = np.eye(self.T)
        self.A1 = np.vstack(
            [np.hstack([I, 0 * I, 0 * I, 0 * I]) for i in range(self.Np)]
        )
        self.A2 = np.vstack(
            [np.hstack([0 * I, I, 0 * I, 0 * I]) for i in range(self.Np)]
        )
        self.A3 = np.vstack(
            [np.hstack([0 * I, 0 * I, I, 0 * I]) for i in range(self.Np)]
        )
        self.A4 = np.vstack(
            [np.hstack([0 * I, 0 * I, 0 * I, I]) for i in range(self.Np)]
        )

    def add_twin(self):
        self.twin = self.sys.get_twin()
        # ram = self.sys.ram
        # self.sys.ram = None
        # mycopy = copy.deepcopy(self.sys)
        # self.sys.ram = ram
        # self.twin = mycopy

    def update_twin(self):
        """
        Something that get_twin should do is to overwrite some attributes
        (those associated to measurements) so that they actually come from
        RAMSES. Of course, I should not run any power flows afterwards, because
        that would overwrite the measurements.

        Important: Twin should modify the value in detectors, which should,
        in turn, be linked to the system. This is because doing
        get_min_NLI will run power flows (see comment in update_twin)

        Idea: add a method to controller called: solve_twin.
        This would run a power flow and update the Detector
        (this should be a method inside Detector:

        An attribute of the Detector is:
            - value
            - observed_corridors
            - controlled_transformers

        A method is
            - update_value(): this fetches get_min... and assigns it to value
            -

        Then, I would never compute the NLI directly. Instead, I would do

            In fact: there should be a way to decide: are measurements
            taken from the static or from the dynamic model?

            This would be the case, for instance, in the Detector:

            uodate_value_static()

            and

            update_value_dynamic()

        This is giving me a headache... I should look into it later.
        """

        # Temporary implementation: perfect measurements
        self.twin = self.sys.get_twin()

    def get_derivatives(self):
        """
        Get all derivatives in one shot.
        """

        # Initialize sensitivity matrices
        self.partial_u_N = np.zeros([self.B, 3 * self.T])
        self.partial_u_VT = np.zeros([self.T, 3 * self.T])
        self.partial_u_VD = np.zeros([self.T, 3 * self.T])

        # Define changes for all sensitivities
        dn = 1e-3
        dP = 1e-3
        dQ = 1e-3

        # Update twin
        if self.twin is None:
            self.add_twin()

        # Copy twin to compute derivatives
        sys = copy.deepcopy(self.twin)

        # Iterate over all substations
        for trafo_no, trafo in enumerate(self.controlled_transformers):
            for attr_no, attr in enumerate(["n", "PL", "QL"]):
                # Run initial power flow
                sys.run_pf(flat_start=False)

                # Evaluate all transmission voltages
                VTs_0 = [
                    sys.get_transformer(transformer).get_HV_bus().V
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all distribution voltages
                VDs_0 = [
                    sys.get_transformer(transformer).get_LV_bus().V
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all NLIs
                NLIs_0 = [
                    sys.get_min_NLI(corr, self.controlled_transformers)
                    for corr in self.observed_corridors
                ]

                # Change the right parameter at the right transformer.
                # Because PL and QL are treated as per-unit values by the whole
                # run_pf machinery, those are the units of these derivatives.
                # Furthermore, the derivatives are computed as if positive dP
                # and dQ meant increment in loads. Hence, that's the meaning
                # of the values spit out by the MPC: if positive, they are
                # load increments, and they always are in pu.
                if attr == "n":
                    sys.get_transformer(trafo).n += dn
                elif attr == "PL":
                    sys.get_transformer(trafo).get_LV_bus().PL += dP
                elif attr == "QL":
                    sys.get_transformer(trafo).get_LV_bus().QL += dQ

                # Run second power flow
                sys.run_pf(flat_start=False)

                # Evaluate all transmission voltages (again)
                VTs_1 = [
                    sys.get_transformer(transformer).get_HV_bus().V
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all distribution voltages (again)
                VDs_1 = [
                    sys.get_transformer(transformer).get_LV_bus().V
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all NLIs (again)
                NLIs_1 = [
                    sys.get_min_NLI(corr, self.controlled_transformers)
                    for corr in self.observed_corridors
                ]

                # Undo changes (to avoid doing multiple deep copies)
                if attr == "n":
                    sys.get_transformer(trafo).n -= dn
                elif attr == "PL":
                    sys.get_transformer(trafo).get_LV_bus().PL -= dP
                elif attr == "QL":
                    sys.get_transformer(trafo).get_LV_bus().QL -= dQ

                # Select delta
                if attr == "n":
                    dx = dn
                elif attr == "PL":
                    dx = dP
                elif attr == "QL":
                    dx = dQ

                # Compute derivatives
                der_NLI = (np.array(NLIs_1) - np.array(NLIs_0)) / dx
                der_VT = (np.array(VTs_1) - np.array(VTs_0)) / dx
                der_VD = (np.array(VDs_1) - np.array(VDs_0)) / dx

                # Compute and store derivatives
                self.partial_u_N[:, attr_no * self.T + trafo_no] = der_NLI
                self.partial_u_VT[:, attr_no * self.T + trafo_no] = der_VT
                self.partial_u_VD[:, attr_no * self.T + trafo_no] = der_VD

        print(" \nThese are the sensitivities of the NLI w.r.t. [r, P, Q]:\n")
        print(self.partial_u_N)
        input("\nPress ENTER to continue\n")

        print(" These are the sensitivities of VT w.r.t. [r, P, Q]:\n")
        print(self.partial_u_VT)
        input("\nPress ENTER to continue\n")

        print(" These are the sensitivities of VD w.r.t. [r, P, Q]:\n")
        print(self.partial_u_VD)
        input("\nPress ENTER to continue\n")

    def get_cache_root(self):
        return self.sys.name + "-cache-"

    def has_cache(self):
        return False
        # return all(os.path.exists(self.get_cache_root() + matrix + '.txt')
        #            for matrix in ['N', 'VT', 'VD'])

    def build_sensitivities(self):
        """
        Stack derivatives into matrices.
        """

        if self.has_cache():
            # Update twin
            if self.twin is None:
                self.add_twin()

            self.D_u_N = np.loadtxt(self.get_cache_root() + "N.txt")
            self.D_u_VT = np.loadtxt(self.get_cache_root() + "VT.txt")
            self.D_u_VD = np.loadtxt(self.get_cache_root() + "VD.txt")

        else:
            # Compute derivatives
            self.get_derivatives()

            # Build sensitivities
            rows_N = []
            rows_VT = []
            rows_VD = []
            for row_no in range(self.Np):
                # Initialize list for columns
                cols_N = []
                cols_VT = []
                cols_VD = []
                # Compute and store each column
                for col_no in range(self.Nc):
                    if col_no <= row_no:
                        cols_N.append(self.partial_u_N)
                        cols_VT.append(self.partial_u_VT)
                        cols_VD.append(self.partial_u_VD)
                    else:
                        cols_N.append(0 * self.partial_u_N)
                        cols_VT.append(0 * self.partial_u_VT)
                        cols_VD.append(0 * self.partial_u_VD)
                # Form row by stacking columns
                row_N = np.hstack(cols_N)
                row_VT = np.hstack(cols_VT)
                row_VD = np.hstack(cols_VD)
                # Save row
                rows_N.append(row_N)
                rows_VT.append(row_VT)
                rows_VD.append(row_VD)
            # Form final matrices by stacking rows
            self.D_u_N = np.vstack(rows_N)
            self.D_u_VT = np.vstack(rows_VT)
            self.D_u_VD = np.vstack(rows_VD)

            # Dump files for next run
            np.savetxt(self.get_cache_root() + "N.txt", self.D_u_N)
            np.savetxt(self.get_cache_root() + "VT.txt", self.D_u_VT)
            np.savetxt(self.get_cache_root() + "VD.txt", self.D_u_VD)

    def get_measurements(self):
        """
        Measurements are taken from self.twin as they are.

        The method System.get_twin() is the one that should introduce
        randomization and corruptions.
        """

        if (
            not hasattr(self, "D_u_N")
            or not hasattr(self, "D_u_VT")
            or not hasattr(self, "D_u_VD")
        ):
            self.build_sensitivities()

        # Update the twin
        self.sys.get_twin()

        # Initialize arrays
        u_meas = np.zeros([3 * self.T, 1])
        NLI_meas = np.zeros([self.B, 1])
        VT_meas = np.zeros([self.T, 1])
        VD_meas = np.zeros([self.T, 1])

        # Measure u
        u_meas[:, 0] = (
            [
                self.twin.get_transformer(t).n
                for t in self.controlled_transformers
            ]
            + [
                self.twin.get_transformer(t).get_LV_bus().PL
                for t in self.controlled_transformers
            ]
            + [
                self.twin.get_transformer(t).get_LV_bus().QL
                for t in self.controlled_transformers
            ]
        )

        # Measure the NLI directly from the detectors
        for i, c in enumerate(self.observed_corridors):
            # Find first (only) detector that matches the requirements
            detectors = [
                d
                for d in self.twin.detectors
                if d.type == "NLI" and d.observed_corridor == c
            ]
            detector = detectors[0]
            # Store its reading
            NLI_meas[i, 0] = detector.get_reading()

        # Measure VT
        VT_meas[:, 0] = [
            self.twin.get_transformer(t).get_HV_bus().V
            for t in self.controlled_transformers
        ]

        # Measure VD
        VD_meas[:, 0] = [
            self.twin.get_transformer(t).get_LV_bus().V
            for t in self.controlled_transformers
        ]

        return u_meas, NLI_meas, VT_meas, VD_meas

    def update_measurement_dependent_matrices(self):
        # Get measurements
        u_meas, NLI_meas, VT_meas, VD_meas = self.get_measurements()

        # Build P matrix (quadratic part of the cost function)
        P00 = self.R1 + self.C2.T @ self.R2 @ self.C2
        P01 = np.zeros([self.Nc * 3 * self.T, 4 * self.T])
        P10 = P01.T
        P11 = self.S

        self.P_matrix = np.vstack(
            [np.hstack([P00, P01]), np.hstack([P10, P11])]
        )

        # Build q matrix (linear part of the cost function)
        q00 = self.C2.T @ self.R2 @ (self.C1 @ u_meas - self.u_star)
        q10 = np.zeros([4 * self.T, 1])

        self.q_matrix = np.vstack([q00, q10])

        # Build G matrix (LHS of constraints)
        I_Nc3T = np.eye(self.Nc * 3 * self.T)
        zero_Nc3T = np.zeros([self.Nc * 3 * self.T, 4 * self.T])
        zero_NpB = np.zeros([self.Np * self.B, 4 * self.T])
        self.G_matrix = np.vstack(
            [
                np.hstack([I_Nc3T, zero_Nc3T]),
                np.hstack([-I_Nc3T, zero_Nc3T]),
                np.hstack([self.C2, zero_Nc3T]),
                np.hstack([-self.C2, zero_Nc3T]),
                np.hstack([self.D_u_N, zero_NpB]),
                np.hstack([-self.D_u_N, zero_NpB]),
                np.hstack([self.D_u_VT, -self.A2]),
                np.hstack([-self.D_u_VT, -self.A1]),
                np.hstack([self.D_u_VD, -self.A4]),
                np.hstack([-self.D_u_VD, -self.A3]),
            ]
        )

        # Build h matrix (RHS of constraints)
        self.h_matrix = np.vstack(
            [
                self.du_upper,
                -self.du_lower,
                self.u_upper - self.C1 @ u_meas,
                -self.u_lower + self.C1 @ u_meas,
                self.NLI_upper - self.F_N @ NLI_meas,
                -self.NLI_lower + self.F_N @ NLI_meas,
                self.VT_upper - self.F_VT @ VT_meas,
                -self.VT_lower + self.F_VT @ VT_meas,
                self.VD_upper - self.F_VD @ VD_meas,
                -self.VD_lower + self.F_VD @ VD_meas,
            ]
        )

    def solve_optimization(self):
        return utils.cvxopt_solve_qp(
            self.P_matrix, self.q_matrix, G=self.G_matrix, h=self.h_matrix
        )

    def get_actions(self):
        # Initialize disturbances
        dists = []

        # Count iteration
        self.current_iter += 1

        # Build everything
        self.update_measurement_dependent_matrices()

        # Solve optimization
        x = self.solve_optimization()
        self.solutions.append(x)

        # Filter out solutions
        dr = x[: self.T]
        dP = x[self.T : 2 * self.T]
        dQ = x[2 * self.T : 3 * self.T]
        slacks = x[-4 * self.T :]

        # Store solutions from this iteration
        self.dr.append(dr)
        self.dP.append(dP)
        self.dQ.append(dQ)
        self.slacks.append(slacks)

        # Send changes in tap ratio to transformers in the 'real' system
        overridable_OLTCs = [
            OLTC for OLTC in self.sys.OLTC_controllers if OLTC.is_overridable
        ]
        for i, OLTC in enumerate(overridable_OLTCs):
            # Add dr
            OLTC.increment_r(dr[i])
            # Try to move the OLTC
            dist = OLTC.try_to_move_OLTC()
            if dist is not None:
                # The method try_to_move_OLTC returns a list of disturbances
                dists += dist

        # Send signals to the substations
        dP_requests = dict(zip(self.controlled_transformers, dP))
        dQ_requests = dict(zip(self.controlled_transformers, dQ))
        for t in self.controlled_transformers:
            coordinator = next(
                (
                    c
                    for c in self.sys.controllers
                    if c.type == "Coordinator" and c.transformer_name == t
                ),
                None,
            )
            if coordinator is not None:
                coordinator.increment_request(
                    delta_P_load_pu=dP_requests[t],
                    delta_Q_load_pu=dQ_requests[t],
                )

        # There is no need to send those changes to the twin because it is
        # updated:
        self.update_twin()

        #
        u_meas, NLI_meas, VT_meas, VD_meas = self.get_measurements()
        print("\n\nMeasurements by MPC: u", u_meas.T, "\n")
        print("Measurements by MPC: NLI", NLI_meas.T, "\n")
        print("Measurements by MPC: VT", VT_meas.T, "\n")
        print("Measurements by MPC: VD", VD_meas.T, "\n")
        print("Disturbances required by MPC", [str(d) for d in dists], "\n")

        return dists


class Coordinator(Controller):
    """
    Coordinator installed at a step-down substation.
    """

    type = "Coordinator"
    sigma_P_min = -5
    sigma_P_max = 5
    sigma_Q_min = -5
    sigma_Q_max = 5

    def __init__(
        self,
        transformer_name,
        min_P_injection_MVA,
        max_P_injection_MVA,
        min_Q_injection_MVA,
        max_Q_injection_MVA,
    ):
        self.sys = None
        self.transformer_name = transformer_name
        self.min_P_injection_MVA = min_P_injection_MVA
        self.max_P_injection_MVA = max_P_injection_MVA
        self.min_Q_injection_MVA = min_Q_injection_MVA
        self.max_Q_injection_MVA = max_Q_injection_MVA
        self.requested_P_injection_MVA = 0
        self.requested_Q_injection_MVA = 0
        self.period = 1
        self.t_last_action = 0
        self.available_injectors = None
        self.sigma = []

    def __str__(self):
        inj_names = [inj.name for inj in self.available_injectors]

        controller_types = [c.type for c in self.sys.controllers]

        controllable_inj = []
        for inj in self.available_injectors:
            prefix = inj.prefix.split(" ")
            if len(prefix) == 2 and prefix[1] in controller_types:
                controllable_inj.append(inj.name)

        data = [
            ["Substation", self.transformer_name],
            ["Injectors downstream", inj_names],
            ["Controllable injectors", controllable_inj],
            ["Period (s)", self.period],
            ["Minimum P injection (MW)", self.min_P_injection_MVA],
            ["Max. P injection (MW)", self.max_P_injection_MVA],
            ["Min. Q injection (Mvar)", self.min_Q_injection_MVA],
            ["Max. A injection (Mvar)", self.max_Q_injection_MVA],
            ["Min. P signal", self.sigma_P_min],
            ["Max. P signal", self.sigma_P_max],
            ["Min. Q signal", self.sigma_Q_min],
            ["Max. Q signal", self.sigma_Q_max],
            ["Signal when OK", 0],
        ]

        table = tabulate.tabulate(data)

        return table

    def increment_request(self, delta_P_load_pu, delta_Q_load_pu):
        """
        This method is the one used by the MPC.
        """

        # print(f'The MPC sent dP = {delta_P_load_pu} pu and dQ = {delta_Q_load_pu} pu')
        # Decrement because the MPC thinks the deltas refer to loads
        self.requested_P_injection_MVA -= delta_P_load_pu * self.sys.Sb
        self.requested_Q_injection_MVA -= delta_Q_load_pu * self.sys.Sb

    def get_actions(self):
        delay = 10e-3
        tk = self.sys.get_t_now() + delay

        # Get signal
        sigma = self.get_sigma()

        # Store values of sigma
        self.sigma.append((sigma["P"], sigma["Q"]))
        print(" Sending sigma =", (sigma["P"], sigma["Q"]))

        # print(f'Coordinator at substation {self.transformer_name} is sending {sigma} to its DERs')
        # print(f'MPC is requesting P = {self.requested_P_injection_MVA} MW and Q = {self.requested_Q_injection_MVA} Mvar from this coordinator')
        # Get available injectors
        injectors = self.get_available_injectors()

        # Apply appropriate controller
        dists = []
        for inj in injectors:
            controller = next(
                (
                    c
                    for c in self.sys.controllers
                    if c.type is not None and c.type in inj.prefix
                ),
                None,
            )
            if controller is not None:
                dists += controller.get_actions(tk, inj, sigma)

        return dists

    @staticmethod
    def interpolate(
        requested_power, power_min, power_max, sigma_min, sigma_max
    ):
        if requested_power < power_min:
            return sigma_min
        elif power_min <= requested_power < 0:
            frac = requested_power / power_min
            return math.ceil(frac * sigma_min)
        elif 0 <= requested_power <= power_max:
            frac = requested_power / power_max
            return math.floor(frac * sigma_max)
        elif power_max < requested_power:
            return sigma_max

    def get_sigma(self):
        """
        Get sigma based on the requested powers.
        """

        sigma_P = self.interpolate(
            self.requested_P_injection_MVA,
            self.min_P_injection_MVA,
            self.max_P_injection_MVA,
            self.sigma_P_min,
            self.sigma_P_max,
        )

        sigma_Q = self.interpolate(
            self.requested_Q_injection_MVA,
            self.min_Q_injection_MVA,
            self.max_Q_injection_MVA,
            self.sigma_Q_min,
            self.sigma_Q_max,
        )

        return {
            "P": sigma_P,
            "Q": sigma_Q,
            "min_P": self.sigma_P_min,
            "max_P": self.sigma_P_max,
            "min_Q": self.sigma_Q_min,
            "max_Q": self.sigma_Q_max,
        }

    def get_available_injectors(self):
        # Maybe initialize available injectors
        if self.available_injectors is None:
            t = self.sys.get_transformer(self.transformer_name)
            buses = self.sys.isolate_buses_by_kV(t.get_LV_bus())
            self.available_injectors = [
                inj for inj in self.sys.injectors if inj.bus in buses
            ]

        return self.available_injectors


class DERA_Controller(Controller):
    type = "DER_A"
    period = np.inf
    t_last_action = 0
    sys = None

    @staticmethod
    def get_ref(sigma, sigma_min, sigma_max, power_0, SN):
        """
        Map sigma (either P or Q) to reference (either P or Q).
        """

        if sigma < 0:
            frac = sigma / abs(sigma_min)
        else:
            frac = sigma / abs(sigma_max)

        return power_0 / SN + frac * (1 - power_0 / SN * np.sign(sigma))

    def get_actions(self, tk, element, sigma):
        """
        Change Pref and Qref according to values of sigma.
        """

        P0 = element.P0
        Q0 = element.Q0
        SN = element.SNOM

        Pref = self.get_ref(sigma["P"], sigma["min_P"], sigma["max_P"], P0, SN)
        Qref = self.get_ref(sigma["Q"], sigma["min_Q"], sigma["max_P"], Q0, SN)

        return [
            sim_interaction.Disturbance(
                tk, element, par_name="Pref", par_value=Pref
            ),
            sim_interaction.Disturbance(
                tk, element, par_name="Qref", par_value=Qref
            ),
        ]
