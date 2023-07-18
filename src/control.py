"""
Classes for controllers and detectors.

These classes 'know' (->) each other according to

    System -> Controller -> Detector
     |  ^___________|_________|  ^
     |___________________________|

All Detectors have methods get_reading() and update_measurements(), whereas all
Controllers have a method get_actions().
"""

# Modules from this repository
import utils
import records
import sim_interaction

# Modules from the standard library
import copy
import os
import math
from collections.abc import Sequence
from typing import Union

# Other modules
import numpy as np
import tabulate


class Detector:
    """
    A detector is a class with a method get_reading() that returns a float.

    Important: they all 'know' the system they are connected to.
    """

    pass


class Controller:
    """
    All non-OLTCs controllers must have an attribute called overrides_OLTCs
    (boolean).
    """

    overrides_OLTCs = False
    type = None

    def get_actions(self):
        """
        Return a list of disturbances.

        All controllers must implement this method.
        """

        pass


class MPC_controller(Controller):
    """
    The MPC controller that forms the basis of this thesis.

    All the matrices and  coefficients were pain-stakingly revised on 2023-07-10
    and 2023-07-11.
    """

    overrides_OLTCs = True

    def __init__(self) -> None:
        # We first bind the controller to an instance of the System class, which
        # is useful for reading information about the network.
        self.sys: "pf_dynamic.System" = None

        # We then initialize the prediction and control horizons and the
        # sampling period. These can be changed later with the methods
        # set_horizons() and set_period().
        self.Np: int = None
        self.Nc: int = None
        self.period: float = None

        # The attribute t_last_action is used to know when to trigger the
        # controller. It is initialized to 0, and updated at every iteration.
        self.t_last_action: float = 0

        # We then initialize containers for the observed corridors and
        # controlled transformers. We don't need to do this for the DERs,
        # because we will send signals to all of them, as long as they are
        # located downstream of a controlled transformer.
        self.observed_corridors: tuple[str, list[str]] = []
        self.controlled_transformers: list[str] = []
        # B is the number of boundary buses, whereas T is the number of
        # controlled transformers. These are used to build the structural
        # matrices later on.
        self.B: int = 0
        self.T: int = 0

        # When solving the optimization, it might be useful to have bounds
        # that depend not only on the bus, but also on the iteration number
        # (i.e., sooner or later within the same prediction horizon). This
        # is kept track of by the attribute current_iter.
        self.current_iter: int = 0

        # After having solved the optimization, we keep track of the solution in
        # the following containers. This was particularly useful in the paper
        # for ISGT North America 2023.
        self.solutions: list[np.ndarray] = []  # decision variables
        self.dr: list[np.ndarray] = []  # OLTC taps
        self.dP: list[np.ndarray] = []  # active power injections
        self.dQ: list[np.ndarray] = []  # reactive power injections
        self.slacks: list[np.ndarray] = []  # slack variables

    def set_period(self, period: float) -> None:
        """
        Set controller's period (time between consecutive actions).
        """

        if not isinstance(period, (int, float)):
            raise TypeError("The period must be a number.")

        if period <= 0:
            raise ValueError("The period must be positive.")

        self.period = period

    def add_observed_corridor(
        self, boundary_bus: str, sending_buses: list[str]
    ) -> None:
        """
        Add corridor by specifying bus names.
        """

        if not isinstance(boundary_bus, str):
            raise TypeError("Argument boundary_bus must be a string.")

        if not isinstance(sending_buses, list):
            raise TypeError("Argument sending_buses must be a list.")

        for bus in sending_buses:
            if not isinstance(bus, str):
                raise TypeError(
                    "Argument sending_buses must be a list of strings."
                )

        # The observed corridor is a tuple[str, list[str]]. It does not have to
        # be inserted in any particular order.
        self.observed_corridors.append((boundary_bus, sending_buses))
        # After adding an observed corridor, we increment the number of boundary
        # buses.
        self.B += 1

    def add_controlled_transformers(self, transformers: list[str]) -> None:
        """
        Add named transformers as being controlled.
        """

        if not isinstance(transformers, list):
            raise TypeError("Argument transformers must be a list.")

        for transformer in transformers:
            if not isinstance(transformer, str):
                raise TypeError(
                    "Argument transformers must be a list of strings."
                )

        # Contrary to the observed corridors, the transformers are simply a list
        # of strings. Hence they can be concatenated.
        self.controlled_transformers += transformers
        # Once again, we increment the number of controlled transformers.
        self.T += len(transformers)

    def set_horizons(self, Np: int, Nc: int) -> None:
        """
        Set prediction and control horizons.
        """

        if not isinstance(Np, int):
            raise TypeError("The prediction horizon must be an integer.")

        if not isinstance(Nc, int):
            raise TypeError("The control horizon must be an integer.")

        if Np <= 0:
            raise ValueError("The prediction horizon must be positive.")

        if Nc <= 0:
            raise ValueError("The control horizon must be positive.")

        if Nc > Np:
            raise ValueError(
                "The control horizon must be smaller than the prediction horizon."
            )

        self.Np = Np
        self.Nc = Nc

    @staticmethod
    def v_bound(
        bus: records.Bus,
        Np: int,
        iter: int,
        half_db_pu: float,
        v_set_pu: float = 1.0,
    ) -> tuple[np.ndarray]:
        """
        Return two arrays of height Np with the voltage bounds of this bus.

        In set_bounds (see below), VT_fun can be replaced by any function (or
        lambda expression) that receives (bus, Np, iter) and behaves like this
        function.
        """

        v_min_pu = (v_set_pu - half_db_pu) * np.ones([Np, 1])
        v_max_pu = (v_set_pu + half_db_pu) * np.ones([Np, 1])

        return v_min_pu, v_max_pu

    @staticmethod
    def power_bound(bus: records.Bus, Nc: int, iter: int) -> tuple[np.ndarray]:
        """
        Return four arrays of height Nc with the power bounds of this bus.

        In set_bounds (see below), P_fun and Q_fun can be replaced by any
        function (or lambda expression) that receives (bus, Nc, iter) and
        behaves like this function.

        Notice that 'p' does not stand for active power necessarily, but for
        any power injection (active or reactive).
        """

        p_min_pu = -1e6 * np.ones([Nc, 1])

        # Multipling by +- 1.0 creates a copy of the array.
        p_max_pu = -1.0 * p_min_pu
        dp_min_pu = 1.0 * p_min_pu
        dp_max_pu = 1.0 * p_max_pu

        return p_min_pu, p_max_pu, dp_min_pu, dp_max_pu

    def set_bounds(
        self,
        # Fixed value, hard-coded
        NLI_min: float = 0.1,
        # Return a tuple of arrays (min, max). We use 0.1 and 0.05 as half
        # deadbands of transmission and distribution voltages, respectively,
        # as is normally dictated in grid codes.
        VT_fun=lambda bus, Np, iter: MPC_controller.v_bound(
            bus=bus, Np=Np, iter=iter, half_db_pu=0.1
        ),
        VD_fun=lambda bus, Np, iter: MPC_controller.v_bound(
            bus=bus, Np=Np, iter=iter, half_db_pu=0.05
        ),
        # Return a tuple of arrays (min, max, min_delta, max_delta)
        P_fun=lambda bus, Nc, iter: MPC_controller.power_bound(
            bus=bus, Nc=Nc, iter=iter
        ),
        Q_fun=lambda bus, Nc, iter: MPC_controller.power_bound(
            bus=bus, Nc=Nc, iter=iter
        ),
    ) -> None:
        """
        Set bounds on u, the voltages, and the NLI.

        These bounds may depend on the bus, the horizons, and the number of
        iterations that the controller has made. This implementation should
        suffice for most experiments.
        """

        # The decision variables are defined for each controlled transformer and
        # for the network downstream of it. To have more knowledge about the
        # transformers, we map names to objects. To build matrices, it's
        # important that they preserve a predictable order.
        trafos = [
            self.sys.get_transformer(t) for t in self.controlled_transformers
        ]

        # We begin with the most straightforward decision variable: the tap
        # ratios. These are bounded by the physical limits of the OLTCs.
        r_min = np.array([[t.OLTC.nmin_pu] for t in trafos])
        r_max = np.array([[t.OLTC.nmax_pu] for t in trafos])
        dr_min = np.array([[-t.OLTC.step_pu] for t in trafos])
        dr_max = np.array([[t.OLTC.step_pu] for t in trafos])
        # We now store them in a list to facilitate the construction of the
        # matrices later on.
        r_bounds = [r_min, r_max, dr_min, dr_max]

        # The voltage bounds are a bit more complicated, and hence we use
        # helper functions.

        def get_u_bound(i: int, k: int) -> np.ndarray:
            """
            Return i-th bound (0=min, 1=max, 2=min_delta, 3=max_delta) at tk.
            """

            # The strategy will be to get the bounds of tap ratios, active
            # power injections, and reactive power injections, and then stack
            # them vertically.

            # Getting the i-th bound of the tap ratios is easy.
            r_bound = r_bounds[i]

            # Getting the i-th bound of the power injections is a bit more
            # complicated. We get the output of P_fun (four binding curves over
            # the control horizon), get the curve we are interested in (index
            # i), and finally get the bound at the control index k we are
            # interested in.
            P_bound = np.array(
                [
                    [
                        P_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[i][k, 0]
                    ]
                    for t in trafos
                ]
            )
            # We then repeat for the reactive power injections.
            Q_bound = np.array(
                [
                    [
                        Q_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[i][k, 0]
                    ]
                    for t in trafos
                ]
            )

            # Finally, we stack them vertically so that the resulting array
            # remains a column vector.
            return np.vstack([r_bound, P_bound, Q_bound])

        # With the previous function, computing the bounds on u reduces to
        # calling the helper function for each type of bound (0 to 3 for min,
        # max, min_delta, max_delta), for each control index (0 to Nc-1), and
        # stacking the results vertically.
        self.u_lower = np.vstack([get_u_bound(0, k) for k in range(self.Nc)])
        self.u_upper = np.vstack([get_u_bound(1, k) for k in range(self.Nc)])
        self.du_lower = np.vstack([get_u_bound(2, k) for k in range(self.Nc)])
        self.du_upper = np.vstack([get_u_bound(3, k) for k in range(self.Nc)])

        # We use a similar strategy for the voltage bounds, i.e. we use helper
        # functions.

        def get_VT_bound(i: int, k: int) -> np.ndarray:
            """
            Get array with i-th bound (min, max) at tk.
            """

            # The logic is completely analogous to the one used for the decision
            # variables. The only difference is that we have to evaluate the
            # voltage bounds at the HV bus of the transformer (VT stands for
            # transmission voltage).

            return np.array(
                [
                    [
                        VT_fun(
                            bus=t.get_HV_bus(),
                            Np=self.Np,
                            iter=self.current_iter,
                        )[i][k, 0]
                    ]
                    for t in trafos
                ]
            )

        # Building the bounds on VT is then a matter of calling the helper
        # function for each type of bound (0 to 1 for min and max), for each
        # prediction index (0 to Np-1), and stacking the results vertically.
        self.VT_lower = np.vstack([get_VT_bound(0, k) for k in range(self.Np)])
        self.VT_upper = np.vstack([get_VT_bound(1, k) for k in range(self.Np)])

        # We now repeat for the distribution voltages.

        def get_VD_bound(i: int, k: int) -> np.ndarray:
            """
            Get array with i-th bound (min, max) at tk.
            """

            return np.array(
                [
                    [
                        VD_fun(
                            bus=t.get_LV_bus(),
                            Np=self.Np,
                            iter=self.current_iter,
                        )[i][k, 0]
                    ]
                    for t in trafos
                ]
            )

        # And iterate and stack:
        self.VD_lower = np.vstack([get_VD_bound(0, k) for k in range(self.Np)])
        self.VD_upper = np.vstack([get_VD_bound(1, k) for k in range(self.Np)])

        # Finally, we set the bounds on the NLI (our stability indicator). The
        # lower bound is taken as a parameter, whereas the upper bound is set to
        # a very large number (to avoid problems that np.inf might cause).
        self.NLI_lower = NLI_min * np.ones([self.Np * self.B, 1])
        # Note: the following line was implemented wrong, as it was setting the
        # lower bound as 1e6 * self.NLI_lower (instead of an addition). This
        # would have failed whenever NLI_min = 0.
        self.NLI_upper = 1e6 * np.ones([self.Np * self.B, 1])

    @staticmethod
    def some_setpoint(
        bus_or_transformer: Union[records.Bus, records.Branch],
        Nc: int,
        iter: int,
    ) -> np.ndarray:
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
            bus_or_transformer=transformer, Nc=Nc, iter=iter
        ),
        # In the case of active and reactive power injections, it might make
        # sense to distinguish among buses, in case some of them have a priority
        # over others.
        P_fun=lambda bus, Nc, iter: MPC_controller.some_setpoint(
            bus_or_transformer=bus, Nc=Nc, iter=iter
        ),
        Q_fun=lambda bus, Nc, iter: MPC_controller.some_setpoint(
            bus_or_transformer=bus, Nc=Nc, iter=iter
        ),
    ) -> None:
        """
        Define setpoint of u.
        """

        # As with the bounds, we first map transformer names to transformer
        # objects and then use a helper function to build the setpoint arrays.

        trafos = [
            self.sys.get_transformer(t) for t in self.controlled_transformers
        ]

        def get_u_setpoints(k: int) -> np.ndarray:
            """
            Get array with setpoints at time tk.
            """

            # Normally, the tap ratio should not have a setpoint, but we
            # include it for completeness. This setpoint can be supressed
            # by imposing no penalization inside the cost function.
            r_setpoint = np.array(
                [
                    [
                        r_fun(
                            transformer=t, Nc=self.Nc, iter=self.current_iter
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )
            # The following setpoints on active and reactive power make sense
            # even for the default P_fun and Q_fun, because they are the zero
            # vectors: we don't want to change the power injections of DERs.
            P_setpoint = np.array(
                [
                    [
                        P_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )
            Q_setpoint = np.array(
                [
                    [
                        Q_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )

            return np.vstack([r_setpoint, P_setpoint, Q_setpoint])

        # At present, we only support setpoints for the decision variables,
        # mainly because of the cost function I chose. However, it would be easy
        # to extend this to the voltages and the NLI (i.e., the outputs).
        self.u_star = np.vstack([get_u_setpoints(k) for k in range(self.Nc)])

    @staticmethod
    def some_u_penalization(
        bus_or_transformer: Union[records.Bus, records.Branch],
        Nc: int,
        iter: int,
    ) -> np.ndarray:
        """
        Return array of height Nc with penalization factors.

        This method is used as a default for penalizations in P, Q, dP, and dQ.
        """

        return np.ones([Nc, 1])

    @staticmethod
    def some_slack_penalization(
        bus: records.Bus, Np: int, iter: int
    ) -> tuple[float]:
        """
        Return a tuple of penalizations for the slacks.

        The first element of the tuple penalizes the lower-bound violations,
        the second one penalizes the upper-bound violations.

        For simplicity, the penalization factors are kept constant across the
        horizon. Hence, the method returns a tuple of floats, not a tuple of
        arrays.
        """

        return 1.0, 1.0

    def set_weights(
        self,
        dr_fun=lambda transformer, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(
            bus_or_transformer=transformer, Nc=Nc, iter=iter
        ),
        dP_fun=lambda bus, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(
            bus_or_transformer=bus, Nc=Nc, iter=iter
        ),
        dQ_fun=lambda bus, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(
            bus_or_transformer=bus, Nc=Nc, iter=iter
        ),
        r_devs_fun=lambda transformer, Nc, iter: 0.0
        * MPC_controller.some_u_penalization(
            bus_or_transformer=transformer, Nc=Nc, iter=iter
        ),
        P_devs_fun=lambda bus, Nc, iter: MPC_controller.some_u_penalization(
            bus_or_transformer=bus, Nc=Nc, iter=iter
        ),
        Q_devs_fun=lambda bus, Nc, iter: MPC_controller.some_u_penalization(
            bus_or_transformer=bus, Nc=Nc, iter=iter
        ),
        slacks_fun=lambda bus, Np, iter: MPC_controller.some_slack_penalization(
            bus=bus, Np=Np, iter=iter
        ),
    ) -> None:
        """
        Define weight matrices to be used in the cost function.

        By default, neither r nor dr are penalized.
        """

        # As with the bounds and the setpoints, we first map transformer names
        # to transformer objects and then use a helper function to build the
        # weight matrices.
        trafos = [
            self.sys.get_transformer(t) for t in self.controlled_transformers
        ]

        def get_du_penalizations(k: int) -> np.ndarray:
            """
            Get array with penalization factors for du at time tk.
            """

            dr_pen = np.array(
                [
                    [
                        dr_fun(
                            transformer=t, Nc=self.Nc, iter=self.current_iter
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )

            dP_pen = np.array(
                [
                    [
                        dP_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )

            dQ_pen = np.array(
                [
                    [
                        dQ_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )

            return np.vstack([dr_pen, dP_pen, dQ_pen])

        factors = np.vstack([get_du_penalizations(k) for k in range(self.Nc)])
        # Transposing gets a row vector, indexing a 1-D array, np.diag converts
        # that 1-D array to a diagonal matrix. The matrix R1 is the one associated
        # to the change in the decision variables.
        self.R1 = np.diag(factors.T[0])

        # We now repeat for the penalizations on the deviations with respect to
        # the setpoints.

        def get_u_devs_penalizations(k: int) -> np.ndarray:
            """
            Get array with penalization factors for u deviations at time k.
            """

            r_devs_pen = np.array(
                [
                    [
                        r_devs_fun(
                            transformer=t, Nc=self.Nc, iter=self.current_iter
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )

            P_devs_pen = np.array(
                [
                    [
                        P_devs_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )

            Q_devs_pen = np.array(
                [
                    [
                        Q_devs_fun(
                            bus=t.get_LV_bus(),
                            Nc=self.Nc,
                            iter=self.current_iter,
                        )[k, 0]
                    ]
                    for t in trafos
                ]
            )

            return np.vstack([r_devs_pen, P_devs_pen, Q_devs_pen])

        factors = np.vstack(
            [get_u_devs_penalizations(k) for k in range(self.Nc)]
        )
        # For the following hack, see above in this same method.
        self.R2 = np.diag(factors.T[0])

        # Build penalizations on the slack variables. Notice that in the list
        # comprehensions below, the 'i' index is the one that runs slower, so
        # that the first half of VT_pen corresponds to the penalizationof the
        # lower bound violations, and the second half to the upper bound
        # violations. The same applies to VD_pen. This could be important later
        # on.
        VT_pen = np.array(
            [
                [
                    slacks_fun(
                        bus=t.get_HV_bus(), Np=self.Np, iter=self.current_iter
                    )[i]
                ]
                for i in range(2)
                for t in trafos
            ]
        )

        VD_pen = np.array(
            [
                [
                    slacks_fun(
                        bus=t.get_LV_bus(), Np=self.Np, iter=self.current_iter
                    )[i]
                ]
                for i in range(2)
                for t in trafos
            ]
        )

        # Notice that for the slack variables, we don't consider variation
        # across the control horizon.
        factors = np.vstack([VT_pen, VD_pen])
        # For the following hack, see above in this same method.
        self.S = np.diag(factors.T[0])

    def __str__(self) -> str:
        """
        Print relevant settings of the MPC controller, mainly for debugging.

        This method will fail if the controller has not been initialized
        properly.
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

    def build_structural_matrices(self) -> None:
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

    def generate_twin(
        self, parameter_randomizations: Sequence[callable] = None
    ) -> None:
        """
        Generate a twin of the system this controller is controlling.
        """

        # To avoid unnecessary duplications, all manipulations and retrievals of
        # the twin and its data will take place through self.sys.twin, instead
        # of self.twin.
        self.sys.generate_twin(
            parameter_randomizations=parameter_randomizations
        )

    def update_twin(
        self,
        measurement_corruption: callable = lambda element, measurement: measurement,
    ) -> None:
        """
        Update twin with (possibly corrupted) measurements.
        """

        # Again, we use self.sys.twin to avoid duplications.
        self.sys.update_twin(measurement_corruption=measurement_corruption)

    def update_derivatives(self) -> None:
        """
        Get all derivatives in one shot.

        In this implementation, the derivatives are with respect to the load,
        not the power injections.
        """

        # Initialize sensitivity matrices
        self.partial_u_N = np.zeros([self.B, 3 * self.T])
        self.partial_u_VT = np.zeros([self.T, 3 * self.T])
        self.partial_u_VD = np.zeros([self.T, 3 * self.T])

        # Define changes for all sensitivities
        dn = 1e-3
        dP = 1e-3
        dQ = 1e-3

        # Copy twin to compute derivatives
        sys = copy.deepcopy(self.sys.twin)

        # Iterate over all substations
        for trafo_no, trafo in enumerate(self.controlled_transformers):
            for attr_no, attr in enumerate(["n", "PL", "QL"]):
                # Run initial power flow. We don't start from a flat profile to
                # take advantage of previous iterations.
                sys.run_pf(flat_start=False)

                # Evaluate all transmission voltages
                VTs_0 = [
                    sys.get_transformer(name=transformer).get_HV_bus().V_pu
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all distribution voltages
                VDs_0 = [
                    sys.get_transformer(name=transformer).get_LV_bus().V_pu
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all NLIs
                NLIs_0 = [
                    sys.get_min_NLI(
                        corridor=corridor,
                        transformer_names=self.controlled_transformers,
                    )
                    for corridor in self.observed_corridors
                ]

                # Change the right parameter at the right transformer.
                # Because PL and QL are treated as per-unit values by the whole
                # run_pf machinery, those are the units of these derivatives.
                # Furthermore, the derivatives are computed as if positive dP
                # and dQ meant increment in loads. Hence, that's the meaning
                # of the values spit out by the MPC: if positive, they are
                # load increments, and they always are in pu.
                if attr == "n":
                    sys.get_transformer(trafo).n_pu += dn
                elif attr == "PL":
                    sys.get_transformer(trafo).get_LV_bus().PL_pu += dP
                elif attr == "QL":
                    sys.get_transformer(trafo).get_LV_bus().QL_pu += dQ

                # Run second power flow
                sys.run_pf(flat_start=False)

                # In the following evaluations, it is important that the
                # iterations take place in the same order as before. This is
                # ensured by using the same iterables (lists).

                # Evaluate all transmission voltages (again)
                VTs_1 = [
                    sys.get_transformer(name=transformer).get_HV_bus().V_pu
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all distribution voltages (again)
                VDs_1 = [
                    sys.get_transformer(name=transformer).get_LV_bus().V_pu
                    for transformer in self.controlled_transformers
                ]
                # Evaluate all NLIs (again)
                NLIs_1 = [
                    sys.get_min_NLI(
                        corridor=corridor,
                        transformer_names=self.controlled_transformers,
                    )
                    for corridor in self.observed_corridors
                ]

                # Undo changes (to avoid doing multiple deep copies)
                if attr == "n":
                    sys.get_transformer(name=trafo).n_pu -= dn
                elif attr == "PL":
                    sys.get_transformer(name=trafo).get_LV_bus().PL_pu -= dP
                elif attr == "QL":
                    sys.get_transformer(name=trafo).get_LV_bus().QL_pu -= dQ

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

        # print(" \nThese are the sensitivities of the NLI w.r.t. [r, P, Q]:\n")
        # print(self.partial_u_N)
        # input("\nPress ENTER to continue\n")

        # print(" These are the sensitivities of VT w.r.t. [r, P, Q]:\n")
        # print(self.partial_u_VT)
        # input("\nPress ENTER to continue\n")

        # print(" These are the sensitivities of VD w.r.t. [r, P, Q]:\n")
        # print(self.partial_u_VD)
        # input("\nPress ENTER to continue\n")

    def build_sensitivities(self):
        """
        Stack derivatives into matrices.
        """

        # Compute derivatives
        self.update_derivatives()

        # Build sensitivities. The following code simply applies structural
        # rules to build the matrices.
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
                # The following inequality makes sure that the sensitivity
                # matrices are lower trapezoidal block matrices.
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

    def get_measurements(self):
        """
        Measurements are taken from self.twin as they are.

        The method System.update_twin() is the one that introduces measurement
        corruptions.
        """

        if (
            not hasattr(self, "D_u_N")
            or not hasattr(self, "D_u_VT")
            or not hasattr(self, "D_u_VD")
        ):
            self.build_sensitivities()

        # Update the twin
        self.sys.update_twin()

        # Initialize arrays
        u_meas = np.zeros([3 * self.T, 1])
        NLI_meas = np.zeros([self.B, 1])
        VT_meas = np.zeros([self.T, 1])
        VD_meas = np.zeros([self.T, 1])

        # Measure u
        u_meas[:, 0] = (
            [
                self.sys.twin.get_transformer(name=transformer_name).n_pu
                for transformer_name in self.controlled_transformers
            ]
            + [
                self.sys.twin.get_transformer(name=transformer_name)
                .get_LV_bus()
                .PL_pu
                for transformer_name in self.controlled_transformers
            ]
            + [
                self.sys.twin.get_transformer(name=transformer_name)
                .get_LV_bus()
                .QL_pu
                for transformer_name in self.controlled_transformers
            ]
        )

        # Measure the NLI directly from the detectors of the system (not the twin)
        for corridor_no, corridor in enumerate(self.observed_corridors):
            # Fetch detector
            detector = next(
                detector
                for detector in self.sys.detectors
                if detector.corridor == corridor and detector.type == "NLI"
            )
            # Store its reading
            NLI_meas[corridor_no, 0] = detector.get_reading()

        # Measure VT (must be read from the twin)
        VT_meas[:, 0] = [
            self.twin.get_transformer(name=transformer_name).get_HV_bus().V_pu
            for transformer_name in self.controlled_transformers
        ]

        # Measure VD (must be read from the twin)
        VD_meas[:, 0] = [
            self.twin.get_transformer(name=transformer_name).get_LV_bus().V_pu
            for transformer_name in self.controlled_transformers
        ]

        return u_meas, NLI_meas, VT_meas, VD_meas

    def update_measurement_dependent_matrices(self) -> None:
        """
        Update all matrices that depend on measurements.
        """

        # Get measurements
        u_meas, NLI_meas, VT_meas, VD_meas = self.get_measurements()

        # Recall that the quadratic program is of the form
        #
        # min 1/2 x^T P x + q^T x
        # s.t. G x <= h
        #
        # where x is the vector of decision variables. In our case, x is
        # of the form [du^T slacks^T]^T, where du is the vector of changes
        # in the decision variables, and slacks is the vector of slack
        # variables. The matrices P, q, G, and h are built below.

        # The following will fail if R1 and R2 are not symmetric, so we
        # should check that.
        assert np.allclose(self.R1, self.R1.T)
        assert np.allclose(self.R2, self.R2.T)

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

    def solve_optimization(self) -> np.ndarray:
        """
        The return type is a 1D np.ndarray, as implemented in cvxopt_solve_qp.
        """
        return utils.cvxopt_solve_qp(
            P=self.P_matrix, q=self.q_matrix, G=self.G_matrix, h=self.h_matrix
        )

    def get_actions(self) -> list[sim_interaction.Disturbance]:
        """
        Return disturbances that the MPC would apply to the system.

        Contrary to the OLTC controllers, this get_actions() method must not
        receive the time now, since its logic does not depend on time. OLTCs, on
        the other hand, behave as timed automata.
        """

        # Initialize disturbances
        dists: list[sim_interaction.Disturbance] = []

        # Count iteration. For instance, if the MPC acts every 10 seconds, then
        # after acting for 1 minute, this counter will be 6. This is useful in
        # case one wants the MPC to be more aggressive at the beginning of the
        # simulation, and then relax a bit.
        self.current_iter += 1

        # Build everything
        self.update_measurement_dependent_matrices()

        # Solve optimization
        x = self.solve_optimization()
        self.solutions.append(x)

        # Filter out solutions. Recall that x is a 1D np.ndarray, as implemented
        # in cvxopt_solve_qp.
        dr = x[: self.T]
        dP = x[self.T : 2 * self.T]
        dQ = x[2 * self.T : 3 * self.T]
        slacks = x[-4 * self.T :]

        # Store solutions from this iteration. The elements of these lists are
        # themselves 1D np.ndarrays.
        self.dr.append(dr)
        self.dP.append(dP)
        self.dQ.append(dQ)
        self.slacks.append(slacks)

        # Send changes in tap ratio to transformers in the 'real' system. A more
        # robust implementation of the following would make sure that the
        # tranformer names match.

        for dr_value, transformer_name in zip(
            dr, self.controlled_transformers
        ):
            # Fetch transformer
            transformer = self.sys.get_transformer(name=transformer_name)
            # Find OLTC controller
            OLTC_controller = transformer.OLTC.OLTC_controller
            # Skip if OLTC is not overridable
            if not OLTC_controller.is_overridable:
                continue
            # Increment cumulative tap ratio
            OLTC_controller.increment_r(dr=dr_value)
            # Try to move the OLTC
            dist = OLTC_controller.try_to_move_OLTC()
            # If the OLTC moved, add the disturbance to the list
            if dist is not None:
                # The method try_to_move_OLTC returns a list of disturbances
                dists += dist


#         # Send signals to the substations
#         dP_requests = dict(zip(self.controlled_transformers, dP))
#         dQ_requests = dict(zip(self.controlled_transformers, dQ))
#         for t in self.controlled_transformers:
#             coordinator = next(
#                 (
#                     c
#                     for c in self.sys.controllers
#                     if c.type == "Coordinator" and c.transformer_name == t
#                 ),
#                 None,
#             )
#             if coordinator is not None:
#                 coordinator.increment_request(
#                     delta_P_load_pu=dP_requests[t],
#                     delta_Q_load_pu=dQ_requests[t],
#                 )

#         # There is no need to send those changes to the twin because it is
#         # updated:
#         self.update_twin()

#         #
#         u_meas, NLI_meas, VT_meas, VD_meas = self.get_measurements()
#         print("\n\nMeasurements by MPC: u", u_meas.T, "\n")
#         print("Measurements by MPC: NLI", NLI_meas.T, "\n")
#         print("Measurements by MPC: VT", VT_meas.T, "\n")
#         print("Measurements by MPC: VD", VD_meas.T, "\n")
#         print("Disturbances required by MPC", [str(d) for d in dists], "\n")

#         return dists


# class Coordinator(Controller):
#     """
#     Coordinator installed at each step-down substation.
#     """

#     type = "Coordinator"
#     sigma_P_min = -5
#     sigma_P_max = 5
#     sigma_Q_min = -5
#     sigma_Q_max = 5

#     def __init__(
#         self,
#         transformer_name,
#         min_P_injection_MVA,
#         max_P_injection_MVA,
#         min_Q_injection_MVA,
#         max_Q_injection_MVA,
#     ):
#         self.sys = None
#         self.transformer_name = transformer_name
#         self.min_P_injection_MVA = min_P_injection_MVA
#         self.max_P_injection_MVA = max_P_injection_MVA
#         self.min_Q_injection_MVA = min_Q_injection_MVA
#         self.max_Q_injection_MVA = max_Q_injection_MVA
#         self.requested_P_injection_MVA = 0
#         self.requested_Q_injection_MVA = 0
#         self.period = 1
#         self.t_last_action = 0
#         self.available_injectors = None
#         self.sigma = []

#     def __str__(self):
#         inj_names = [inj.name for inj in self.available_injectors]

#         controller_types = [c.type for c in self.sys.controllers]

#         controllable_inj = []
#         for inj in self.available_injectors:
#             prefix = inj.prefix.split(" ")
#             if len(prefix) == 2 and prefix[1] in controller_types:
#                 controllable_inj.append(inj.name)

#         data = [
#             ["Substation", self.transformer_name],
#             ["Injectors downstream", inj_names],
#             ["Controllable injectors", controllable_inj],
#             ["Period (s)", self.period],
#             ["Minimum P injection (MW)", self.min_P_injection_MVA],
#             ["Max. P injection (MW)", self.max_P_injection_MVA],
#             ["Min. Q injection (Mvar)", self.min_Q_injection_MVA],
#             ["Max. A injection (Mvar)", self.max_Q_injection_MVA],
#             ["Min. P signal", self.sigma_P_min],
#             ["Max. P signal", self.sigma_P_max],
#             ["Min. Q signal", self.sigma_Q_min],
#             ["Max. Q signal", self.sigma_Q_max],
#             ["Signal when OK", 0],
#         ]

#         table = tabulate.tabulate(data)

#         return table

#     def increment_request(self, delta_P_load_pu, delta_Q_load_pu):
#         """
#         This method is the one used by the MPC.
#         """

#         # print(f'The MPC sent dP = {delta_P_load_pu} pu and dQ = {delta_Q_load_pu} pu')
#         # Decrement because the MPC thinks the deltas refer to loads
#         self.requested_P_injection_MVA -= delta_P_load_pu * self.sys.Sb
#         self.requested_Q_injection_MVA -= delta_Q_load_pu * self.sys.Sb

#     def get_actions(self):
#         delay = 10e-3
#         tk = self.sys.get_t_now() + delay

#         # Get signal
#         sigma = self.get_sigma()

#         # Store values of sigma
#         self.sigma.append((sigma["P"], sigma["Q"]))
#         print(" Sending sigma =", (sigma["P"], sigma["Q"]))

#         # print(f'Coordinator at substation {self.transformer_name} is sending {sigma} to its DERs')
#         # print(f'MPC is requesting P = {self.requested_P_injection_MVA} MW and Q = {self.requested_Q_injection_MVA} Mvar from this coordinator')
#         # Get available injectors
#         injectors = self.get_available_injectors()

#         # Apply appropriate controller
#         dists = []
#         for inj in injectors:
#             controller = next(
#                 (
#                     c
#                     for c in self.sys.controllers
#                     if c.type is not None and c.type in inj.prefix
#                 ),
#                 None,
#             )
#             if controller is not None:
#                 dists += controller.get_actions(tk, inj, sigma)

#         return dists

#     @staticmethod
#     def interpolate(
#         requested_power, power_min, power_max, sigma_min, sigma_max
#     ):
#         if requested_power < power_min:
#             return sigma_min
#         elif power_min <= requested_power < 0:
#             frac = requested_power / power_min
#             return math.ceil(frac * sigma_min)
#         elif 0 <= requested_power <= power_max:
#             frac = requested_power / power_max
#             return math.floor(frac * sigma_max)
#         elif power_max < requested_power:
#             return sigma_max

#     def get_sigma(self):
#         """
#         Get sigma based on the requested powers.
#         """

#         sigma_P = self.interpolate(
#             self.requested_P_injection_MVA,
#             self.min_P_injection_MVA,
#             self.max_P_injection_MVA,
#             self.sigma_P_min,
#             self.sigma_P_max,
#         )

#         sigma_Q = self.interpolate(
#             self.requested_Q_injection_MVA,
#             self.min_Q_injection_MVA,
#             self.max_Q_injection_MVA,
#             self.sigma_Q_min,
#             self.sigma_Q_max,
#         )

#         return {
#             "P": sigma_P,
#             "Q": sigma_Q,
#             "min_P": self.sigma_P_min,
#             "max_P": self.sigma_P_max,
#             "min_Q": self.sigma_Q_min,
#             "max_Q": self.sigma_Q_max,
#         }

#     def get_available_injectors(self):
#         # Maybe initialize available injectors
#         if self.available_injectors is None:
#             t = self.sys.get_transformer(self.transformer_name)
#             buses = self.sys.isolate_buses_by_kV(starting_bus=t.get_LV_bus())
#             self.available_injectors = [
#                 inj for inj in self.sys.injectors if inj.bus in buses
#             ]

#         return self.available_injectors


# class DERA_Controller(Controller):
#     """
#     Local controller of each DERA.
#     """

#     type = "DER_A"
#     period = np.inf
#     t_last_action = 0
#     sys = None

#     @staticmethod
#     def get_ref(sigma, sigma_min, sigma_max, power_0, SN):
#         """
#         Map sigma (either P or Q) to reference (either P or Q).
#         """

#         if sigma < 0:
#             frac = sigma / abs(sigma_min)
#         else:
#             frac = sigma / abs(sigma_max)

#         return power_0 / SN + frac * (1 - power_0 / SN * np.sign(sigma))

#     def get_actions(self, tk, element, sigma):
#         """
#         Change Pref and Qref according to values of sigma.
#         """

#         P0 = element.P0
#         Q0 = element.Q0
#         SN = element.SNOM

#         Pref = self.get_ref(sigma["P"], sigma["min_P"], sigma["max_P"], P0, SN)
#         Qref = self.get_ref(sigma["Q"], sigma["min_Q"], sigma["max_P"], Q0, SN)

#         return [
#             sim_interaction.Disturbance(
#                 tk, element, par_name="Pref", par_value=Pref
#             ),
#             sim_interaction.Disturbance(
#                 tk, element, par_name="Qref", par_value=Qref
#             ),
#         ]
