"""
A module for the OLTC_controller class.
"""

# Modules from this repository
import records
import control
import sim_interaction

# Modules from the standard library

# Other modules


class OLTC_controller(control.Controller, records.DCTL):
    """
    A class that reimplements the OLTC controller from RAMSES.

    Doing this reimplementation is necessary in order to have full control over
    the OLTC position, which is not possible in RAMSES.
    """

    prefix: str = "DCTL LTC2"

    def __init__(
        self, OLTC: records.OLTC, delay_1: float, delay_2: float, dir: int
    ) -> None:
        """
        Initialize the OLTC controller.

        All other parameters (not having to do with time or motion) are read
        from the OLTC object.
        """

        # Initialize attributes from arguments
        self.OLTC = OLTC
        self.delay_1 = delay_1  # time to wait before the first action
        self.delay_2 = delay_2  # time to wait before the remaining actions
        self.dir = dir  # direction of the OLTC

        # We now initialize the remaining attributes. The name and the voltage
        # setpoint are simply inherited from the OLTC object.
        self.name: str = OLTC.trafo.name
        self.v_setpoint_pu: float = self.OLTC.v_setpoint_pu
        # The parameters that are written to the .dat file are the delays
        # and the half deadband. On the one hand, the delays are set to the
        # values given as arguments.
        self.delay_RAMSES_1: float = delay_1
        self.delay_RAMSES_2: float = delay_2
        # On the other hand, the half deadband is set to 2. This is to make
        # sure that the DCTL in RAMSES will never act by itself. The trick will
        # be, instead, to change this half deadband precisely when we want to
        # trigger a movement of the OLTC. Setting the value to 2 allows
        # voltages from 0 pu to 1 pu, which covers all plausible voltages in
        # the system.
        self.initial_half_db_pu_RAMSES: float = 1.0
        # For internal use in this class, especially when this controller is
        # overriden by the MPC, we keep track of cumulative changes in the tap
        # ratio. Furthermore, we keep a copy of the half deadband, which will
        # be modified strategically, and we also initialize this OLTC as being
        # non-overridable.
        self.cumulative_dr: float = 0
        self.half_db_pu: float = self.initial_half_db_pu_RAMSES
        self.is_overridable: bool = False

        # The OLTC behaves like a Finite State Machine (FSM). The following
        # attributes are used to store the state of the FSM, as well as to
        # keep track of time between actions.
        self.db_state: str = "within"
        self.tau: float = 0
        self.t_last_action: float = 0

    def increment_r(self, dr: float) -> None:
        """
        Add a change in the tap ratio (r) to the cumulative change.
        """

        self.cumulative_dr += dr

    def make_overridable(self) -> None:
        """
        Make the OLTC overridable, so that it can be controlled by the MPC.
        """

        self.is_overridable = True

    def get_disturbances(
        self,
        ocurrence_time: float,
        new_v_setpoint_pu: float,
        new_half_db_pu: float,
    ) -> list[sim_interaction.Disturbance]:
        """
        Return disturbances to change voltage setpoint and half deadband.
        """

        # For the purpose of defining the disturbances, we compute the changes
        # in the parameters. This has to do with the fact that
        # Disturbance.__str__ returns CHGPRM <object acted on> <parameter name>
        # <parameter value> 0.0 and this is interpreted as an increment.
        delta_v_setpoint_pu = new_v_setpoint_pu - self.v_setpoint_pu
        delta_half_db_pu = new_half_db_pu - self.half_db_pu
        # For use in the internal FSM of this class, however, we need to store
        # the new values of the parameters. They are states (attributes) that
        # are incremented by the disturbances.
        self.v_setpoint_pu += delta_v_setpoint_pu
        self.half_db_pu += delta_half_db_pu

        # Returning the disturbances as a list is useful for concatenating them
        # with other lists of disturbances.
        return [
            sim_interaction.Disturbance(
                ocurrence_time=ocurrence_time,
                object_acted_on=self,
                par_name="Vsetpt",
                par_value=delta_v_setpoint_pu,
            ),
            sim_interaction.Disturbance(
                ocurrence_time=ocurrence_time,
                object_acted_on=self,
                par_name="DB",
                par_value=delta_half_db_pu,
            ),
        ]

    def increase_voltages(
        self, ocurrence_time: float
    ) -> list[sim_interaction.Disturbance]:
        """
        Move the OLTC so that it increases the voltage of the controlled bus.
        """

        # We first apply that change in the static model.
        self.OLTC.increase_voltages()

        # We then return the disturbances that will be applied in the dynamic
        # model.
        return self.get_disturbances(
            ocurrence_time=ocurrence_time,
            # Setting the voltage setpoint to 2 pu and having a half deadband
            # of 0 pu (no tolerance) will force the OLTC to increase the
            # voltage.
            new_v_setpoint_pu=2,
            new_half_db_pu=0,
        ) + self.get_disturbances(
            # To make sure that the OLTC has moved once, we apply the next
            # disturbance halfway between the first action and the second
            # action.
            ocurrence_time=ocurrence_time
            + self.delay_RAMSES_1
            + 0.5 * self.delay_RAMSES_2,
            # The setpoint returns to 1.0 pu and the half deadband returns to 2
            # pu (large tolerance), to ensure that the OLTC remains idle.
            new_v_setpoint_pu=1,
            new_half_db_pu=2,
        )

    def reduce_voltages(
        self, ocurrence_time: float
    ) -> list[sim_interaction.Disturbance]:
        """
        Move the OLTC so that it decreases the voltage of the controlled bus.
        """

        # The logic is completely analogous to the increase_voltages method.

        self.OLTC.reduce_voltages()

        return self.get_disturbances(
            ocurrence_time=ocurrence_time,
            # Here's the only slight difference with respect to
            # increase_voltages: setting the voltage setpoint to 0 pu and
            # having a half deadband of 0 pu (no tolerance) will force the OLTC
            # to decrease the voltage.
            new_v_setpoint_pu=0,
            new_half_db_pu=0,
        ) + self.get_disturbances(
            ocurrence_time=ocurrence_time
            + self.delay_RAMSES_1
            + 0.5 * self.delay_RAMSES_2,
            new_v_setpoint_pu=1,
            new_half_db_pu=2,
        )

    def try_to_move_OLTC(self) -> list[sim_interaction.Disturbance]:
        """
        Try to move the OLTC in the direction it prefers.

        The only thing that could stop this movement it is that the cumulative
        dr is not enough to trigger an integer change in the tap ratio.

        Returns None if the OLTC could not move in neither direction.
        """

        # We start by measuring the current time, using the system bound to the
        # OLTC object.
        t_now = self.OLTC.trafo.sys.get_t_now()

        # We then compute the time at which the disturbance will take place.
        # Introducing this delay is important because in RAMSES one cannot send
        # disturbances that take place in the present. When working with the
        # MPC, this delay will be very small. When working with the FSM, this
        # delay will admittedly be a few seconds, but the response should still
        # be very similar to the "real" response.
        dt = self.delay_RAMSES_1 / 2

        # If the tap ratio has increased beyond one integer step, we apply it
        # and return the corresponding disturbances. Recall that an increase in
        # the tap ratio is a decrease in the voltage.
        if self.cumulative_dr > self.OLTC.step:
            self.cumulative_dr -= self.OLTC.step
            return self.reduce_voltages(t_now + dt)

        # Handle opposite case:
        elif self.cumulative_dr < -self.OLTC.step:
            self.cumulative_dr += self.OLTC.step
            return self.increase_voltages(t_now + dt)

    def measure_v(self) -> float:
        """
        Measure the voltage of the controlled bus.
        """

        ram = self.OLTC.trafo.sys.ram

        return ram.getBusVolt([self.OLTC.controlled_bus.name])[0]

    def v_is_below(self) -> bool:
        """
        Check if the voltage of the controlled bus is below the deadband.
        """

        return (
            self.measure_v() < self.OLTC.v_setpoint_pu - self.OLTC.half_db_pu
        )

    def v_is_above(self) -> bool:
        """
        Check if the voltage of the controlled bus is above the deadband.
        """

        return (
            self.measure_v() > self.OLTC.v_setpoint_pu + self.OLTC.half_db_pu
        )

    def v_is_within(self) -> bool:
        """
        Check if the voltage of the controlled bus is within the deadband.
        """

        return not self.v_is_above() and not self.v_is_below()

    def get_actions(self, t_now: float) -> list[sim_interaction.Disturbance]:
        """
        Return disturbances that the FSM would apply at the current time.
        """

        # The FSM is, more specifically, a timed automaton. It is therefore
        # very important to keep track of the time that has elapsed since the
        # last action.
        elapsed = t_now - self.t_last_action

        # The disturbances that will be returned are stored in a list.
        dists = []

        # The following transitions are based on current measurements and
        # the states.

        if self.db_state == "within" and not self.v_is_within():
            # Change state
            self.db_state = "outside/iddle"
            # Store new values
            self.tau = self.delay_1
            self.t_last_action = t_now

        elif self.db_state == "outside/iddle" and self.v_is_within():
            # Change state
            self.db_state = "within"
            self.t_last_action = t_now  # Not even used in the state 'within'

        elif self.db_state == "outside/iddle" and elapsed > self.tau:
            # Change state
            self.db_state = "outside/active"
            # Store new values
            self.t_last_action = t_now

        # Do actions and remaining transitions
        if self.db_state == "outside/active":
            # Do actions
            if self.v_is_below():
                dists = self.increase_voltages(t_now)
            elif self.v_is_above():
                dists = self.reduce_voltages(t_now)
            # Change state
            self.db_state = "outside/iddle"
            # Store new values
            self.tau = self.delay_2

        # Return disturbances
        return dists

    def get_pars(self) -> list[records.Parameter]:
        """
        Return the parameters of the OLTC controller for RAMSES
        """

        return [
            records.Parameter("trafo", self.OLTC.trafo.name),
            records.Parameter("bus", self.OLTC.controlled_bus.name),
            records.Parameter("dir", self.dir),
            records.Parameter("nmin", 100 * self.OLTC.nmin_pu),
            records.Parameter("nmax", 100 * self.OLTC.nmax_pu),
            records.Parameter(
                "npos", self.OLTC.positions_down + 1 + self.OLTC.positions_up
            ),
            records.Parameter("tol", self.initial_half_db_pu_RAMSES),
            records.Parameter("vset", self.OLTC.v_setpoint_pu),
            records.Parameter("delay1", self.delay_RAMSES_1),
            records.Parameter("delay2", self.delay_RAMSES_2),
        ]
