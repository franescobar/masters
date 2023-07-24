"""
A module with benchmarking algorithms.

These algorithms are:
    - the LIVES method from Vournas and Van Cutsem: 10.1109/TPWRS.2008.926425,
    - the controller from Pabón et al.: 10.1109/TPWRS.2020.3027949,
    - the under-voltage load shedding scheme from Pabón et al. (see above).
"""

# Modules from this repository
import control
import records
import sim_interaction

# Modules from the standard library

# Other modules
import numpy as np

class PabonController(control.Controller):
    """
    The controller from Pabón et al.: 10.1109/TPWRS.2020.3027949.

    The parameters have the following meaning:
        - transformer: the transformer controlled by this controller,
        - period: the period of the controller (in seconds),
        - epsilon_pu: the tolerance for the distribution voltage (in pu),
        - delta_pu: the tolerance for the transmission voltage (in pu),
        - increase_rate_pu_per_s: the rate at which the controlled DERAs
              increase their reactive power (in pu/s).
    """

    def __init__(self,
                 transformer: records.Branch,
                 period: float = 1,
                 epsilon_pu: float = 0.02,
                 delta_pu: float = 0.01,
                 increase_rate_pu_per_s: float = 0.01,
                 ) -> None:

        # Initialize the system governed by the controller. This attribute is
        # then overwritten by pf_dynamic.System's add_controllers method.
        self.sys: "pf_dynamic.System" = None

        # We then initialize the other attributes that are common to all
        # controllers.
        self.t_last_action: float = 0
        self.period: float = period

        # The following attributes are specific to this controller.
        self.transformer: records.Branch = transformer
        self.epsilon_pu: float = epsilon_pu
        self.delta_pu: float = delta_pu
        self.increase_rate_pu_per_s: float = increase_rate_pu_per_s

        # These attributes will be updated as new measurements are obtained.
        self.state: str = "IDLE"
        self.Vt_min: float = np.nan
        self.Vd_max: float = np.nan

        # Knowing which DERs are controlled by this controller is imposible during
        # the initialization, since the controller is not yet added to the system
        # (it is being created in that very moment). We therefore initialize the
        # following attribute as an empty list, and then update it in the
        # update_controlled_DERAs method.
        self.controlled_DERAs: list[records.DERA] = []

    def update_controlled_DERAs(self) -> None:
        """
        Return a list of the DERAs governed by this controller.

        This method is called during the initialization of the controller.

        Since this controller has access to the system and the transformer, it
        can take advantage of get_LV_bus() and isolate_by_kV() to locate the
        buses downstream of the transformer. Then, one can look at the injector
        dictionary and find the DERAs located at these buses.
        """

        # We begin by fetching the LV bus of the controlled transformer.
        LV_bus_name = self.transformer.get_LV_bus().name
        LV_bus = self.sys.bus_dict[LV_bus_name]

        # We then isolate the buses downstream of the transformer.
        downstream_buses = self.sys.isolate_buses_by_kV(starting_bus=LV_bus)

        # We then look at the injectors dictionary and find the DERAs located at
        # these buses.
        for bus in downstream_buses:
            for injector in self.sys.bus_to_injectors[bus]:
                if isinstance(injector, records.DERA):
                    self.controlled_DERAs.append(injector)

    def get_measured_voltages(self) -> tuple[float, float]:
        """
        Return the voltages at the transformer's HV and LV buses.

        Since this controller has access to the system, it can access the
        sys.ram.getBusVolt() method to obtain the voltages at the buses.
        """

        # We begin by fetching the HV and LV buses of the controlled transformer.
        HV_bus = self.transformer.get_HV_bus()
        LV_bus = self.transformer.get_LV_bus()

        # We then obtain the voltages at these buses.
        Vt = self.sys.ram.getBusVolt(busNames=[HV_bus.name])[0]
        Vd = self.sys.ram.getBusVolt(busNames=[LV_bus.name])[0]

        return Vt, Vd

    def update_state(self) -> None:
        """
        Update the state of the controller between "IDLE" and "ACTIVE".
        """

        # If the controller is already active, there is nothing to do. It will
        # stay there forever. We can think of this as if the controller would
        # be locked, waiting for manual intervention.
        if self.state == "ACTIVE":
            return None

        # If, instead, the controller is idle, we check if it should become
        # active. This happens whenever any "NLI" detector in the system
        # registers a negative reading.
        for detector in self.sys.detectors:
            if detector.type == "NLI":
                if detector.get_reading() < 0:
                    self.state = "ACTIVE"
                    # If the method came this far, it means that the change of
                    # state has just happened. We therefore measure the voltages
                    # and store them as Vt_min and Vd_max.
                    self.Vt_min, self.Vd_max = self.get_measured_voltages()
                    break

    def freeze_r(self) -> list[sim_interaction.Disturbance]:
        """
        Return a disturbance (list) to freeze the transformer's tap ratio.
        """

        # To ensure that the OLTC does not move, we simply enlarge its
        # deadband so that it is never triggered.
        self.transformer.OLTC.OLTC_controller.half_db_pu = 1.0
        self.transformer.OLTC.OLTC_controller.v_setpoint_pu = 1.0

        # This method does not return any disturbance, but we make it return
        # an empty list for consistency with the other methods. Since it
        # modifies the OLTC object stored in Python, it does not need to send
        # anything to RAMSES.
        return []

    def freeze_Q(self) -> list[sim_interaction.Disturbance]:
        """
        Return a disturbance (list) to freeze the DERAs' reactive power.
        """

        # The implementation is trivial:
        return []

    def increase_r(self) -> list[sim_interaction.Disturbance]:
        """
        Return a disturbance (list) to increase the transformer's tap ratio.
        """

        # To ensure that the OLTC increases r (decrease the distribution
        # voltage), we simply set its deadband to zero and its setpoint to a
        # value below the current voltage.
        self.transformer.OLTC.OLTC_controller.half_db_pu = 0.0
        self.transformer.OLTC.OLTC_controller.v_setpoint_pu = 0.0

        # As in freeze_r(), we modify the OLTC object stored in Python, so
        # there is no need to send anything to RAMSES.
        return []


    def decrease_r(self) -> list[sim_interaction.Disturbance]:
        """
        Return a disturbance (list) to decrease the transformer's tap ratio.
        """

        # To ensure that the OLTC decreases r (increase the distribution
        # voltage), we simply set its deadband to zero and its setpoint to a
        # value above the current voltage.
        self.transformer.OLTC.OLTC_controller.half_db_pu = 0.0
        self.transformer.OLTC.OLTC_controller.v_setpoint_pu = 2.0

        # As in freeze_r(), we modify the OLTC object stored in Python, so
        # there is no need to send anything to RAMSES.
        return []

    def increase_Q(self) -> list[sim_interaction.Disturbance]:
        """
        Return a disturbance (list) to increase the DERAs' reactive power.
        """

        # Since the command must be sent to (possibly more than) one DERA,
        # we store the disturbances in a list.
        dists = []

        # We then loop over the DERAs and generate the disturbances. One tricky
        # bit is figuring out the ocurrence time. I'll put 0.02 seconds, which
        # is larger than the larger allowed time step in RAMSES (0.01 s) and
        # 50 times smaller than the controller's period (1 s).
        ocurrence_time = self.sys.get_t_now() + 0.02
        # We also need to figure out the duration of the disturbance. One
        # reasonable value will be self.period - 2 * 0.02 s = self.period - 0.04 s.
        # The first 0.02 s are there to cover the ocurrence time, and the second
        # 0.02 s are to ensure that the disturbance is over before the next
        # evaluation of this controller.
        duration = self.period - 2 * 0.02
        # Knowing the duration, we can set the total increment. This can be
        # done outside the for loop because the Qref parameter is, as seen in
        # the DER_A.txt file generated by CODEGEN, in the DER's base MVA.
        increment = self.increase_rate_pu_per_s * duration

        for DERA in self.controlled_DERAs:
            dist = sim_interaction.Disturbance(
                ocurrence_time=ocurrence_time,
                object_acted_on=DERA,
                par_name="Qref",
                par_value=increment,
                duration=duration,
                # The units are set as "" so that the disturbance is interpreted
                # as an increment, not a setpoint value.
                units="",
            )

            dists.append(dist)

        return dists

    def get_region_number(self) -> int:
        """
        Return the region (1 to 4) of the (Vt, Vd) space the controller is in.
        """

        # Apply the logic in Fig. 4 of Pabón et al. (see above). This logic
        # depends solely on the voltages, so we measure them.
        Vt, Vd = self.get_measured_voltages()

        # In the following conditionals, we use <= instead of <. Since the
        # voltages are floats, this should not be a problem. We do it because
        # it is better to land in two regions (the first one to be tested
        # will be chosen) than in none.

        if (
            # The operating point is to the left of the Vt_min line...
            Vt <= self.Vt_min
            and
            # ... and below the dashed line.
            Vd <= self.Vd_max
        ):
            return 1

        elif (
            # The operating point is above the dashed line.
            self.Vd_max <= Vd
        ):
            return 2

        elif (
            # The operating point is to the right of the Vt_min + delta line...
            self.Vt_min + self.delta_pu <= Vt
            and
            # ... and below the Vd_max - epsilon line.
            Vd <= self.Vd_max - self.epsilon_pu
        ):
            return 3

        elif (
            # The operating point is in the shaded region, which can be
            # split in three subregions:
            (
                # 1) the shaded rectangle at the bottom left
                self.Vt_min <= Vt <= self.Vt_min + self.delta_pu
                and
                Vd <= self.Vd_max - self.epsilon_pu
            )
            or
            (
                # 2) the shaded rectangle at the top left
                self.Vt_min <= Vt <= self.Vt_min + self.delta_pu
                and
                self.Vd_max - self.epsilon_pu <= Vd <= self.Vd_max
            )
            or
            (
                # 3) the shaded rectangle at the top right
                self.Vt_min + self.delta_pu <= Vt
                and
                self.Vd_max - self.epsilon_pu <= Vd <= self.Vd_max
            )
        ):
            return 4

        else:
            raise ValueError (
                "The voltages fell into a region that should not exist."
            )

    def get_actions(self) -> list[sim_interaction.Disturbance]:
        """
        Return the actions to be taken by the controller.

        This method will be called by pf_dynamic.System's follow_controllers()
        method, and it orchestrates the controller behavior. It is an
        implementation of the algorithm in Fig. 4 of Pabón et al. (see above).
        """

        # We first identify the DERAs that must be acted upon, in case they
        # have not been identified.
        if self.controlled_DERAs == []:
            self.update_controlled_DERAs()

        # We begin by updating the state of the controller.
        self.update_state()

        # The disturbances are stored in a list, which must be returned by the
        # get_actions() method of any controller.
        dists = []

        # If the controller is idle, there is nothing to do.
        if self.state == "IDLE":
            return dists

        # If the controller is active, we need figure out the disturbances that
        # must be sent. This depends on the region.
        region_number = self.get_region_number()

        # Once the region number is known, the control boils down to sending
        # the right disturbances:
        if region_number == 1:
            # Freeze r and increase Q of DERAs
            dists += self.freeze_r()
            dists += self.increase_Q()

        elif region_number == 2:
            # Increase r and freeze Q of DERAs
            dists += self.increase_r()
            dists += self.freeze_Q()

        elif region_number == 3:
            # Decrease r and freeze Q of DERAs
            dists += self.decrease_r()
            dists += self.freeze_Q()

        elif region_number == 4:
            # Freeze r and freeze Q of DERAs
            dists += self.freeze_r()
            dists += self.freeze_Q()

        else:
            raise ValueError("Invalid region number.")

        # Finally, simply return the list of disturbances.
        return dists


class LIVES(control.Detector):
    def __init__(self):
        pass

    def get_value(self):
        pass

class Load_Shedder(control.Controller):
    pass