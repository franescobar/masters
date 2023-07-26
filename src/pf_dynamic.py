"""
An object-oriented power-system simulator, for static and dynamic studies.
"""

# Modules from this repository
import pf_static
import records
import control
import sim_interaction
import oltc

# Modules from the standard library
import copy
import bisect
from collections.abc import Sequence


class System(pf_static.StaticSystem):
    """
    A representation of the power system capable of dynamic simulations.
    """

    def __init__(
        self, name: str = "", pu: bool = True, base_MVA: float = 100
    ) -> None:
        """
        Initialize a system called 'name'.

        The argument 'pu' determines whether the system parameters are given in
        per-unit or in SI units.
        """

        # Initialize static system
        super().__init__(name=name, pu=pu, base_MVA=base_MVA)

        # Initialize bound simulator
        self.ram = None

        # Initialize container for all RAMSES records
        self.records: list[records.Record] = []

        # Initialize attributes
        self.disturbances: list[sim_interaction.Disturbance] = []
        self.detectors: list[control.Detector] = []
        self.controllers: list[control.Controller] = []
        self.OLTC_controllers: list[oltc.OLTC_controller] = []

        # Initialize twin
        self.twin = None

    def add_record(self, record: records.Record) -> None:
        """
        Add a record, i.e. an element with no effect on the power flow.

        This method is useful for keeping track of DCTLs, nominal frequency,
        synchronous machines, and so on, when exporting to RAMSES.
        """

        if not isinstance(record, records.Record):
            raise RuntimeError(
                f"Record {record} is not an instance of Record."
            )

        if record in self.records:
            raise RuntimeError(
                f"Record {record} is already in the system's records."
            )

        for attribute in ["prefix", "name"]:
            attr = getattr(record, attribute, None)
            if not attr and attr != "":
                raise RuntimeError(
                    f"Attribute {attribute} "
                    f"is missing from {record.__class__.__name__}."
                )

        for method in ["get_pars"]:
            attr = getattr(record, method, None)
            if not attr or not callable(attr):
                raise RuntimeError(
                    f"Method {method} "
                    f"is missing from {record.__class__.__name__}."
                )

        bisect.insort(self.records, record)

    def set_frequency(self, fnom: float) -> None:
        """
        Set nominal frequency of the system.
        """

        if fnom <= 0:
            raise RuntimeError("Frequency must be positive.")

        for record in self.records:
            if isinstance(record, records.Frequency):
                raise RuntimeError(
                    f"Frequency has already been set to {record.pars[0]}."
                )

        self.add_record(records.Frequency(fnom=fnom))

    def add_OLTC_controller(
        self, OLTC_controller: oltc.OLTC_controller
    ) -> None:
        """
        Add a single OLTC controller to the system.
        """

        if not isinstance(OLTC_controller, oltc.OLTC_controller):
            raise RuntimeError(
                f"OLTC controller {OLTC_controller} "
                f"is not an instance of OLTC_controller."
            )

        if OLTC_controller in self.OLTC_controllers:
            raise RuntimeError(
                f"OLTC controller {OLTC_controller} "
                f"is already in the system's OLTC controllers."
            )

        self.OLTC_controllers.append(OLTC_controller)

    def import_dynamic_data(self, filename: str) -> None:
        """
        Import dynamic data from a RAMSES file.
        """

        # Since synchronous machines are usually defined in several lines, the
        # parser must remember what it read in previous lines. Hence it must be
        # a finite-state machine (FSM). We can keep track of the state with a
        # boolean variables, as there are only two states.
        reading_machine = False
        # We also need to remember the governor type, the generator name, and
        # the parameters read so far.
        governor_type = None
        gen_name = None
        pars = []
        # These parameters will be reset after having read each generator,
        # specifically after having read the governor.

        # The parameters from all other records are easier to parse.

        with open(filename, "r") as f:
            for line in f:
                # We split the lines at the spaces, which are the delimiters
                # used in RAMSES.
                words = line.split()

                # There might be empty lines in the file, which are ignored.
                if len(words) == 0:
                    continue

                # After reaching this point, we know that the line is not
                # empty. To parse the line, we simply check the first word to
                # know what record to define.

                # We start by implementing the parsing of frequency, but this
                # does mean that the frequency is the first record in the file.
                # This logic boils down to using the set_frequency method.
                if words[0] == "FNOM":
                    self.set_frequency(fnom=float(words[1]))

                # We now treat OLTCs. For most systems I have worked with in
                # RAMSES, these are modeled with the record DCTL LTC2.
                if words[0] == "DCTL" and words[1] == "LTC2":
                    # To add the OLTC as an attribute of the transformer, we
                    # must access to the transformer object. This is stored in
                    # the StaticSystem's dictionary of transformers. By
                    # construction, we know that this transformer is unique.
                    transformer = self.transformer_dict[words[3]]
                    # We then read the dynamic parameters of the OLTC, which
                    # are the direction of the tap changer (1 for direct and -1
                    # for reverse), and the delays in seconds (first and
                    # subsequent delays).
                    dir = int(float(words[5]))
                    delay_1 = float(words[11])
                    delay_2 = float(words[12])
                    # Having read the parameters, we create the OLTC controller
                    # instance, and then add it to the system. OLTC controllers
                    # are special in the sense that they are both a controller,
                    # as in the case of the MPC controller, and a record.
                    # Notice that the OLTC controller requires an instance of
                    # the (static) OLTC, which was read from the ARTERE file.
                    DCTL = oltc.OLTC_controller(
                        OLTC=transformer.OLTC,
                        delay_1=delay_1,
                        delay_2=delay_2,
                        dir=dir,
                    )
                    # We now make three additions: (1) add the OLTC controller
                    # as a record that will be written to the .dat file for
                    # RAMSES, (2) add it as a controller that will be followed
                    # by the system at periodic intervals, and (3) bind it to
                    # the transformer.
                    self.add_record(DCTL)
                    self.add_OLTC_controller(DCTL)
                    transformer.OLTC.OLTC_controller = DCTL

                # Possibly overwrite transformer names (as of 2023-02-07, I
                # don't recall why this is needed).
                elif words[0] == "TRFO":
                    candidates = self.get_branches_between(
                        bus_name_1=words[2], bus_name_2=words[3]
                    )
                    if len(candidates) == 1:
                        candidates[0].name = words[1]

                # We now come to the point of parsing generators. This is taken
                # care of in the remaining elif statements.

                # The first block reads information from the synchronous
                # machine that is located in the same line as SYNC_MACH.
                elif words[0] == "SYNC_MACH":
                    # For the purpose of the parser, reading_machine must
                    # simply be a boolean, but we define it as a string for the
                    # sake of clarity. Recall that any non-empty string
                    # evaluates to True.
                    reading_machine = "Machine"
                    # We then store the generator name and parameters.
                    gen_name = words[1]
                    pars += words[1:]

                # The second block reads information from the exciter. We are
                # leaving many records out since we concentrate on the Nordic
                # system.
                elif len(words) > 1 and words[0:2] == ["EXC", "GENERIC1"]:
                    # If we are reading the exciter, then we must have read the
                    # synchronous machine before. We then create the
                    # synchronous machine instance. Notice that all parameters
                    # are passed as strings, which means that the __str__
                    # method of the Parameter class will have no effect. Hence
                    # all parameters are printed as they are in the .dat file.
                    bus = self.get_bus(pars[1])
                    sm = records.SYNC_MACH(gen_name, bus, *pars[6:])
                    # We then bind the synchronous machine to the generator.
                    # Generators are initialized with self.machine = None.
                    self.get_generator(gen_name).machine = sm
                    # All that is left is to change the state of the parser,
                    # reset the parameters, and save the exciter parameters
                    # from the current line.
                    reading_machine = "Exciter"
                    pars = []
                    pars += words[2:]

                # The third block reads information from the governor.
                # For the Nordic, there are two possibilities for the governor:
                # CONSTANT and HYDRO_GENERIC1.
                elif len(words) > 1 and words[0] == "TOR":
                    # As before, if we are reading the governor, then we must
                    # have read the exciter before. We then create the exciter
                    # instance and bind it to the generator.
                    exc = records.GENERIC1(*pars)
                    self.get_generator(gen_name).exciter = exc
                    # We then change the state of the parser, read the governor
                    # type (this is needed to know what constructor to call),
                    # and reset the parameters.
                    reading_machine = "Governor"
                    governor_type = words[1]
                    pars = []
                    # Notice that if the governor is CONSTANT, then there are
                    # no parameters to read.
                    pars = pars + words[2:] if len(words) > 3 else pars

                # This block applies actions that depend on the FSM state.
                elif reading_machine:
                    pars += words

                # Finally, this block terminates the reading of a machine.
                if reading_machine and ";" in line:
                    # Depending on the governor type, we call the right
                    # constructor.
                    if governor_type == "HYDRO_GENERIC1":
                        # The semicolon is still present in the line and must
                        # be left out, since it's added by the __str__ method
                        # of the Record class.
                        gov = records.HYDRO_GENERIC1(*pars[:-1])
                    elif governor_type == "CONSTANT":
                        # The semicolon was already left out.
                        gov = records.CONSTANT()
                    # We then bind the governor to the generator and reset
                    # all variables for the next machine.
                    self.get_generator(gen_name).governor = gov
                    governor_type = None
                    gen_name = None
                    pars = []
                    reading_machine = False

    def export_to_RAMSES(self, filename: str) -> None:
        """
        Export system to a RAMSES file.
        """

        # The exporting boils down to opening the .dat file and writing the
        # records taking advantage of their __str__ method.
        with open(filename, "w") as datfile:
            # System's name:
            datfile.write(f"! {self.name}\n\n")

            # Nominal frequency:
            fnom_record = next(
                r for r in self.records if isinstance(r, records.Frequency)
            )
            datfile.write(f"# Nominal frequency\n{fnom_record}\n")

            # Buses:
            datfile.write("\n# Buses\n")
            for bus in self.buses:
                datfile.write(f"{bus}\n")

            # Transmission lines:
            datfile.write("\n# Transmission lines\n")
            for line in self.lines:
                datfile.write(str(line) + "\n")

            # Transformers:
            datfile.write("\n# Transformers\n")
            for trafo in self.transformers:
                datfile.write(f"{trafo}\n")

            # OLTCs:
            datfile.write("\n# OLTCs\n")
            OLTCs = filter(
                lambda r: isinstance(r, oltc.OLTC_controller), self.records
            )
            for OLTC in OLTCs:
                datfile.write(f"{OLTC}\n")

            # Slack bus:
            datfile.write("\n# Slack\n")
            slack_gens = [g for g in self.generators if g.bus is self.slack]
            # If there are no generators at the slack, then we add a stiff
            # (infinite-power) network.
            if len(slack_gens) == 0:
                inf_bus = records.Thevenin("MACHINF", self.slack)
                datfile.write(f"{inf_bus}\n")
            # Otherwise, we add (possibly more than one) generator(s).
            else:
                for gen in slack_gens:
                    datfile.write(f"{gen}\n")

            # Remaining generators:
            datfile.write("\n# Remaining generators\n")
            for gen in self.generators:
                if gen.bus is not self.slack:
                    datfile.write(f"{gen}\n")

            # Remaining injectors (as long as they inject anything):
            datfile.write("\n# Remaining injectors\n")
            for inj in self.injectors:
                if abs(inj.get_P()) > 1e-9 or abs(inj.get_Q()) > 1e-9:
                    datfile.write(f"{inj}\n")

            # Stiff loads:
            if any(bus.has_stiff_load for bus in self.buses):
                datfile.write("\n# Stiff loads at buses\n")
                for bus in self.buses:
                    if bus.has_stiff_load:
                        stiff_load = records.Load(
                            name=f"SL-{bus.name}",
                            bus=bus,
                            P0_MW=bus.PL_pu * self.base_MVA,
                            Q0_Mvar=bus.QL_pu * self.base_MVA,
                        )
                        datfile.write(f"{stiff_load}\n")

            # Write initial conditions
            datfile.write("\n# Initial conditions\n")
            for bus in self.buses:
                v0_record = records.InitialVoltage(bus)
                datfile.write(f"{v0_record}\n")

    def add_disturbances(
        self, disturbances: Sequence[sim_interaction.Disturbance]
    ) -> None:
        """
        Add new disturbances to the (sorted) disturbances list.

        We don't do type checking here because this method might be called
        too many times.
        """

        for disturbance in disturbances:
            if disturbance not in self.disturbances:
                bisect.insort(self.disturbances, disturbance)

    def add_detector(self, detector: control.Detector) -> None:
        """
        Add detector to the system.
        """

        if not isinstance(detector, control.Detector):
            raise RuntimeError(
                f"Detector is not an instance of control.Detector."
            )

        if detector in self.detectors:
            raise RuntimeError(
                f"Detector is already in the system's detectors."
            )

        detector.sys = self
        self.detectors.append(detector)

    def add_controllers(
        self, controllers: Sequence[control.Controller]
    ) -> None:
        """
        Add controllers that the system will have to follow.
        """

        for controller in controllers:
            if not isinstance(controller, control.Controller):
                raise RuntimeError(
                    f"Controller is not an instance of Controller."
                )

            if controller in self.controllers:
                raise RuntimeError(
                    f"Controller is already in the system's " f"controllers."
                )

            if controller.sys is not None:
                raise RuntimeError(
                    "You want a controller to act " "on two systems"
                )

            controller.sys = self
            self.controllers.append(controller)

    def get_t_now(self) -> float:
        """
        Get simulation time now.
        """

        return self.ram.getSimTime()

    def get_disturbances_until(
        self, t: float
    ) -> list[sim_interaction.Disturbance]:
        """
        Get disturbances until time t (inclusive).

        It is very important that the disturbances that are selected are
        immediately removed from the list of disturbances. Otherwise, they
        would be applied twice.
        """

        # Determine slicing point using dummy disturbance. Setting
        # object_acted_on to "solver" prevents that Disturbance.__init__ raises
        # an error.
        i = bisect.bisect_right(
            self.disturbances,
            sim_interaction.Disturbance(
                ocurrence_time=t, object_acted_on="solver", par_value=""
            ),
        )

        # Select disturbances and update list of disturbances, removing the
        # ones that have already been selected.
        selected_disturbances = self.disturbances[:i]
        self.disturbances = self.disturbances[i:]

        return selected_disturbances

    def send_disturbance(
        self, disturbance: sim_interaction.Disturbance
    ) -> None:
        """
        Send a single disturbance to RAMSES.
        """

        true_ocurrence_time = self.ram.getSimTime() + disturbance.duration
        # The 1 is added just in case
        if true_ocurrence_time + 1 < self.ram.getInfTime():
            # Not adding this conditional may raise an error:
            # You're trying to add a disturbance beyond the STOP time.
            print(f"Sending disturbance at {disturbance.ocurrence_time + disturbance.duration}")
            print(f"Descriptor is {str(disturbance)}")
            self.ram.addDisturb(
                # We add the duration because otherwise RAMSES will complain and
                # say we're trying to apply a disturbance in the past.
                t_dist=disturbance.ocurrence_time + disturbance.duration,
                disturb=str(disturbance)
            )

    def send_disturbances(
        self, disturbances: Sequence[sim_interaction.Disturbance]
    ) -> None:
        """
        Send disturbances and update system topology.
        """

        # Even though the user or a controller may ask for a disturbance to be
        # applied, this could be impossible. For example, this would be the
        # case if a controller asked to shed load by opening the circuit
        # breakers of a branch that were already out of operation. To handle
        # those cases, where RAMSES would probably raise an error, we use a
        # boolean variable to either apply or ignore the disturbance.

        for disturbance in disturbances:
            apply = True  # assume that the disturbance can be applied
            element = disturbance.object_acted_on

            if isinstance(element, records.Branch):
                # When the disturbance is directed to a branch, we check if the
                # disturbance asks for a different status (open or closed) by
                # comparing a boolean with either 1.0 or 0.0. Take into account
                # that True == 1.0 and False == 0.0.
                if element.in_operation != round(disturbance.par_value):
                    if element.in_operation:
                        if element.branch_type == "Line":
                            self.get_line(name=element.name).disconnect()
                        elif element.branch_type == "Transformer":
                            self.get_transformer(
                                name=element.name
                            ).disconnect()
                    else:
                        if element.branch_type == "Line":
                            self.get_line(name=element.name).connect()
                        elif element.branch_type == "Transformer":
                            self.get_transformer(name=element.name).connect()
                # If the branch status is already the one asked for, then we
                # don't apply the disturbance.
                else:
                    apply = False

            elif isinstance(element, records.Bus):
                # If the disturbance is applied to a bus, we ignore it if the
                # bus is not connected. We can rely on the is_connected
                # attribute because buses can only be disconnected by opening a
                # line, and because both the connect() and disconnect() methods
                # update the system connectivity and overwrite the is_connected
                # attribute when needed.
                if not element.is_connected:
                    apply = False

                # We do not check for short circuit or clearance durations
                # because they are not modeled in Python.

            elif isinstance(element, records.Injector):
                # Finally, if the disturbance is applied to an injector, we
                # ignore it if the injector is not connected, which in turn can
                # be inferred from the bus connectivity.
                if not element.bus.is_connected:
                    apply = False

            # We now apply the disturbance if it has not been ignored.
            if apply:
                self.send_disturbance(disturbance=disturbance)

    def send_disturbances_until(self, t: float) -> None:
        """
        Send all disturbances that take place before time t.
        """

        self.send_disturbances(disturbances=self.get_disturbances_until(t=t))

    def follow_controllers(self) -> None:
        """
        Fetch controller commands and add disturbances to queue.
        """

        # Apply actions from non-OLTC controllers
        if self.has_contingency():
            for c in self.controllers:
                if self.get_t_now() - c.t_last_action > c.period:
                    # Apply actions
                    self.add_disturbances(c.get_actions())
                    # Restart controller's timer
                    c.t_last_action = self.get_t_now()

        # Apply actions from OLTCs acting autonomously
        for OLTC in self.OLTC_controllers:
            if (not OLTC.is_overridable) or (
                OLTC.is_overridable
                and not any(c.overrides_OLTCs for c in self.controllers)
            ):
                self.add_disturbances(OLTC.get_actions(t_now=self.get_t_now()))

    def update_detectors(self) -> None:
        """
        Update values from detectors.

        Detectors can be NLI detectors, voltage and current meters,
        and so on.
        """

        for d in self.detectors:
            ind_to_update = self.get_t_now() - d.t_last_measurement > d.period
            d.update_measurements(self.get_t_now(), ind_to_update)

    def generate_twin(
        self, parameter_randomizations: Sequence[callable] = None
    ) -> None:
        """
        Generate a twin of the system so it's used by controllers.

        The twin is generated at a certain point in time, and the parameters
        might be corrupted.

        Ideally, the twin should be generated before the disaggregation.

        Each parameter randomization can be called as

            parameter_randomization(system: System)

        and returns nothing, but instead modifies the system that is passed.
        """

        # The twin will be a deep copy of the system. However, we must be
        # careful with the RAMSES simulator instance, which is not copyable. As
        # a workaround, we temporarily remove it, create the copy, and then
        # restore it.
        ram = self.ram
        self.ram = None
        self.twin = copy.deepcopy(self)
        self.ram = ram

        # We then apply parameter randomizations, which must be such that they
        # modify the system that is passed.
        if parameter_randomizations is not None:
            for parameter_randomization in parameter_randomizations:
                parameter_randomization(self.twin)

        # Having created the twin, update it with the latest measurements
        # available from the simulator?

    def update_twin(
        self, measurement_corruption: callable = lambda element, measurement: measurement
    ) -> None:
        """
        Update the twin by incorporating measurements from the simulation.

        The measurement corruption can be called as

            measurement_corruption(element: records.Record,
                                   measurement: float)

        and returns a float (or a list of floats, as in the case of branches).
        Inside it, it implements a sort of multiple dispatch, where the type of
        the element determines the corruption.

        By default, the corruption is the identity function.
        """

        if self.twin is None:
            raise RuntimeError(
                f"Twin has not been generated. "
                f"Call {self.__class__.__name__}.generate_twin() first, "
                f"and do it when specifying the controller, before "
                f"disaggregating the system."
            )

        # Update topology
        for branch, twin_branch in zip(self.branches, self.twin.branches):
            twin_branch.in_operation = branch.in_operation

        # Update connectivity
        self.twin.update_connectivity()

        # Measuring the voltages is needed to close the feedback loop of the
        # MPC. In fact, some of the matrices that this controller uses
        # internally are dependent on measurements. Notice that we *must* use
        # the RAMSES simulator of this instance to retrieve the measurements,
        # but we *can* read the bus names from the twin. This is because these
        # names are plain strings, not instances of a custom class.
        for bus in self.twin.buses:
            bus.V_pu = measurement_corruption(
                element=bus,
                # The following index might be unnecessary.
                measurement=self.ram.getBusVolt(busNames=[bus.name])[0],
            )

        # The same is true for tap ratios. The method getObs() is different
        # from getBusVolt() in the sense that it returns a list, and hence we
        # must index it.
        for transformer in self.twin.transformers:
            if (
                transformer.has_OLTC
                and transformer.OLTC.OLTC_controller is not None
            ):
                transformer.n_pu = measurement_corruption(
                    element=transformer.OLTC,
                    measurement=self.ram.getObs(
                        ["DCTL"],
                        [transformer.OLTC.OLTC_controller.name],
                        ["ratio"],
                    )[0],
                )

        # The MPC also requires measuring the NLI, but this is retrieved
        # directly from the detectors.

        # Therefore, the last set of measurements required by the MPC are the
        # DER powers. Because we cannot measure the DER powers individually, we
        # use an approximation: after measuring the total power served by the
        # transformer, we substract the power that the model (twin) expects
        # from the voltage-sensitive load. This difference is then stored in
        # the PL_pu and QL_pu attributes of the secondary bus, and it is
        # considered to be the DER contribution. We expect the MPC to
        # compensate for this approximation.

        # We apply the previous logic to each transformer.
        for transformer in self.twin.transformers:

            # If the transformer is not a step-down transformer, we skip it.
            if not any(
                inj.bus is transformer.get_LV_bus() for inj in self.twin.injectors
                if isinstance(inj, records.Load)
            ):
                continue

            # We start by measuring the voltage-sensitive load. Note that the
            # method get_sensitive_load_MW_Mvar calls the methods get_P() and
            # get_Q() of each Load injector connected to the secondary bus. In
            # turn, these methods read the attribute V_pu, which was updated a
            # few lines ago.
            (
                sensitive_P_load_MW,
                sensitive_Q_load_Mvar,
            ) = self.twin.get_sensitive_load_MW_Mvar(transformer.get_LV_bus())

            # Then, we measure the total power entering the transformer.
            power_into_transformer_pu = measurement_corruption(
                element=transformer,
                measurement=self.ram.getBranchPow([transformer.name])[0],
            )

            # This measurement can be translated into the total MW or Mvar
            # *supplied* by the transformer.
            if transformer.get_LV_bus() is transformer.from_bus:
                # The "from" and "to" buses in this class are, in RAMSES, the
                # "to" and "from" buses, respectively. Hence the indices below.
                # The minus sign is because served power = - power entering the
                # transformer.
                P_from_served_pu = -power_into_transformer_pu[2]
                Q_from_served_pu = -power_into_transformer_pu[3]
                # We then convert to MW and Mvar.
                total_P_load_MW = P_from_served_pu * self.twin.base_MVA
                total_Q_load_Mvar = Q_from_served_pu * self.twin.base_MVA
            else:
                # This handles the case when the secondary, LV bus is the
                # "to" bus.
                P_to_served_pu = -power_into_transformer_pu[0]
                Q_to_served_pu = -power_into_transformer_pu[1]
                # Again, we convert to MW and Mvar.
                total_P_load_MW = P_to_served_pu * self.twin.base_MVA
                total_Q_load_Mvar = Q_to_served_pu * self.twin.base_MVA

            # Once we know the total power served by the transformer, we can
            # compute the DER contribution, which is the insensitive load.
            insensitive_P_load_pu = (
                total_P_load_MW - sensitive_P_load_MW
            ) / self.twin.base_MVA
            insensitive_Q_load_pu = (
                total_Q_load_Mvar - sensitive_Q_load_Mvar
            ) / self.twin.base_MVA
            # Finally, we save the DER contribution to the transformer's
            # secondary bus.
            transformer.get_LV_bus().PL_pu = insensitive_P_load_pu
            transformer.get_LV_bus().QL_pu = insensitive_Q_load_pu
