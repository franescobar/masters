"""
A module for defining experiments on power systems, run with RAMSES.
"""

# Modules from this repository
import pf_dynamic  # for specifying the simulated system
import sim_interaction  # for specifying disturbances and observables
import control  # for specifying controllers
import nli
import visual  # for specifying visualizations
import records
import metrics

# Modules from the standard libray
import time  # for appending a timestamp to the experiment's name
from typing import Union  # for annotating observed objects
import os  # for creating directories and checking if files exist
import copy  # for deep copying systems and controllers before simulations
import bisect  # for inserting observables into sorted lists
import tabulate # for tabulating metrics

# Other modules
import numpy as np

try:
    import pyramses
    from pyramses import simulator
    print("Module 'pyramses' was succesfully imported!")
except ModuleNotFoundError:
    print("Module 'pyramses' could not be imported!")


def get_timestamp(
    opening_delimiter: str = "[", closing_delimiter: str = "]"
) -> str:
    """
    Return formatted timestampt.

    This timestamp is useful when creating the directories' name and when
    documenting the experiment. The format is easy to read and sort.
    """

    format = f"{opening_delimiter}%Y-%m-%d %H %M %S{closing_delimiter}"

    return time.strftime(format, time.localtime())


class Randomization:
    """
    A class for specifying randomizations of the experiment.
    """

    def __init__(self, description: str) -> None:
        self.description = description


class Experiment:
    """
    A class for specifying, running, and documenting experiments.

    Things I still need to figure out:
        - how to measure effort and performance,
        - how to manage files, and
        - how to time the experiment with decorators.
    """

    # Automatically-generated input files
    syst_filename = "system.dat"
    sett_filename = "settings.dat"
    obse_filename = "observables.dat"
    dist_filename = "disturbances.dat"
    # Output files
    init_filename = "init.trace"
    cont_filename = "continuous.trace"
    disc_filename = "discrete.trace"
    outp_filename = "output.trace"
    traj_filename = "obs.trj"
    # Directory names
    inp_dir = "0_Input"
    out_dir = "1_RAMSES output"
    vis_dir = "2_Visualizations"
    obs_dir = "3_Explicit observables"
    obs_children_dirs = ["BUS", "SYNC", "INJEC", "SHUNT", "DCTL", "BRANCH"]
    met_dir = "4_Metrics"
    det_dir = "5_Detectors"
    common_children_dirs = [
        inp_dir,
        out_dir,
        vis_dir,
        obs_dir,
        met_dir,
        det_dir,
    ]

    # Other filenames
    des_filename = "6_description.txt"
    log_filename = "7_log.txt"
    ana_filename = "8_analysis.txt"
    sum_filename = "9_summary.txt"
    common_filenames = [des_filename, log_filename, ana_filename, sum_filename]

    # Default setting names
    setting_names = [
        "PLOT_STEP",
        "GP_REFRESH_RATE",
        "DISP_PROF",
        "T_LOAD_REST",
        "OMEGA_REF",
        "S_BASE",
        "NEWTON_TOLER",
        "FIN_DIFFER",
        "FULL_UPDATE",
        "SKIP_CONV",
        "LATENCY",
        "SCHEME",
        "NB_THREADS",
        "SPARSE_SOLVER",
        "OMP",
        "NET_FREQ_UPD",
    ]

    solver_setting_names = [
        "disc_method",
        "max_h",
        "min_h",
        "latency",
        "upd_over",
    ]

    def __init__(
        self,
        name: str,
        DLL_dir: Union[str, None] = None,
        repetitions: int = 1,
        must_document=False,
    ) -> None:
        if len(name) > 10:
            raise RuntimeError(
                "Experiment name must not exceed 10 characters."
            )

        if DLL_dir is not None and not os.path.isdir(DLL_dir):
            raise RuntimeError(
                f"RAMSES DLL directory {DLL_dir} does not exist."
            )

        if not isinstance(repetitions, int) or repetitions < 1:
            raise RuntimeError(
                f"Number of repetitions must be a positive int."
            )

        self.name = name  # experiment's name
        self.DLL_dir = DLL_dir  # location of the RAMSES DLL
        self.repetitions = repetitions  # repetitions used to test performance
        self.must_document = (
            must_document  # if True, a dedicated folder is created
        )
        self.description = "No description was provided."
        self.systems: list[tuple[str, pf_dynamic.System]] = []
        self.controllers: list[tuple[str, list[control.Controller]]] = [
            ("No control", control.Controller())
        ]
        self.disturbances: list[
            tuple[str, list[sim_interaction.Disturbance]]
        ] = [
            ("No dist.", None) # All experiments include case without disturbance
        ]
        self.observables: list[sim_interaction.Observable] = []
        self.randomizations: list[Randomization] = [
            ("Not random", Randomization("Not random"))
        ]
        self.visualizations: list[visual.Visualization] = []
        self.metrics: list[metrics.Metric] = []

        self.settings = {
            "PLOT_STEP": 0.001,
            "GP_REFRESH_RATE": 1.0,
            "DISP_PROF": "T",
            "T_LOAD_REST": 0.005,
            "OMEGA_REF": "COI",
            "S_BASE": 100,
            "NEWTON_TOLER": (0.001, 0.001, 0.0001),
            "FIN_DIFFER": (0.00001, 0.00001),
            "FULL_UPDATE": "T",
            "SKIP_CONV": "F",
            "SCHEME": "DE",
            "NB_THREADS": 2,
            "SPARSE_SOLVER": "KLU",
            "LICENSE": "apetros@pm.me 94A55D745302D072449DF2DEA71ABA678990393074634763603FDC73E4CC481D",
        }

        self.solver_settings = {
            "disc_method": "BD",
            "max_h": 0.001,
            "min_h": 0.001,
            "latency": 0,
            "upd_over": "ABL",
        }

        self.warning_voltages = (0.85, 1.0)
        self.error_voltages = (0.7, 1.2)
        self.disturbance_window = 5
        self.horizon = 200
        self.pyramses_step = 20e-3
        self.current_repetition = 1

    def describe(self, description: str) -> None:
        """
        Includes stuff known a priori.
        """

        if len(description) > 100:
            raise RuntimeError(
                "Description should have no more than 100 chars"
            )

        self.description = description

    def add_system(self, description: str, system: pf_dynamic.System) -> None:
        """
        Add a system to the experiment.

        To prevent side effects, each system is deep-copied before the
        addition.
        """

        if len(description) > 10:
            raise RuntimeError(
                "Description should have no more than 10 characters."
            )

        if not isinstance(system, pf_dynamic.System):
            raise RuntimeError(
                f"System {system} is not an instance of the System class."
            )

        # Each system is actually a tuple (description: str, system:
        # pf_dynamic.System), where the tuple is useful for creating
        # directories and printing to terminal.
        element = (description, copy.deepcopy(system))

        # When adding a system, it's not necessary to sort it since the notion
        # of ordering is not define for systems (unlike for disturbances, for
        # instance).
        self.systems.append(element)

    def add_disturbances(
        self, description: str, *disturbances: sim_interaction.Disturbance
    ) -> None:
        """
        Add disturbance(s) that are applied simultaneously.

        Because they are not changed during the experiment, they don't need to
        be deep-copied.
        """

        if len(description) > 10:
            raise RuntimeError(
                "Description should have no more than 10 characters."
            )

        for dist in disturbances:
            if not isinstance(dist, sim_interaction.Disturbance):
                raise RuntimeError(
                    f"Disturbance {dist} "
                    f"is not an instance of the Disturbance class."
                )

        # As with systems, each disturbance is actually a tuple (description:
        # str, disturbances: list[Disturbance]), where the tuple is useful for
        # creating directories and printing to terminal. Note that several
        # disturbances can be applied in batch.
        element = (description, sorted(list(disturbances)))

        # These batches of disturbances are not sorted, but each list of
        # disturbances is sorted internally.
        self.disturbances.append(element)

    def add_observables(
        self, *observables: sim_interaction.Observable
    ) -> None:
        """
        Add explicit observable.

        All remaining observables are defined automatically by instances of the
        visualization class.

        As with disturbances, they need not be deep-copied.

        Note finally that observables do not require a description, since they
        don't need to go into the directory name.
        """

        for obs in observables:
            if not isinstance(obs, sim_interaction.Observable):
                raise RuntimeError(
                    f"Observable {obs} "
                    f"is not an instance of the Observable class."
                )

            # Add observable to a sorted list
            bisect.insort(self.observables, obs)

    def add_controllers(
        self, description: str, *controllers: control.Controller
    ) -> None:
        """
        Add controller(s) that act simultaneously on the system.

        To prevent side effects, a deep copy of each controller is created
        first.

        Controllers are not sorted, since the notion does not make much sense.
        """

        if len(description) > 10:
            raise RuntimeError(
                "Description should have no more than 10 characters."
            )

        for con in controllers:
            if not isinstance(con, control.Controller):
                raise RuntimeError(
                    f"Controller {con} "
                    f"is not an instance of the Controller class."
                )

            if not hasattr(con, "overrides_OLTCs"):
                raise RuntimeError(
                    f"Controller {con} "
                    f"does not have an overrides_OLTCs attribute."
                )

            if not hasattr(con, "get_actions"):
                raise RuntimeError(
                    f"Controller {con} does not have a get_actions method."
                )

        element = (description, [copy.deepcopy(con) for con in controllers])
        self.controllers.append(element)

    def add_randomization(self):
        """
        Specify randomization of the network parameters (maybe a function).
        """

    def add_metrics(self, *defined_metrics: metrics.Metric) -> None:
        """
        Add metrics to compare effort of network elements (can be functions).
        """

        for metric in defined_metrics:
            if not isinstance(metric, metrics.Metric):
                raise RuntimeError(
                    f"Metric {metric} "
                    f"is not an instance of the Metric class."
                )
            
            self.metrics.append(metric)

    def add_visualizations(self, *visualizations: visual.Visualization) -> None:
        """
        Add an instance of the Visualization class (see visual.py).
        """

        for vis in visualizations:
            if not isinstance(vis, visual.Visualization):
                raise RuntimeError(
                    f"Visualization {vis} "
                    f"is not an instance of the Visualization class."
                )

            # Visualizations need neither a description nor an ordering.
            self.visualizations.append(vis)

    def set_RAMSES_settings(
        self, settings_dict: dict[str, Union[str, float]]
    ) -> None:
        """
        Set the RAMSES settings from a dictionary.

        Some of these settings are included in the disturbance files, other
        in the settings.dat file.

        Important settings are:
            'PLOT_STEP':        time [s]
            'GP_REFRESH_RATE':  time_interval [s]
            'DISP_PROF':        enable (T or F)
            'T_LOAD_REST':      time_constant [s]
            'OMEGA_REF':        reference (SYN or COI)
            'S_BASE':           base_power [MVA]
            'NEWTON_TOLER':     net_tol inj_rel_tol inj_abs_tol
            'FIN_DIFFER':       proportional_val abs_val
            'FULL_UPDATE':      enable (T or F)
            'SKIP_CONV':        enable (T or F)
            'LATENCY':          time_window [s] early_stop (0 or 1)
            'SCHEME':           scheme (DE for decomposed or IN for integrated)
            'NB_THREADS':       N (for large systems, number of physical cores)
            'SPARSE_SOLVER':    solver (KLU)
            'OMP':              assignment method (STA, DYN, GUI) chunk (int)
            'NET_FREQ_UPD':     enable (T or F)
        """

        for setting, value in settings_dict.items():
            if setting not in self.setting_names:
                raise RuntimeError(f"Unknown setting {setting}.")
            else:
                self.settings[setting] = value

    def set_solver_and_horizon(
        self,
        solver_settings_dict: dict[str, Union[str, float]],
        horizon: float,
    ) -> None:
        """
        Write solver settings and horizon to the disturbances file.

        Settings are:
            'disc_method':  TR, BE or BD
            'max_h':        maximum step [s]
            'min_h':        minimum step [s]
            'latency':      latency [pu]
            'upd_over':     ALL, NET, ABL or NOT
        """

        if horizon <= 0:
            raise RuntimeError("Horizon must be positive.")

        self.horizon = horizon

        for setting, value in solver_settings_dict.items():
            if setting not in self.solver_setting_names:
                raise RuntimeError(f"Unknown setting {setting}.")
            else:
                self.solver_settings[setting] = value

    def get_dir(self) -> str:
        """
        Generate directory name for the experiment.
        """

        if self.must_document:
            if self.repetitions == 1:
                return f"{get_timestamp()} {self.name}"
            else:
                return (
                    f"{get_timestamp()} {self.name} "
                    f"({self.current_repetition})"
                )
        else:
            return self.name

    def get_settings_str(self) -> str:
        """
        Return settings in RAMSES format.
        """

        # Define separation between columns
        col_sep = 1
        end_sep = 1
        end_char = ";"

        def val2str(values: Union[tuple, str, float]) -> str:
            """
            Convert one (or several) setting value(s) to a string.

            This function makes sure that all numbers are printed as floats
            and that scientific notation is never used.
            """

            if isinstance(values, tuple):
                return " ".join(
                    np.format_float_positional(val) for val in values
                )
            elif not isinstance(values, str):
                return np.format_float_positional(values)
            else:
                return str(values)

        # Infer column widths
        setting_width = max(len(s) for s in self.settings.keys()) + 1
        val_width = (
            max(len(val2str(v)) for v in self.settings.values()) + end_sep
        )

        # Initialize
        text = "# Simulation settings\n\n"

        # Write settings
        for setting, val in self.settings.items():
            text += (
                f"${setting.ljust(setting_width)}"
                f"{val2str(val).ljust(val_width)}"
                f"{end_char}\n"
            )

        return text

    def get_solver_and_horizon_str(self) -> str:
        """
        Return solver settings in RAMSES format.
        """

        # Define simulation interval
        interval = (0, self.horizon)

        # Formar numbers as strings
        t0 = f"{interval[0]:.3f}"
        tf = f"{interval[1]:.3f}"
        width = max(len(w) for w in [t0, tf])

        def val2str(val):
            """
            Convert a setting value to a string.

            This function makes sure that all numbers are printed as floats
            and that scientific notation is never used.
            """

            return np.format_float_positional(val)

        # Include solver settings. Single quotes are used so that the f-string
        # works.
        head = (
            f"{t0.rjust(width)} CONTINUE SOLVER "
            f'{self.solver_settings["disc_method"]} '
            f'{val2str(self.solver_settings["max_h"])} '
            f'{val2str(self.solver_settings["min_h"])} '
            f'{val2str(self.solver_settings["latency"])} '
            f'{self.solver_settings["upd_over"]}\n'
        )

        tail = f"{tf.rjust(width)} STOP\n"

        return head + tail

    def get_observables_str(self) -> str:
        """
        Return observables in RAMSES format.
        """

        return "".join(
            f"{obs}\n" for obs in sorted(list(set(self.observables)))
        )

    def case2str(
        self,
        system_description: str,
        disturbance_description: str,
        controller_description: str,
        randomization_description: str,
    ) -> str:
        """
        Generate case directory name.
        """

        return (
            f"{self.get_dir()}/"
            f"{system_description}, {disturbance_description}, "
            f"{controller_description}, {randomization_description}"
        )

    def init_files_and_dirs(self) -> None:
        """
        Create directory structure and create input files.

        The structure is, in terms of the class' attributes,

            - dir
                - case_dir
                    - inp_dir
                    - out_dir
                    - obs_dir
                        - obs_children_dirs[i]
                    - vis_dir
                    - met_dir
                    - des_filename
                    - log_filename
                    - ana_filename
                    - sum_filename

        Files that are created when calling this method are, again, in terms
        of the class' attributes,

            - obse_filename,
            - sett_filename,
            - dist_filename

        """

        # Create names
        case_dirs = [
            self.case2str(
                system_description=sys[0],
                disturbance_description=dis[0],
                controller_description=con[0],
                randomization_description=ran[0],
            )
            for sys in self.systems
            for dis in self.disturbances
            for con in self.controllers
            for ran in self.randomizations
        ]

        # Create experiment's directory
        if not os.path.exists(self.get_dir()):
            os.mkdir(self.get_dir())

        # Initialize files and directories
        for case_dir in case_dirs:
            # Create case directories
            if not os.path.exists(case_dir):
                os.mkdir(case_dir)

            # Create their children directories
            for child_dir in self.common_children_dirs:
                name = os.path.join(case_dir, child_dir)
                if not os.path.exists(name):
                    os.mkdir(name)
                # Create folder for each observable type
                if child_dir == self.obs_dir:
                    for obs_child_dir in self.obs_children_dirs:
                        name = os.path.join(case_dir, child_dir, obs_child_dir)
                        if not os.path.exists(name):
                            os.mkdir(name)

            # Initialize RAMSES input
            with open(
                os.path.join(case_dir, self.inp_dir, self.obse_filename), "w"
            ) as f:
                f.write(self.get_observables_str())

            with open(
                os.path.join(case_dir, self.inp_dir, self.sett_filename), "w"
            ) as f:
                f.write(self.get_settings_str())

            with open(
                os.path.join(case_dir, self.inp_dir, self.dist_filename), "w"
            ) as f:
                f.write(self.get_solver_and_horizon_str())

    def __str__(self) -> str:
        """
        Return experiment description for printing it to the terminal.
        """

        return self.description

    def run_simulation(
        self, cwd: str, sys: pf_dynamic.System, record_traces: bool = False
    ) -> None:
        """
        Run simulation of a single system.

        To do during the simulation:
        - Collect measurements for visualization
        - Collect measurements for performance metrics
        - Time parts of the code
        """

        # Define simulation settings (almost always constant)
        h = self.pyramses_step
        time_values = np.arange(0, self.horizon, h)

        # Make the OLTCs 'instantaneous' in terms of simulation settings. This
        # has to be done before creating the .dat file.
        for OLTC in sys.OLTC_controllers:
            OLTC.delay_RAMSES_1 = OLTC.delay_RAMSES_2 = 3 / 4 * h

        def map2inp(filename: str) -> str:
            """
            Append path to RAMSES' input folder.
            """

            return os.path.join(cwd, self.inp_dir, filename)

        sys.export_to_RAMSES(map2inp(self.syst_filename))

        # Configure simulation and add the input files
        case = pyramses.cfg()
        case.addData(map2inp(self.syst_filename))  # system description
        case.addData(map2inp(self.sett_filename))  # RAMSES settings
        case.addObs(map2inp(self.obse_filename))  # observables
        case.addDst(map2inp(self.dist_filename))  # initial disturbances

        def map2out(filename: str) -> str:
            """
            Append path to RAMSES' output folder.
            """

            return os.path.join(cwd, self.out_dir, filename)

        # Add output files
        case.addInit(map2out(self.init_filename))  # initialization trace
        case.addTrj(map2out(self.traj_filename))  # trajectory file

        # The output trace should be removed before any simulation. Otherwise,
        # the results of previous simulations will be appended to the file and
        # debugging will be difficult.
        out_trace = map2out(self.outp_filename)
        if os.path.exists(out_trace):
            os.remove(out_trace)
        case.addOut(out_trace)  # output trace

        if record_traces:
            case.addCont(map2out(self.cont_filename))  # continuous trace, slow
            case.addDisc(map2out(self.disc_filename))  # discrete trace, slow

        # Create simulation instance using the custom DLL
        sys.ram = pyramses.sim(custLibDir=self.DLL_dir)
        sys.ram.execSim(case, 0)

        # Get bus names
        monitored_buses = sys.ram.getAllCompNames("BUS")

        # Read stopping criteria
        vmin, vmax = self.error_voltages

        for tk in time_values:
            # Display progress
            perc = int(round(100 * tk / self.horizon))
            if 0 < tk and np.isclose(perc % 1, 0, atol=1e-3):
                # print("", end=f"\rSimulation progress: {perc} %")
                print(f"Simulation progress is {100*tk/self.horizon:.2f}")

            # Simulate if voltages are OK
            try:
                voltages = sys.ram.getBusVolt(monitored_buses)
                if tk > self.disturbance_window and not all(
                    vmin < v < vmax for v in voltages
                ):
                    print("\nThere are undervoltages after the disturbance window.")
                    break
                sys.ram.contSim(tk)
            except:
                print(sys.ram.getLastErr())
                break

            # At all iterations, if the simulation went fine, update controllers
            # and send disturbances
            sys.update_detectors()
            sys.follow_controllers()
            sys.send_disturbances_until(t=tk + h)

        # Print empty line
        print("")

        # Finish simulation
        sys.ram.endSim()

    def run(self):
        """
        Run the experiment.

        This run can include multiple systems, multiple controllers, multiple
        sets of disturbances, and multiple randomizations.
        """

        # Add observables from visualizations

        # Iterate over all cases
        for sys_description, system in self.systems:
            for con in self.controllers:
                for dis in self.disturbances:
                    for ran in self.randomizations:
                        # Create a deep copy of the system
                        ram = system.ram
                        system.ram = None
                        sys = copy.deepcopy(system)
                        sys.ram = ram

                        # Add controllers to the system (con[1] is iterable)
                        if con[0] != "No control":
                            print(con[1])
                            sys.add_controllers(con[1])

                        # Add observables from detectors
                        for d in sys.detectors:
                            self.add_observables(*d.get_required_observables())

                        # Add disturbances to the system (dis[1] is iterable)
                        if dis[0] != "No dist.": 
                            sys.add_disturbances(dis[1])

                        # Generate working directory
                        cwd = self.case2str(
                            system_description=sys_description,
                            disturbance_description=dis[0],
                            controller_description=con[0],
                            randomization_description=ran[0],
                        )

                        # Initialize directories
                        self.init_files_and_dirs()

                        # Run simulation
                        self.run_simulation(cwd=cwd, sys=sys)

                        def map2out(filename: str) -> str:
                            """
                            Append path to RAMSES' output folder.
                            """

                            return os.path.join(cwd, self.out_dir, filename)

                        # Add output files
                        ext = pyramses.extractor(map2out(self.traj_filename))

                        # Generate results
                        self.build_visualizations(system=sys, extractor=ext)
                        self.analyze_experiment()
                        self.document_experiment()

                        # Remove the trj
                        os.remove(map2out(filename=self.traj_filename))

                        def map2metric(filename: str) -> str:
                            """
                            Append path to metrics folder.
                            """

                            return os.path.join(cwd, self.met_dir, filename)

                        # Write performance metrics to file
                        with open(map2metric("metrics.txt"), "w") as f:
                            data = []
                            
                            for metric in self.metrics:
                                value = metric.evaluate(
                                    system=sys, extractor=ext
                                )
                                row = [metric.name, value, metric.units]
                                data.append(row)

                            headers = ["Metric description", "Value", "Units"]
                            precision = (0, ".4f", 0)

                            table = tabulate.tabulate(
                                tabular_data=data,
                                headers=headers,
                                floatfmt=precision,
                            )

                            f.write(table + "\n")

    def build_visualizations(self,
                             system: pf_dynamic.System,
                             extractor: pyramses.extractor) -> None:
        """
        Build visualizations so that they can be compiled in LaTeX.
        """

        for visualization in self.visualizations:
            visualization.generate(system=system, extractor=extractor)

    def analyze_experiment(self) -> None:
        """
        Analyze the experiment and write results to a file.

        Some of the questions this analysis should answer are:
            - Which controller was better?
            - What major actions took place?
            - How did some metrics change from one repetition to another?
        """

    def document_experiment(self) -> None:
        """
        Organize results (visualizations, analysis, raw output) in a folder.
        """
