"""
Test functions for the 'experiment' module.
"""

# Module to be tested
from experiment import get_timestamp, Experiment, VoltageIntegral, ReactiveMargin, TapMovements

# Modules from this repository
import pf_dynamic
import sim_interaction
from test_sim_interaction import (
    get_record_examples,
    get_disturbance_examples,
    get_observable_examples,
)
from test_pf_dynamic import get_dynamic_nordic
import control
import records
import nli
import visual

# Modules from the standard library
import os
import shutil

# Other modules
import numpy as np


def test_get_timestamp():
    """
    Test retrieval of current timestamp.
    """

    print(get_timestamp())
    looks_nice = input("Does the timestamp look correct (check date)? (y/n) ")
    assert looks_nice == "y", "Timestamp is incorrect."


# Create dummy DLL file
dummy_dll_path = r"C:\Users\FranciscoEscobarPrad\Desktop"


def get_experiments() -> tuple[Experiment, Experiment]:
    """
    Get two experiments for testing.
    """

    exp = Experiment(name="Test", DLL_dir=dummy_dll_path, repetitions=1)
    exp_multiple = Experiment(
        name="Test", DLL_dir=dummy_dll_path, repetitions=3
    )

    return exp, exp_multiple


def test_Experiment_initialization():
    """
    Test class attributes of the Experiment class.
    """

    exp, exp_multiple = get_experiments()

    # Test class attributes
    assert exp.syst_filename == "system.dat"
    assert exp.sett_filename == "settings.dat"
    assert exp.obse_filename == "observables.dat"
    assert exp.dist_filename == "disturbances.dat"
    assert exp.init_filename == "init.trace"
    assert exp.cont_filename == "continuous.trace"
    assert exp.disc_filename == "discrete.trace"
    assert exp.outp_filename == "output.trace"
    assert exp.traj_filename == "obs.trj"
    assert exp.inp_dir == "0_Input"
    assert exp.out_dir == "1_RAMSES output"
    assert exp.vis_dir == "2_Visualizations"
    assert exp.obs_dir == "3_Explicit observables"
    assert exp.obs_children_dirs == [
        "BUS",
        "SYNC",
        "INJEC",
        "SHUNT",
        "DCTL",
        "BRANCH",
    ]
    assert exp.met_dir == "4_Metrics"
    assert exp.det_dir == "5_Detectors"
    assert exp.common_children_dirs == [
        exp.inp_dir,
        exp.out_dir,
        exp.vis_dir,
        exp.obs_dir,
        exp.met_dir,
        exp.det_dir,
    ]
    assert exp.des_filename == "6_description.txt"
    assert exp.log_filename == "7_log.txt"
    assert exp.ana_filename == "8_analysis.txt"
    assert exp.sum_filename == "9_summary.txt"
    assert exp.common_filenames == [
        exp.des_filename,
        exp.log_filename,
        exp.ana_filename,
        exp.sum_filename,
    ]
    assert exp.setting_names == [
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
    assert exp.solver_setting_names == [
        "disc_method",
        "max_h",
        "min_h",
        "latency",
        "upd_over",
    ]

    # Test instance attributes
    assert exp.name == "Test", "Experiment name is incorrect."
    assert exp.DLL_dir == dummy_dll_path, "DLL directory is incorrect."
    assert exp.repetitions == 1, "Number of repetitions is incorrect."
    assert (
        exp_multiple.repetitions == 3
    ), "Number of repetitions is incorrect (should be several)."
    assert not exp.must_document, "Experiment must not be documented."

    # Test experiment description
    assert exp.description == (
        "No description was provided."
    ), "Experiment description is incorrect."

    # Test initial (empty) containers
    assert exp.systems == [], "Systems should be empty."
    # assert exp.controllers == [], "Controllers should be empty."
    assert exp.disturbances == [], "Disturbances should be empty."
    assert exp.observables == [], "Observables should be empty."
    # assert exp.randomizations == [], "Randomizations should be empty."
    assert exp.metrics == [], "Metrics should be empty."

    # Test RAMSES settings
    assert exp.settings == {
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

    }, "RAMSES settings are incorrect."

    # Test solver and time horizon
    assert exp.solver_settings == {
        "disc_method": "BD",
        "max_h": 0.001,
        "min_h": 0.001,
        "latency": 0.0,
        "upd_over": "ABL",
    }

    # Test remaining parameters
    assert exp.warning_voltages == (
        0.85,
        1.0,
    ), "Warning voltages are incorrect."
    assert exp.error_voltages == (0.7, 1.2), "Error voltages are incorrect."
    assert exp.disturbance_window == 5, "Disturbance window is incorrect."
    assert exp.horizon == 200, "Time horizon is incorrect."
    assert np.isclose(exp.pyramses_step, 20e-3), "PyRAMSES step is incorrect."
    assert exp.current_repetition == 1, "Current repetition is incorrect."


def test_describe():
    """
    Test experiment description (stuff known a priori).
    """

    exp, _ = get_experiments()

    assert (
        exp.description == "No description was provided."
    ), "Experiment description is incorrect."

    exp.describe("Test description.")

    assert (
        exp.description == "Test description."
    ), "Experiment description is incorrect."


def test_add_system():
    """
    Test system addition to an experiment.
    """

    exp, _ = get_experiments()

    class TestSystem(pf_dynamic.System):
        pass

    exp.add_system(description="Descript.", system=TestSystem())

    assert len(exp.systems) == 1, "System was not added to experiment."
    assert (
        exp.systems[0][0] == "Descript."
    ), "System description is incorrect after addition."

    # Test deep copying
    sys = TestSystem()
    exp.add_system(description="Second", system=sys)
    assert (
        exp.systems[1][0] == "Second" and exp.systems[1][1] is not sys
    ), "System was not deep copied."

    # Test detection of invalid system types
    class WrongSystem:
        pass

    try:
        exp.add_system(description="Wrong", system=WrongSystem())
        assert False, "System of type 'WrongSystem' should not be allowed."
    except RuntimeError:
        pass
    else:
        assert False, "System of type 'WrongSystem' should not be allowed."


def test_add_disturbances():
    """
    Test disturbance addition to an experiment.
    """

    exp, _ = get_experiments()

    (
        dist_bus,
        dist_branch,
        dist_inj,
        dist_dctl,
        dist_solver,
    ) = get_disturbance_examples()

    exp.add_disturbances(
        "Examples", dist_bus, dist_branch, dist_inj, dist_dctl, dist_solver
    )

    assert (
        len(exp.disturbances) == 1
    ), "Disturbances were not added in batch to the experiment."
    assert (
        exp.disturbances[0][0] == "Examples"
    ), "Disturbance description is incorrect after addition."
    assert (
        exp.disturbances[0][1][-1] is dist_solver
        and exp.disturbances[0][1][-2] is dist_dctl
    ), "Disturbances were not added in the correct order."


def test_add_observables():
    """
    Test addition of observables to an experiment.
    """

    exp, _ = get_experiments()

    (
        obs_bus,
        obs_branch,
        obs_gen,
        obs_exc,
        obs_inj,
        obs_dctl,
    ) = get_observable_examples()

    exp.add_observables(
        obs_bus, obs_branch, obs_gen, obs_exc, obs_inj, obs_dctl
    )

    assert (
        len(exp.observables) == 6
    ), "Observables were not added in batch to the experiment."
    # The following tests use exp.observables[1] because the first observable
    # is, according to the sorting, the branch observable, so that the bus
    # observable comes second.
    assert (
        exp.observables[1].observed_object is obs_bus.observed_object
    ), "Observable object is incorrect after addition."
    assert (
        exp.observables[1].obs_name == "BV"
    ), "Observable name is incorrect after addition."
    assert (
        exp.observables[1].element_type == "BUS"
    ), "Observable element type is incorrect after addition."


def test_add_controllers():
    """
    Test addition of controllers to an experiment.
    """

    exp, _ = get_experiments()

    class TestController(control.Controller):
        pass

    C1 = TestController()
    C2 = TestController()

    assert hasattr(
        C1, "overrides_OLTCs"
    ), "Controller lacks mandatory attribute. Did control.Controller change?"
    assert hasattr(
        C1, "get_actions"
    ), "Controller lacks mandatory method. Did control.Controller change?"

    exp.add_controllers("Test", C1, C2)

    assert (
        len(exp.controllers) == 2
    ), "Controllers were not added in batch to the experiment."
    assert (
        exp.controllers[1][0] == "Test"
    ), "Controller description is incorrect after addition."

    # Test deep copying
    assert (
        exp.controllers[1][1][-1] is not C2
    ), "Controllers were not added in the correct order."
    assert (
        exp.controllers[1][1][-2] is not C1
    ), "Controllers were not added in the correct order."


def test_add_randomization():
    """
    Test introduction of randomization to experiment.
    """

    # Not implemented yet. Nothing to test.


def test_add_metrics():
    """
    Test addition of performance metrics to experiment.
    """

    # Not implemented yet. Nothing to test.


def test_add_visualization():
    """
    Test addition of visualizations to experiment.
    """

    # Not implemented yet. Nothing to test.


def test_set_RAMSES_settings():
    """
    Test setting of RAMSES settings.
    """

    exp, _ = get_experiments()

    # Test for invalid settings
    try:
        exp.set_RAMSES_settings(settings_dict={"UNKNOWN": 0.0})
        assert False, "Invalid setting should not be allowed."
    except RuntimeError:
        pass
    else:
        assert False, "Invalid setting should not be allowed."

    assert "NET_FREQ_UPD" not in exp.settings, "Setting should not be default."

    # Set some values
    exp.set_RAMSES_settings(
        settings_dict={
            "PLOT_STEP": 0.002,
            "NET_FREQ_UPD": "T",
        }
    )

    assert np.isclose(
        exp.settings["PLOT_STEP"], 0.002
    ), "Setting was not changed."
    assert exp.settings["NET_FREQ_UPD"] == "T", "Setting was not changed."


def test_set_solver_and_horizon():
    """
    Test setting of solver and time horizon.
    """

    exp, _ = get_experiments()

    # Test for invalid settings
    try:
        exp.set_solver_and_horizon(
            solver_settings_dict={"UNKNOWN": 0.0}, horizon=100.0
        )
        assert False, "Invalid setting should not be allowed."
    except RuntimeError:
        pass
    else:
        assert False, "Invalid setting should not be allowed."

    # Test for invalid horizon
    try:
        exp.set_solver_and_horizon(solver_settings_dict={}, horizon=0.0)
        assert False, "Invalid horizon should not be allowed."
    except RuntimeError:
        pass
    else:
        assert False, "Invalid horizon should not be allowed."

    try:
        exp.set_solver_and_horizon(solver_settings_dict={}, horizon=-1.0)
        assert False, "Invalid horizon should not be allowed."
    except RuntimeError:
        pass
    else:
        assert False, "Invalid horizon should not be allowed."

    # Set some values
    exp.set_solver_and_horizon(
        solver_settings_dict={"disc_method": "BE"}, horizon=100.0
    )

    assert (
        exp.solver_settings["disc_method"] == "BE"
    ), "Setting was not changed."


def test_get_dir():
    """
    Test retrieval of directory name for experiment.
    """

    exp, exp_multiple = get_experiments()

    # Test directory name when not documented
    assert "Test" in exp.get_dir(), "Directory must contain experiment name."
    year_month_day = get_timestamp().split(" ")[0]
    assert (
        year_month_day not in exp.get_dir()
    ), "Directory name must contain timestamp only when documented."

    # Manually change documentation flag
    exp.must_document = True
    assert "Test" in exp.get_dir(), "Directory must contain experiment name."
    assert (
        year_month_day in exp.get_dir()
    ), "Directory name must contain timestamp when documented."

    # Return flag to its original value
    exp.must_document = False

    # Change flag of other experiment
    exp_multiple.must_document = True

    # Test directory name for several repetitions
    assert (
        "(1)" in exp_multiple.get_dir()
    ), "Directory must contain repetition number."
    exp_multiple.current_repetition += 1
    assert (
        "(2)" in exp_multiple.get_dir()
    ), "Directory must contain repetition number."

    # Return flag to its original value
    exp_multiple.must_document = False


def test_get_settings_str():
    """
    Test printing of RAMSES settings for experiment.
    """

    # Get experiments
    exp, _ = get_experiments()

    # assert "" + (
    #     "# Simulation settings\n\n"
    #     "$PLOT_STEP       0.001              ;\n"
    #     "$GP_REFRESH_RATE 1.                 ;\n"
    #     "$DISP_PROF       T                  ;\n"
    #     "$T_LOAD_REST     0.005              ;\n"
    #     "$OMEGA_REF       COI                ;\n"
    #     "$S_BASE          100.               ;\n"
    #     "$NEWTON_TOLER    0.001 0.001 0.0001 ;\n"
    #     "$FIN_DIFFER      0.00001 0.00001    ;\n"
    #     "$FULL_UPDATE     T                  ;\n"
    #     "$SKIP_CONV       F                  ;\n"
    #     "$SCHEME          DE                 ;\n"
    #     "$NB_THREADS      2.                 ;\n"
    #     "$SPARSE_SOLVER   KLU                ;\n"
    # ) in exp.get_settings_str(), "RAMSES settings string is incorrect."


def test_get_solver_and_horizon_str():
    """
    Test printing of solver (with settings) and time horizon for experiment.
    """

    # Get experiments
    exp, _ = get_experiments()

    # Compare string with solver settings
    assert exp.get_solver_and_horizon_str() == (
        f"  0.000 CONTINUE SOLVER BD 0.001 0.001 0. ABL\n"
        f"{exp.horizon:.3f} STOP\n"
    ), "Solver and time horizon string is incorrect."


def test_get_observables_str():
    """
    Test printing of RAMSES observables.
    """

    # Get experiments
    exp, _ = get_experiments()

    # Generate dummy observables. Usually, these would be added with the
    # 'add_observables' method, but we do it manually here to decouple this
    # test from the 'add_observables' method.
    for object in get_record_examples():
        exp.observables.append(
            sim_interaction.Observable(observed_object=object, obs_name="Test")
        )

    assert exp.get_observables_str() == (
        "BRANCH Example branch\n"
        "BUS Example bus\n"
        "DCTL Example DCTL\n"
        "INJEC Example load\n"
        "SHUNT Example shunt\n"
        "SYNC Example generator\n"
    )


def test_case2str():
    """
    Test conversion of RAMSES case to string.
    """

    # Get experiments
    exp, _ = get_experiments()

    assert exp.case2str(
        system_description="Sys",
        disturbance_description="Dist",
        controller_description="Cont",
        randomization_description="Rand",
    ) == ("Test/Sys, Dist, Cont, Rand"), "Case string is incorrect."


def get_full_experiment() -> Experiment:
    """
    Get an experiment for testing with all components.
    """

    exp = Experiment(
        name="Dir. test",
        DLL_dir=dummy_dll_path,
        repetitions=3,
        must_document=True,
    )

    exp.describe("This experiment simply builds the directory tree.")

    # Add a system to the experiment
    class TestSystem(pf_dynamic.System):
        pass

    exp.add_system(description="Test syst.", system=TestSystem())

    # Add disturbances
    (
        dist_bus,
        dist_branch,
        dist_inj,
        dist_dctl,
        dist_solver,
    ) = get_disturbance_examples()
    exp.add_disturbances(
        "Test dist.", dist_bus, dist_branch, dist_inj, dist_dctl, dist_solver
    )

    # Add observables
    (
        obs_bus,
        obs_branch,
        obs_gen,
        obs_exc,
        obs_inj,
        obs_dctl,
    ) = get_observable_examples()
    exp.add_observables(
        obs_bus, obs_branch, obs_gen, obs_exc, obs_inj, obs_dctl
    )

    # Add controllers
    class TestController(control.Controller):
        pass

    C1 = TestController()
    exp.add_controllers("Test cont.", C1)

    return exp


def test_init_files_and_dirs():
    """
    Test initialization of file- and directory tree.
    """

    exp = get_full_experiment()
    exp.init_files_and_dirs()
    looks_nice = input("Does the directory tree look correct? (y/n) ")
    assert looks_nice == "y", "Directory tree is incorrect."

    def is_test_dir(dir: str) -> bool:
        """
        Check if directory is the test directory.
        """

        return "[" in dir and "]" in dir and "Dir. test" in dir

    found_test_dir = False
    for dir in os.listdir("."):
        if os.path.isdir(dir) and is_test_dir(dir):
            shutil.rmtree(dir)
            found_test_dir = True

    if not found_test_dir:
        raise RuntimeError("Test directory was not found.")


def test_str():
    """
    Test string representation of experiment for printing in terminal.
    """

    # Get experiments
    exp, _ = get_experiments()

    assert str(exp) == (
        "No description was provided."
    ), "Experiment string is incorrect."


def test_run_simulation():
    """
    Test the execution of a single simulation.
    """

    # Import both the static and the dynamic data of the Nordic
    nordic = get_dynamic_nordic(solve=True)
    nordic.import_dynamic_data(
        filename=os.path.join(
            "networks",
            "Nordic",
            "Nordic test system",
            "dyn_A.dat"
        )
    )

    # Make all loads voltage sensitive. Otherwise, the syste, will be small-
    # signal unstable and the simulation will diverge after a few seconds.
    for injector in nordic.injectors:
        if isinstance(injector, records.Load):
            # These are the values recommended by the Task Force:
            injector.make_voltage_sensitive(alpha=1, beta=2)

    # Initialie experiment
    exp = Experiment(
        name="run_sim()",
        DLL_dir=r"C:\Users\FranciscoEscobarPrad\Desktop\URAMSES-1.2\Release_intel_w64"
    )

    # As a single observable, add the voltage magnitude at bus 4041.
    obs = sim_interaction.Observable(
        observed_object=nordic.get_bus(name="1041"),
        obs_name="BV"
    )
    exp.add_observables(obs)

    # Add disturbances
    exp.add_disturbances(
        "4032-4044",
        sim_interaction.Disturbance(
            ocurrence_time=1.0,
            object_acted_on=nordic.get_bus(name="4032"),
            par_name="fault",
            par_value=1e-4
        ),
        sim_interaction.Disturbance(
            ocurrence_time=1.1,
            object_acted_on=nordic.get_bus(name="4032"),
            par_name="clearance",
            par_value=None
        ),
        sim_interaction.Disturbance(
            ocurrence_time=1.1,
            object_acted_on=nordic.get_branches_between(
                bus_name_1="4032",
                bus_name_2="4044"
            )[0],
            par_name="status",
            par_value=0
        )
    )

    nordic.add_disturbances(exp.disturbances[0][1])

    # Add the NLI detectors
    nordic.add_detector(
        detector=nli.NLI(
            observed_corridor=("4041", ["4031"]),
            h=20e-3,
            delta_T=7,
            tau_s=1,
            epsilon=1e-3
        )
    )

    nordic.add_detector(
        detector=nli.NLI(
            observed_corridor=("4042", ["4021", "4032"]),
            h=20e-3,
            delta_T=7,
            tau_s=1,
            epsilon=1e-3
        )
    )

    for generator in nordic.generators:
        nordic.add_detector(
            detector=nli.FieldCurrent(machine_name=generator.name)
        )

    # Visualize the NLI
    exp.add_visualizations(
        visual.NLI_plots(receiving_buses=["4041", "4042"])
    )

    exp.set_solver_and_horizon(
        solver_settings_dict={}, horizon=40.0
    )

    # Add all bus voltages as observables
    for bus in nordic.buses:
        exp.add_observables(
            sim_interaction.Observable(
                observed_object=bus,
                obs_name="BV"
            )
        )

    # Add all field currents as observables
    for generator in nordic.generators:
        exp.add_observables(
            sim_interaction.Observable(
                observed_object=generator,
                obs_name="if"
            )
        )

    # Add all transformers as observables
    for record in nordic.records:
        if isinstance(record, records.DCTL):
            exp.add_observables(
                sim_interaction.Observable(
                    observed_object=record,
                    obs_name=None
                )
            )

    # print(nordic.detectors)
    # exit()
    # Adding the system should always be the last thing.
    exp.add_system(description="Nordic", system=nordic)

    exp.add_metrics(VoltageIntegral())
    exp.add_metrics(ReactiveMargin())
    exp.add_metrics(TapMovements())



    # Create the directory test_sim() for the simulation, as well as the input
    # and output directories. We prefer to create these directories from
    # scratch to decouple this test from that of self.init_files_and_dirs.
    dir = "test_sim"
    if not os.path.isdir(dir):
        os.mkdir(dir)
    else:
        delete = input("Directory test_sim already exists. Remove? (y/n) ")
        if delete == "y":
            shutil.rmtree(dir)
            os.mkdir(dir)
        else:
            print("Exiting...")
    os.mkdir(os.path.join(dir, "0_Input"))
    os.mkdir(os.path.join(dir, "1_RAMSES output"))

    # Add the observables file:
    with open(os.path.join(dir, "0_Input", exp.obse_filename), "w") as f:
        f.write(exp.get_observables_str())

    # Add the settings file:
    with open(os.path.join(dir, "0_Input", exp.sett_filename), "w") as f:
        f.write(exp.get_settings_str())

    # Add the disturbance file with the solver settings and the horizon:
    with open(os.path.join(dir, "0_Input", exp.dist_filename), "w") as f:
        f.write(exp.get_solver_and_horizon_str())

    # Run the actual simulation.
    # exp.run_simulation(
    #     cwd=dir,
    #     sys=nordic,
    # )

    exp.run()

    # # Visualize the only voltage that was added to the observables.
    # import pyramses
    # data = pyramses.extractor(os.path.join(dir, "1_RAMSES output", exp.traj_filename))
    # data.getBus("1041").mag.plot()

    # Since no disturbances were applied, the plotted voltage should be a
    # horizontal line.
    looks_nice = input("Is the plotted voltage a straight line? (y/n) ")

    # Erase directory before raising AssertionError
    shutil.rmtree(dir)

    assert looks_nice == "y", "Simulation was not successful."

def test_run():
    """
    Test execution of experiment.
    """


def test_build_visualizations():
    """
    Test building of visualizations.
    """

    # Not implemented yet. Nothing to test.


def test_analyze_experiment():
    """
    Test analysis of experiment.
    """

    # Not implemented yet. Nothing to test.


def test_document_experiment():
    """
    Test documentation of experiment.
    """

    # Not implemented yet. Nothing to test.


if __name__ == "__main__":
    test_get_timestamp()
    test_Experiment_initialization()
    test_describe()
    test_add_system()
    test_add_disturbances()
    test_add_observables()
    test_add_controllers()
    test_add_randomization()  # Not implemented yet
    test_add_metrics()  # Not implemented yet
    test_add_visualization()  # Not implemented yet
    test_set_RAMSES_settings()
    test_set_solver_and_horizon()
    test_get_dir()
    test_get_settings_str()
    test_get_solver_and_horizon_str()
    test_get_observables_str()
    test_case2str()
    test_init_files_and_dirs()
    test_str()
    test_run_simulation()
    test_run()
    test_build_visualizations()  # Not implemented yet
    test_analyze_experiment()  # Not implemented yet
    test_document_experiment()  # Not implemented yet

    print("Module 'experiment' passed all tests!")
