"""
A test of the MPC algorithm.

This test is performed at this level as it involves too many modules and is
close to a finished product.
"""

import sys
sys.path.append("src")

# Modules from this repository
import test_systems
import experiment
import sim_interaction
import metrics
import nli
import visual
import records
import utils


# Modules from the standard library


# Other modules

# The following functions define experiments ready to be run.

def simulate_DERAs(penetration: float = 0.1,
                   disaggregate: bool = False) -> experiment.Experiment:
    """
    Simulate the DERAs with and without the line tripping.
    """

    # Import the test system with a certain DERA penetration.
    if not disaggregate:
        nordic = test_systems.get_Nordic_with_DERAs(penetration=penetration)
    else:
        nordic = test_systems.get_disaggregated_nordic(penetration=penetration,
                                                       coverage_percentage=0.2,
                                                       filename="disaggregated_nordic.txt")

    # Add the detectors to the system (NLIs and field currents).
    nordic.add_detector(
        detector=nli.NLI(
            observed_corridor=("4041", ["4031"]),
            h=20e-3,
            delta_T=7,
            tau_s=1,
            epsilon=1e-3,
        )
    )

    nordic.add_detector(
        detector=nli.NLI(
            observed_corridor=("4042", ["4021", "4032"]),
            h=20e-3,
            delta_T=7,
            tau_s=1,
            epsilon=1e-3,
        )
    )

    for generator in nordic.generators:
        nordic.add_detector(
            detector=nli.FieldCurrent(machine_name=generator.name)
        )


    # Initialize the experiment.
    exp = experiment.Experiment(
        name="Big DERAs",
        DLL_dir=r"C:\Users\FranciscoEscobarPrad\Desktop\URAMSES-1.2\Release_intel_w64"
    )

    # Add to the experiment the desired disturbances (the case without
    # disturbance will be run by default).
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

    # # Add to the experiment the desired visualizations.
    # exp.add_visualizations(
    #     visual.NLI_plots(
    #         receiving_buses=["4041", "4042"]
    #     ),
    #     visual.CentralVoltages()
    # )

    # Add to the experiment all transmission quantities as observables
    # (implementing a nicer interface would be expensive).
    for bus in nordic.buses:
        exp.add_observables(
            sim_interaction.Observable(
                observed_object=bus,
                obs_name="BV"
            )
        )

    for generator in nordic.generators:
        exp.add_observables(
            sim_interaction.Observable(
                observed_object=generator,
                obs_name="if"
            )
        )

    for record in nordic.records:
        if isinstance(record, records.DCTL):
            exp.add_observables(
                sim_interaction.Observable(
                    observed_object=record,
                    obs_name=None
                )
            )

    for injector in nordic.injectors:
        if isinstance(injector, records.DERA):
            exp.add_observables(
                sim_interaction.Observable(
                    observed_object=injector,
                    obs_name="Pgen"
                ),
                sim_interaction.Observable(
                    observed_object=injector,
                    obs_name="Qgen"
                )
            )

    # Add to the experiment the system.
    exp.add_system(description="Big DERAs", system=nordic)

    # Add to the experiment the metrics.
    exp.add_metrics(
        metrics.VoltageIntegral(only_central=True),
        metrics.ReactiveMargin(only_central=True),
        metrics.ControlEffort(power_type="P"),
        metrics.ControlEffort(power_type="Q"),
        metrics.TapMovements(only_central=True),
        metrics.NLI(),
        metrics.PowerReserve(),
        metrics.ActivatedOELs(only_central=False),
    )

    # Set the solver and the horizon
    exp.set_solver_and_horizon(
        solver_settings_dict={
            # Enlarging the maximum step does not seem to make a difference
            "max_h": 0.02,
            "min_h": 0.005
        },
        horizon=200,
    )

    exp.set_RAMSES_settings(
        settings_dict={
            "SPARSE_SOLVER": "ma41",
            # "SKIP_CONV": "T",
            "NB_THREADS": 4,
            # "FULL_UPDATE": "F",
        }
    )

    return exp, nordic

if __name__ == "__main__":

    #############################################
    # Experiment: Simulation of DERAs without MPC
    #############################################

    p = 0.2
    exp, nordic = simulate_DERAs(penetration=p, disaggregate=True)
    print(f"Running experiment with DERA penetration = {p*100:.0f} %...")
    with utils.Timer(name="Full experiment"):
        exp.run()

