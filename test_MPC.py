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
import benchmark
import control


# Modules from the standard library


# Other modules
import numpy as np

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

    # Add to the experiment the desired visualizations.
    exp.add_visualizations(
        visual.NLIPlots(
            receiving_buses=["4041", "4042"]
        ),
        visual.CentralVoltages(),
        visual.FieldCurrents(),
        visual.DERAPowers(),
        visual.VoltageTrajectories()
    )

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
    # nordic.generate_twin()

    # Make the OLTCs overridable
    for transformer in nordic.transformers:
        if transformer.touches(location="CENTRAL"):
            transformer.OLTC.OLTC_controller.is_overridable = True

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
            # "max_h": 0.001, "min_h": 0.001
            "max_h": 0.001, "min_h": 0.001
        },
        # This should be enough to observe the control
        horizon=10*60,
        # horizon=9
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

    # p = 0.2
    p = 0.2
    exp, nordic = simulate_DERAs(penetration=p, disaggregate=False)
    # print(nordic)

    # Add Pabons controller to the experiment
    PC_controllers = [
        benchmark.PabonController(
            transformer=transformer,
            period=1,
            epsilon_pu=0.02,
            delta_pu=0.01,
            increase_rate_pu_per_s=0.01,
        )
        for transformer in nordic.transformers
        if transformer.touches(location="CENTRAL")
    ]

    mpc = control.MPC_controller()
    mpc.sys = nordic

    central_transformers = [
        transformer.name 
        for transformer in nordic.transformers
        if transformer.touches(location="CENTRAL")
    ]

    # Set controlled transformers
    mpc.add_controlled_transformers(transformers=central_transformers)

    # Add observed corridors
    mpc.add_observed_corridor(
        boundary_bus="4041",
        sending_buses=["4031"],
    )
    mpc.add_observed_corridor(
        boundary_bus="4042",
        sending_buses=["4021", "4032"],
    )

    # Set horizons
    mpc.set_horizons(Np=3, Nc=3)

    # Set period
    mpc.set_period(10)

    # Set default setpoints
    mpc.define_setpoints()

    def no_power(bus: records.Bus, Nc: int, iter: int) -> tuple[float]:
        p_min_pu = -1e-3 * np.ones([Nc, 1])
        p_max_pu = - 1.0 * p_min_pu
        dp_min_pu = 0.5 * p_min_pu
        dp_max_pu = 0.5 * p_max_pu

        return p_min_pu, p_max_pu, dp_min_pu, dp_max_pu
    
    bounds = {}
    
    coordinators = []
    for transformer_name in central_transformers:
        # Get load at secondary bus
        LV_bus = nordic.get_transformer(name=transformer_name).get_LV_bus()
        load = next(inj
                    for inj in nordic.injectors
                    if isinstance(inj, records.Load) and inj.bus is LV_bus)
        # Get load powers
        true_P_MW = load.P0_MW * p
        true_Q_Mvar = 0
        # Get load estimation
        perc_error = 0.2
        estimated_P_MW = true_P_MW * (1 + np.random.uniform(-0.1, 0.1))
        estimated_Q_Mvar = true_Q_Mvar * (1 + np.random.uniform(-0.1, 0.1))
        # Room for injections
        room_P_MW = estimated_P_MW / 4
        # Generate coordinator
        coordinator = control.Coordinator(
            transformer_name=transformer_name,
            min_P_injection_MVA=-9 * room_P_MW,
            max_P_injection_MVA= room_P_MW,
            min_Q_injection_MVA=-5 * room_P_MW,
            max_Q_injection_MVA= 5 * room_P_MW,
        )
        coordinators.append(coordinator)
        print(coordinator)

        bounds[LV_bus.name] = (-9*room_P_MW/nordic.base_MVA,
                               room_P_MW/nordic.base_MVA, 
                               -5 * room_P_MW/nordic.base_MVA, 
                               5 * room_P_MW/nordic.base_MVA)

    max_dp_pu = 1e-6
    max_dq_pu = 1.0

    def p_bound(bus: records.Bus, Nc: int, iter: int) -> tuple[np.ndarray]:

        # p_min_pu = bounds[bus.name][0] * np.ones([Nc, 1])
        # p_max_pu = bounds[bus.name][1] * np.ones([Nc, 1])
        p_min_pu = - 100 * np.ones([Nc, 1])
        p_max_pu = - p_min_pu
        dp_min_pu = - max_dp_pu * np.ones([Nc, 1])
        dp_max_pu =   max_dp_pu * np.ones([Nc, 1])

        return p_min_pu, p_max_pu, dp_min_pu, dp_max_pu

    def q_bound(bus: records.Bus, Nc: int, iter: int) -> tuple[np.ndarray]:

        p_min_pu = -100 * np.ones([Nc, 1]) # 20 * bounds[bus.name][2] * np.ones([Nc, 1])
        p_max_pu =  100 * np.ones([Nc, 1]) # 20 * bounds[bus.name][3] * np.ones([Nc, 1])
        dp_min_pu = - max_dq_pu * np.ones([Nc, 1])
        dp_max_pu =   max_dq_pu * np.ones([Nc, 1])

        return p_min_pu, p_max_pu, dp_min_pu, dp_max_pu

    # Set default bounds
    mpc.set_bounds(
        NLI_min=0.09,
        # NLI_fun=control.MPC_controller.NLI_bound,
        P_fun = p_bound,
        Q_fun = q_bound,
        # P_fun=no_power,
        # Q_fun=no_power 
    )

    # print(mpc.NLI_lower)
    # print(mpc.NLI_upper)

    # exit()

    # Set default weights
    def slacks_fun(bus: records.Bus, Np: int, iter: int) -> tuple[float, float]:

        return 1e4, 1e4

    # def P_devs_fun(bus: records.Bus, Nc: int, iter: int) -> tuple[float, float]:

    #     return 1e8 * np.ones([Nc, 1])
    no_pen = lambda bus, Nc, iter: \
        0.0 * control.MPC_controller.some_u_penalization(bus, Nc, iter)

    f_deltas = lambda bus, Nc, iter: \
        5e-3 * control.MPC_controller.some_u_penalization(bus, Nc, iter)
    f_deltas_heavier = lambda bus, Nc, iter: \
        1e-1 * control.MPC_controller.some_u_penalization(bus, Nc, iter)
    f_dr = lambda transformer, Nc, iter: \
        1e-3 * control.MPC_controller.some_u_penalization(transformer, Nc, iter)
    f_dP = f_deltas_heavier
    f_dQ = f_deltas
    f_dev_P = f_dev_Q = no_pen

    mpc.set_weights(
        # slacks_fun=slacks_fun,
        dr_fun=f_dr, 
        dP_fun=f_dP,
        dQ_fun=f_dQ,
        P_devs_fun=f_dev_P,
        Q_devs_fun=f_dev_Q,
        # P_devs_fun=P_devs_fun,
        # Q_devs_fun=P_devs_fun,
    )

    # import sys
    # np.set_printoptions(threshold=sys.maxsize)
    print(mpc)
    # input("Check the MPC...")
    mpc.sys = None
    # exit()


   

    # exit()

    DERAC = control.DERA_Controller()

    exp.add_controllers("Pabon only", *PC_controllers)
    exp.add_controllers("MPC only", mpc, *coordinators, DERAC)

    print(f"Running experiment with DERA penetration = {p*100:.0f} %...")
    with utils.Timer(name="Full experiment"):
        exp.run(remove_trj=False)

