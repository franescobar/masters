"""
This file specifies the experiments from the master's thesis

    "Control of distributed energy resources and load tap changers to enhance
    the long-term voltage stability of transmission networks."

While the experiments should be reproducible by looking at this file and
systems.py, these two programs rely on an extensive library of functions and
classes located elsewhere in this repository. These functions and classes handle
all aspects of the simulations, from initializing the power flow cases to
processing the results.

Francisco Escobar (fescobar@ieee.org)
August 2023, San José
"""

# \section{Preliminaries}

import sys
sys.path.append("src")

from benchmark import PabonController
import control
from experiment import Experiment
import metrics
import nli
import records
import pf_dynamic
from sim_interaction import Disturbance, Observable
import systems
from utils import Timer
import visual

# From the standard library:
from typing import Union

# From third-party packages:
import numpy as np

# \section{Base experiment}

def get_base_experiment(
    name: str,
    penetration: float = 0.2,
    frequency: float = 0.15,
    headroom: float = 0.2,
    disaggregate: bool = False,
) -> tuple[Experiment, pf_dynamic.System]:
    if disaggregate:
        system = systems.get_TD_system(
            penetration,
            frequency,
            headroom,
            N_monitored_buses=50, # should be fine
            N_monitored_DERs=50, # should be fine
        )
    else:
        system = systems.get_T_system(penetration, frequency, headroom)

    H = 20e-3
    DELTA_T = 7
    TAU_S = 1
    EPSILON = 1e-3
    system.add_detector(
        nli.NLI(("4041", ["4031"]), H, DELTA_T, TAU_S, EPSILON)
    )
    system.add_detector(
        nli.NLI(("4042", ["4021", "4032"]), H, DELTA_T, TAU_S, EPSILON)
    )

    for generator in system.generators:
        system.add_detector(nli.FieldCurrent(generator.name))

    exp = Experiment(
        name,
        DLL_dir=(
            r"C:\Users\FranciscoEscobarPrad\Desktop"
            r"\URAMSES-1.2\Release_intel_w64"
        ),
    )

    bus_4032 = system.get_bus("4032")
    line_4032_4044 = system.get_branches_between("4032", "4044")

    exp.add_disturbances(
        "4032-4044",
        Disturbance(1.0, bus_4032, "fault", 1e-4), # 1e-4 is the fault reactance in pu
        Disturbance(1.1, bus_4032, "clearance", None),
        Disturbance(1.1, line_4032_4044, "status", 0),
    )

    exp.add_visualizations(
        visual.NLIPlots(receiving_buses=("4041", "4042")),
        visual.CentralVoltages(),
        visual.FieldCurrents(),
        visual.DERAPowers(),
    )

    for bus in system.buses:
        if bus.base_kV > 100.0 or hasattr(bus, "is_monitored"):
            exp.add_observables(Observable(bus, "BV"))

    for generator in system.generators:
        exp.add_observables(Observable(generator, "if"))

    for record in system.records:
        if isinstance(record, records.DCTL):
            exp.add_observables(Observable(record, "all"))

    for injector in system.injectors:
        if isinstance(injector, records.DERA) and hasattr(
            injector, "is_monitored"
        ):
            exp.add_observables(
                Observable(injector, "Pgen"), Observable(injector, "Qgen")
            )

    for transformer in system.transformers:
        if transformer.touches("CENTRAL"):
            transformer.OLTC.OLTC_controller.is_overridable = True

    exp.add_system("Large" if disaggregate else "Small", system)

    exp.add_metrics(
        metrics.VoltageIntegral(only_central=True),
        metrics.NLI(),
        metrics.TapMovements(only_central=True),
        metrics.ControlEffort(power_type="P"),
        metrics.ControlEffort(power_type="Q"),
        metrics.PowerReserve(),
        metrics.ReactiveMargin(only_central=True),
        metrics.ActivatedOELs(only_central=False),
    )

    # Cite paper by Fabozzi noting that the step can
    # make discrete events to shift.
    MINUTES = 10
    exp.set_solver_and_horizon(
        solver_settings_dict={"max_h": 0.001, "min_h": 0.001},
        horizon=MINUTES * 60,
    )

    exp.set_RAMSES_settings(
        settings_dict={"SPARSE_SOLVER": "ma41", "NB_THREADS": 4}
    )

    return exp, system


if __name__ == "__main__":
    # Constants:
    PENETRATION = 0.20
    FREQUENCY = 0.15
    HEADROOM = 0.20

    # Experiments with small system:
    exp, T_system = get_base_experiment(
        "Small sys.", PENETRATION, FREQUENCY, HEADROOM, disaggregate=False
    )

    # \section{Benchmark controller}

    central_transformers = [
        t for t in T_system.transformers if t.touches("CENTRAL")
    ]

    local_controllers = [
        PabonController(
            transformer=t,
            period=1,
            epsilon_pu=0.02,
            delta_pu=0.01,
            increase_rate_pu_per_s=0.01,
        )
        for t in central_transformers
    ]

    exp.add_controllers("Pabon only", *local_controllers)

    # \section{Proposed \gls{MPC} scheme}

    # \subsection{\gls{MPC} controller}
    mpc = control.MPC_controller()
    # This will be undone later on.
    mpc.sys = T_system

    mpc.add_controlled_transformers([t.name for t in central_transformers])
    mpc.add_observed_corridor("4041", ["4031"])
    mpc.add_observed_corridor("4042", ["4021", "4032"])
    # Done as seen in the literature
    mpc.set_horizons(Np=3, Nc=3)
    mpc.set_period(10)

    # Has no effect
    mpc.define_setpoints()

    # In the following, notice that they are trajectories (curves) and not
    # single values.
    def get_power_bound(max_delta_pu: float) -> callable:
        def bound(bus: records.Bus, Nc: int, iter: int) -> tuple[np.ndarray]:
            return (
                -100 * np.ones([Nc, 1]),  # minimum power in pu
                100 * np.ones([Nc, 1]),  # maximum power in pu
                -max_delta_pu * np.ones([Nc, 1]),  # min. delta in pu
                max_delta_pu * np.ones([Nc, 1]),  # max. delta in pu
            )

        return bound

    def get_voltage_bound(V_min_pu: float, V_max_pu: float) -> callable:
        def bound(bus: records.Bus, Np: int, iter: int) -> tuple[np.ndarray]:
            return (
                V_min_pu * np.ones([Np, 1]),  # minimum voltage in pu
                V_max_pu * np.ones([Np, 1]),  # maximum voltage in pu
            )
        return bound

    # All bounds are being, indeed, set
    mpc.set_bounds(
        P_fun=get_power_bound(max_delta_pu=1e-6), # suppressed
        Q_fun=get_power_bound(max_delta_pu=1.0), # 100 Mvar per time step
        NLI_min=0.09,
        VD_fun=get_voltage_bound(V_min_pu=0.975, V_max_pu=1.025),
        VT_fun=get_voltage_bound(V_min_pu=0.900, V_max_pu=1.100),
    )

    def get_input_weight(value: float) -> callable:
        def weight(
            element: Union[records.Bus, records.Branch], Nc: int, iter: int
        ) -> np.ndarray:
            return value * np.ones([Nc, 1])

        return weight

    def get_slack_weight(value: float) -> callable:
        def weight(element, Np: int, iter: int) -> tuple[float, float]:
            return value, value

        return weight

    def no_input_weight(
        element: Union[records.Bus, records.Branch], Nc: int, iter: int
    ) -> np.ndarray:
        return np.zeros([Nc, 1])

    # All weights are, indeed, being set.
    mpc.set_weights(
        dr_fun=get_input_weight(1e-3), # the least expensive
        dP_fun=get_input_weight(0.0), # suppressed anyway
        dQ_fun=get_input_weight(5e-3), # more expensive than tap changes
        r_devs_fun=no_input_weight,
        P_devs_fun=no_input_weight,
        Q_devs_fun=no_input_weight,
        slacks_fun=get_slack_weight(1.0),  # voltage slacks
        NLI_slacks_fun=get_slack_weight(10.0),  # NLI slacks (most expensive)
    )

    # This reset is important
    mpc.sys = None

    # \subsection{\gls{CO} instances}

    def is_load(injector: records.Injector) -> bool:
        return isinstance(injector, records.Load)

    coordinators = []
    for transformer in central_transformers:
        secondary = T_system.get_transformer(transformer.name).get_LV_bus()
        load = next(filter(is_load, T_system.bus_to_injectors[secondary]))

        # Estimate parameters
        np.random.seed(3)
        estimated_penetration = systems.randomize_normally(PENETRATION)
        estimated_headroom = systems.randomize_normally(HEADROOM)

        estimated_DER_P_MW = load.P0_MW * estimated_penetration
        estimated_Snom_MVA = estimated_DER_P_MW / (1 - estimated_headroom)

        CO = control.Coordinator(
            transformer_name=transformer.name,
            min_P_injection_MVA=-estimated_Snom_MVA - estimated_DER_P_MW,
            max_P_injection_MVA=estimated_Snom_MVA - estimated_DER_P_MW,
            min_Q_injection_MVA=-estimated_Snom_MVA,
            max_Q_injection_MVA=estimated_Snom_MVA,
        )

        coordinators.append(CO)

    # \subsection{Local \gls{DER} controller}

    DER_controller = control.DERA_Controller()

    exp.add_controllers("MPC only", mpc, *coordinators, DER_controller)

    with Timer("Experiment on T-only system"):
        exp.run(remove_trj=False)

    # \section{Experiment on the \gls{T-D} system}
    # Experiments with large system:
    # exp, system = get_base_experiment("Big sys.", PENETRATION, FREQUENCY, HEADROOM, disaggregate=True)
    # with Timer("Experiment on T-D system"):
    #     exp.run(remove_trj=True)
