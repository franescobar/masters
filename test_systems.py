"""
Test systems used throughout this project.

The following functions are defined:

    - get_dynamic_Nordic() -> pf.System:

"""

import sys

sys.path.append("src")

# Modules from this repository
import pf_dynamic as pf
import records as rec
import utils

# Modules from the standard library
import os
import copy
import random

# Other modules
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


def check_nordic_power_flow(nordic: pf.System) -> None:
    nordic.run_pf()

    # We first check the slack power.
    S_slack_MVA = nordic.get_S_slack_MVA()
    assert np.isclose(
        S_slack_MVA, 2137.35 + 1j * 377.4, atol=1e-1
    ), "The slack power of the Nordic is wrong."

    # We now check the transmission voltages.
    bus_name_to_voltage: dict[str, complex] = {}
    filename = os.path.join(
        "src", "networks", "Nordic", "Nordic test system", "lf_A.dat"
    )
    with open(filename) as f:
        for line in f:
            words = line.split()
            if len(words) == 0 or words[0].strip() == "":
                continue
            if words[0] == "LFRESV":
                bus_name = words[1]
                mag = float(words[2])
                rad = float(words[3])
                voltage = mag * np.exp(1j * rad)
                bus_name_to_voltage[bus_name] = voltage

    for bus in nordic.buses:
        if bus.name in bus_name_to_voltage:
            assert np.isclose(
                bus.get_phasor_V(),
                bus_name_to_voltage[bus.name],
                atol=1e-4,
            ), "The voltages of the Nordic are wrong."


def get_dynamic_Nordic() -> pf.System:
    """
    Return the Nordic test system, ready to be run in an experiment.

    The power flows have already been solved and the loads have been made
    voltage-sensitive. By default, we use alpha=1.0 (constant current) and
    beta=2.0 (constant impedance).
    """

    # We first import the power-flow data.
    nordic = pf.System.import_ARTERE(
        filename=os.path.join(
            "src", "networks", "Nordic", "Nordic test system", "lf_A.dat"
        ),
        system_name="Nordic Test System - Case A - Sensitive loads",
        use_injectors=True,
    )

    # To ensure homogeneity in the experiments, we compute the power flows.
    nordic.run_pf()

    # Just in case: we verify the power flows by comparing the slack power and
    # the bus voltages with those reported by the Task Force that documented the
    # system.
    check_nordic_power_flow(nordic=nordic)

    # We also make all the loads voltage-sensitive.
    for inj in nordic.injectors:
        if isinstance(inj, rec.Load):
            inj.make_voltage_sensitive(alpha=1.0, beta=2.0)

    # Making all the loads voltage sensitive should not change the power flows,
    # so we check them again just to make sure.
    check_nordic_power_flow(nordic=nordic)

    # We then import the dynamic data
    nordic.import_dynamic_data(
        filename=os.path.join(
            "src", "networks", "Nordic", "Nordic test system", "dyn_A.dat"
        )
    )

    # We now create the twin before adding any DERAs
    nordic.generate_twin()

    # Finally, we return the system.
    return nordic


def get_Nordic_with_DERAs(penetration: float) -> pf.System:
    """
    Return the Nordic populated with DERAs but at the same operating point.

    The parameter 'penetration' is defined as P_DER / P_load, where P_DER is the
    total power of the DERAs and P_load is the total load power. This means that
    0 <= penetration <= 1.

    Take into account that penetration=0.5 already causes the original load to
    double. This may cause the simulation to fail or the system to become
    unstable quicker.
    """

    # First, we load the system as before.
    nordic = get_dynamic_Nordic()

    # We then populate the system with DERAs.
    load_factor = 1.0 / (1.0 - penetration)
    DERA_factor = penetration / (1.0 - penetration)

    # Locate the load buses from the central region.
    load_buses = [
        transformer.get_LV_bus()
        for transformer in nordic.transformers
        if transformer.touches(location="CENTRAL")
    ]

    # For each load bus, we scale the load and add a DERA.
    for bus in load_buses:
        # Find the load connected to this bus
        load = next(
            inj
            for inj in nordic.injectors
            if isinstance(inj, rec.Load) and inj.bus is bus
        )
        # Get its active power
        P_load_MW = load.get_P()
        # Scale the load
        load.scale_P_by(factor=load_factor)
        # Add a DERA with the right power
        DERA = rec.DERA(
            name=f"DERA_{bus.name}",
            bus=bus,
            P0_MW=abs(P_load_MW) * DERA_factor,
            Q0_Mvar=0.0,
            Snom_MVA=1.2 * abs(P_load_MW) * DERA_factor,
        )
        nordic.store_injector(inj=DERA)

    # Having made these changes should not affect the power flows, so we check
    # them again.
    check_nordic_power_flow(nordic=nordic)

    # Finally, we return the system.
    return nordic


def get_UKGDS(
    V_pu: float,
    theta_radians: float,
    factor_deviation: float,
    P_max_MW: float,
    P_average_MW: float,
    V_min_pu: float = 0.95,
    V_max_pu: float = 1.05,
) -> pf.System:
    """
    Return the UKGDS test system.
    """

    # Load the static system
    UKGDS = pf.System.import_ARTERE(
        filename=os.path.join("src", "networks", "UKGDS", "DN.dat"),
        system_name="UKGDS",
        base_MVA=100.0,
        use_injectors=True,
    )

    # Remove both transformers
    step_down_1, step_down_2 = UKGDS.transformers
    UKGDS.remove_branch(branch=step_down_1)
    UKGDS.remove_branch(branch=step_down_2)

    # Remove old slack bus
    UKGDS.remove_bus(bus=UKGDS.slack)

    # Redefine a slack bus
    slack_bus = rec.Slack(
        V_pu=V_pu,
        theta_radians=theta_radians,
        PL_pu=0.0,
        QL_pu=0.0,
        G_pu=0.0,
        B_pu=0.0,
        base_kV=step_down_1.get_LV_bus().base_kV,
        bus_type="Slack",
        V_min_pu=0.95,
        V_max_pu=1.05,
        name=step_down_1.get_LV_bus().name,
    )

    # Replace the old secondary by the new slack
    UKGDS.replace_bus(
        old_bus=step_down_1.get_LV_bus(),
        new_bus=slack_bus,
    )

    # Reset loads
    loads = [inj for inj in UKGDS.injectors if isinstance(inj, rec.Load)]
    for load in loads:
        load.set_P_to(PL_MW=0.0)
        load.set_Q_to(QL_Mvar=0.0)

    # Change the base voltage and power of the system
    base_kV_new = 20.0
    base_MVA_new = 100.0
    for element in UKGDS.lines + UKGDS.buses:
        if isinstance(element, rec.Branch):
            base_kV_old = element.from_bus.base_kV
        elif isinstance(element, rec.Bus):
            base_kV_old = element.base_kV
        element.change_base(
            base_MVA_old=UKGDS.base_MVA,
            base_MVA_new=base_MVA_new,
            base_kV_old=base_kV_old,
            base_kV_new=base_kV_new,
        )
    UKGDS.base_MVA = base_MVA_new

    # Run initial power flow
    UKGDS.run_pf(tol=1e-12)

    # The following helper function is useful for the optimization.

    def run_pf_with_loads(x: np.ndarray) -> None:
        """
        Run the power flow with the loads given by x.
        """

        for load_value, load in zip(x, loads):
            load.set_P_to(PL_MW=load_value)
            load.set_Q_to(QL_Mvar=load_value / 10)

        UKGDS.run_pf(tol=1e-12)

    # We now want to fit more loads into that network using optimization. To do
    # so, we define the objective function.

    def cost(x: np.ndarray) -> float:
        """
        Penalize deviation from original allocation but maximize power.
        """

        # factor_deviation = factor_deviation
        factor_allocated_power = 1 - factor_deviation

        # We first run the power flow with the loads given by x.
        run_pf_with_loads(x=x)

        # We then compute the cost
        cost = 0.0
        for load in loads:
            cost += factor_deviation * (-load.get_P() - P_average_MW) ** 2
        cost -= factor_allocated_power * UKGDS.get_S_slack_MVA().real

        return cost

    # We define the constraints
    def voltages(x: np.ndarray) -> np.ndarray:
        """
        We compute the voltages in order to define the constrants.
        """

        # We first run the power flow with the loads given by x.
        run_pf_with_loads(x=x)

        # We then compute the cost
        return np.array([bus.V_pu for bus in UKGDS.buses])

    NLC = NonlinearConstraint(
        fun=voltages,
        lb=V_min_pu * np.ones(len(UKGDS.buses)),
        ub=V_max_pu * np.ones(len(UKGDS.buses)),
    )

    LC = LinearConstraint(
        A=np.eye(len(loads)),
        lb=np.zeros(len(loads)),
        ub=P_max_MW * np.ones(len(loads)),
    )

    # We define the initial guess
    x0 = np.array([-inj.get_P() for inj in loads])

    # We solve the optimization problem
    res = minimize(
        fun=cost,
        x0=x0,
        constraints=[NLC, LC],
        # A tolerance of +- 1 MW is acceptable and gives a good speed.
        tol=1.0,
    )

    # We now run the power flow with the optimal loads.
    run_pf_with_loads(x=res.x)

    return UKGDS


def get_UKGDS_pool(
    V_min_pu: float = 0.95, V_max_pu: float = 1.05
) -> list[pf.System]:
    """
    Return a pool of UKGDS systems.
    """

    UKGDS_small = get_UKGDS(
        V_pu=1.0,
        theta_radians=0.0,
        factor_deviation=0.0,
        P_max_MW=1.5,
        P_average_MW=1.0,
        V_min_pu=V_min_pu,
        V_max_pu=V_max_pu,
    )

    UKGDS_small.match_power(
        P_desired_MW=99.0,
        Q_desired_Mvar=9.9,  # This should not be 0.0, as otherwise Newton will fail
        V_desired_pu=1.00,
        theta_desired_radians=0.0,
        tol=1e-6,
        max_iters=10,
        use_OLTCs=True,
    )

    UKGDS_medium = get_UKGDS(
        V_pu=1.0,
        theta_radians=0.0,
        factor_deviation=0.0,
        P_max_MW=2.5,
        P_average_MW=1.0,
        V_min_pu=V_min_pu,
        V_max_pu=V_max_pu,
    )

    UKGDS_large = get_UKGDS(
        V_pu=1.0,
        theta_radians=0.0,
        factor_deviation=0.0,
        P_max_MW=3.0,
        P_average_MW=1.0,
        V_min_pu=V_min_pu,
        V_max_pu=V_max_pu,
    )

    UKGDS_very_large = get_UKGDS(
        V_pu=1.0,
        theta_radians=0.0,
        factor_deviation=0.0,
        P_max_MW=4.0,
        P_average_MW=1.0,
        V_min_pu=V_min_pu,
        V_max_pu=V_max_pu,
    )

    return (
        [copy.deepcopy(UKGDS_small) for _ in range(1)]
        + [copy.deepcopy(UKGDS_medium) for _ in range(1)]
        + [copy.deepcopy(UKGDS_large) for _ in range(1)]
        + [copy.deepcopy(UKGDS_very_large) for _ in range(1)]
    )


def get_disaggregated_nordic(
    penetration: float, coverage_percentage: float, filename: str = None
) -> pf.System:
    """
    Return the disaggregated Nordic populated with DERAs.

    penetration is defined as P_DER / P_load, where P_DER is the total power of
    the DERAs and P_load is the total load power. This means that
    0 <= penetration <= 1.

    coverage is defined as the percentage of the load buses that contain a DERA.
    """

    # First, we load the system as before.
    nordic = get_dynamic_Nordic()

    # We now create a pool of UKGDS that are useful for the disaggregation.
    pool = get_UKGDS_pool(V_min_pu=0.965)

    # Identify central buses
    central_buses = [
        transformer.get_LV_bus()
        for transformer in nordic.transformers
        if transformer.touches(location="CENTRAL")
    ]

    original_loads = {}
    # We now disaggregate the loads in the central area.
    for bus in central_buses:
        # Save the original load in a dictionary (useful for defining the DERAs)
        original_loads[bus.name] = nordic.get_sensitive_load_MW_Mvar(bus=bus)[
            0
        ]
        # Disaggregatek
        print(f"Disaggregating bus {bus.name}.")
        _, largest_mismatch = nordic.disaggregate_load(
            bus=bus,
            systems=pool,
        )
        # Assert that the largest mismatch is small enough
        assert largest_mismatch < 1e-6, "The largest mismatch is too large."
        # Check power flow
        check_nordic_power_flow(nordic=nordic)

    # We now add the DERAs to the system
    for bus in central_buses:
        # Get neighboring buses with loads
        neighbors = nordic.isolate_buses_by_kV(starting_bus=bus)
        neighbors_with_loads = [
            neighbor
            for neighbor in neighbors
            # We will count as load anything above 0.5 MW
            if nordic.get_sensitive_load_MW_Mvar(bus=neighbor)[0] > 0.5
        ]

        # Get a random sample respecting the coverage percentage
        random_neighbors_with_loads = random.sample(
            population=neighbors_with_loads,
            k=int(np.floor(coverage_percentage * len(neighbors_with_loads))),
        )

        # Get the power per DERA
        DERA_factor = penetration / (1.0 - penetration)
        P_DERA_MW = (
            original_loads[bus.name]
            * DERA_factor
            / len(random_neighbors_with_loads)
        )

        # Add a DERA to each neighboring bus with load
        total_P_DERA_MW = 0
        for neighbor in random_neighbors_with_loads:
            # Add the DERA
            DERA = rec.DERA(
                name=f"D{neighbor.name}",
                bus=neighbor,
                P0_MW=P_DERA_MW,
                Q0_Mvar=0.0,
                Snom_MVA=1.2 * P_DERA_MW,
            )
            nordic.store_injector(inj=DERA)
            # Increment the load
            load = next(
                inj
                for inj in nordic.bus_to_injectors[neighbor]
                if isinstance(inj, rec.Load)
            )
            load.increment_P_by(delta_P_MW=P_DERA_MW)
            total_P_DERA_MW += P_DERA_MW
        print(f"Original load at bus {bus.name} is {original_loads[bus.name]:.4f} MW.")
        print(f"Total DERA power at bus {bus.name} is {total_P_DERA_MW:.4f} MW.")

    raise NotImplementedError(
        "Make sure that the secondary voltages are marked"
        " as having location = CENTRAL"
        " otherwise the visualizations will fail"
        " then test with a small horizon"
        " Ideally, change the name of the secondary to detect those buses: keep 1, 2, 5, etc"
    )

    # Check one final time that the power flow is correct
    check_nordic_power_flow(nordic=nordic)

    # Write system to file
    if filename is not None:
        with open(filename, "w") as f:
            f.write(f"Original case remained unchanged.\n")
            f.write(nordic.generate_table())

    return nordic


if __name__ == "__main__":
    nordic = get_disaggregated_nordic(
        penetration=0.2,
        coverage_percentage=0.4,
        filename="disaggregated_nordic.txt",
    )
