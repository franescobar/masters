"""
Test systems used throughout this project.

The following functions are defined:

    - get_dynamic_Nordic() -> pf.System:

"""

# Modules from this repository
import pf_dynamic as pf
import records as rec

# Modules from the standard library
import os

# Other modules
import numpy as np

def check_nordic_power_flow(nordic: pf.System) -> None:

    nordic.run_pf()

    # We first check the slack power.
    S_slack_MVA = nordic.get_S_slack_MVA()
    assert np.isclose(
        S_slack_MVA, 2137.4 + 1j * 377.4, atol=1e-1
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
        use_injectors=True
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
    load_buses = [transformer.get_LV_bus()
                  for transformer in nordic.transformers
                  if transformer.touches(location="CENTRAL")]

    # For each load bus, we scale the load and add a DERA.
    for bus in load_buses:
        # Find the load connected to this bus
        load = next(inj
                    for inj in nordic.injectors
                    if isinstance(inj, rec.Load)
                    and inj.bus is bus)
        # Get its active power
        P_load_MW = load.get_P()
        # Scale the load
        load.scale_P_by(factor=load_factor)
        # Add a DERA with the right power
        DERA = rec.DERA(name=f"DERA_{bus.name}",
                        bus=bus,
                        P0_MW=abs(P_load_MW) * DERA_factor,
                        Q0_Mvar=0.0,
                        Snom_MVA=1.2*abs(P_load_MW) * DERA_factor)
        nordic.store_injector(inj=DERA)

    # Having made these changes should not affect the power flows, so we check
    # them again.
    check_nordic_power_flow(nordic=nordic)

    # Finally, we return the system.
    return nordic




# UKGDS

# T-D system (combines the previous two)