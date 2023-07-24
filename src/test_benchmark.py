"""
Test functions for the "benchmark" module.
"""

# Module to be tested
from benchmark import *

# Modules from this repository
import pf_dynamic
import records
import control
import test_control

# Modules from the standard library
import os

# Other modules
import numpy as np

# The following is not very elegant: we redefine some functions from
# ../test_systems.py. We do this because managing the imports is tricky
# otherwise.


def check_nordic_power_flow(nordic: pf_dynamic.System) -> None:
    nordic.run_pf()

    # We first check the slack power.
    S_slack_MVA = nordic.get_S_slack_MVA()
    assert np.isclose(
        S_slack_MVA, 2137.35 + 1j * 377.4, atol=1e-1
    ), "The slack power of the Nordic is wrong."

    # We now check the transmission voltages.
    bus_name_to_voltage: dict[str, complex] = {}
    filename = os.path.join(
        "networks", "Nordic", "Nordic test system", "lf_A.dat"
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


def get_dynamic_Nordic() -> pf_dynamic.System:
    """
    Return the Nordic test system, ready to be run in an experiment.

    The power flows have already been solved and the loads have been made
    voltage-sensitive. By default, we use alpha=1.0 (constant current) and
    beta=2.0 (constant impedance).
    """

    # We first import the power-flow data.
    nordic = pf_dynamic.System.import_ARTERE(
        filename=os.path.join(
            "networks", "Nordic", "Nordic test system", "lf_A.dat"
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
        if isinstance(inj, records.Load):
            inj.make_voltage_sensitive(alpha=1.0, beta=2.0)

    # Making all the loads voltage sensitive should not change the power flows,
    # so we check them again just to make sure.
    check_nordic_power_flow(nordic=nordic)

    # We then import the dynamic data
    nordic.import_dynamic_data(
        filename=os.path.join(
            "networks", "Nordic", "Nordic test system", "dyn_A.dat"
        )
    )

    # Finally, we return the system.
    return nordic


def get_Nordic_with_DERAs(penetration: float) -> pf_dynamic.System:
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
            if isinstance(inj, records.Load) and inj.bus is bus
        )
        # Get its active power
        P_load_MW = load.get_P()
        # Scale the load
        load.scale_P_by(factor=load_factor)
        # Add a DERA with the right power
        DERA = records.DERA(
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


def get_system(disaggregate: bool = False) -> pf_dynamic.System():
    if disaggregate:
        raise NotImplementedError("Disaggregation not implemented yet.")

    else:
        return get_Nordic_with_DERAs(penetration=0.1)

class NewRAMSESImitator(test_control.RAMSESImitator):

    def __init__(self, system: pf_dynamic.System) -> None:
        self.system = system

    def getSimTime(self) -> float:

        return np.pi

def get_controller_and_system() -> tuple[PabonController, pf_dynamic.System]:
    nordic = get_system(disaggregate=False)

    PC = PabonController(
        transformer=nordic.get_branches_between(
            bus_name_1="1045", bus_name_2="5"
        )[0],
        period=1,
        epsilon_pu=0.02,
        delta_pu=0.01,
        increase_rate_pu_per_s=0.01,
    )

    nordic.add_controllers(
        controllers=[PC],
    )

    nordic.ram = NewRAMSESImitator(system=nordic)

    return PC, nordic


def test_PabonController_init():
    """
    Test the initialization of the PabonController class.

    This function also tests the get_controlled_DERAs method.
    """

    PC, nordic = get_controller_and_system()

    # Make assertions about the attributes
    assert PC.sys is nordic, "The system is not being stored correctly."
    assert nordic.controllers == [
        PC
    ], "The controller is not being stored correctly."

    assert np.isclose(
        PC.t_last_action, 0
    ), "The last action time is not being initialized correctly."
    assert np.isclose(
        PC.period, 1
    ), "The period is not being stored correctly."

    assert (
        PC.transformer
        is nordic.get_branches_between(bus_name_1="1045", bus_name_2="5")[0]
    ), "The transformer is not being stored correctly."
    assert (
        PC.epsilon_pu == 0.02
    ), "The tolerance for the distribution voltage is not being stored correctly."
    assert (
        PC.delta_pu == 0.01
    ), "The tolerance for the transmission voltage is not being stored correctly."
    assert (
        PC.increase_rate_pu_per_s == 0.01
    ), "The increase rate is not being stored correctly."

    assert PC.state == "IDLE", "The state is not being initialized correctly."
    assert np.isnan(
        PC.Vt_min
    ), "The minimum transmission voltage is not being initialized correctly."
    assert np.isnan(
        PC.Vd_max
    ), "The maximum distribution voltage is not being initialized correctly."

    # Test the update_controlled_DERAs method
    PC.update_controlled_DERAs()

    assert (
        len(PC.controlled_DERAs) == 1
    ), "The number of controlled DERAs is wrong."
    assert (
        PC.controlled_DERAs[0].bus.name == "5"
    ), "The controlled DERA is wrong."

class NLIImitator(control.Detector):

    def __init__(self) -> None:
        self.type = "NLI"

    def get_reading(self) -> float:
        return - 1.0

def test_get_measured_voltages():
    """
    Test the measurement of voltages (using a RAMSES imitator).
    """

    PC, nordic = get_controller_and_system()

    Vt, Vd = PC.get_measured_voltages()

    assert np.isclose(
        Vt, nordic.get_bus(name="1045").V_pu
    ), "The transmission voltage is not being measured correctly."
    assert np.isclose(
        Vd, nordic.get_bus(name="5").V_pu
    ), "The distribution voltage is not being measured correctly."

def test_update_state():
    """
    Test the update of the state between "IDLE" and "ACTIVE".
    """

    PC, nordic = get_controller_and_system()

    # If there is no NLI detector with a negative reading, the controller should be
    # in the "IDLE" state.

    PC.update_state()
    assert PC.state == "IDLE", "The state is not being updated correctly."

    # As soon as we add a detector with a negative reading and update the state,
    # the controller should be in the "ACTIVE" state.
    nordic.add_detector(detector=NLIImitator())
    PC.update_state()
    assert PC.state == "ACTIVE", "The state is not being updated correctly."

    # The following should have updated the minimum transmission voltage and the
    # maximum distribution voltage.
    assert np.isclose(
        PC.Vt_min, nordic.get_bus(name="1045").V_pu
    ), "The minimum transmission voltage is not being updated correctly."
    assert np.isclose(
        PC.Vd_max, nordic.get_bus(name="5").V_pu
    ), "The maximum distribution voltage is not being updated correctly."

    # If we remove the detector, the controller should remain in the "ACTIVE"
    # state.
    nordic.detectors = []
    PC.update_state()
    assert PC.state == "ACTIVE", "The state is not being updated correctly."


def test_freeze_r():
    """
    Test creation of a disturbance to freeze the transformer's tap ratio.
    """

    PC, nordic = get_controller_and_system()

    assert PC.freeze_r() == [], "The freeze_r method is not working correctly."

    transformer = nordic.get_branches_between(bus_name_1="1045", bus_name_2="5")[0]
    OLTC_controller = transformer.OLTC.OLTC_controller

    assert OLTC_controller.half_db_pu > 0.5, "The tap ratio is not being frozen correctly."
    assert np.isclose(
        OLTC_controller.v_setpoint_pu, 1.0
    ), "The tap ratio is not being frozen correctly."


def test_freeze_Q():
    """
    Test creation of a disturbance to freeze the DERAs' reactive power.
    """

    PC, nordic = get_controller_and_system()

    assert PC.freeze_Q() == [], "The freeze_Q method is not working correctly."


def test_increase_r():
    """
    Test creation of a disturbance to increase the transformer's tap ratio.
    """

    PC, nordic = get_controller_and_system()

    assert PC.increase_r() == [], "The increase_r method is not working correctly."

    transformer = nordic.get_branches_between(bus_name_1="1045", bus_name_2="5")[0]
    OLTC_controller = transformer.OLTC.OLTC_controller

    assert np.isclose(
        OLTC_controller.half_db_pu, 0
    ), "The tap ratio is not being increased correctly."

    assert np.isclose(
        OLTC_controller.v_setpoint_pu, 0
    ), "The tap ratio is not being increased correctly."


def test_decrease_r():
    """
    Test creation of a disturbance to decrease the transformer's tap ratio.
    """

    PC, nordic = get_controller_and_system()

    assert PC.decrease_r() == [], "The decrease_r method is not working correctly."

    transformer = nordic.get_branches_between(bus_name_1="1045", bus_name_2="5")[0]
    OLTC_controller = transformer.OLTC.OLTC_controller

    assert np.isclose(
        OLTC_controller.half_db_pu, 0
    ), "The tap ratio is not being decreased correctly."
    assert np.isclose(
        OLTC_controller.v_setpoint_pu, 2
    ), "The tap ratio is not being decreased correctly."


def test_increase_Q():
    """
    Test creation of a disturbance to increase the DERAs' reactive power.
    """

    PC, nordic = get_controller_and_system()
    PC.update_controlled_DERAs()

    print([str(dist) for dist in PC.increase_Q()])



def test_get_region_number():
    """
    Test the retrieval of the region in the (Vt, Vd) plane where the system is.
    """

    PC, nordic = get_controller_and_system()

    # Include a detector with a negative reading.
    nordic.add_detector(detector=NLIImitator())

    # We store the initial voltages.
    Vt0, Vd0 = PC.get_measured_voltages()

    PC.update_state()
    # Assert that Vt_min and Vd_max are being updated correctly, as well as the
    # state.
    assert np.isclose(
        PC.Vt_min, nordic.get_bus(name="1045").V_pu
    ), "The minimum transmission voltage is not being updated correctly."
    assert np.isclose(
        PC.Vd_max, nordic.get_bus(name="5").V_pu
    ), "The maximum distribution voltage is not being updated correctly."
    assert PC.state == "ACTIVE", "The state is not being updated correctly."

    # Because of the <= operators used throughout the get_region_number() method,
    # the system should be in region 1.
    assert PC.get_region_number() == 1, "The region number is wrong."

    # Let us now move the voltages to region 1.
    nordic.get_bus(name="1045").V_pu -= 0.01
    nordic.get_bus(name="5").V_pu -= 0.01

    assert PC.get_region_number() == 1, "The region number is wrong."

    # Let us move them to region 2 (this leaves us 0.01 above Vdmax).
    nordic.get_bus(name="1045").V_pu += 0.00
    nordic.get_bus(name="5").V_pu += 0.02

    assert PC.get_region_number() == 2, "The region number is wrong."

    # It should stay there even if we move a lot to the right (this leaves
    # us delta/2 to the right of the boundary).
    nordic.get_bus(name="1045").V_pu += 0.01 + PC.delta_pu / 2

    assert PC.get_region_number() == 2, "The region number is wrong."

    # Now we move down into region 4 (this leaves us in the center
    # of the rectangle of region 4).
    nordic.get_bus(name="5").V_pu -= 0.01 + PC.epsilon_pu / 2

    assert PC.get_region_number() == 4, "The region number is wrong."

    # And finally we move slightly to the left into region 3.
    nordic.get_bus(name="1045").V_pu += PC.delta_pu/2 - 1e-9
    nordic.get_bus(name="5").V_pu -= + PC.epsilon_pu / 2 - 1e-9

    # We should still be in region 4.
    assert PC.get_region_number() == 4, "The region number is wrong."

    # Now we move slightly into region 3
    nordic.get_bus(name="1045").V_pu += 2e-9
    nordic.get_bus(name="5").V_pu -= 2e-9

    assert PC.get_region_number() == 3, "The region number is wrong."

    # Moving up a little bit should return us to region 4.
    nordic.get_bus(name="5").V_pu += 2e-9

    assert PC.get_region_number() == 4, "The region number is wrong."

    # We return a move a little bit to the left into region 3.
    nordic.get_bus(name="5").V_pu -= 2e-9
    nordic.get_bus(name="1045").V_pu -= 2e-9

    assert PC.get_region_number() == 4, "The region number is wrong."

    # Finally, we test that a wide range of possible voltages always land
    # on a region:
    for delta_Vt in np.linspace(-0.2, 0.2, 100):
        nordic.get_bus(name="1045").V_pu = Vt0 + delta_Vt
        for delta_Vd in np.linspace(-0.2, 0.2, 100):
            nordic.get_bus(name="5").V_pu = Vd0 + delta_Vd
            # We now check that the region number is always between 1 and 4.
            assert (
                PC.get_region_number() in [1, 2, 3, 4]
            )

def test_get_actions():
    """
    Test the retrieval of the actions to be taken by a PabonController.
    """


    # Testing this is difficult, and hence it will be done externally in RAMSES,
    # not in this file.


if __name__ == "__main__":
    test_PabonController_init()
    test_get_measured_voltages()
    test_update_state()
    test_freeze_r()
    test_freeze_Q()
    test_increase_r()
    test_decrease_r()
    test_increase_Q()
    test_get_region_number()
    test_get_actions()

    print("Module 'benchmark' passed all tests!")
