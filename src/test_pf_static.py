"""
Test functions for the 'pf_static' module.
"""

# Module to be tested
from pf_static import *

# Modules from this repository
import utils

# Modules from the standard library
from collections.abc import Sequence
import random
import copy
import os

# Other modules
import numpy as np


def test_init():
    """
    Test initialization of a StaticSystem.
    """

    print("Testing initialization of StaticSystem...")

    sys = StaticSystem(name="Test system", pu=True, base_MVA=200)

    # Test attributes from the constructor
    assert sys.name == "Test system"
    assert hasattr(sys, "pu")
    assert hasattr(sys, "base_MVA")

    # Test status
    assert sys.status == "unsolved"

    # Test slack bus
    assert sys.slack is None

    # Test containers
    assert sys.PQ_buses == []
    assert sys.PV_buses == []
    assert sys.non_slack_buses == []
    assert sys.buses == []
    assert sys.lines == []
    assert sys.transformers == []
    assert sys.branches == []
    assert sys.generators == []
    assert sys.injectors == []
    assert sys.bus_dict == {}
    assert sys.line_dict == {}
    assert sys.transformer_dict == {}
    assert sys.gen_dict == {}
    assert sys.inj_dict == {}
    assert sys.bus_to_injectors == {}
    assert sys.bus_to_generators == {}


def test_conversions_pu_ohm():
    """
    Test conversions between pu, ohms, and mhos.
    """

    print("Testing conversions between pu, ohms, and mhos...")

    # Define example quantities
    Sb = 200
    Vb = 10
    Zb = Vb**2 / Sb
    Z_ohm = 1 + 1j

    # Initialize dummy system
    sys = StaticSystem(base_MVA=Sb)

    # Test pu2ohm and ohm2pu
    assert np.isclose(
        sys.ohm2pu(Z_ohm=Z_ohm, base_kV=Vb),
        Z_ohm / Zb,
    ), "Conversion from ohm to pu failed."

    assert np.isclose(
        sys.pu2ohm(Z_pu=Z_ohm / Zb, base_kV=Vb),
        Z_ohm,
    ), "Conversion from pu to ohm failed."

    # Test mho2pu and pu2mho
    assert np.isclose(
        sys.mho2pu(Y_mho=1 / Z_ohm, base_kV=Vb), 1 / (Z_ohm / Zb)
    ), "Conversion from mho to pu failed."

    assert np.isclose(
        sys.pu2mho(Y_pu=1 / (Z_ohm / Zb), base_kV=Vb), 1 / Z_ohm
    ), "Conversion from pu to mho failed."


def test_bus_additions():
    """
    Test addition of buses to the system.
    """

    print("Testing addition of buses to the system...")

    # Define dummy system
    sys = StaticSystem()

    # Add PV buses
    PV1 = sys.add_PV(V_pu=1.02, PL=1.0, name="PV1")
    PV2 = sys.add_PV(V_pu=1.03, PL=0.9, name="PV2")

    # Add slack bus
    slack = sys.add_slack(V_pu=1.02, name="slack")

    # Add PQ buses
    PQ1 = sys.add_PQ(PL=1.0, QL=0.5, name="PQ1")
    PQ2 = sys.add_PQ(PL=1.0, QL=0.5, name="PQ2")

    # Add more buses in random order
    PQ3 = sys.add_PQ(PL=1.0, QL=0.5, name="PQ3")
    PQ4 = sys.add_PQ(PL=1.0, QL=0.5, name="PQ4")
    PV3 = sys.add_PV(V_pu=1.02, PL=1.1, name="PV3")

    # Test construction of bus list
    assert sys.buses == [
        slack,
        PQ1,
        PQ2,
        PQ3,
        PQ4,
        PV1,
        PV2,
        PV3,
    ], "Bus list is not sorted in order slack, PQ, PV."

    # Test construction of bus dictionaries
    assert sys.bus_dict == {
        "slack": slack,
        "PQ1": PQ1,
        "PQ2": PQ2,
        "PQ3": PQ3,
        "PQ4": PQ4,
        "PV1": PV1,
        "PV2": PV2,
        "PV3": PV3,
    }, "Bus dictionary is wrong."

    # Test initialization of injector and generator dictionaries
    for bus in sys.buses:
        assert sys.bus_to_injectors[bus] == [], "Inj. dictionary is wrong."
        assert sys.bus_to_generators[bus] == [], "Gen. dictionary is wrong."


def test_branch_additions():
    """
    Test addition of branches to the system.
    """

    print("Testing addition of branches to the system...")

    # Define dummy system
    sys = StaticSystem()

    # Add example transformers
    b1 = sys.add_slack(V_pu=1.0, name="slack")
    b2 = sys.add_PQ(PL=1.0, QL=0.5, name="PQ1")

    # Add transformer
    T1 = sys.add_transformer(from_bus=b1, to_bus=b2, X=0.1, name="T1-2")

    # Add line
    L1 = sys.add_line(from_bus=b1, to_bus=b2, X=0.1, name="L1-2")
    L2 = sys.add_line(from_bus=b1, to_bus=b2, X=0.1, name="L1-2B")

    # Add another transformer
    T2 = sys.add_transformer(from_bus=b1, to_bus=b2, X=0.1, name="T1-2B")

    # Add another line
    L3 = sys.add_line(from_bus=b1, to_bus=b2, X=0.1, name="L1-2C")

    # Test construction of lists
    assert sys.lines == [L1, L2, L3], "Line not added correctly."
    assert sys.transformers == [T1, T2], "Transformer not added correctly."
    assert sys.branches == [
        L1,
        L2,
        L3,
        T2,
        T1,
    ], "Branch list not sorted correctly."

    # Test construction of dictionaries
    assert sys.line_dict == {
        "L1-2": L1,
        "L1-2B": L2,
        "L1-2C": L3,
    }, "Line dictionary is wrong."


def test_generator_additions():
    """
    Test addition of generators to the system.
    """

    print("Testing addition of generators to the system...")

    # Define dummy system
    sys = StaticSystem()

    # Add buses
    slack = sys.add_slack(V_pu=1.0, name="slack")
    PV1 = sys.add_PV(V_pu=1.0, PL=1.0, name="PV1")
    PQ1 = sys.add_PQ(PL=1.0, QL=0.5, name="PQ1")

    # Test that generators can only be added to slack and PV buses
    try:
        gen_slack = records.Generator(PG_MW=100, bus=slack, name="gen_slack")
        sys.store_generator(gen=gen_slack)
        assert False, "Generator added to slack bus."
    except AssertionError:
        pass

    try:
        gen_PV = records.Generator(PG_MW=100, bus=PV1, name="gen_PV")
        sys.store_generator(gen=gen_PV)
        assert False, "Generator added to PV bus."
    except AssertionError:
        pass

    try:
        gen = records.Generator(PG_MW=100, bus=PQ1, name="gen_PQ")
        sys.store_generator(gen=gen)
        assert False, "Generator added to PQ bus."
    except Exception:
        pass

    # Test that generators cannot be duplicated
    try:
        gen_slack_2 = copy.deepcopy(gen_slack)
        sys.store_generator(gen=gen_slack_2)
        assert False, "Generator added twice."
    except RuntimeError:
        pass

    # Test containers
    assert len(sys.generators) == 2, "Generator list is wrong."
    assert len(sys.gen_dict) == 2, "Generator dictionary is wrong."
    assert (
        sys.gen_dict["gen_slack"] is gen_slack
    ), "Generator dictionary is wrong."
    assert sys.gen_dict["gen_PV"] is gen_PV, "Generator dictionary is wrong."
    assert sys.bus_to_generators[slack] == [
        gen_slack
    ], "Generator dictionary is wrong."
    assert sys.bus_to_generators[PV1] == [
        gen_PV
    ], "Generator dictionary is wrong."


def test_injector_additions():
    """
    Test addition of injectors to the system.
    """

    print("Testing addition of injectors to the system...")

    class TestInjector(records.Injector):
        """
        Dummy injector class.
        """

        prefix = "TEST"

        def __init__(self, bus: records.Bus, name: str) -> None:
            self.bus = bus
            self.name = name

        def get_pars(self) -> None:
            """
            Each injector must implement this method.
            """
            pass

    # Define dummy system with a single bus and injector
    sys = StaticSystem()
    bus = sys.add_PQ(PL=0, QL=0, name="PQ1")
    inj_1 = TestInjector(bus=bus, name="inj_1")
    sys.store_injector(inj_1)

    # Test containers
    assert sys.injectors == [inj_1], "Injector not added correctly."
    assert sys.inj_dict == {"inj_1": inj_1}, "Injector dictionary is wrong."
    assert sys.bus_to_injectors[bus] == [
        inj_1
    ], "Injector dictionary is wrong."

    # Test that injectors cannot be duplicated
    try:
        inj_2 = copy.deepcopy(inj_1)
        sys.store_injector(inj_2)
        assert False, "Injector added twice."
    except RuntimeError:
        pass

    class WrongInjector:
        """
        Dummy injector class that should raise RuntimeError.
        """

        def __init__(self, bus: records.Bus, name: str) -> None:
            self.bus = bus
            self.name = name

    inj_2 = WrongInjector(bus=bus, name="inj_2")
    try:
        sys.store_injector(inj_2)
    except RuntimeError:
        pass

    # Test containers once again
    assert sys.injectors == [
        inj_1
    ], "Wrong injector was added inspite of missing arguments."
    assert sys.inj_dict == {"inj_1": inj_1}, "Injector dictionary is wrong."
    assert sys.bus_to_injectors[bus] == [
        inj_1
    ], "Injector dictionary is wrong."


def get_fivebus(solve: bool = True) -> StaticSystem:
    """
    Return five-bus system from Duncan Glover, example 6.9.
    """

    sys = StaticSystem(
        name="Five-bus system from Duncan Glover, example 6.9.\n"
        "Results are reported in table 6.6."
    )

    # Add buses
    b1 = sys.add_slack(V_pu=1.0, base_kV=15, name="Bus 1")
    b2 = sys.add_PQ(PL=8, QL=2.8, base_kV=345, name="Bus 2")
    b3 = sys.add_PV(V_pu=1.05, PL=0.8 - 5.2, base_kV=15, name="Bus 3")
    b4 = sys.add_PQ(PL=0, QL=0, base_kV=345, name="Bus 4")
    b5 = sys.add_PQ(PL=0, QL=0, base_kV=345, name="Bus 5")

    # Add lines
    sys.add_line(
        from_bus=b2, to_bus=b4, R=0.009, X=0.1, total_B=1.72, name="Line 2-4"
    )
    sys.add_line(
        from_bus=b2, to_bus=b5, R=0.0045, X=0.05, total_B=0.88, name="Line 2-5"
    )
    sys.add_line(
        from_bus=b4,
        to_bus=b5,
        R=0.00225,
        X=0.025,
        total_B=0.44,
        name="Line 4-5",
    )

    # Add transformers
    sys.add_transformer(from_bus=b1, to_bus=b5, R=0.0015, X=0.02, name="T1-5")
    sys.add_transformer(from_bus=b3, to_bus=b4, R=0.00075, X=0.01, name="T3-4")

    # Possibly run a power flow
    if solve:
        sys.run_pf()

    return sys


def test_build_Y():
    """
    Test construction of the nodal admittance matrix.
    """

    print("Testing construction of the nodal admittance matrix...")

    # Load five-bus system from Duncan Glover, example 6.9
    sys = get_fivebus()

    # Build Y matrix and store it as attribute Y of sys
    sys.build_Y()

    # The following are textbook values:
    Y21, Y23 = 0, 0
    Y22 = 2.67828 - 28.4590j
    Y24 = -0.89276 + 9.91964j
    Y25 = -1.78552 + 19.83932j
    textbook_values = [Y21, Y22, Y24, Y25, Y23]

    # Compare computed values with textbook values
    for textbook_value, computed_value in zip(textbook_values, sys.Y[1, :]):
        assert np.isclose(
            textbook_value, computed_value
        ), "Element of Y is wrong."


def test_build_J():
    """
    Test construction of the Jacobian matrix.
    """

    print("Testing construction of the Jacobian matrix...")

    # Load five-bus system from Duncan Glover, example 6.9
    sys = get_fivebus(solve=False)

    # Build Jacobian for the conditions described in the texrbook
    for bus in sys.buses:
        bus.V_pu = 1.0
        bus.theta_deg = 0
    sys.PV_buses[0].V_pu = 1.05

    # Build relevant matrices
    sys.build_Y()
    sys.build_J()
    sys.build_full_J()

    # Textbook entry of the Jacobian is:
    textbook_entry = -9.91964

    # The following could be done with sys.get_bus(name="Bus 2") but we want
    # to decouple the testing of methods.
    b2 = next(bus for bus in sys.buses if bus.name == "Bus 2")
    b4 = next(bus for bus in sys.buses if bus.name == "Bus 4")

    index_b2 = sys.buses.index(b2)
    index_b4 = sys.buses.index(b4)

    # Computed entry is:
    computed_entry = sys.full_J[index_b2, index_b4]

    # Compare entries:
    assert np.isclose(textbook_entry, computed_entry), "Element of J is wrong."

    # The Jacobian matrix, when properly built, should satisfy the fact that
    # adding all derivatives of a single quantify with respect to all the
    # voltage angles should yield zero. This is because shifting all angles by
    # the same amount should not change any other quantify (the phase reference
    # is arbitrary).
    N = len(sys.buses)
    assert np.allclose(
        sys.full_J[:, :N].sum(axis=1), np.zeros([N, 1])
    ), "J is not balanced."


def test_flows():
    """
    Test computation of power flows.
    """

    print("Testing computation of power flows...")

    # Load five-bus system from Duncan Glover, example 6.9
    sys = get_fivebus(solve=False)

    # Ensure a flat start
    for bus in sys.buses:
        bus.V_pu = 1.0
        bus.theta_deg = 0
        bus.B_pu = 0

    # Remove shunt admittances for easy evaluation
    for branch in sys.branches:
        branch.from_Y_pu = branch.to_Y_pu = 0

    # Build nodal admittance matrix
    sys.build_Y()

    # For a flat start (actually for any condition with equal voltages)
    # at all buses, the flows out of each bus should be zero.
    assert np.allclose(
        sys.get_S_towards_network(), np.zeros([5, 1])
    ), "S_towards_network is not zero."

    # Build power mismatches
    sys.build_F()

    # N is used for shifting the slice when testing Q mismatch in PQ buses.
    N = len(sys.non_slack_buses)

    for bus_index, bus in enumerate(sys.non_slack_buses):
        # Test for P mismatch
        if isinstance(bus, records.PQ) or isinstance(bus, records.PV):
            assert np.allclose(
                sys.F[bus_index, 0], bus.PL_pu
            ), "Mismatch of active power is wrong."

        # Test for Q mismatch
        if isinstance(bus, records.PQ):
            assert np.allclose(
                sys.F[bus_index + N, 0], bus.QL_pu
            ), "Mismatch of reactive power is wrong."


def test_updates():
    """
    Test update of bus voltages and power injections towards the network.
    """

    print("Testing update of bus voltages and power injections...")

    # Load five-bus system from Duncan Glover, example 6.9
    sys = get_fivebus(solve=True)

    # Test that voltages are not 1 pu
    for bus in sys.non_slack_buses:
        assert not np.isclose(
            bus.V_pu, 1.0
        ), "Voltage is 1.0 pu after power flow."

    # Enforce null angles in PQ + PV, and flat voltage in PQ buses
    N = len(sys.non_slack_buses)
    M = len(sys.PQ_buses)
    x = np.vstack([np.zeros([N, 1]), np.ones([M, 1])])
    sys.update_v(x=x)

    # Ensure flat voltage in PV buses
    for bus in sys.PV_buses:
        bus.V_pu = 1.0

    # Test that voltages are flat
    for bus in sys.non_slack_buses:
        assert np.isclose(bus.get_phasor_V(), 1.0), "Voltage update is wrong."

    # Make the slack voltage flat
    sys.slack.V_pu = 1.0
    sys.slack.theta_radians = 0

    # Remove shunt admittances for easy evaluation of power flows
    for branch in sys.branches:
        branch.from_Y_pu = branch.to_Y_pu = 0

    # Test power update (should be zero everywhere due to flat voltages)
    sys.build_Y()
    sys.update_S()
    for bus in sys.buses:
        assert np.isclose(
            bus.P_to_network_pu, 0
        ), "Active power update is wrong."
        assert np.isclose(
            bus.Q_to_network_pu, 0
        ), "Reactive power update is wrong."


def get_nordic(solve: bool = True) -> StaticSystem:
    """
    Return the Nordic Test System - Case A, as described in the ARTERE file.
    """

    nordic = StaticSystem.import_ARTERE(
        filename=os.path.join(
            "networks", "Nordic", "Nordic test system", "lf_A.dat"
        ),
        system_name="Nordic Test System - Case A",
        use_injectors=True,
    )

    if solve:
        nordic.run_pf()

    return nordic


def test_run_pf():
    """
    Test computation of power flows.
    """

    print("Testing computation of power flows...")

    # Load five-bus system from Duncan Glover, example 6.9
    sys = get_fivebus(solve=True)

    # Obtain all five buses. When using pf_static, buses should be fetched with
    # the get_bus method, but, as above, we want to decouple the testing of
    # methods.
    b1 = next(bus for bus in sys.buses if bus.name == "Bus 1")
    b2 = next(bus for bus in sys.buses if bus.name == "Bus 2")
    b3 = next(bus for bus in sys.buses if bus.name == "Bus 3")
    b4 = next(bus for bus in sys.buses if bus.name == "Bus 4")
    b5 = next(bus for bus in sys.buses if bus.name == "Bus 5")

    # The following are the voltages reported in the textbook:
    v_textbook = [
        utils.pol2rect(1, 0),  # Slack
        utils.pol2rect(0.834, -22.407),
        utils.pol2rect(1.050, -0.597),
        utils.pol2rect(1.019, -2.834),
        utils.pol2rect(0.974, -4.548),
    ]

    # Compare computed values with textbook values
    for bus, v in zip([b1, b2, b3, b4, b5], v_textbook):
        assert np.isclose(
            bus.get_phasor_V(), v, atol=1e-3
        ), "Voltage is wrong."

    # We now test the string that reports the status. This is not something
    # that requires "correctness", but it is useful to test anyway in case one
    # screws up.
    assert (
        sys.status == "solved (max |F| < 0.1 W) in 5 iterations"
    ), "Power flow did not converge."

    # Test computation of power flows with the Nordic test system. This is an
    # indirect test of the import_ARTERE method.
    nordic = get_nordic(solve=True)

    # Start with an easy test: the name.
    assert (
        nordic.name == "Nordic Test System - Case A"
    ), "System name not set correctly."

    # Make sure that the dimensions are right. These are reported in the
    # paper by the Task Force (or in the ARTERE file).
    assert len(nordic.buses) == 74, "Number of buses is wrong."
    assert len(nordic.injectors) == 11 + 22, "Number of injectors is wrong."
    assert (
        len([inj for inj in nordic.injectors if isinstance(inj, records.Load)])
        == 22
    ), "Number of voltage-sensitive loads is wrong."
    assert (
        len(
            [inj for inj in nordic.injectors if isinstance(inj, records.Shunt)]
        )
        == 11
    ), "Number of shunts is wrong."
    assert len(nordic.lines) == 52, "Number of lines is wrong."

    def has_generator(bus: records.Bus) -> bool:
        """
        Return True if the bus has a generator.
        """
        return any(gen.bus is bus for gen in nordic.generators)

    def has_load(bus: records.Bus) -> bool:
        """
        Return True if the bus has a voltage-sensitive load.
        """
        return any(
            inj.bus is bus
            for inj in nordic.injectors
            if isinstance(inj, records.Load)
        )

    # Count step-up transformers (those that connect to a generator)
    step_up_transformers = [
        transformer
        for transformer in nordic.transformers
        if has_generator(transformer.get_LV_bus())
    ]

    assert (
        len(step_up_transformers) == 20
    ), "Number of step-up transformers is wrong."

    # Count step-down transformers (those that connect to a load)
    step_down_transformers = [
        transformer
        for transformer in nordic.transformers
        if has_load(transformer.get_LV_bus())
    ]

    assert (
        len(step_down_transformers) == 22
    ), "Number of step-down transformers is wrong."

    # Count generators
    assert len(nordic.generators) == 20, "Number of generators is wrong."

    # Finally, we verify that the power flows are correct by comparing
    # the power flows at the slack bus with the values reported in the
    # paper by the Task Force.
    S_slack_MVA = (
        nordic.slack.P_to_network_pu + 1j * nordic.slack.Q_to_network_pu
    ) * nordic.base_MVA

    assert np.isclose(
        S_slack_MVA, 2137.4 + 1j * 377.4, atol=1e-1
    ), "Slack power is wrong."


def test_get_named_elements():
    """
    Test retrieval of named elements.

    All reference values are taken from the report by the Task Force.
    """

    print("Testing retrieval of named elements...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Test retrieval of generation buses
    assert nordic.get_bus(name="g20") is nordic.slack, "Slack bus not found."
    bus_g10 = nordic.get_bus(name="g10")
    assert np.isclose(
        bus_g10.V_pu, 1.0157, atol=1e-4
    ), "Voltage of bus g10 is wrong."
    assert np.isclose(
        np.rad2deg(bus_g10.theta_radians), 0.99, atol=1e-2
    ), "Angle of bus g10 is wrong."

    # To increase certainty, we also test the power injection at bus g10.
    gen_g10 = next(gen for gen in nordic.generators if gen.bus is bus_g10)
    assert np.isclose(
        gen_g10.PG_MW, 600
    ), "Active power injection at bus g10 is wrong."

    # Test retrieval of transmission buses
    assert np.isclose(
        nordic.get_bus(name="1045").V_pu, 1.0111, atol=1e-4
    ), "Voltage of bus 1045 is wrong."
    assert np.isclose(
        np.rad2deg(nordic.get_bus(name="1045").theta_radians),
        -71.66,
        atol=1e-2,
    ), "Angle of bus 1045 is wrong."

    # Test retrieval of distribution (load) buses
    bus = nordic.get_bus(name="71")
    assert np.isclose(
        bus.V_pu, 1.0028, atol=1e-4
    ), "Voltage of bus 71 is wrong."
    assert np.isclose(
        np.rad2deg(bus.theta_radians), -7.80, atol=1e-2
    ), "Angle of bus 71 is wrong."

    # To increase certainty, we also test the power injection at bus 71.
    load_71 = next(
        inj
        for inj in nordic.injectors
        if isinstance(inj, records.Load) and inj.bus is bus
    )
    assert np.isclose(load_71.P0_MW, 300), "Active power of bus 71 is wrong."
    assert np.isclose(
        load_71.Q0_Mvar, 83.8
    ), "Reactive power of bus 71 is wrong."

    # Test retrieval of transmission lines
    line = nordic.get_line(name="4022-4031-2")
    base_kV = line.from_bus.base_kV
    assert np.isclose(
        nordic.pu2ohm(Z_pu=line.R_pu, base_kV=base_kV), 6.40
    ), "Resistance of line 4022-4031-2 is wrong."
    assert np.isclose(
        nordic.pu2ohm(Z_pu=line.X_pu, base_kV=base_kV), 64.0
    ), "Reactance of line 4022-4031-2 is wrong."
    assert np.isclose(
        nordic.pu2mho(Y_pu=line.from_Y_pu.imag, base_kV=base_kV), 375.42e-6
    ), "Susceptance of line 4022-4031-2 is wrong."
    assert np.isclose(
        line.Snom_MVA, 1400
    ), "Rating of line 4022-4031-2 is wrong."

    # Test retrieval of transformers
    transformer = nordic.get_transformer(name="g15")
    assert transformer.get_HV_bus() is nordic.get_bus(
        name="4047"
    ), "HV bus of g15 is wrong."
    assert transformer.get_LV_bus() is nordic.get_bus(
        name="g15"
    ), "LV bus of g15 is wrong."
    assert np.isclose(
        utils.change_base(
            quantity=transformer.X_pu,
            Sb_old=nordic.base_MVA,
            Sb_new=transformer.Snom_MVA,
            type="Z",
        ),
        0.15,
    ), "Reactance of g15 is wrong."
    assert np.isclose(transformer.Snom_MVA, 1200), "Rating of g15 is wrong."
    assert np.isclose(transformer.n_pu, 1.05), "Turns ratio of g15 is wrong."

    # Test retrieval of branches between buses
    assert (
        len(
            nordic.get_branches_between(
                bus_name_1="63", bus_name_2="62", warn=False
            )
        )
        == 0
    ), "There should be no branches between buses 63 and 62."

    step_down = nordic.get_branches_between(
        bus_name_1="63", bus_name_2="4063"
    )[0]

    assert step_down.get_HV_bus() is nordic.get_bus(
        name="4063"
    ), "HV bus of step-down transformer is wrong."
    assert step_down.get_LV_bus() is nordic.get_bus(
        name="63"
    ), "LV bus of step-down transformer is wrong."

    assert (
        len(nordic.get_branches_between(bus_name_1="1041", bus_name_2="1045"))
        == 2
    ), "There should be two branches between 1041 and 1045."

    # Test retrieval of generators
    gen_g20 = nordic.get_generator(name="g20")
    assert np.isclose(gen_g20.PG_MW, 0), "Active power of g20 is wrong."

    gen_g10 = nordic.get_generator(name="g10")
    assert np.isclose(gen_g10.PG_MW, 600), "Active power of g10 is wrong."

    # Test retrieval of injectors
    load_2 = nordic.get_injector(name="L02")
    assert np.isclose(load_2.P0_MW, 330), "Active power of L2 is wrong."
    assert np.isclose(load_2.Q0_Mvar, 71), "Reactive power of L2 is wrong."

    # Try to get elements that do not exist
    try:
        nordic.get_bus(name="does not exist")
        assert False, "Bus that does not exist was found."
    except RuntimeError:
        pass

    try:
        nordic.get_line(name="does not exist")
        assert False, "Line that does not exist was found."
    except RuntimeError:
        pass

    try:
        nordic.get_transformer(name="does not exist")
        assert False, "Transformer that does not exist was found."
    except RuntimeError:
        pass

    try:
        nordic.get_generator(name="does not exist")
        assert False, "Generator that does not exist was found."
    except RuntimeError:
        pass

    try:
        nordic.get_injector(name="does not exist")
        assert False, "Injector that does not exist was found."
    except RuntimeError:
        pass


def test_power_retrieval():
    """
    Test retrieval of consumed and generated powers.
    """

    print("Testing retrieval of consumed and generated powers...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Get power consumption
    assert (
        nordic.get_bus_load_MVA(bus=nordic.slack, attr="P") is None
    ), "Active load at slack bus should be zero."
    assert (
        nordic.get_bus_load_MVA(bus=nordic.slack, attr="Q") is None
    ), "Reactive load at slack bus should be zero."

    assert np.isclose(
        float(
            nordic.get_bus_load_MVA(bus=nordic.get_bus(name="71"), attr="P")
        ),
        300,
    ), "Active load at bus 71 is wrong."
    assert np.isclose(
        float(
            nordic.get_bus_load_MVA(bus=nordic.get_bus(name="71"), attr="Q")
        ),
        83.8,
    ), "Reactive load at bus 71 is wrong."

    bus_72 = nordic.get_bus(name="72")

    assert np.isclose(
        nordic.get_bus_load_MVA(bus=bus_72, attr="P"), 2000
    ), "Active power of loads at bus 72 is wrong."
    assert np.isclose(
        nordic.get_bus_load_MVA(bus=bus_72, attr="Q"), 396.1
    ), "Reactive power of loads at bus 72 is wrong."

    bus_4011 = nordic.get_bus(name="4011")
    assert (
        nordic.get_bus_load_MVA(bus=bus_4011, attr="P") is None
    ), "Load threshold is not working."

    # Get power generation
    assert np.isclose(
        float(nordic.get_bus_generation_MVA(bus=nordic.slack, attr="P")),
        2137.4,
        atol=1e-1,
    ), "Active generation at slack bus should be 2137.4 MW."

    assert np.isclose(
        float(nordic.get_bus_generation_MVA(bus=nordic.slack, attr="Q")),
        377.4,
        atol=1e-1,
    ), "Reactive generation at slack bus should be 377.4 Mvar."

    assert np.isclose(
        float(
            nordic.get_bus_generation_MVA(
                bus=nordic.get_bus(name="g10"), attr="P"
            )
        ),
        600,
        atol=1e-1,
    ), "Active generation at bus g10 is wrong."

    assert np.isclose(
        float(
            nordic.get_bus_generation_MVA(
                bus=nordic.get_bus(name="g10"), attr="Q"
            )
        ),
        255.7,
        atol=1e-1,
    ), "Reactive generation at bus g10 is wrong."

    bus_g2 = nordic.get_bus(name="g2")

    assert np.isclose(
        nordic.get_bus_generation_MVA(bus=bus_g2, attr="P"), 300
    ), "Active power of generators at bus g2 is wrong."
    assert np.isclose(
        nordic.get_bus_generation_MVA(bus=bus_g2, attr="Q"), 17.2, atol=1e-1
    ), "Reactive power of generators at bus g2 is wrong."

    assert (
        nordic.get_bus_generation_MVA(bus=bus_4011, attr="P") is None
    ), "Generation threshold is not working."
    assert (
        nordic.get_bus_generation_MVA(bus=bus_4011, attr="Q") is None
    ), "Generation threshold is not working."


def test_get_P_and_G():
    """
    Test retrieval of active power and conductance
    """

    print("Testing retrieval of active power and conductance...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Test against values from Vournas, Lambrou, and Mandoulidis (2017);
    # see Fig. 13
    P_publication = 13.3
    G_publication = 12.0

    # Retrieve values
    P, G = nordic.get_P_and_G(boundary_bus="4041", sending_buses=["4031"])

    # Compare
    assert np.isclose(P, P_publication, atol=1e-1), "Active power is wrong."
    assert np.isclose(G, G_publication, atol=1e-1), "Reactive power is wrong."


def test_get_min_NLI():
    """
    Test retrieval of minimum NLI at a bus.
    """

    print("Testing retrieval of minimum NLI...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Make all loads voltage-sensitive
    for inj in nordic.injectors:
        if isinstance(inj, records.Load):
            inj.make_voltage_sensitive(alpha=1, beta=2)

    # Remove one line that is known to be critical
    nordic.get_branches_between(bus_name_1="4032", bus_name_2="4044")[
        0
    ].in_operation = False

    # Run another power flow
    nordic.run_pf()

    # Define corridor for computing the current contributions
    corridor = ("4041", ["4031"])

    # Define transformers that will be perturbed
    transformer_names = [
        transformer.name
        for transformer in nordic.transformers
        if transformer.touches(location="CENTRAL")
    ]

    # The NLI trajectory is initialized with infinity so that testing for
    # reduction returns True in the first iteration.
    NLI_trajectory = [np.inf]

    # In the following loop, we let the transformers act and check that the
    # NLI is decreasing (the expected behavior).
    while nordic.run_pf(warn=False):
        NLI = nordic.get_min_NLI(
            corridor=corridor, transformer_names=transformer_names
        )

        if NLI == NLI_trajectory[-1]:
            return None

        assert NLI <= NLI_trajectory[-1], "NLI is not decreasing."
        NLI_trajectory.append(NLI)

        # Add perturbation
        for transformer_name in transformer_names:
            transformer = nordic.get_transformer(name=transformer_name)
            transformer.OLTC.act()
            # Maybe the following could be tried in the future?
            # transformer.OLTC.increase_voltages()

    else:
        return None


def test_str():
    """
    Test string representation of the system.
    """

    print("Testing string representation...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # The method generate_table should print a thorough report of the system.
    print(nordic.generate_table())

    looks_nice = input("Does the table look correct? (y/n) ")
    assert looks_nice == "y", "Table does not look correct."

    # On the other hand, __str__ (used by print) should only print a report of
    # the bus quantities.
    print(nordic)

    looks_nice = input("Does the reduced table look correct? (y/n) ")
    assert looks_nice == "y", "Reduced table does not look correct."


def test_graph_drawing():
    """
    Test drawing of voltage plot.
    """

    print("Testing drawing of voltage plot...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    nordic.draw_network(
        parameter="V_pu", display=True, title="Voltage magnitude"
    )

    looks_nice = input("Does the voltage magnitude plot look correct? (y/n) ")
    assert looks_nice == "y", "Voltage magnitude plot does not look correct."


def test_graph_theoretic_methods():
    """
    Test graph-theoretic methods.
    """

    print("Testing graph-theoretic methods...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Test subsystem definition
    bus_numbers = [4041, 4031, 4032, 4044, "g12"]

    buses = [
        nordic.get_bus(name=str(bus_number)) for bus_number in bus_numbers
    ]

    subsystem = nordic.get_subsystem(buses=buses)

    assert len(subsystem.buses) == 5, "Number of buses in subsystem is wrong."
    assert (
        len(subsystem.branches) == 6
    ), "Number of branches in subsystem is wrong."

    lines = [
        branch for branch in subsystem.branches if branch.branch_type == "Line"
    ]
    assert len(lines) == 5, "Number of lines in subsystem is wrong."

    transformers = [
        branch
        for branch in subsystem.branches
        if branch.branch_type == "Transformer"
    ]
    assert (
        len(transformers) == 1
    ), "Number of transformers in subsystem is wrong."

    for branch in subsystem.branches:
        assert (
            branch not in nordic.branches
        ), "Branch in subsystem is not a copy."
    for bus in subsystem.buses:
        assert bus not in nordic.buses, "Bus in subsystem is not a copy."

    # Test isolation by kV
    bus_4032 = nordic.get_bus(name="4032")
    buses_400_kV = nordic.isolate_buses_by_kV(starting_bus=bus_4032)

    bus_names = [
        "4011",
        "4012",
        "4021",
        "4022",
        "4031",
        "4032",
        "4041",
        "4042",
        "4043",
        "4044",
        "4045",
        "4046",
        "4047",
        "4051",
        "4061",
        "4062",
        "4063",
        "4071",
        "4072",
    ]

    solution = {nordic.get_bus(name=name) for name in bus_names}

    assert buses_400_kV == solution, "Isolation by kV fails for 400 kV."

    bus_1044 = nordic.get_bus(name="1044")
    buses_130_kV = nordic.isolate_buses_by_kV(starting_bus=bus_1044)

    bus_names = ["1041", "1042", "1043", "1044", "1045"]

    solution = {nordic.get_bus(name=name) for name in bus_names}

    assert buses_130_kV == solution, "Isolation by kV fails for 130 kV."

    # Test isolation by radius
    center = nordic.get_bus(name="4043")

    bus_numbers_r1 = [4047, 4046, 4042, 4044, 43]
    bus_numbers_r2 = [47, "g15", 46, 1044, "g14", 42, 4021, 4032, 4041, 4045]

    buses_r1 = {
        nordic.get_bus(name=str(bus_number)) for bus_number in bus_numbers_r1
    }
    buses_r2 = {
        nordic.get_bus(name=str(bus_number)) for bus_number in bus_numbers_r2
    }

    subsystem_r1 = nordic.isolate_by_radius(starting_bus=center, r=1)
    subsystem_r2 = nordic.isolate_by_radius(starting_bus=center, r=2)

    def buses2names(buses: Sequence) -> set[str]:
        return {bus.name for bus in buses}

    assert buses2names(subsystem_r1.buses) == {center.name} | buses2names(
        buses_r1
    ), "Isolation by radius fails for r=1."
    assert buses2names(subsystem_r2.buses) == {center.name} | buses2names(
        buses_r1
    ) | buses2names(buses_r2), "Isolation by radius fails for r=2."

    # Test update connectivity
    nordic = get_nordic()

    lines_4044_4045 = nordic.get_branches_between(
        bus_name_1="4044", bus_name_2="4045"
    )
    lines_4062_4045 = nordic.get_branches_between(
        bus_name_1="4062", bus_name_2="4045"
    )

    assert (
        len(lines_4044_4045) == 2
    ), "Number of lines between 4044 and 4045 is wrong."
    assert (
        len(lines_4062_4045) == 1
    ), "Number of lines between 4062 and 4045 is wrong."

    lines_4044_4045[0].in_operation = False
    lines_4044_4045[1].in_operation = False
    lines_4062_4045[0].in_operation = False

    nordic.update_connectivity(reference_bus=nordic.slack)

    connected_buses = list(filter(lambda bus: bus.is_connected, nordic.buses))
    assert len(connected_buses) == len(
        nordic.buses
    ), "Number of connected buses is wrong."

    # Restore lines
    lines_4044_4045[0].in_operation = True
    lines_4044_4045[1].in_operation = True
    lines_4062_4045[0].in_operation = True

    # Remove critical lines from operation
    lines_4046_4047 = nordic.get_branches_between(
        bus_name_1="4046", bus_name_2="4047"
    )
    lines_4043_4047 = nordic.get_branches_between(
        bus_name_1="4043", bus_name_2="4047"
    )

    assert (
        len(lines_4046_4047) == 1
    ), "Number of lines between 4046 and 4047 is wrong."
    assert (
        len(lines_4043_4047) == 1
    ), "Number of lines between 4043 and 4047 is wrong."

    lines_4046_4047[0].in_operation = False
    lines_4043_4047[0].in_operation = False

    nordic.update_connectivity(reference_bus=nordic.slack)

    connected_buses = list(filter(lambda bus: bus.is_connected, nordic.buses))
    disconnected_buses = list(
        filter(lambda bus: not bus.is_connected, nordic.buses)
    )

    bus_4047 = nordic.get_bus(name="4047")
    bus_47 = nordic.get_bus(name="47")
    bus_g15 = nordic.get_bus(name="g15")

    assert set(disconnected_buses) == {
        bus_4047,
        bus_47,
        bus_g15,
    }, "Disconnected buses are wrong."
    assert set(connected_buses) == set(nordic.buses) - {
        bus_4047,
        bus_47,
        bus_g15,
    }, "Connected buses are wrong."

    # Restore lines
    lines_4046_4047[0].in_operation = True
    lines_4043_4047[0].in_operation = True

    # Remove critical transformers from operation (using disconnect)
    boundary_transformers = nordic.get_branches_between(
        bus_name_1="4044", bus_name_2="1044"
    ) + nordic.get_branches_between(bus_name_1="4045", bus_name_2="1045")

    for transformer in boundary_transformers:
        transformer.disconnect()

    bus_1044 = nordic.get_bus(name="1044")
    bus_4044 = nordic.get_bus(name="4044")
    bus_4045 = nordic.get_bus(name="4045")
    buses_130_kV = nordic.isolate_buses_by_kV(starting_bus=bus_1044)
    isolated_buses = set()
    for bus in buses_130_kV:
        isolated_buses.update(
            nordic.isolate_buses_by_radius(starting_bus=bus, r=1)
        )

    isolated_buses -= {bus_4044, bus_4045}

    assert isolated_buses == set(
        filter(lambda bus: not bus.is_connected, nordic.buses)
    ), "Isolated buses are wrong."

    # Restore transformers
    for transformer in boundary_transformers:
        transformer.connect()

    assert (
        set(filter(lambda bus: not bus.is_connected, nordic.buses)) == set()
    ), "Isolated buses are wrong."


def test_contingency_check():
    """
    Test detection of contingencies.
    """

    print("Testing detection of contingencies...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Choose a branch at random and test detection
    branch = random.choice(nordic.branches)
    branch.disconnect()
    assert nordic.has_contingency(), "Branch contingency not detected."
    branch.connect()
    assert not nordic.has_contingency(), "Branch contingency detected."

    # Choose a generator at random and test detection
    generator = random.choice(nordic.generators)
    generator.trip()
    assert nordic.has_contingency(), "Generator contingency not detected."
    generator.trip_back()
    assert not nordic.has_contingency(), "Generator contingency detected."

    # Choose a shunt at random and test detection
    shunts = [
        inj for inj in nordic.injectors if isinstance(inj, records.Shunt)
    ]
    shunt = random.choice(shunts)
    shunt.trip()
    assert nordic.has_contingency(), "Shunt contingency not detected."
    shunt.trip_back()
    assert not nordic.has_contingency(), "Shunt contingency detected."


def test_get_sensitive_load():
    """
    Test retrieval of voltage-sensitive loads at buses.
    """

    print("Testing retrieval of voltage-sensitive loads...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Test sensitive load at bus 47
    bus_47 = nordic.get_bus(name="47")
    P_MW, Q_Mvar = nordic.get_sensitive_load_MW_Mvar(bus=bus_47)
    assert np.isclose(
        P_MW, 100
    ), "Active power of sensitive load at bus 47 is wrong."
    assert np.isclose(
        Q_Mvar, 44
    ), "Reactive power of sensitive load at bus 47 is wrong."


def test_voltage_checkers():
    """
    Test detection of under- and overvoltages.
    """

    print("Testing detection of under- and overvoltages...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Test overvoltages with known conditions from the Nordic
    assert nordic.has_overvoltages(), "Overvoltages not detected."
    assert not nordic.has_undervoltages(), "Undervoltages detected."

    # Induce an undervoltage and detectit
    nordic.buses[0].V_pu = 0.94
    assert nordic.has_undervoltages(), "Undervoltages not detected."


def test_power_matching():
    """
    Test algorithm for matching a certain slack power.
    """

    print("Testing power matching...")

    # Load the Nordic Test System - Case A
    nordic = get_nordic(solve=True)

    # Extract bus 31
    bus_31 = nordic.get_bus(name="31")
    load_31 = next(
        inj
        for inj in nordic.injectors
        if isinstance(inj, records.Load) and inj.bus is bus_31
    )

    # Scale powers at all buses
    x = np.array([[0.9], [0.8]])
    nordic.scale_powers(x=x)
    nordic.run_pf()

    # Make sure that power at bus 31 (chosen at random) was scaled correctly
    assert np.isclose(
        -load_31.get_P(), 0.9 * 100
    ), "Active power of load 31 is wrong."
    assert np.isclose(
        -load_31.get_Q(), 0.8 * 24.7
    ), "Reactive power of load 31 is wrong."

    # Load the Nordic again to start from scratch
    nordic = get_nordic(solve=True)

    # Verify slack power before scaling
    S_slack_MVA = nordic.get_S_slack_MVA()
    assert np.isclose(
        S_slack_MVA, 2137.4 + 1j * 377.4, atol=1e-1
    ), "Slack power is wrong."

    # Scale power from all generators
    for gen in nordic.generators:
        gen.PG_MW *= 0.9

    # After scaling the generators and now scaling the loads, the slack power
    # should scale by about the same factor (plus minus 400).
    S_slack_MVA_scaled = nordic.get_scaled_S_slack_MVA(x=x)
    assert np.isclose(
        S_slack_MVA_scaled, 2137.4 + 1j * 377.4, atol=400
    ), "Scaled slack power is wrong."

    # Load the Nordic again to start from scratch
    nordic = get_nordic(solve=False)

    # Scaling by a factor of 1 should not change the slack power
    x = np.ones([2, 1])
    S_slack_MVA_scaled = nordic.get_scaled_S_slack_MVA(x=x)
    assert np.isclose(
        S_slack_MVA_scaled, 2137.4 + 1j * 377.4, atol=1e-1
    ), "Scaled slack power is wrong."

    # Test mismatch computation
    x = np.array([[0.9], [0.8]])
    S_MVA = nordic.get_scaled_S_slack_MVA(x=x)
    mismatch = nordic.get_slack_mismatch_MW_Mvar(
        x=x, P_desired_MW=2137.4, Q_desired_Mvar=377.4
    )
    assert np.isclose(
        mismatch[0, 0], S_MVA.real - 2137.4, atol=1e-1
    ), "Mismatch is wrong."
    assert np.isclose(
        mismatch[1, 0], S_MVA.imag - 377.4, atol=1e-1
    ), "Mismatch is wrong."

    # Load the Nordic again to start from scratch
    nordic = get_nordic(solve=True)

    # In the base case of the Nordic, voltage correction should always
    # converge.
    assert nordic.correct_voltages(), "Voltage correction failed."

    # After having corrected the voltages, all voltages should be within the
    # LTC limits.
    for transformer in nordic.transformers:
        if transformer.has_OLTC:
            assert np.isclose(
                transformer.get_LV_bus().V_pu,
                transformer.OLTC.v_setpoint_pu,
                atol=transformer.OLTC.half_db_pu,
            ), "OLTC did not converge."

    # Load the Nordic again to start from scratch
    nordic = get_nordic(solve=True)

    # Match the slack power to 2300 MW and 400 Mvar. The angle of the slack is
    # set to 45 degrees simply to make the test more interesting.
    nordic.match_power(
        P_desired_MW=2300.0,
        Q_desired_Mvar=400.0,
        V_desired_pu=1.02,
        theta_desired_radians=np.pi / 4,
        tol=1e-6,
    )

    # Run the power flow and make sure that the match was successful
    nordic.run_pf()
    assert np.isclose(nordic.slack.V_pu, 1.02, atol=1e-4), "Voltage is wrong."
    assert np.isclose(
        nordic.slack.theta_radians, np.pi / 4, atol=1e-4
    ), "Angle is wrong."
    assert np.isclose(
        nordic.get_S_slack_MVA(), 2300.0 + 1j * 400.0, atol=1e-6
    ), "Slack power is wrong."


def get_UKGDS(solve=True):
    """
    Return the United Kingdom Generic Distribution System.
    """

    # Load the system
    UKGDS = StaticSystem.import_ARTERE(
        filename="networks/UKGDS/DN.dat",
        system_name="United-Kingdom Generic Distribution System",
        base_MVA=10.0,
        use_injectors=True,
    )

    # Initialize the loads at reasonable values
    for inj in UKGDS.injectors:
        if isinstance(inj, records.Load):
            inj.P0_MW = inj.allocated_P0_MW = 0.1
            inj.Q0_Mvar = inj.allocated_Q0_Mvar = 0.1 / 10

    # Modify transformers
    for transformer in UKGDS.transformers:
        # Add OLTC
        transformer.add_OLTC(
            positions_up=16, positions_down=16, step_pu=1 / 100
        )
        # Move to nominal tap position
        transformer.n_pu = 1.0

    return UKGDS


def test_disaggregation():
    """
    Test disaggregation of loads.
    """

    print("Testing load disaggregation...")

    # Load TN and DN
    nordic = get_nordic(solve=True)
    UKGDS = get_UKGDS(solve=True)

    # Increase consumption of UKGDS
    UKGDS.match_power(
        P_desired_MW=30,
        Q_desired_Mvar=5,
        V_desired_pu=1.0,
        theta_desired_radians=0.0,
    )

    # Create a pool of template systems
    pool = [copy.deepcopy(UKGDS) for _ in range(100)]

    # Identify buses from the central region
    central_buses = [bus for bus in nordic.buses if bus.location == "CENTRAL"]

    # Disaggregate loads
    with open("expanded_nordic.txt", "w") as f:
        for bus in central_buses[5:6]:
            f.write(f"Disaggregating bus {bus.name}...\n")
            f.write(f"P = {nordic.get_bus_load_MVA(bus=bus, attr='P')}\n")
            f.write(f"Q = {nordic.get_bus_load_MVA(bus=bus, attr='Q')}\n")
            max_F = nordic.disaggregate_load(bus=bus, systems=pool)[1]
            f.write(f"Maximum mismatch: {max_F}\n")
        nordic.run_pf()
        f.write(nordic.generate_table())

    looks_nice = input(
        "Check expanded_nordic.txt. Does it look correct? (y/n) "
    )

    assert looks_nice == "y", "Expanded Nordic does not look correct."

    # assert False, "Method should fail when base_kV are different, as here."


if __name__ == "__main__":
    test_init()
    test_conversions_pu_ohm()
    test_bus_additions()
    test_branch_additions()
    test_generator_additions()
    test_injector_additions()
    test_build_Y()
    test_build_J()
    test_flows()
    test_updates()
    test_run_pf()
    test_get_named_elements()
    test_power_retrieval()
    test_get_P_and_G()
    test_get_min_NLI()
    test_str()
    test_graph_drawing()
    test_graph_theoretic_methods()
    test_contingency_check()
    test_get_sensitive_load()
    test_voltage_checkers()
    test_power_matching()
    test_disaggregation()

    print("Module 'pf_static' passed all tests!")
