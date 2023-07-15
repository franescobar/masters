"""
Test functions for the 'pf_dynamic' module.

This file tests all methods except for three:

    1. System.update_detectors,
    2. System.follow_controllers, and
    3. System.get_twin.

Since they rely on the RAMSES simulator, they are tested in test_experiment.py
"""

# Module to be tested
from pf_dynamic import *

# Modules from this repository
import pf_static
import records
import test_sim_interaction
import control
import oltc

# Modules from the standard library
import os

# Other modules
import numpy as np


def test_System_initialization():
    """
    Test initialization of pf_dynamic.System.
    """

    sys = System()

    assert isinstance(
        sys, pf_static.StaticSystem
    ), "System must be a child of StaticSystem."
    assert sys.ram is None, "RAMSES simulator should be None."
    assert sys.records == [], "There should be no records."
    assert sys.disturbances == [], "There should be no disturbances."
    assert sys.detectors == [], "There should be no detectors."
    assert sys.controllers == [], "There should be no controllers."
    assert sys.OLTC_controllers == [], "There should be no OLTC controllers."


def test_add_record():
    """
    Test the addition of records to the system.
    """

    sys = System()
    exc = test_sim_interaction.get_record_examples()[3]
    sys.add_record(record=exc)

    assert sys.records == [exc], "Records list is incomplete."
    assert sys.records[0].name == "Example generator"

    # Test that a record cannot be added twice
    try:
        sys.add_record(record=exc)
        assert False, "Record should not be added twice."
    except RuntimeError:
        pass
    except:
        assert False, "Record should not be added twice."

    # Test that records are sorted by name
    dctl = test_sim_interaction.get_record_examples()[6]
    sys.add_record(record=dctl)
    assert sys.records == [dctl, exc], "Records are not sorted by name."


def test_set_frequency():
    """
    Test the setting of the system nominal frequency.
    """

    sys = System()
    sys.set_frequency(fnom=60)
    f = sys.records[0].pars[0].value
    assert np.isclose(f, 60), "Frequency should be 60 Hz."

    # Test that the frequency cannot be set twice
    try:
        sys.set_frequency(fnom=50)
        assert False, "Frequency should not be set twice."
    except RuntimeError:
        pass
    except:
        assert False, "Frequency should not be set twice."


def test_add_OLTC_controller():
    """
    Test the addition of OLTC controllers in particular.
    """

    sys = System()

    class TestOLTCController(oltc.OLTC_controller):
        def __init__(self) -> None:
            pass

        def __str__(self) -> str:
            return "Test OLTC controller"

    OLTC = TestOLTCController()

    # Test that an OLTC controller can be added
    sys.add_OLTC_controller(OLTC_controller=OLTC)
    assert sys.OLTC_controllers == [
        OLTC
    ], "OLTC controller list is incomplete."

    # Test that an OLTC controller cannot be added twice
    try:
        sys.add_OLTC_controller(OLTC_controller=OLTC)
        assert False, "OLTC controller should not be added twice."
    except RuntimeError:
        pass
    except:
        assert False, "OLTC controller should not be added twice."

    # Test that only subclasses of oltc.OLTC_controller can be added
    try:
        sys.add_OLTC_controller(OLTC_controller=control.Controller())
        assert False, "Controllers must be a subclass of oltc.OLTC_controller."
    except RuntimeError:
        pass
    except:
        assert False, "Controllers must be a subclass of oltc.OLTC_controller."


def get_dynamic_nordic(solve: bool = True) -> System:
    """
    Return a dynamic version of the Nordic test system.

    We load the system from scratch instead of using test_pf_static.get_nordic
    because we want it to be an instance of pf_dynamic.System, not an instance
    of pf_static.StaticSystem.
    """

    nordic = System().import_ARTERE(
        filename=os.path.join(
            "networks", "Nordic", "Nordic test system", "lf_A.dat"
        ),
        system_name="Nordic Test System - Case A",
        use_injectors=True,
    )

    if solve:
        nordic.run_pf()

    return nordic


def test_import_dynamic_data():
    """
    Test the import of dynamic data from .dat files in RAMSES format.
    """

    nordic = get_dynamic_nordic(solve=True)

    nordic.import_dynamic_data(
        filename=os.path.join(
            "networks", "Nordic", "Nordic test system", "dyn_A.dat"
        )
    )

    # Count the number of OLTC controllers
    N = 0
    for t in nordic.transformers:
        if t.has_OLTC and t.OLTC.OLTC_controller is not None:
            N += 1
    assert N == 22, "There should be 22 OLTCs."

    # Check records bound to generators
    for g in nordic.generators:
        assert g.machine is not None, "Generator should have a machine."
        assert g.exciter is not None, "Generator should have an exciter."
        assert g.governor is not None, "Generator should have a governor."

    # Check existence of frequency record
    assert any(
        isinstance(record, records.Frequency) for record in nordic.records
    ), "There should be a frequency record."


def test_export_to_RAMSES():
    """
    Test the export of dynamic data (possibly modified) to .dat files.
    """

    nordic = get_dynamic_nordic(solve=True)

    nordic.import_dynamic_data(
        filename=os.path.join(
            "networks", "Nordic", "Nordic test system", "dyn_A.dat"
        )
    )

    # Create test .dat file (if it does not exist)
    if not os.path.exists("dyn_A_test.dat"):
        nordic.export_to_RAMSES(filename="dyn_A_test.dat")
    else:
        raise RuntimeError("File 'dyn_A_test.dat' already exists.")

    looks_nice = input("Check dyn_A_test.dat. Does it look correct? (y/n) ")

    # Delete test .dat file before (possibly) raising an assertion error
    os.remove("dyn_A_test.dat")

    assert looks_nice == "y", "The RAMSES file does not look correct."


def test_add_disturbances():
    """
    Test the addition of disturbances to the system.
    """

    (
        dist_inj,
        dist_dctl,
        dist_solver,
    ) = test_sim_interaction.get_disturbance_examples()[-3:]

    sys = System()

    sys.add_disturbances(disturbances=[dist_solver, dist_dctl, dist_inj])

    assert sys.disturbances == [
        dist_inj,
        dist_dctl,
        dist_solver,
    ], "Disturbances were not added in the correct order."


class DetectorImitator(control.Detector):
    pass


def test_add_detector():
    """
    Test the addition of detectors to the system.
    """

    detector = DetectorImitator()
    sys = System()

    sys.add_detector(detector=detector)

    assert len(sys.detectors) == 1, "Detector list is incomplete."

    # Test that the detector must be a child of control.Detector
    class WrongDetector:
        pass

    try:
        sys.add_detector(detector=WrongDetector())
        assert False, "Detector should be a child of control.Detector."
    except RuntimeError:
        pass
    except:
        assert False, "Detector should be a child of control.Detector."

    # Test that a detector cannot be added twice
    try:
        sys.add_detector(detector=detector)
        assert False, "Detector should not be added twice."
    except RuntimeError:
        pass
    except:
        assert False, "Detector should not be added twice."


class ControllerImitator(control.Controller):
    def __init__(self) -> None:
        self.sys = None


def test_add_controllers():
    """
    Test the addition of controllers to the system.
    """

    controller = ControllerImitator()
    sys = System()

    sys.add_controllers(controllers=[controller])
    assert len(sys.controllers) == 1, "There should be one controller."

    # Test that the controller must be a child of control.Controller
    class WrongController:
        pass

    try:
        sys.add_controllers(controllers=[WrongController()])
        assert False, "Controller should be a child of control.Controller."
    except RuntimeError:
        pass
    except:
        assert False, "Controller should be a child of control.Controller."

    # Test that a controller cannot be added twice
    try:
        sys.add_controllers(controllers=[controller])
        assert False, "Controller should not be added twice."
    except RuntimeError:
        pass
    except:
        assert False, "Controller should not be added twice."

    # Test that a controller cannot act on two systems
    controller = ControllerImitator()
    controller.sys = True

    try:
        sys.add_controllers(controllers=[controller])
        assert False, "Controller should not act on two systems."
    except RuntimeError:
        pass
    except:
        assert False, "Controller should not act on two systems."


class RAMSESImitator:
    """
    This class imitates the essential functionality of the RAMSES simulator.
    """

    def __init__(self) -> None:
        self.t_now = np.pi

    def getSimTime(self) -> float:
        return self.t_now

    def addDisturb(self, t_dist: float, disturb: str) -> None:
        print(f"Sending disturbance: t_dist = {t_dist}, disturb = {disturb}")


def test_get_t_now():
    """
    Test the retrieval of the current simulation time.
    """

    sys = System()
    sys.ram = RAMSESImitator()

    assert np.isclose(
        sys.get_t_now(), np.pi
    ), "Current simulation time should be pi."


def test_get_disturbances_until():
    """
    Test the retrieval of disturbances until a given time.
    """

    (
        dist_inj,
        dist_dctl,
        dist_solver,
    ) = test_sim_interaction.get_disturbance_examples()[-3:]

    sys = System()
    sys.add_disturbances(disturbances=[dist_solver, dist_dctl, dist_inj])

    # Test the incremental retrieval of disturbances. This type of retrieval
    # should also remove some disturbances from the list at each call.
    assert (
        sys.get_disturbances_until(t=0) == []
    ), "There should be no disturbances."
    assert sys.get_disturbances_until(t=1 - 1e-3) == []
    assert sys.get_disturbances_until(t=1 + 1e-3) == [dist_inj]
    assert sys.get_disturbances_until(t=2 - 1e-3) == []
    assert sys.get_disturbances_until(t=2 + 1e-3) == [dist_dctl]
    assert sys.get_disturbances_until(t=3 - 1e-3) == []
    assert sys.get_disturbances_until(t=3 + 1e-3) == [dist_solver]
    assert sys.get_disturbances_until(t=np.inf) == []

    # Send the disturbances again and test the retrieval of all disturbances
    # at once.
    sys.add_disturbances(disturbances=[dist_solver, dist_dctl, dist_inj])
    assert sys.get_disturbances_until(t=np.inf) == [
        dist_inj,
        dist_dctl,
        dist_solver,
    ]


def test_send_disturbance():
    """
    Test the sending of a single disturbance (string) to the system.
    """

    sys = System()
    sys.ram = RAMSESImitator()

    dist_inj = test_sim_interaction.get_disturbance_examples()[-3]
    sys.send_disturbance(disturbance=dist_inj)

    looks_nice = input("Does the disturbance look correct? (y/n) ")
    assert looks_nice == "y", "The disturbance was not sent correctly."


def test_send_disturbances():
    """
    Test the sending of a list of disturbances (strings) to the system.

    This function tests both send_disturbances and send_disturbances_until.
    """

    nordic = get_dynamic_nordic(solve=True)
    nordic.ram = RAMSESImitator()

    # Disturbance: Short circuit at bus 2
    dist_sc = test_sim_interaction.Disturbance(
        ocurrence_time=1,
        object_acted_on=nordic.get_bus(name="2"),
        par_name="fault",
        par_value=0.1,
    )

    # Disturbance: Change in load power at bus 2
    injector = next(
        inj
        for inj in nordic.injectors
        if isinstance(inj, records.Load) and inj.bus.name == "2"
    )
    dist_inj = test_sim_interaction.Disturbance(
        ocurrence_time=0.1,
        object_acted_on=injector,
        par_name="P0",
        par_value=0.1,
    )

    # Disturbance: Opening of branch between buses 1042 and 2
    dist_opening = test_sim_interaction.Disturbance(
        ocurrence_time=1.1,
        object_acted_on=nordic.get_branches_between(
            bus_name_1="1042", bus_name_2="2"
        )[0],
        par_name="status",
        par_value=0,
    )

    # Disturbance: (Another) short circuit at bus 2
    dist_second_sc = test_sim_interaction.Disturbance(
        ocurrence_time=2,
        object_acted_on=nordic.get_bus(name="2"),
        par_name="fault",
        par_value=0.1,
    )

    # Disturbance: (Another) opening of branch between buses 1042 and 2
    dist_second_opening = test_sim_interaction.Disturbance(
        ocurrence_time=2.1,
        object_acted_on=nordic.get_branches_between(
            bus_name_1="1042", bus_name_2="2"
        )[0],
        par_name="status",
        par_value=0,
    )

    # Disturbance: (Another) change in load power at bus 2
    dist_second_inj = test_sim_interaction.Disturbance(
        ocurrence_time=2.2,
        object_acted_on=injector,
        par_name="P0",
        par_value=0.1,
    )

    nordic.add_disturbances(
        disturbances=[
            dist_sc,
            dist_inj,
            dist_opening,
            dist_second_sc,
            dist_second_opening,
            dist_second_inj,
        ]
    )

    # Sending these disturbances will result in them being printed in the
    # terminal.
    print("")
    nordic.send_disturbances_until(t=3)

    # The add_disturbances method should have put all disturbances in
    # chronological order.
    looks_nice = input(
        "\nWere the disturbances sent in chronological order? (y/n) "
    )
    assert (
        looks_nice == "y"
    ), "The disturbances were not sent in chronological order."

    # The send_disturbances method should have ignored disturbances that made
    # no sense.
    print("\nThe following disturbances should have been sent:\n")
    print("\t1. Change in load power at bus 2.")
    print("\t2. Short circuit at bus 2.")
    print("\t3. Opening of branch between buses 1042 and 2.")
    looks_nice = input("\nWere the correct disturbances sent? (y/n) ")
    assert looks_nice == "y", "Wrong disturbances were sent."


def test_twins():
    """
    Test the generation and update of twins.
    """

    nordic = get_dynamic_nordic(solve=True)

    def randomize_normally(parameter: float) -> float:
        return np.random.normal(loc=parameter, scale=0.1 * parameter)

    # Define dummy parameter randomization
    def randomize_loads(sys: System) -> None:
        for inj in sys.injectors:
            if isinstance(inj, records.Load):
                # Modify the active power
                inj.allocated_P0_MW -= 30
                inj.P0_MW = inj.allocated_P0_MW
                # Modify the reactive power
                inj.allocated_Q0_Mvar -= 30
                inj.Q0_Mvar = inj.allocated_Q0_Mvar

    def randomize_branches(sys: System) -> None:
        for branch in sys.branches:
            branch.R_pu = randomize_normally(branch.R_pu)
            branch.X_pu = randomize_normally(branch.X_pu)
            branch.from_Y_pu = randomize_normally(
                branch.from_Y_pu.real
            ) + 1j * randomize_normally(branch.from_Y_pu.imag)
            branch.to_Y_pu = randomize_normally(
                branch.to_Y_pu.real
            ) + 1j * randomize_normally(branch.to_Y_pu.imag)

    # Test the generation of a (randomized) twin
    nordic.generate_twin(
        parameter_randomizations=[randomize_loads, randomize_branches],
    )

    nordic.twin.run_pf()
    print(nordic.generate_table())
    print(nordic.twin.generate_table())

    looks_nice = input("Does the twin look correct? (y/n) ")
    assert looks_nice == "y", "The twin was not generated correctly."

    # Overwrite original twin
    nordic.generate_twin()
    nordic.twin.run_pf()

    # In the model, make the loads voltage sensitive
    for inj in nordic.twin.injectors:
        if isinstance(inj, records.Load):
            inj.make_voltage_sensitive(alpha=1, beta=2)

    # Test the update of a twin
    class RAMSESImitator:
        def getBusVolt(self, busNames: list[str]) -> list[float]:
            return [0.95 for _ in busNames]

        def getObs(
            self,
            comp_type: list[str],
            comp_name: list[str],
            obs_name: list[str],
        ) -> list[float]:
            return [0.98 for _ in comp_type]

        def getBranchPow(self, branchName: list[str]) -> list[float]:
            return [[-np.e, -np.e, -np.e, -np.e] for _ in branchName]

    nordic.ram = RAMSESImitator()

    # Store values to make assertions about the insensitive loads (bus 3)
    P3 = (0.95 / nordic.twin.get_bus(name="3").V_pu) ** 1 * 260
    Q3 = (0.95 / nordic.twin.get_bus(name="3").V_pu) ** 2 * 83.8

    nordic.update_twin()

    # Make assertions about the voltages
    for bus in nordic.twin.buses:
        assert np.isclose(bus.V_pu, 0.95), "Voltage should be 1.0 p.u."

    # Make assertions about the tap ratios
    for transformer in nordic.twin.transformers:
        if (
            transformer.has_OLTC
            and transformer.OLTC.OLTC_controller is not None
        ):
            assert np.isclose(
                transformer.n_pu, 0.98
            ), "Tap ratio should be 0.99"

    assert np.isclose(
        nordic.twin.get_bus(name="3").PL_pu,
        np.e - P3 / 100,
    )

    assert np.isclose(
        nordic.twin.get_bus(name="3").QL_pu,
        np.e - Q3 / 100,
    )


if __name__ == "__main__":
    # test_System_initialization()
    # test_add_record()
    # test_set_frequency()
    # test_add_OLTC_controller()
    # test_import_dynamic_data()
    # test_export_to_RAMSES()
    # test_add_disturbances()
    # test_add_detector()
    # test_add_controllers()
    # test_get_t_now()
    # test_get_disturbances_until()
    # test_send_disturbance()
    # test_send_disturbances()
    test_twins()

    print("Module 'pf_dynamic' passed all tests!")
