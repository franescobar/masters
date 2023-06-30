"""
Test functions for the 'sim_interaction' module.
"""

# Module to be tested
from sim_interaction import Observable, Disturbance

# Modules from this repository
import records

# Modules from the standard library
import copy


def get_record_examples() -> (
    tuple[
        records.Bus,
        records.Branch,
        records.Generator,
        records.EXC,
        records.Injector,
        records.DCTL,
    ]
):
    """
    Return dummy examples of records for testing purposes.
    """

    # Initialize bus
    bus = records.PV(
        V_pu=1.0,
        theta_radians=0.0,
        PL_pu=0.0,
        QL_pu=0.0,
        G_pu=0.0,
        B_pu=0.0,
        base_kV=1.0,
        bus_type="Slack",
        V_min_pu=0.95,
        V_max_pu=1.05,
        name="Example bus",
    )

    bus_2 = copy.deepcopy(bus)

    # Initialize branch
    branch = records.Branch(
        from_bus=bus,
        to_bus=bus_2,
        X_pu=0.1,
        R_pu=0.1,
        from_Y_pu=0.0,
        to_Y_pu=0.0,
        n_pu=1.0,
        branch_type="Line",
        Snom_MVA=1.0,
        name="Example branch",
        sys=None,
    )

    # Initialize generator
    gen = records.Generator(
        PG_MW=0.0,
        bus=bus,
        name="Example generator",
    )

    # Initialize exciter (all parameters are set to 1.0 for simplicity)
    exc = records.GENERIC1(*(1.0 for _ in range(23)))
    exc.name = "Example generator"

    # Initialize injector (e.g., load)
    inj = records.Load(
        name="Example load",
        bus=bus,
        P0_MW=0.0,
        Q0_Mvar=0.0,
    )

    # Initialize shunt
    shunt = records.Shunt(
        name="Example shunt",
        bus=bus,
        Mvar_at_Vnom=100,
    )

    # Initialize DCTL
    dctl = records.DCTL()
    dctl.name = "Example DCTL"

    return bus, branch, gen, exc, inj, shunt, dctl


def get_disturbance_examples() -> (
    tuple[
        Disturbance,
        Disturbance,
        Disturbance,
        Disturbance,
        Disturbance,
        Disturbance,
    ]
):
    """
    Return dummy examples of disturbances for testing purposes.
    """

    bus, branch, gen, exc, inj, shunt, dctl = get_record_examples()

    # Test initialization of disturbances on buses
    dist_bus = Disturbance(
        ocurrence_time=1.0,
        object_acted_on=bus,
        par_name="fault",
        par_value=3.14,
    )

    # Test initialization of disturbances on branches
    dist_branch = Disturbance(
        ocurrence_time=1.0,
        object_acted_on=branch,
        par_name="status",
        par_value=1.0,
    )

    # Test initialization of disturbances on injectors
    dist_inj = Disturbance(
        ocurrence_time=1.0,
        object_acted_on=inj,
        par_name="P0_MW",
        par_value=3.14,
    )

    # Test initialization of disturbances on DCTLS
    dist_dctl = Disturbance(
        ocurrence_time=2.0, object_acted_on=dctl, par_name="P", par_value=3.14
    )

    # Test initialization of disturbances on the solver
    dist_solver = Disturbance(
        ocurrence_time=3.0,
        object_acted_on="solver",
        par_name="some_value",
        par_value="CONTINUE SOLVER BD 0.02 0.001 0. ALL",
    )

    return (
        dist_bus,
        dist_branch,
        dist_inj,
        dist_dctl,
        dist_solver,
    )


def test_Disturbance():
    """
    Test the class for disturbances.
    """

    bus, branch, _, _, inj, _, _ = get_record_examples()
    (
        dist_bus,
        dist_branch,
        dist_inj,
        dist_dctl,
        dist_solver,
    ) = get_disturbance_examples()

    # Test correct initializations
    assert str(dist_bus) == "FAULT BUS Example bus 0.0 3.14"
    assert str(dist_branch) == "BREAKER BRANCH Example branch 1 1"
    assert str(dist_inj) == "CHGPRM INJ Example load P0_MW 3.14 SETP 0.0"
    assert str(dist_dctl) == "CHGPRM DCTL Example DCTL P 3.14 0.0"
    assert str(dist_solver) == "CONTINUE SOLVER BD 0.02 0.001 0. ALL"

    # Test another disturbance on branches
    another_dist_branch = Disturbance(
        ocurrence_time=1.0,
        object_acted_on=branch,
        par_name="status",
        par_value=0.0,
    )
    assert str(another_dist_branch) == "BREAKER BRANCH Example branch 0 0"

    # Test incorrect initializations

    try:
        Disturbance(
            ocurrence_time=-1.0,
            object_acted_on=bus,
            par_name="fault",
            par_value=1.0,
        )
        assert (
            False
        ), "Disturbance with negative ocurrence time should not be allowed."
    except RuntimeError:
        pass
    else:
        assert (
            False
        ), "Disturbance with negative ocurrence time should not be allowed."

    try:
        Disturbance(
            ocurrence_time=1.0,
            object_acted_on=bus,
            par_name="some parameter",
            par_value=1.0,
        )
        assert False, "Bus parameter should be either 'fault' or 'clearance'."
    except RuntimeError:
        pass
    else:
        assert False, "Bus parameter should be either 'fault' or 'clearance'."

    try:
        Disturbance(
            ocurrence_time=1.0,
            object_acted_on=branch,
            par_name="some parameter",
            par_value=1.0,
        )
        assert False, "Branch parameter should be 'status'."
    except RuntimeError:
        pass
    else:
        assert False, "Branch parameter should be 'status'."

    try:
        Disturbance(
            ocurrence_time=1.0,
            object_acted_on=branch,
            par_name="status",
            par_value=2.0,
        )
    except RuntimeError:
        pass
    else:
        assert False, "Branch parameter value should be either 0 or 1."

    try:
        Disturbance(
            ocurrence_time=3.0,
            object_acted_on="solver",
            par_name="some_value",
            par_value=0.1,
        )
        assert False, "Solver parameter should be a string."
    except RuntimeError:
        pass
    else:
        assert False, "Solver parameter should be a string."

    # Test ordering of disturbances
    assert (
        dist_inj < dist_dctl < dist_solver
    ), "Disturbances should be ordered by ocurrence time."

    # Test disturbance equality
    dist_inj_2 = Disturbance(
        ocurrence_time=1.0,
        object_acted_on=inj,
        par_name="P0_MW",
        par_value=3.14,
    )
    dist_inj_3 = Disturbance(
        ocurrence_time=1.0 + 1e-4,
        object_acted_on=inj,
        par_name="P0_MW",
        par_value=3.14,
    )

    assert dist_inj == dist_inj_2, "Disturbances should be equal."

    dist_inj_4 = Disturbance(
        ocurrence_time=1.0 + 2e-3,
        object_acted_on=inj,
        par_name="P0_MW",
        par_value=2.0,
    )

    assert dist_inj == dist_inj_3, "Disturbances should be equal within 1e-3."
    assert dist_inj != dist_inj_4, "Disturbances should not be equal."

    return dist_bus, dist_branch, dist_inj, dist_dctl, dist_solver


def test_Observable():
    """
    Test the class for observables.
    """

    bus, branch, gen, exc, inj, _, dctl = get_record_examples()

    # Test observables for buses
    obs_bus = Observable(observed_object=bus, obs_name="BV")

    assert obs_bus.observed_object is bus, "Observable object is incorrect."
    assert obs_bus.obs_name == "BV", "Observable name is incorrect."
    assert (
        obs_bus.element_type == "BUS"
    ), "Observable element type is incorrect."

    # Test observables for branches
    obs_branch = Observable(observed_object=branch, obs_name="PF")

    assert (
        obs_branch.observed_object is branch
    ), "Observable object is incorrect."
    assert obs_branch.obs_name == "PF", "Observable name is incorrect."
    assert (
        obs_branch.element_type == "BRANCH"
    ), "Observable element type is incorrect."

    # Test observables for generators
    obs_gen = Observable(observed_object=gen, obs_name="PG")

    assert obs_gen.observed_object is gen, "Observable object is incorrect."
    assert obs_gen.obs_name == "PG", "Observable name is incorrect."
    assert (
        obs_gen.element_type == "SYNC"
    ), "Observable element type is incorrect."

    # Test observables for exciters
    obs_exc = Observable(observed_object=exc, obs_name="EFD")

    assert obs_exc.observed_object is exc, "Observable object is incorrect."
    assert obs_exc.obs_name == "EFD", "Observable name is incorrect."
    assert (
        obs_exc.element_type == "SYNC"
    ), "Observable element type is incorrect."

    # Test observables for injectors
    obs_inj = Observable(observed_object=inj, obs_name="P")

    assert obs_inj.observed_object is inj, "Observable object is incorrect."
    assert obs_inj.obs_name == "P", "Observable name is incorrect."
    assert (
        obs_inj.element_type == "INJEC"
    ), "Observable element type is incorrect."

    # Test observables for DCTLs
    obs_dctl = Observable(observed_object=dctl, obs_name="P")

    assert obs_dctl.observed_object is dctl, "Observable object is incorrect."
    assert obs_dctl.obs_name == "P", "Observable name is incorrect."
    assert (
        obs_dctl.element_type == "DCTL"
    ), "Observable element type is incorrect."

    # Test any other type
    class TestOther:
        pass

    try:
        obs = Observable(observed_object=TestOther(), obs_name="P")
        assert False, "Observable of type 'TestOther' should not be allowed."
    except RuntimeError:
        pass
    else:
        assert False, "Observable of type 'TestOther' should not be allowed."

    # Test conversion to string
    obs = Observable(observed_object=bus, obs_name="BV")

    assert str(obs) == "BUS Example bus", "Observable string is incorrect."

    # Test equality of observables
    obs_2 = Observable(observed_object=bus, obs_name="BV")

    assert obs == obs_2, "Observables should be equal."
    assert obs is not obs_2, "Observables should not be identical."

    return obs_bus, obs_branch, obs_gen, obs_exc, obs_inj, obs_dctl


if __name__ == "__main__":
    test_Disturbance()
    test_Observable()

    print("Module 'sim_interaction' passed all tests!")
