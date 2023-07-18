"""
Test functions for the 'control' module.
"""

# Module to be tested
from control import *

# Modules from this repository
from test_pf_dynamic import get_dynamic_nordic
import records
import pf_dynamic

# Modules from the standard library

# Other modules
import numpy as np

# This is simply to make sure that a parameter randomization passes some tests
# below.
np.random.seed(3)


def test_MPC_Controller_init():
    """
    Test initialization of the MPC_Controller class.
    """

    mpc = MPC_controller()

    assert (
        mpc.sys is None
    ), "The MPC system should be None after initialization."

    assert (
        mpc.Np is None
    ), "The MPC prediction horizon should be None after initialization."
    assert (
        mpc.Nc is None
    ), "The MPC control horizon should be None after initialization."
    assert (
        mpc.period is None
    ), "The MPC period should be None after initialization."

    assert np.isclose(
        mpc.t_last_action, 0
    ), "The MPC time of the last action should be 0 after initialization."

    assert (
        mpc.observed_corridors == []
    ), "The MPC observed corridors should be empty after initialization."
    assert (
        mpc.controlled_transformers == []
    ), "The MPC controlled transformers should be empty after initialization."
    assert (
        mpc.B == 0
    ), "The number of boundary buses should be 0 after initialization."
    assert (
        mpc.T == 0
    ), "The number of controlled transformers should be 0 after initialization."

    assert (
        mpc.current_iter == 0
    ), "The current iteration should be 0 after initialization."

    assert (
        mpc.solutions == []
    ), "The solution history should be empty after initialization."
    assert (
        mpc.dr == []
    ), "The tap increment history should be empty after initialization."
    assert (
        mpc.dP == []
    ), "The power increment history should be empty after initialization."
    assert (
        mpc.dQ == []
    ), "The reactive power increment history should be empty after initialization."
    assert (
        mpc.slacks == []
    ), "The slack history should be empty after initialization."


def test_set_period():
    """
    Test setting the period of an MPC controller.
    """

    mpc = MPC_controller()

    mpc.set_period(1)

    assert mpc.period == 1, "The MPC period should be 1 after setting it to 1."

    # Test that the period must be a number
    try:
        mpc.set_period("1")
    except TypeError:
        pass
    else:
        raise AssertionError("The MPC period must be a number.")

    # Test that the period must be positive
    try:
        mpc.set_period(-1)
    except ValueError:
        pass
    else:
        raise AssertionError("The MPC period must be positive.")


def test_add_observed_corridor():
    """
    Test addition of an observed corridor to an MPC controller.
    """

    mpc = MPC_controller()

    mpc.add_observed_corridor(
        boundary_bus="4041", sending_buses=["4030", "4032"]
    )

    assert mpc.observed_corridors == [
        ("4041", ["4030", "4032"])
    ], "The MPC observed corridors were not added correctly."
    assert (
        mpc.B == 1
    ), "The number of boundary buses should be 1 after adding a single corridor."

    # Test that the boundary bus must be a string
    try:
        mpc.add_observed_corridor(
            boundary_bus=4041, sending_buses=["4030", "4032"]
        )
    except TypeError:
        pass
    else:
        raise AssertionError("The boundary bus must be a string.")

    # Test that the sending buses must be a list
    try:
        mpc.add_observed_corridor(boundary_bus="4041", sending_buses="4030")
    except TypeError:
        pass
    else:
        raise AssertionError("The sending buses must be a list.")

    # Test that the sending buses must be a list of strings
    try:
        mpc.add_observed_corridor(
            boundary_bus="4041", sending_buses=[4030, 4032]
        )
    except TypeError:
        pass
    else:
        raise AssertionError("The sending buses must be a list of strings.")


def test_add_controlled_transformer():
    """
    Test addition of a controlled transformer to an MPC controller.
    """

    mpc = MPC_controller()

    mpc.add_controlled_transformers(transformers=["4041-4042", "4042-4043"])

    assert mpc.controlled_transformers == [
        "4041-4042",
        "4042-4043",
    ], "The MPC controlled transformers were not added correctly."
    assert (
        mpc.T == 2
    ), "The number of controlled transformers should be 2 after adding two transformers."

    # Test that the transformers must be a list
    try:
        mpc.add_controlled_transformers(transformers="4041-4042")
    except TypeError:
        pass
    else:
        raise AssertionError("The transformers must be a list.")

    # Test that the transformers must be a list of strings
    try:
        mpc.add_controlled_transformers(transformers=[4041, 4042])
    except TypeError:
        pass
    else:
        raise AssertionError("The transformers must be a list of strings.")


def test_set_horizons():
    """
    Test setting the control and prediction horizons of an MPC controller.
    """

    mpc = MPC_controller()

    mpc.set_horizons(Np=10, Nc=5)

    assert (
        mpc.Np == 10
    ), "The MPC prediction horizon should be 10 after setting it to 10."

    assert (
        mpc.Nc == 5
    ), "The MPC control horizon should be 5 after setting it to 5."

    # Test that the horizons must be integers
    try:
        mpc.set_horizons(Np=10.0, Nc=5.0)
    except TypeError:
        pass
    else:
        raise AssertionError("The MPC horizons must be integers.")

    # Test that the horizons must be positive
    try:
        mpc.set_horizons(Np=-10, Nc=-5)
    except ValueError:
        pass
    else:
        raise AssertionError("The MPC horizons must be positive.")


def test_set_bounds():
    """
    Test setting the bounds of the MPC optimization.
    """

    # Test flat voltage bounds
    v_min_pu, v_max_pu = MPC_controller.v_bound(
        bus=None, Np=3, iter=2, half_db_pu=0.01, v_set_pu=1.01
    )

    assert v_min_pu.shape == (3, 1), "The shape of v_min_pu is incorrect."
    assert v_max_pu.shape == (3, 1), "The shape of v_max_pu is incorrect."

    assert np.allclose(
        v_min_pu, np.array([[1.0], [1.0], [1.0]])
    ), "The lower voltage bounds are incorrect."

    assert np.allclose(
        v_max_pu, np.array([[1.02], [1.02], [1.02]])
    ), "The upper voltage bounds are incorrect."

    # Test power bounds from the DERs
    p_min_pu, p_max_pu, dp_min_pu, dp_max_pu = MPC_controller.power_bound(
        bus=None,
        Nc=4,
        iter=2,
    )

    assert p_min_pu.shape == (4, 1), "The shape of p_min_pu is incorrect."
    assert p_max_pu.shape == (4, 1), "The shape of p_max_pu is incorrect."
    assert dp_min_pu.shape == (4, 1), "The shape of dp_min_pu is incorrect."
    assert dp_max_pu.shape == (4, 1), "The shape of dp_max_pu is incorrect."

    assert np.allclose(
        p_min_pu, np.array([[-1e6], [-1e6], [-1e6], [-1e6]])
    ), "The lower power bounds are incorrect."
    assert np.allclose(
        p_max_pu, np.array([[1e6], [1e6], [1e6], [1e6]])
    ), "The upper power bounds are incorrect."
    assert np.allclose(
        dp_min_pu, np.array([[-1e6], [-1e6], [-1e6], [-1e6]])
    ), "The lower power increment bounds are incorrect."
    assert np.allclose(
        dp_max_pu, np.array([[1e6], [1e6], [1e6], [1e6]])
    ), "The upper power increment bounds are incorrect."

    # Create test controller
    mpc = MPC_controller()

    # Load nordic to make meaningful tests
    nordic = get_dynamic_nordic()
    nordic.add_controllers(controllers=[mpc])

    # Set controlled transformer
    mpc.add_controlled_transformers(
        transformers=[
            t.name
            for t in nordic.transformers
            if t.touches(location="CENTRAL")
        ]
    )

    S = len(mpc.controlled_transformers)
    assert S == 11, "The number of controlled transformers is incorrect."

    # Set horizons
    mpc.set_horizons(Np=5, Nc=5)

    # Set observed corridor
    mpc.add_observed_corridor(boundary_bus="4041", sending_buses=["4031"])
    mpc.add_observed_corridor(
        boundary_bus="4042", sending_buses=["4021", "4032"]
    )

    # Set default bounds
    mpc.set_bounds(NLI_min=0.1)

    # Make general assertions on the dimensions
    assert mpc.u_lower.shape == (
        3 * S * mpc.Nc,
        1,
    ), "The shape of u_lower is incorrect."
    assert mpc.u_upper.shape == (
        3 * S * mpc.Nc,
        1,
    ), "The shape of u_upper is incorrect."
    assert mpc.du_lower.shape == (
        3 * S * mpc.Nc,
        1,
    ), "The shape of du_lower is incorrect."
    assert mpc.du_upper.shape == (
        3 * S * mpc.Nc,
        1,
    ), "The shape of du_upper is incorrect."

    assert mpc.VT_lower.shape == (
        S * mpc.Np,
        1,
    ), "The shape of VT_lower is incorrect."
    assert mpc.VT_upper.shape == (
        S * mpc.Np,
        1,
    ), "The shape of VT_upper is incorrect."
    assert mpc.VD_lower.shape == (
        S * mpc.Np,
        1,
    ), "The shape of VD_lower is incorrect."
    assert mpc.VD_upper.shape == (
        S * mpc.Np,
        1,
    ), "The shape of VD_upper is incorrect."

    B = len(mpc.observed_corridors)

    assert mpc.NLI_lower.shape == (
        B * mpc.Np,
        1,
    ), "The shape of NLI_lower is incorrect."
    assert mpc.NLI_upper.shape == (
        B * mpc.Np,
        1,
    ), "The shape of NLI_upper is incorrect."

    # Test bounds on tap ratios
    for control_index in range(mpc.Nc):
        r_lower = mpc.u_lower[
            control_index * 3 * S * mpc.Nc : control_index * 3 * S * mpc.Nc
            + S,
            0,
        ]
        r_upper = mpc.u_upper[
            control_index * 3 * S * mpc.Nc : control_index * 3 * S * mpc.Nc
            + S,
            0,
        ]
        dr_lower = mpc.du_lower[
            control_index * 3 * S * mpc.Nc : control_index * 3 * S * mpc.Nc
            + S,
            0,
        ]
        dr_upper = mpc.du_upper[
            control_index * 3 * S * mpc.Nc : control_index * 3 * S * mpc.Nc
            + S,
            0,
        ]

        assert np.allclose(r_lower, 0.88 * np.ones((S, 1)))
        assert np.allclose(r_upper, 1.20 * np.ones((S, 1)))
        assert np.allclose(dr_lower, -0.01 * np.ones((S, 1)))
        assert np.allclose(dr_upper, 0.01 * np.ones((S, 1)))

    # Test bounds on power injections with default bounds
    for control_index in range(mpc.Nc):
        p_lower = mpc.u_lower[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]
        p_upper = mpc.u_upper[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]
        dp_lower = mpc.du_lower[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]
        dp_upper = mpc.du_upper[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]

        assert np.allclose(p_lower, -1e6 * np.ones((S, 1)))
        assert np.allclose(p_upper, 1e6 * np.ones((S, 1)))
        assert np.allclose(dp_lower, -1e6 * np.ones((S, 1)))
        assert np.allclose(dp_upper, 1e6 * np.ones((S, 1)))

        q_lower = mpc.u_lower[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]
        q_upper = mpc.u_upper[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]
        dq_lower = mpc.du_lower[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]
        dq_upper = mpc.du_upper[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]

        assert np.allclose(q_lower, -1e6 * np.ones((S, 1)))
        assert np.allclose(q_upper, 1e6 * np.ones((S, 1)))
        assert np.allclose(dq_lower, -1e6 * np.ones((S, 1)))
        assert np.allclose(dq_upper, 1e6 * np.ones((S, 1)))

    # Test bounds on the NLI
    for prediction_index in range(mpc.Np):
        nli_lower = mpc.NLI_lower[
            prediction_index * B : prediction_index * B + B, 0
        ]
        nli_upper = mpc.NLI_upper[
            prediction_index * B : prediction_index * B + B, 0
        ]

        assert np.allclose(nli_lower, 0.1 * np.ones((B, 1)))
        assert np.allclose(nli_upper, 1e6 * np.ones((B, 1)))

    def P_fun(bus: records.Bus, Nc: int, iter: int) -> np.ndarray:
        """
        Custom function for power bound.
        """

        p_min_pu = np.zeros((Nc, 1))
        p_max_pu = np.ones((Nc, 1))
        dp_min_pu = -0.1 * np.ones((Nc, 1))
        dp_max_pu = 0.1 * np.ones((Nc, 1))

        return p_min_pu, p_max_pu, dp_min_pu, dp_max_pu

    def VT_fun(bus: records.Bus, Np: int, iter: int) -> np.ndarray:
        """
        Custom function for transmission voltage bound.
        """

        v_min_pu = np.ones((Np, 1))
        v_max_pu = 1.1 * np.ones((Np, 1))

        return v_min_pu, v_max_pu

    def VD_fun(bus: records.Bus, Np: int, iter: int) -> np.ndarray:
        """
        Custom function for distribution voltage bound.
        """

        v_min_pu = 1.025 * np.ones((Np, 1))
        v_max_pu = 1.075 * np.ones((Np, 1))

        return v_min_pu, v_max_pu

    # Reset bounds
    mpc.set_bounds(
        NLI_min=0,
        P_fun=P_fun,
        Q_fun=P_fun,
        VT_fun=VT_fun,
        VD_fun=VD_fun,
    )

    # Test bounds on voltages
    for prediction_index in range(mpc.Np):
        vt_lower = mpc.VT_lower[
            prediction_index * S : prediction_index * S + S, 0
        ]
        vt_upper = mpc.VT_upper[
            prediction_index * S : prediction_index * S + S, 0
        ]

        assert np.allclose(vt_lower, 1.0 * np.ones((S, 1)))
        assert np.allclose(vt_upper, 1.1 * np.ones((S, 1)))

        vd_lower = mpc.VD_lower[
            prediction_index * S : prediction_index * S + S, 0
        ]
        vd_upper = mpc.VD_upper[
            prediction_index * S : prediction_index * S + S, 0
        ]

        assert np.allclose(vd_lower, 1.025 * np.ones((S, 1)))
        assert np.allclose(vd_upper, 1.075 * np.ones((S, 1)))

    # Test bounds on power injections
    for control_index in range(mpc.Nc):
        p_lower = mpc.u_lower[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]
        p_upper = mpc.u_upper[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]
        dp_lower = mpc.du_lower[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]
        dp_upper = mpc.du_upper[
            control_index * 3 * S * mpc.Nc
            + S : control_index * 3 * S * mpc.Nc
            + 2 * S,
            0,
        ]

        assert np.allclose(p_lower, 0.0 * np.ones((S, 1)))
        assert np.allclose(p_upper, 1.0 * np.ones((S, 1)))
        assert np.allclose(dp_lower, -0.1 * np.ones((S, 1)))
        assert np.allclose(dp_upper, 0.1 * np.ones((S, 1)))

        q_lower = mpc.u_lower[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]
        q_upper = mpc.u_upper[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]
        dq_lower = mpc.du_lower[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]
        dq_upper = mpc.du_upper[
            control_index * 3 * S * mpc.Nc
            + 2 * S : control_index * 3 * S * mpc.Nc
            + 3 * S,
            0,
        ]

        assert np.allclose(q_lower, 0.0 * np.ones((S, 1)))
        assert np.allclose(q_upper, 1.0 * np.ones((S, 1)))
        assert np.allclose(dq_lower, -0.1 * np.ones((S, 1)))
        assert np.allclose(dq_upper, 0.1 * np.ones((S, 1)))


def test_define_setpoints():
    """
    Test definition of setpoints for all inputs u.
    """

    # Create test controller and load nordic
    nordic = get_dynamic_nordic()
    mpc = MPC_controller()
    nordic.add_controllers(controllers=[mpc])

    # Set controlled transformers
    mpc.add_controlled_transformers(
        transformers=[
            t.name
            for t in nordic.transformers
            if t.touches(location="CENTRAL")
        ]
    )

    # Set horizons
    mpc.set_horizons(Np=5, Nc=5)

    # Set default setpoints
    mpc.define_setpoints()

    S = len(mpc.controlled_transformers)

    # Make general assertions on the dimensions
    assert mpc.u_star.shape == (
        3 * S * mpc.Nc,
        1,
    ), "The shape of u_star is incorrect."

    # Test setpoint values
    assert np.allclose(
        mpc.u_star, np.zeros((3 * S * mpc.Nc, 1))
    ), "The default setpoints are incorrect."


def test_set_weights():
    """
    Test setting of weights for the cost function.
    """

    # Create test controller and load nordic
    nordic = get_dynamic_nordic()
    mpc = MPC_controller()
    nordic.add_controllers(controllers=[mpc])

    # Set controlled transformers
    mpc.add_controlled_transformers(
        transformers=[
            t.name
            for t in nordic.transformers
            if t.touches(location="CENTRAL")
        ]
    )

    S = len(mpc.controlled_transformers)

    # Set horizons
    mpc.set_horizons(Np=5, Nc=5)

    # Set default weights
    mpc.set_weights()

    # Make general assertions on the dimensions
    assert mpc.R1.shape == (
        3 * S * mpc.Nc,
        3 * S * mpc.Nc,
    ), "The shape of R1 is incorrect."
    assert np.isclose(
        np.sum(mpc.R1), 0
    ), "The sum of R1 should be 0 since changes are not penalized."
    assert mpc.R2.shape == (
        3 * S * mpc.Nc,
        3 * S * mpc.Nc,
    ), "The shape of R2 is incorrect."
    assert np.isclose(
        np.sum(mpc.R2), 2 * S * mpc.Nc
    ), "The sum of R2 should be equal to 2*S*Nc."
    assert mpc.S.shape == (4 * S, 4 * S), "The shape of S is incorrect."
    assert np.isclose(np.sum(mpc.S), 4 * S)


def test_MPC_Controller_str():
    """
    Test initialization of the MPC_Controller class.
    """

    # Create test controller and load nordic
    nordic = get_dynamic_nordic()
    mpc = MPC_controller()
    nordic.add_controllers(controllers=[mpc])

    # Set controlled transformers
    mpc.add_controlled_transformers(
        transformers=[
            t.name
            for t in nordic.transformers
            if t.touches(location="CENTRAL")
        ]
    )

    # Add observed corridor
    mpc.add_observed_corridor(boundary_bus="4041", sending_buses=["4031"])
    mpc.add_observed_corridor(
        boundary_bus="4042", sending_buses=["4021", "4032"]
    )

    # Set horizons
    mpc.set_horizons(Np=5, Nc=5)

    # Set period
    mpc.set_period(10)

    # Set default setpoints
    mpc.define_setpoints()

    # Set default weights
    mpc.set_weights()

    # Set default bounds
    mpc.set_bounds(NLI_min=0.1)

    # Test string representation
    with open("test_control_str.txt", "w") as f:
        f.write(str(mpc))


def get_nordic_and_mpc(controlled_transformers: list[str] = None):
    """
    Create test and load nordic.
    """

    # Create test controller and load nordic
    nordic = get_dynamic_nordic()

    # Make the loads voltage sensitive
    for inj in nordic.injectors:
        if isinstance(inj, records.Load):
            inj.make_voltage_sensitive(alpha=1, beta=2)

    mpc = MPC_controller()
    nordic.add_controllers(controllers=[mpc])

    # Set controlled transformers
    if controlled_transformers is None:
        mpc.add_controlled_transformers(
            transformers=[
                t.name
                for t in nordic.transformers
                if t.touches(location="CENTRAL")
            ]
        )
    else:
        mpc.add_controlled_transformers(transformers=controlled_transformers)

    # Add observed corridor
    mpc.add_observed_corridor(boundary_bus="4041", sending_buses=["4031"])
    mpc.add_observed_corridor(
        boundary_bus="4042", sending_buses=["4021", "4032"]
    )

    # Set horizons (different to test harsher conditions)
    mpc.set_horizons(Np=5, Nc=3)

    # Set period
    mpc.set_period(10)

    # Set default setpoints
    mpc.define_setpoints()

    # Set default bounds
    mpc.set_bounds(NLI_min=0.1)

    # Set default weights
    mpc.set_weights()

    return nordic, mpc


def test_build_structural_matrices():
    """
    Test construction of structural matrices for the MPC optimization.
    """

    nordic, mpc = get_nordic_and_mpc()

    mpc.build_structural_matrices()

    S = len(mpc.controlled_transformers)
    B = len(mpc.observed_corridors)

    # Make general assertions on the dimensions
    assert mpc.C1.shape == (
        3 * S * mpc.Nc,
        3 * S,
    ), "The shape of C1 is incorrect."
    assert mpc.C2.shape == (
        3 * S * mpc.Nc,
        3 * S * mpc.Nc,
    ), "The shape of C2 is incorrect."
    assert mpc.F_N.shape == (B * mpc.Np, B), "The shape of F_N is incorrect."
    assert (
        mpc.F_VT.shape == mpc.F_VD.shape == (S * mpc.Np, S)
    ), "The shape of F_VT and F_VD is incorrect."
    assert mpc.A1.shape == (S * mpc.Np, 4 * S), "The shape of A1 is incorrect."
    assert mpc.A2.shape == (S * mpc.Np, 4 * S), "The shape of A2 is incorrect."
    assert mpc.A3.shape == (S * mpc.Np, 4 * S), "The shape of A3 is incorrect."
    assert mpc.A4.shape == (S * mpc.Np, 4 * S), "The shape of A4 is incorrect."

    # Assert content of C1
    assert np.isclose(np.sum(mpc.C1), 3 * S * mpc.Nc)

    # Assert content of C2
    assert np.isclose(np.sum(mpc.C2), mpc.Nc * (mpc.Nc + 1) / 2 * 3 * S)

    # Assert content of F_N
    assert np.isclose(np.sum(mpc.F_N), B * mpc.Np)

    # Assert content of F_VT
    assert np.isclose(np.sum(mpc.F_VT), S * mpc.Np)

    # Assert content of F_VD
    assert np.isclose(np.sum(mpc.F_VD), S * mpc.Np)

    # Assert content of A1, A2, ...
    for offset, attr in enumerate(["A1", "A2", "A3", "A4"]):
        assert np.allclose(
            getattr(mpc, attr)[:, offset * S : (offset + 1) * S],
            np.vstack([np.eye(S) for _ in range(mpc.Np)]),
        )
        assert np.isclose(np.sum(getattr(mpc, attr)), S * mpc.Np)


def test_generate_twin():
    """
    Test addition of twins (to avoid trouble with the original system).
    """

    nordic, mpc = get_nordic_and_mpc()

    assert (
        nordic.twin is None
    ), "The nordic twin should be None before generating it."

    mpc.generate_twin()

    assert (
        nordic.twin is not None
    ), "The nordic twin should not be None after generating it."

    # If no parameter randomizations were passed, the twin should converge to
    # the same power flow solution as the original system.
    nordic.run_pf()
    nordic.twin.run_pf()

    assert (
        nordic.generate_table() == nordic.twin.generate_table()
    ), "The twin should converge to the same power flow as the original system."

    # If a parameter randomization is specified, the strings should be different,
    # but all voltages, for instance, should be similar.

    def load_randomization(system: pf_dynamic.System) -> pf_dynamic.System:
        for line in system.lines:
            line.R_pu += np.random.normal(loc=0, scale=0.1 * line.R_pu)
            line.X_pu += np.random.normal(loc=0, scale=0.1 * line.X_pu)

    mpc.generate_twin(parameter_randomizations=[load_randomization])

    nordic.run_pf()
    nordic.twin.run_pf()

    assert (
        nordic.generate_table() != nordic.twin.generate_table()
    ), "The twin should not converge to the same power flow as the original system."

    # However...
    for nordic_bus, twin_bus in zip(nordic.buses, nordic.twin.buses):
        assert np.allclose(
            nordic_bus.V_pu, twin_bus.V_pu, atol=0.01
        ), "The twin should converge to similar voltages as the original system."


def test_update_twin():
    """
    Test updating of twins.
    """

    # Not done because implementation is straightforward and it would require
    # implementing a RAMSES imitator. Exiting...


class RAMSESImitator:
    """
    A RAMSES imitator with perfect measurements.
    """

    def __init__(self) -> None:
        self.system = get_nordic_and_mpc()[0]
        self.system.get_branches_between(bus_name_1="4032", bus_name_2="4044")[
            0
        ].disconnect()
        self.system.run_pf()

    def getBusVolt(self, busNames: list[str]) -> list[float]:
        return [self.system.get_bus(name=bus).V_pu for bus in busNames]

    def getObs(
        self, comp_type: list[str], comp_name: list[str], obs_name: list[str]
    ) -> list[float]:
        if comp_type != len(comp_type) * ["DCTL"]:
            raise NotImplementedError("Only DCTLs are implemented.")

        # Find the associated transformer
        ratios = []
        for name in comp_name:
            transformer = next(
                transformer
                for transformer in self.system.transformers
                if transformer.name == name
            )
            ratios.append(transformer.n_pu)

        return ratios

    def getBranchPow(self, branchName: list[str]) -> list[float]:
        flows = []
        for name in branchName:
            branch = next(
                branch
                for branch in self.system.branches
                if branch.name == name
            )
            values = branch.get_pu_flows()[2:4] + branch.get_pu_flows()[:2]
            flows.append(values)

        return flows


def test_get_derivatives():
    """
    Test retrieval of all partial derivatives.
    """

    # For simplicity, only include two transformers
    controlled_transformers = [
        "1-1041",
        "2-1042",
        "3-1043",
    ]  # , "4-1044", "5-1045"]

    nordic, mpc = get_nordic_and_mpc(
        # Comment if you want to test with some tranformers
        controlled_transformers=controlled_transformers
    )

    nordic.get_branches_between(bus_name_1="4032", bus_name_2="4044")[
        0
    ].disconnect()
    nordic.run_pf()

    # The simulator will provide perfect measurements.
    nordic.ram = RAMSESImitator()

    # Test that the measurements are correct:
    for bus in nordic.buses:
        assert np.isclose(
            bus.V_pu, nordic.ram.getBusVolt([bus.name])[0]
        ), "The bus voltages are incorrect."
    for transformer in nordic.transformers:
        assert np.isclose(
            transformer.n_pu,
            nordic.ram.getObs(["DCTL"], [transformer.name], [""])[0],
        ), "The transformer ratios are incorrect."

    mpc.generate_twin()
    mpc.update_twin()
    mpc.update_derivatives()

    # Make assertions about the dimensions
    assert mpc.partial_u_N.shape == (
        len(mpc.observed_corridors),
        3 * len(mpc.controlled_transformers),
    ), "The shape of partial_u_N is incorrect."
    assert mpc.partial_u_VT.shape == (
        len(mpc.controlled_transformers),
        3 * len(mpc.controlled_transformers),
    ), "The shape of partial_u_VT is incorrect."
    assert mpc.partial_u_VD.shape == (
        len(mpc.controlled_transformers),
        3 * len(mpc.controlled_transformers),
    ), "The shape of partial_u_VD is incorrect."

    # Make assertions about the signs
    # NLIs
    assert np.all(
        mpc.partial_u_N[:, : len(mpc.controlled_transformers)] > 0
    ), "NLI must increase with a tap ratio increase."
    assert np.all(
        mpc.partial_u_N[
            :,
            len(mpc.controlled_transformers) : 2
            * len(mpc.controlled_transformers),
        ]
        < 0
    ), "NLI must decrease with an active load increase."
    assert np.all(
        mpc.partial_u_N[
            :,
            2
            * len(mpc.controlled_transformers) : 3
            * len(mpc.controlled_transformers),
        ]
        < 1e-1
    ), "NLI must essentially decrease with a reactive load increase."

    # VTs
    assert np.all(
        mpc.partial_u_VT[:, : len(mpc.controlled_transformers)] > 0
    ), "VT must increase with a tap ratio increase."
    assert np.all(
        mpc.partial_u_VT[
            :,
            len(mpc.controlled_transformers) : 3
            * len(mpc.controlled_transformers),
        ]
        < 0
    ), "VT must decrease with any load increase."

    # VDs
    assert np.all(
        np.diag(mpc.partial_u_VD[:, : len(mpc.controlled_transformers)]) < 0
    ), "VD must decrease with an OWN tap ratio increase."
    T = len(mpc.controlled_transformers)
    for row_no in range(T):
        for col_no in range(T):
            if row_no != col_no:
                assert (
                    mpc.partial_u_VD[row_no, col_no] > 0
                ), "VD must increase with any OTHER tap ratio increase."

    # Making assertions about the numbers using the Jacobian is not possible
    # since the Jacobian assumes changes in the entire load at a bus, whereas
    # this program computes it with respect to the injections from the DERs
    # only.


def test_build_sensitivities():
    """
    Test construction of sensitivity matrices.
    """

    # For simplicity, only include two transformers
    controlled_transformers = [
        "1-1041",
        "2-1042",
        "3-1043",
    ]  # , "4-1044", "5-1045"]

    nordic, mpc = get_nordic_and_mpc(
        # Comment if you want to test with some tranformers
        controlled_transformers=controlled_transformers
    )

    nordic.get_branches_between(bus_name_1="4032", bus_name_2="4044")[
        0
    ].disconnect()
    nordic.run_pf()

    # The simulator will provide perfect measurements.
    nordic.ram = RAMSESImitator()

    mpc.generate_twin()
    mpc.update_twin()
    mpc.build_sensitivities()

    # Make assertions about the dimensions
    assert mpc.D_u_N.shape == (
        mpc.Np * len(mpc.observed_corridors),
        3 * len(mpc.controlled_transformers) * mpc.Nc,
    ), "The shape of D_u_N is incorrect."

    assert mpc.D_u_VT.shape == (
        mpc.Np * len(mpc.controlled_transformers),
        3 * len(mpc.controlled_transformers) * mpc.Nc,
    ), "The shape of D_u_VT is incorrect."

    assert mpc.D_u_VD.shape == (
        mpc.Np * len(mpc.controlled_transformers),
        3 * len(mpc.controlled_transformers) * mpc.Nc,
    ), "The shape of D_u_VD is incorrect."


def test_get_measurements():
    """
    Test retrieval of measurements in RAMSES.
    """

    # Will work better if tested in RAMSES.


def test_update_measurement_dependent_matrices():
    """
    Test updating of measurement-dependent matrices (feedback loop).
    """

    # Will work better if tested in RAMSES.


def test_solve_optimization():
    """
    Test solving of the MPC optimization.
    """

    # Will work better if tested in RAMSES.


def test_get_actions():
    """
    Test retrieval of the MPC actions, needed in any Controller class.
    """

    # Will work better if tested in RAMSES.


def test_Coordinator_init_and_str():
    pass


def test_increment_request():
    pass


def test_get_actions():
    pass


def test_interpolate():
    pass


def test_get_sigma():
    pass


def test_DERA_Controller():
    pass


if __name__ == "__main__":
    # Tests of the Controller class
    test_MPC_Controller_init()
    test_set_period()
    test_add_observed_corridor()
    test_add_controlled_transformer()
    test_set_horizons()
    test_set_bounds()
    test_define_setpoints()
    test_set_weights()
    test_MPC_Controller_str()
    test_build_structural_matrices()
    test_generate_twin()
    test_update_twin()
    test_get_derivatives()
    test_build_sensitivities()
    test_get_measurements()
    test_update_measurement_dependent_matrices()
    test_solve_optimization()
    test_get_actions()

    # Tests of the Coordinator class
    test_Coordinator_init_and_str()
    test_increment_request()
    test_get_actions()
    test_interpolate()
    test_get_sigma()

    # Tests of the DERA_Controller class
    test_DERA_Controller()

    print("Module 'control' passed all tests!")
