"""
Test functions for the "records" module.
"""

from records import *
import numpy as np


def test_Parameter():
    # Test alphanumeric parameters
    p = Parameter(name="Test parameter", value="Test value", digits=4)
    assert p.name == "Test parameter"
    assert p.value == "Test value"
    assert p.digits == 4, "Significant digits are wrong."
    assert str(p) == "Test value"

    # Test numeric parameters with default digits
    p = Parameter(name="Test parameter", value=3.14159265)
    assert str(p) == "3.14159265"

    # Test numeric parameters with specified digits
    p = Parameter(name="Test parameter", value=0, digits=4)
    assert str(p) == "0.0"

    p = Parameter(name="Test parameter", value=-3.141592, digits=4)
    assert str(p) == "-3.142"

    p = Parameter(name="Test parameter", value=3.141592, digits=4)
    assert str(p) == "3.142"

    p = Parameter(name="Test parameter", value=1.53293e4, digits=4)
    assert str(p) == "15329.0"

    p = Parameter(name="Test parameter", value=1.53293e-4, digits=4)
    assert str(p) == "0.0001532"

    p = Parameter(name="Test parameter", value=1, digits=4)
    assert str(p) == "1.0"

    p = Parameter(name="Test parameter", value=-1, digits=4)
    assert str(p) == "-1.0"


def test_Record():
    class TestRecord(Record):
        """A test record."""

        def __init__(
            self, prefix: str, name: str, example_value: float
        ) -> None:
            self.prefix = prefix
            self.name = name
            self.example_value = example_value
            self.ind_offset = "  "

        def get_pars(self) -> list[Parameter]:
            return [
                Parameter(name="Example", value=self.example_value, digits=4)
            ]

    record = TestRecord(prefix="TR", name="Example", example_value=3.14159265)

    assert str(record) == "  TR Example 3.142;"


def test_Injector():
    inj = Injector()
    assert inj.prefix == ""


def test_DCTL():
    """
    Nothing to test.
    """

    pass


# Define example bus for subsequent tests
b = Bus(
    V_pu=1.0,
    theta_radians=np.pi / 4,
    PL_pu=1.0,
    QL_pu=0.1,
    G_pu=0.5,
    B_pu=0.5,
    base_kV=230,
    bus_type="PQ",
    V_min_pu=0.95,
    V_max_pu=1.05,
    name="ExampleBus",
)


def test_Bus():
    assert b.prefix == "BUS"
    assert b.V_pu == 1.0
    assert np.isclose(b.theta_radians, np.pi / 4)
    assert b.PL_pu == 1.0
    assert b.QL_pu == 0.1
    assert b.G_pu == 0.5
    assert b.B_pu == 0.5
    assert b.base_kV == 230
    assert b.bus_type == "PQ"

    assert np.isclose(b.get_phasor_V(), (1 + 1j) / np.sqrt(2))
    assert str(b.get_pars()[0]) == "230.0"

    # Test change of base
    b.change_base_power(Sb_old=100, Sb_new=200)

    assert np.isclose(b.PL_pu, 0.5) and np.isclose(b.allocated_PL_pu, 0.5)
    assert np.isclose(b.QL_pu, 0.05) and np.isclose(b.allocated_QL_pu, 0.05)

    assert np.isclose(b.G_pu, 0.25) and np.isclose(b.B_pu, 0.25)


def test_Thevenin():
    eq = Thevenin(name="ExampleThevenin", bus=b)
    assert (
        str(eq) == "INJEC THEVEQ ExampleThevenin ExampleBus "
        "1.0 1.0 0.0 0.0 10000000.0;"
    )


def test_Frequency():
    f = Frequency(fnom=60)
    assert str(f) == "FNOM 60.0;"


def test_InitialVoltage():
    v = InitialVoltage(bus=b)
    assert str(v) == "LFRESV ExampleBus 1.0 0.78539816;"


# Define example machine for subsequent tests
sm = SYNC_MACH(
    name="g1",
    bus=b,
    Snom_MVA=800,
    Pnom_MW=760,
    H=3,
    D=0,
    IBRATIO=0.95,
    model="XT",
    Xl=0.15,
    Xd=1.1,
    Xdp=0.25,
    Xdpp=0.2,
    Xq=0.7,
    Xqp="*",
    Xqpp=0.2,
    m=0.1,
    n=6.0257,
    Ra=0,
    Td0p=5.00,
    Td0pp=0.05,
    Tq0p="*",
    Tq0pp=0.1,
)


def test_SYNC_MACH():
    assert str(sm) == (
        "SYNC_MACH g1 ExampleBus 1.0 1.0 0.0 0.0 800.0 760.0 3.0 0.0 "
        "0.95 XT 0.15 1.1\n"
        "    0.25 0.2 0.7 * 0.2 0.1 6.0257 0.0 5.0 0.05 * 0.1"
    )


def test_EXC():
    """
    Nothing to test.
    """

    pass


# Define example exciter for subsequent tests
exc = GENERIC1(
    iflim=1.8991,
    d=-0.1,
    f=0.0,
    s=1.0,
    k1=100.0,
    k2=-1.0,
    L1=-11.0,
    L2=10.0,
    G=70.0,
    Ta=10.0,
    Tb=20.0,
    Te=0.1,
    L3=0.0,
    L4=4.0,
    SPEEDIN=1.0,
    KPSS=75.0,
    Tw=15.0,
    T1=0.2,
    T2=0.01,
    T3=0.2,
    T4=0.01,
    DVMIN=-0.1,
    DVMAX=0.1,
)


def test_GENERIC1():
    assert str(exc) == (
        "    EXC GENERIC1 1.8991 -0.1 0.0 1.0 100.0 -1.0 -11.0 "
        "10.0 70.0 10.0 20.0 0.1\n"
        "        0.0 4.0 1.0 75.0 15.0 0.2 0.01 0.2 0.01 -0.1 0.1"
    )


def test_CONSTANT():
    gov = CONSTANT()
    assert str(gov) == "    TOR CONSTANT;"


# Define example governor for subsequent tests
gov = HYDRO_GENERIC1(
    sigma=0.04,
    Tp=2.0,
    Qv=0.0,
    Kp=2.0,
    Ki=0.4,
    Tsm=0.2,
    limzdot=0.1,
    Tw=1.0,
)


def test_HYDRO_GENERIC1():
    assert (
        str(gov) == "    TOR HYDRO_GENERIC1 0.04 2.0 0.0 2.0 0.4 0.2 0.1 1.0;"
    )


b_gen = PV(
    V_pu=1.0,
    theta_radians=np.pi / 4,
    PL_pu=1.0,
    QL_pu=0.0,
    G_pu=0.0,
    B_pu=0.0,
    base_kV=230,
    bus_type="PV",
    V_min_pu=0.95,
    V_max_pu=1.05,
    name="ExampleGenerationBus",
)


def test_Generator():
    gen = Generator(PG_MW=100, bus=b_gen, name="ExampleGenerator")

    gen.machine = sm
    gen.exciter = exc
    gen.governor = gov

    assert gen.in_operation
    gen.trip()
    assert not gen.in_operation
    gen.trip_back()
    assert gen.in_operation


def test_Shunt():
    sh = Shunt(name="ExampleShunt", bus=b, Mvar_at_Vnom=100)

    assert str(sh) == "SHUNT ExampleShunt ExampleBus 100.0 1.0;"
    assert np.isclose(sh.get_Q(), 100)
    assert np.isclose(sh.get_dQ_dV(), 2 * sh.get_Q() * sh.bus.V_pu)


def test_Load():
    load = Load(name="ExampleLoad", bus=b, P0_MW=100, Q0_Mvar=50)

    assert str(load) == (
        "INJEC LOAD ExampleLoad ExampleBus 0.0 0.0 -100.0 -50.0 "
        "0.0 1.0 0.0 0.0 0.0 0.0\n    0.0 1.0 0.0 0.0 0.0 0.0;"
    )

    assert np.isclose(load.get_P(), -100)
    assert np.isclose(load.get_Q(), -50)
    assert np.isclose(load.get_dP_dV(), 0)
    assert np.isclose(load.get_dQ_dV(), 0)

    load.make_voltage_sensitive(alpha=2, beta=3)

    assert str(load) == (
        "INJEC LOAD ExampleLoad ExampleBus 0.0 0.0 -100.0 -50.0 "
        "0.0 1.0 2.0 0.0 0.0 0.0\n    0.0 1.0 3.0 0.0 0.0 0.0;"
    )

    assert np.isclose(load.get_P(), -100)
    assert np.isclose(load.get_Q(), -50)
    assert np.isclose(load.get_dP_dV(), -2 * load.P0_MW * load.bus.V_pu)
    assert np.isclose(load.get_dQ_dV(), -3 * load.Q0_Mvar * load.bus.V_pu)


# Define buses for subsequent tests
b1 = Bus(
    V_pu=1.0,
    theta_radians=np.pi / 4,
    PL_pu=1.0,
    QL_pu=0.0,
    G_pu=0.0,
    B_pu=0.0,
    base_kV=230,
    bus_type="PQ",
    V_min_pu=0.95,
    V_max_pu=1.05,
    name="Bus1",
)

b2 = Bus(
    V_pu=1.0,
    theta_radians=np.pi / 4,
    PL_pu=1.0,
    QL_pu=0.0,
    G_pu=0.0,
    B_pu=0.0,
    base_kV=230,
    bus_type="PQ",
    V_min_pu=0.95,
    V_max_pu=1.05,
    name="Bus2",
)

b3 = Bus(
    V_pu=1.0,
    theta_radians=np.pi / 4,
    PL_pu=1.0,
    QL_pu=0.0,
    G_pu=0.0,
    B_pu=0.0,
    base_kV=13.8,
    bus_type="PQ",
    V_min_pu=0.95,
    V_max_pu=1.05,
    name="Bus3",
)


def test_Branch_and_OLTC():
    class MySystem:
        """
        A Sustem() class implementing the bare minimum to test Branch().
        """

        base_MVA: float = 100
        slack: None = None

        def pu2ohm(self, pu: float, kV: float) -> float:
            return pu * kV**2 / self.base_MVA

        def pu2mho(self, pu: float, kV: float) -> float:
            return 0 if pu == 0 else 1 / self.pu2ohm(1 / pu, kV)

        def update_connectivity(self, reference_bus) -> None:
            pass

    br = Branch(
        from_bus=b1,
        to_bus=b2,
        X_pu=0.3,
        R_pu=0.1,
        from_Y_pu=0.01 + 0.02j,
        to_Y_pu=0.01 + 0.02j,
        n_pu=1.0,
        branch_type="Line",
        Snom_MVA=250.0,
        name="Bus1-Bus2",
        sys=MySystem(),
    )
    b1.location = "Area1"
    b2.location = "Area2"

    # Test conversion to string
    assert str(br) == (
        f"LINE Bus1-Bus2 Bus1 Bus2 52.9 158.7 37.8072 " f"250.0 1.0 1.0;"
    )

    # Test adjacency to location
    assert br.touches(location="Area1")
    assert br.touches(location="Area2")
    assert not br.touches(location="Area3")

    # Test connection and disconnection
    assert br.in_operation
    br.disconnect()
    assert not br.in_operation
    br.connect()
    assert br.in_operation

    # Test fetching of HV/LV buses
    assert br.get_HV_bus() is br.to_bus
    assert br.get_LV_bus() is br.to_bus

    # Test creation of LTCs
    br.add_OLTC(
        positions_up=10,
        positions_down=20,
        step_pu=0.01,
        v_setpoint_pu=1.0,
        half_db_pu=0.01,
    )

    # Test LTC parameters
    assert br.has_OLTC
    assert br.OLTC.positions_up == 10
    assert br.OLTC.positions_down == 20
    assert br.OLTC.step_pu == 0.01
    assert br.OLTC.v_setpoint_pu == 1.0
    assert br.OLTC.half_db_pu == 0.01

    # Test inferred parameters
    assert br.OLTC.pos == 0
    assert br.OLTC.nmax_pu == 1.1
    assert br.OLTC.nmin_pu == 0.8
    assert br.OLTC.controlled_bus is br.to_bus
    assert br.OLTC.OLTC_controller is None

    # Test increase voltages
    br.OLTC.increase_voltages()
    assert br.n_pu == 0.99
    assert br.OLTC.pos == -1

    # Test decrease voltages
    br.OLTC.reduce_voltages()
    assert br.n_pu == 1.0
    assert br.OLTC.pos == 0
    br.OLTC.reduce_voltages()
    assert br.n_pu == 1.01
    assert br.OLTC.pos == 1

    # Test natural action
    assert not br.OLTC.act()
    br.to_bus.V_pu = 0.985
    assert br.OLTC.act()
    assert br.OLTC.pos == 0
    assert br.n_pu == 1.0


def test_DERA():
    der = DERA(name="ExampleDERA", bus=b3, P0_MW=100, Q0_Mvar=50, Snom_MVA=200)

    assert str(der) == (
        "INJEC DER_A ExampleDERA Bus3 0.0 0.0 100.0 50.0 200.0 0.02 -0.0006 "
        "0.0006 20.0\n"
        "    0.0 -99.0 99.0 10.0 0.1 0.0 1.0 -99.0 99.0 1.0 0.0 1.0 1.0 5.0 "
        "0.02 1.0\n"
        "    -0.01 0.01 8.0 -1.0 1.0 0.02 1.2 0.942 0.16 1.03 0.16 0.6 0.16 "
        "0.45 0.16\n"
        "    1.15 0.16 1.2 0.16 0.1 0.3 0.02 2.0 0.8 1.0 0.02;"
    )


def test_INDMACH1():
    mac = INDMACH1(
        name="ExampleINDMACH1",
        bus=b,
        P0_MW=100,
        Q0_Mvar=50,
        Snom_MVA=200,
        RS=0.01,
        LLS=0.1,
        LSR=0.01,
        RR=0.01,
        LLR=0.1,
        H=3,
        A=0.0,
        B=0.0,
        LF=0.0,
    )

    assert str(mac) == (
        "INJEC INDMACH1 ExampleINDMACH1 ExampleBus 0.0 0.0 -100.0 -50.0 200.0 "
        "0.01 0.1\n"
        "    0.01 0.01 0.1 3.0 0.0 0.0 0.0;"
    )


if __name__ == "__main__":
    test_Parameter()
    test_Record()
    test_Injector()
    test_DCTL()
    test_Bus()
    test_Thevenin()
    test_Frequency()
    test_InitialVoltage()
    test_SYNC_MACH()
    test_EXC()
    test_GENERIC1()
    test_CONSTANT()
    test_HYDRO_GENERIC1()
    test_Generator()
    test_Shunt()
    test_Load()
    test_Branch_and_OLTC()
    test_DERA()
    test_INDMACH1()

    print("Module 'records' passed all tests!")
