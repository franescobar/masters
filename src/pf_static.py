"""
An object-oriented power-flow solver.

This program runs power-flow studies in networks of arbitrary size. It performs
slightly worse than well-established modules, such as pypower and pandapower,
but its syntax facilitates the creation of large-scale networks. Instead of
using indices to test for identity, look for neighbors, etc, the program
exploits object-oriented programming.

One salient feature of the solver is that it supports voltage-dependent loads.
The implementation uses the concept of an injector: an object connected to a PQ
bus whose active and reactive powers are arbitrary functions of the terminal
voltage. An injector can model devices such as smart inverters with P and Q
control.
"""


# Modules from the standard library
import copy
import bisect
import warnings
from collections.abc import Sequence, Container

# Modules from this repository
import records
import subsum
import utils

# Other modules
import numpy as np
import scipy.sparse
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import tabulate

# numpy's settings
np.set_printoptions(linewidth=np.inf)


class StaticSystem:
    """
    A representation of the power system.

    This class is the main interface between the user and the power-flow
    solver. It contains methods for adding buses, lines, transformers,
    generators, injectors, and controllers. It also contains methods for
    running the power-flow study and importing data from ARTERE.
    """

    def __init__(
        self, name: str = "", pu: bool = True, base_MVA: float = 100
    ) -> None:
        """
        Initialize a system called 'name'.

        The argument 'pu' determines whether the system parameters are given in
        per-unit or in SI units.
        """

        # User-defined attributes
        self.name = name
        self.pu = pu
        self.base_MVA = base_MVA

        # Simulation-related attributes
        self.status = "unsolved"  # status of the power-flow study

        # Bus-related attributes
        self.slack = None
        self.PQ_buses = []
        self.PV_buses = []
        self.non_slack_buses = []  # equal to PQ_buses + PV_buses
        self.buses = []  # all buses = [slack] + non_slack_buses

        # Branch-related attributes
        self.lines = []
        self.transformers = []
        self.branches = []  # all branches = lines + transformers

        # Element-related attributes
        self.generators = []  # synchronous generators connected to PV buses
        self.injectors = []  # injectors connected to PQ buses

        # Dictionaries for quick access
        self.bus_dict = {}
        self.line_dict = {}
        self.transformer_dict = {}
        self.gen_dict = {}
        self.inj_dict = {}
        self.bus_to_injectors = {}
        self.bus_to_generators = {}

    def ohm2pu(self, Z_ohm: complex, base_kV: float) -> complex:
        """
        Convert impedance from ohms to pu (in the system's base).
        """

        base_impedance = base_kV**2 / self.base_MVA

        return Z_ohm / base_impedance

    def pu2ohm(self, Z_pu: complex, base_kV: float) -> complex:
        """
        Convert impedance from pu (in the system's base) to ohms.
        """

        base_impedance = base_kV**2 / self.base_MVA

        return Z_pu * base_impedance

    def mho2pu(self, Y_mho: complex, base_kV: float) -> complex:
        """
        Convert admittance 'Y' from mhos (siemens) to pu (in system's base).
        """

        return (
            0
            if Y_mho == 0
            else 1 / self.ohm2pu(Z_ohm=1 / Y_mho, base_kV=base_kV)
        )

    def pu2mho(self, Y_pu: complex, base_kV) -> complex:
        """
        Convert admittance 'Y' from pu (in system's base) to mhos (siemens).
        """

        return (
            0 if Y_pu == 0 else 1 / self.pu2ohm(Z_pu=1 / Y_pu, base_kV=base_kV)
        )

    def store_bus(self, bus: records.Bus) -> None:
        """
        Store bus keeping self.buses sorted: slack -> PQ -> PV.
        """

        if bus.name in self.bus_dict:
            raise RuntimeError(f"Bus {bus.name} already exists.")

        # Add bus to the list of buses (irrespective of bus type)
        bisect.insort(self.buses, bus)

        # Classify bus in remaining containers
        if isinstance(bus, records.Slack):
            self.slack = bus
        elif isinstance(bus, records.PQ):
            bisect.insort(self.PQ_buses, bus)
            bisect.insort(self.non_slack_buses, bus)
        elif isinstance(bus, records.PV):
            bisect.insort(self.PV_buses, bus)
            bisect.insort(self.non_slack_buses, bus)

        # Store in dictionary for quick access to named elements
        self.bus_dict[bus.name] = bus

        # Initialize entries in dictionaries for quick access to injectors and
        # generators
        self.bus_to_injectors[bus] = []
        self.bus_to_generators[bus] = []

    def remove_bus(self, bus: records.Bus) -> None:
        """
        Remove bus from the system.
        """

        # Remove bus from the list of buses (irrespective of bus type)
        self.buses.remove(bus)

        # Remove bus depending on its type
        if isinstance(bus, records.Slack):
            self.slack = None
        elif isinstance(bus, records.PQ):
            self.PQ_buses.remove(bus)
            self.non_slack_buses.remove(bus)
        elif isinstance(bus, records.PV):
            self.PV_buses.remove(bus)
            self.non_slack_buses.remove(bus)

        # Remove bus from dictionary for quick access to named elements
        del self.bus_dict[bus.name]

        # Remove injectors and generators connected to bus
        for inj in self.bus_to_injectors[bus]:
            self.remove_injector(inj)

        for gen in self.bus_to_generators[bus]:
            self.remove_generator(gen)

        # Remove entries in dictionaries for quick access to injectors and
        # generators
        del self.bus_to_injectors[bus]
        del self.bus_to_generators[bus]

    def replace_bus(self, old_bus: records.Bus, new_bus: records.Bus) -> None:
        """
        Replace bus in the system.
        """

        # Keep the only piece of information that is required from the old bus
        injectors = self.bus_to_injectors[old_bus]
        generators = self.bus_to_generators[old_bus]

        # Remove the old bus and add the new one
        self.remove_bus(bus=old_bus)
        self.store_bus(bus=new_bus)

        # Rewrite the injectors and generators associated to the old bus
        # (since store_bus erases them)
        self.bus_to_injectors[new_bus] = injectors
        self.bus_to_generators[new_bus] = generators

        # Replace the old bus in branches
        for branch in self.branches:
            if branch.from_bus is old_bus:
                branch.from_bus = new_bus
            if branch.to_bus is old_bus:
                branch.to_bus = new_bus

        # Replace the old bus in generators
        for gen in self.generators:
            if gen.bus is old_bus:
                gen.bus = new_bus

        # Replace the old bus in injectors
        for inj in self.injectors:
            if inj.bus is old_bus:
                inj.bus = new_bus


    def store_branch(self, branch: records.Branch) -> None:
        """
        Store branch keeping self.branches sorted: lines -> transformer.
        """

        if (
            branch.name in self.line_dict
            or branch.name in self.transformer_dict
        ):
            raise RuntimeError(f"Branch {branch.name} already exists.")

        # Add branch to the list of branches (irrespective of branch type)
        bisect.insort(self.branches, branch)

        # Classify branch in remaining containers. The addition to the
        # dictionaries is done for quick access to named elements.
        if branch.branch_type == "Line":
            self.lines.append(branch)
            self.line_dict[branch.name] = branch
        elif branch.branch_type == "Transformer":
            self.transformers.append(branch)
            self.transformer_dict[branch.name] = branch

    def remove_branch(self, branch: records.Branch) -> None:
        """
        Remove branch from the system.
        """

        # Remove branch from the list of branches (irrespective of branch type)
        self.branches.remove(branch)

        # Remove branch depending on its type
        if branch.branch_type == "Line":
            self.lines.remove(branch)
            del self.line_dict[branch.name]
        elif branch.branch_type == "Transformer":
            self.transformers.remove(branch)
            del self.transformer_dict[branch.name]

    def store_generator(self, gen: records.Generator) -> None:
        """
        Add a (large) generator to a PV or slack bus.

        It's very important that the generated power is in MW.

        There's no need to test for connection to the generator's bus because
        this is an argument of the generator's constructor anyway. Testing was
        important for injectors because they could be implemented by the user
        outside of this program.
        """

        if gen.name in self.gen_dict:
            raise RuntimeError(f"Generator {gen.name} already exists.")

        self.gen_dict[gen.name] = gen
        bisect.insort(self.generators, gen)
        bisect.insort(self.bus_to_generators[gen.bus], gen)

    def remove_generator(self, gen: records.Generator) -> None:
        """
        Remove generator from the system.
        """

        self.generators.remove(gen)
        del self.gen_dict[gen.name]
        self.bus_to_generators[gen.bus].remove(gen)

    def store_injector(self, inj: records.Injector) -> None:
        """
        Add an injector, which is anything that has the methods required below.

        It's very important that powers are in MVA and derivatives in MVA/pu.

        To facilitate exporting data, it's desirable that injectors have a
        prefix, a name, and a list of parameters.
        """

        if inj.name in self.inj_dict:
            raise RuntimeError(f"Injector {inj.name} already exists.")

        # Test if the injector is connected to a bus
        attr = getattr(inj, "bus", None)
        if not attr:
            raise RuntimeError(
                f"Each instance of {inj.__class__.__name__} "
                "must be connected to a bus"
            )

        # Test for missing methods
        for method in ["get_P", "get_Q", "get_dP_dV", "get_dQ_dV", "get_pars"]:
            attr = getattr(inj, method, None)
            if not attr or not callable(attr):
                raise RuntimeError(
                    f"Method {method} is missing from {inj.__class__.__name__}"
                )

        # Add injector
        self.inj_dict[inj.name] = inj
        bisect.insort(self.injectors, inj)
        bisect.insort(self.bus_to_injectors[inj.bus], inj)

    def remove_injector(self, inj: records.Injector) -> None:
        """
        Remove injector from the system.
        """

        self.injectors.remove(inj)
        del self.inj_dict[inj.name]
        self.bus_to_injectors[inj.bus].remove(inj)

    def add_slack(
        self,
        V_pu: float,
        name: str,
        theta_radians: float = 0,
        PL: float = 0,
        QL: float = 0,
        G: float = 0,
        B: float = 0,
        base_kV: float = np.nan,
        V_min_pu: float = 0.95,
        V_max_pu: float = 1.05,
        pu: bool = None,
    ) -> records.Bus:
        """
        Add the slack bus to the system.

        The boolean 'pu' determines whether PL, QL, G, and B are in per-unit or
        in SI units: MW, Mvar, mho, and mho, respectively.

        If 'pu' is not specified, the global setting is inherited.
        """

        if self.slack is not None:
            raise RuntimeError("The system already has a slack bus!")

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            G = self.mho2pu(Y_mho=G, base_kV=base_kV)
            B = self.mho2pu(Y_mho=B, base_kV=base_kV)
            PL = PL / self.base_MVA
            QL = QL / self.base_MVA

        # Initialize and store bus
        slack = records.Slack(
            V_pu=V_pu,
            theta_radians=theta_radians,
            PL_pu=PL,
            QL_pu=QL,
            G_pu=G,
            B_pu=B,
            base_kV=base_kV,
            bus_type="Slack",
            V_min_pu=V_min_pu,
            V_max_pu=V_max_pu,
            name=name,
        )

        self.store_bus(slack)

        # Returning the bus is important, so that the user can use the returned
        # value to specify the connectivity of branches and other elements.
        return slack

    def add_PQ(
        self,
        PL: float,
        QL: float,
        name: str,
        G: float = 0,
        B: float = 0,
        base_kV: float = np.nan,
        V_min_pu: float = 0.95,
        V_max_pu: float = 1.05,
        pu: bool = None,
    ) -> records.Bus:
        """
        Add a PQ (uncontrolled) bus to the system.

        The boolean 'pu' determines whether PL, QL, G, and B are in per-unit or
        in SI units: MW, Mvar, mho, and mho, respectively.

        If 'pu' is not specified, the global setting is inherited.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            G = self.mho2pu(Y_mho=G, base_kV=base_kV)
            B = self.mho2pu(Y_mho=B, base_kV=base_kV)
            PL = PL / self.base_MVA
            QL = QL / self.base_MVA

        # Initialize and store bus
        PQ = records.PQ(
            V_pu=1,  # voltage magnitude is computed later
            theta_radians=0,  # angle is computed later
            PL_pu=PL,
            QL_pu=QL,
            G_pu=G,
            B_pu=B,
            base_kV=base_kV,
            bus_type="PQ",
            V_min_pu=V_min_pu,
            V_max_pu=V_max_pu,
            name=name,
        )

        self.store_bus(PQ)

        # Returning the bus is important, so that the user can use the returned
        # value to specify the connectivity of branches and other elements.
        return PQ

    def add_PV(
        self,
        V_pu: float,
        PL: float,
        name: str,
        QL: float = 0,
        G: float = 0,
        B: float = 0,
        base_kV: float = np.nan,
        V_min_pu: float = 0.95,
        V_max_pu: float = 1.05,
        pu: bool = None,
    ) -> records.Bus:
        """
        Add a PV (controlled) bus to the system.

        The boolean 'pu' determines whether PL, QL, G, and B are in per-unit or
        in SI units: MW, Mvar, mho, and mho, respectively.

        If 'pu' is not specified, the global setting is inherited.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            G = self.mho2pu(Y_mho=G, base_kV=base_kV)
            B = self.mho2pu(Y_mho=B, base_kV=base_kV)
            PL = PL / self.base_MVA
            QL = QL / self.base_MVA

        # Build bus
        PV = records.PV(
            V_pu=V_pu,
            theta_radians=0,  # angle is computed later
            PL_pu=PL,
            QL_pu=QL,
            G_pu=G,
            B_pu=B,
            base_kV=base_kV,
            bus_type="PV",
            V_min_pu=V_min_pu,
            V_max_pu=V_max_pu,
            name=name,
        )

        self.store_bus(PV)

        return PV

    def add_line(
        self,
        from_bus: records.Bus,
        to_bus: records.Bus,
        X: float,
        name: str,
        R: float = 0,
        total_G: float = 0,
        total_B: float = 0,
        pu: bool = None,
        Snom_MVA: float = np.nan,
    ) -> records.Branch:
        """
        Add transmission line or transformer to the system.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            base_kV = from_bus.base_kV
            R = self.ohm2pu(Z_ohm=R, base_kV=base_kV)
            X = self.ohm2pu(Z_ohm=X, base_kV=base_kV)
            total_G = self.mho2pu(Y_mho=total_G, base_kV=base_kV)
            total_B = self.mho2pu(Y_mho=total_B, base_kV=base_kV)

        # Build a branch with n = 1
        total_Y = total_G + 1j * total_B
        branch = records.Branch(
            from_bus=from_bus,
            to_bus=to_bus,
            X_pu=X,
            R_pu=R,
            from_Y_pu=total_Y / 2,
            to_Y_pu=total_Y / 2,
            n_pu=1,
            branch_type="Line",
            Snom_MVA=Snom_MVA,
            name=name,
            sys=self,
        )

        # Add branch to the system
        self.store_branch(branch)

        return branch

    def add_transformer(
        self,
        from_bus: records.Bus,
        to_bus: records.Bus,
        X: float,
        name: str,
        R: float = 0,
        total_G: float = 0,
        total_B: float = 0,
        n_pu: float = 1,
        pu: bool = None,
        Snom_MVA: float = np.nan,
    ) -> records.Branch:
        """
        Add transformer to the system.

        The transformer is modeled as a branch with n != 1 (in general). The
        following convention is used:

        from   n:1   R+jX      to
        |------0 0---xxxx------|

        Ratio n is turns_from/turns_to. Impedance is on the side of the :1,
        i.e. the 'to' side. The 'to' voltage is thus used for normalization.
        """

        # Inherit pu from class if not specified
        if pu is None:
            pu = self.pu

        if not pu:
            base_kV = from_bus.base_kV
            R = self.ohm2pu(Z_ohm=R, base_kV=base_kV)
            X = self.ohm2pu(Z_ohm=X, base_kV=base_kV)
            total_G = self.mho2pu(Y_mho=total_G, base_kV=base_kV)
            total_B = self.mho2pu(Y_mho=total_B, base_kV=base_kV)

        # Build branch with (possibly) n != 1
        total_Y = total_G + 1j * total_B
        branch = records.Branch(
            from_bus=from_bus,
            to_bus=to_bus,
            X_pu=X,
            R_pu=R,
            from_Y_pu=total_Y / 2,
            to_Y_pu=total_Y / 2,
            n_pu=n_pu,
            branch_type="Transformer",
            Snom_MVA=Snom_MVA,
            name=name,
            sys=self,
        )

        # Add branch to the system
        self.store_branch(branch)

        return branch

    def build_Y(self) -> None:
        """
        Build the bus admittance matrix and store it as an attribute Y.

        Only elements that are in operation are taken into account. Note that
        transformers with off-nominal tap ratio lead to modified B_shunt,
        as explained in section 3.8 of Duncan Glover.

        Although Y is usually sparse, here it's stored as a numpy array because
        that's more readable. The conversion to a sparse-matrix datatype is
        done when computing the entries of the Jacobian.
        """

        # Initialize Y matrix
        N = len(self.buses)
        self.Y = np.zeros([N, N], dtype=complex)

        # Add contributions due to admittances at buses
        for i, bus in enumerate(self.buses):
            self.Y[i, i] += bus.G_pu + 1j * bus.B_pu

        # Add contributions from lines
        for line in self.lines:
            if line.in_operation:
                # Get bus indices
                i = self.buses.index(line.from_bus)
                j = self.buses.index(line.to_bus)
                # Get series impedance
                Y_series = 1 / (line.R_pu + 1j * line.X_pu)
                # Add contributions
                self.Y[i, i] += line.from_Y_pu + Y_series
                self.Y[j, j] += line.to_Y_pu + Y_series
                self.Y[i, j] -= Y_series
                self.Y[j, i] -= Y_series

        # Add contributions from transformers (requires taking n into account)
        for transformer in self.transformers:
            if transformer.in_operation:
                # Get bus indices
                i = self.buses.index(transformer.from_bus)
                j = self.buses.index(transformer.to_bus)
                # Get series impedance
                Y_series = 1 / (transformer.R_pu + 1j * transformer.X_pu)
                new_Y_series = Y_series / transformer.n_pu
                # Add contributions
                new_from_Y = (
                    transformer.from_Y_pu + Y_series
                ) / transformer.n_pu**2 - new_Y_series
                new_to_Y = transformer.to_Y_pu + Y_series - new_Y_series
                self.Y[i, i] += new_from_Y + new_Y_series
                self.Y[j, j] += new_to_Y + new_Y_series
                self.Y[i, j] -= new_Y_series
                self.Y[j, i] -= new_Y_series

    def build_dS_dV(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Build partial derivatives for all buses.

        This method exploits sparsity and the fact that Y and V are available.

        For details, see https://matpower.org/docs/TN2-OPF-Derivatives.pdf
        """

        V = np.array([bus.get_phasor_V() for bus in self.buses])
        ib = range(len(V))
        Ybus = scipy.sparse.csr_matrix(self.Y)

        Ibus = Ybus * V
        diagV = scipy.sparse.csr_matrix((V, (ib, ib)))
        diagIbus = scipy.sparse.csr_matrix((Ibus, (ib, ib)))
        diagVnorm = scipy.sparse.csr_matrix((V / np.abs(V), (ib, ib)))

        dS_dVm = (
            diagV * np.conj(Ybus * diagVnorm) + np.conj(diagIbus) * diagVnorm
        )
        dS_dVa = 1j * diagV * np.conj(diagIbus - Ybus * diagV)

        # Convert back to arrays
        dS_dVm = dS_dVm.toarray()
        dS_dVa = dS_dVa.toarray()

        # Add new terms due to injectors
        for inj in self.injectors:
            i = self.buses.index(inj.bus)
            # Derivatives are substracted because S_injected is substracted
            # from delta_S
            dS_dVm[i, i] -= (
                inj.get_dP_dV() + 1j * inj.get_dQ_dV()
            ) / self.base_MVA

        return dS_dVm, dS_dVa

    def build_J(self) -> None:
        """
        Build Jacobian by calling dS_dV and extracting relevant derivatives.
        """

        dS_dVm, dS_dVa = self.build_dS_dV()

        M = len(self.PQ_buses)
        J11 = dS_dVa[1:, 1:].real
        J12 = dS_dVm[1:, 1 : M + 1].real
        J21 = dS_dVa[1 : M + 1, 1:].imag
        J22 = dS_dVm[1 : M + 1, 1 : M + 1].imag

        self.J = np.vstack([np.hstack([J11, J12]), np.hstack([J21, J22])])

    def build_full_J(self) -> None:
        """
        Build full Jacobian including slack and PV buses (usually not in J).

        One interesting fact is that

            self.full_J[:, :N].sum(axis=1) = zeros([N, 1])

        where N = len(self.buses). This happens because varying all angles by
        the same amount leaves the power flows unchanged.

        Since this matrix is not used in the actual power flow calculation, it
        is only meant to be called on demand, for instance when extracting
        sensitivities.
        """

        dS_dVm, dS_dVa = self.build_dS_dV()

        J11 = dS_dVa.real
        J12 = dS_dVm.real
        J21 = dS_dVa.imag
        J22 = dS_dVm.imag

        self.full_J = np.vstack([np.hstack([J11, J12]), np.hstack([J21, J22])])

    def get_S_towards_network(self) -> np.ndarray:
        """
        Return vector with powers exiting each bus towards the network.

        This method is useful for computing the power mismatches at every
        iteration of NR and for obtaining the consumption at each bus without
        having to distinguish between true and allocated load.
        """

        V = np.array([bus.get_phasor_V() for bus in self.buses])
        ib = range(len(V))
        Ybus = scipy.sparse.csr_matrix(self.Y)

        Ibus = Ybus * V
        diagV = scipy.sparse.csr_matrix((V, (ib, ib)))

        S_to_network_pu = diagV * np.conj(Ybus * np.asmatrix(V).T)

        return S_to_network_pu

    def build_F(self) -> None:
        """
        Build mismatch vector using the voltages that are currently available.

        The first M rows correspond to P and Q mismatches of PQ buses, whereas
        the remaining rows correspond to P of PV buses.
        """

        # Get total power 'injected' by loads (negative load)
        S_injected = np.array(
            [[-bus.PL_pu - 1j * bus.QL_pu] for bus in self.buses],
            dtype=complex,
        )

        # Add contributions due to injectors
        for inj in self.injectors:
            i = self.buses.index(inj.bus)
            S_injected[i, 0] += (
                inj.get_P() + 1j * inj.get_Q()
            ) / self.base_MVA

        # Add contributions due to generators
        for gen in self.generators:
            i = self.buses.index(gen.bus)
            S_injected[i, 0] += gen.PG_MW / self.base_MVA

        # Compute mismatch (ideally, power towards netw. - injected power = 0)
        delta_S = self.get_S_towards_network() - S_injected

        M = len(self.PQ_buses)
        F00 = delta_S[1:, 0].real
        F10 = delta_S[1 : M + 1, 0].imag

        self.F = np.vstack([F00, F10])

    def update_v(self, x: np.ndarray) -> None:
        """
        Update angle of all non-slack buses and voltage magnitude of PQ buses.
        """

        # Update angles
        for bus_no, bus in enumerate(self.non_slack_buses):
            bus.theta_radians = x[bus_no, 0]

        # Update magnitude
        for bus_no, bus in enumerate(self.PQ_buses):
            bus.V_pu = x[len(self.non_slack_buses) + bus_no, 0]

    def update_S(self) -> None:
        """
        Update P and Q consumption at all buses.

        To get the power demanded by an injector at a PQ bus, recall that it's
        possible to call get_P() and get_Q(), which will use the most recent
        voltage to get P and Q.
        """

        # Get power going to the network
        SL = -self.get_S_towards_network()

        # Add it as an attribute to the bus
        for bus_no, bus in enumerate(self.buses):
            bus.P_to_network_pu = -SL[bus_no, 0].real
            bus.Q_to_network_pu = -SL[bus_no, 0].imag

    def run_pf(
        self,
        tol: float = 1e-9,
        max_iters: int = 20,
        flat_start: bool = False,
        warn: bool = True,
    ) -> bool:
        """
        Run AC power-flow study using the Newton-Raphson method.

        The return value is True if the power flow converged and False
        otherwise.

        The threshold on the number of buses for applying LU factorization
        is hard-coded as 1000, based on experiments.
        """

        # Decide if LU decomposition is used when solving J @ dx = F
        LU_is_needed = len(self.buses) > 1000

        # Test for slack bus
        if self.slack is None:
            raise RuntimeError("The system must have a slack bus!")

        # Test for injectors at slack or PV buses
        for inj in self.injectors:
            if inj.bus not in self.PQ_buses:
                raise RuntimeError("Injectors can only be placed at PQ buses!")

        # Test for generators at slack or PQ buses. The condition
        # gen.PG_MW != 0 makes it possible to add dummy generators
        # (gen.PG_MW == 0) at the slack, which is useful, in turn, when
        # exporting into RAMSES.
        for gen in self.generators:
            if gen.bus not in self.PV_buses and gen.PG_MW != 0:
                raise RuntimeError(
                    "Generators can only be placed at PV buses!"
                )

        # Build nodal admittance matrix
        self.build_Y()

        # Ensure flat start
        x0 = np.vstack(
            [
                np.zeros([len(self.non_slack_buses), 1]),  # angles
                np.ones([len(self.PQ_buses), 1]),
            ]
        )  # magnitudes

        # Initialize
        x = x0
        if flat_start:
            self.update_v(x)
        iters = 0
        self.build_F()
        if np.linalg.norm(self.F, np.inf) < tol:  # test for lucky guess
            return True
        self.build_J()

        # Run Newton-Raphson method
        while np.linalg.norm(self.F, np.inf) > tol and iters < max_iters:
            # Update x
            if LU_is_needed:
                lu, piv = scipy.linalg.lu_factor(self.J)
                dx = scipy.linalg.lu_solve((lu, piv), self.F)
                x -= dx
            else:
                x -= np.matmul(np.linalg.inv(self.J), self.F)
            # Update operating point
            self.update_v(x)
            # Update matrices for next iteration
            self.build_F()
            self.build_J()
            # Count iteration
            iters += 1

        # Update complex powers
        self.update_S()

        # Update status
        if iters < max_iters:
            tol_W = round(tol * self.base_MVA * 1e6, 3)
            self.status = f"solved (max |F| < {tol_W} W) in {iters} iterations"
            return True
        else:
            self.status = f"non-convergent after {iters} iterations"
            if warn:
                warnings.warn(
                    f"Newton-Raphson did not converge after "
                    f"{iters} iterations."
                )
            return False

    @classmethod
    def import_ARTERE(
        cls,
        filename: str,
        system_name: str,
        base_MVA: float = 100,
        use_injectors: bool = False,
    ) -> "StaticSystem":
        """
        Import system from ARTERE file.

        The boolean 'use_injectors' determines whether loads and shunts are
        imported as injectors or as attributes of the buses.
        """

        # Initialize system and containers
        sys = cls(name=system_name, base_MVA=base_MVA)
        bus_types = {}  # bus_name: bus_type
        bus_objects = {}  # bus_name: bus_object
        v_setpoints = {}  # gen_name: V_pu
        generation_MW = {}  # gen_name: PG_MW
        locations = {}  # bus_name: location
        gen_names = {}  # bus_name: gen_name

        # Traverse file and save, initially, all buses as PQ buses
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0] == "BUS":
                    bus_name = words[1]
                    bus_types[bus_name] = "PQ"
                # Take advantage of this iteration and store location of the
                # buses
                elif len(words) > 0 and words[0] == "BUSPART":
                    bus_name = words[2]
                    location = words[1]
                    locations[bus_name] = location

        # Traverse file again, correct for PV buses, and store generator data
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0] == "GENER":
                    gen_name = words[1]
                    bus_name = words[2]
                    bus_types[gen_name] = "PV"
                    v_setpoints[gen_name] = float(words[6])
                    generation_MW[gen_name] = float(words[4])
                    gen_names[bus_name] = gen_name

        # Traverse file again and correct the slack
        with open(filename, "r") as f:
            for line in f:
                words = line.split()
                if len(words) > 0 and words[0] == "SLACK":
                    bus_name = words[1].strip(";")
                    bus_types[bus_name] = "Slack"

        # Traverse file again (last time) to call constructors and populate the
        # system
        with open(filename, "r") as f:
            for line in f:
                words = line.split()

                # Skip empty rows
                if len(words) == 0:
                    continue

                # Import buses
                if words[0] == "BUS":
                    # Read parameters
                    bus_name = words[1]
                    base_kV = float(words[2])
                    if use_injectors:
                        PL_MW = 0
                        QL_Mvar = 0
                        B_mho = 0
                    else:
                        PL_MW = float(words[3])
                        QL_Mvar = float(words[4])  # load Q
                        QS_Mvar = float(words[5])  # shunt Q
                        B_mho = utils.var2mho(Mvar_3P=QS_Mvar, kV_LL=base_kV)

                    # Call the right constructor
                    if bus_types[bus_name] == "PQ":
                        # Add bus
                        b = sys.add_PQ(
                            PL=PL_MW,
                            QL=QL_Mvar,
                            B=B_mho,
                            base_kV=base_kV,
                            name=bus_name,
                            pu=False,
                        )

                    elif bus_types[bus_name] == "PV":
                        # Add bus
                        b = sys.add_PV(
                            V_pu=v_setpoints[bus_name],
                            PL=PL_MW,
                            B=B_mho,
                            base_kV=base_kV,
                            name=bus_name,
                            pu=False,
                        )
                        # Add generator
                        gen = records.Generator(
                            PG_MW=generation_MW[bus_name],
                            bus=b,
                            name=gen_names[bus_name],
                        )
                        sys.store_generator(gen=gen)

                    elif bus_types[bus_name] == "Slack":
                        # Add bus
                        b = sys.add_slack(
                            V_pu=v_setpoints[bus_name],
                            B=B_mho,
                            base_kV=base_kV,
                            name=bus_name,
                            pu=False,
                        )
                        # Add dummy generator to the slack
                        gen = records.Generator(
                            PG_MW=0, bus=b, name=gen_names[bus_name]
                        )
                        sys.store_generator(gen)

                    # Save location and bus object
                    if bus_name in locations:
                        b.location = locations[bus_name]
                    bus_objects[bus_name] = b

                    # Create injectors
                    if use_injectors and bus_types[bus_name] == "PQ":
                        # Define load object
                        load_name = f"L{int(bus_name):02d}"
                        P_MW = float(words[3])
                        Q_Mvar = float(words[4])
                        if abs(P_MW) > 1e-6 or abs(Q_Mvar) > 1e-6:
                            load = records.Load(
                                name=load_name,
                                bus=b,
                                P0_MW=P_MW,
                                Q0_Mvar=Q_Mvar,
                            )
                            sys.store_injector(load)

                        # Define shunt object
                        QS_Mvar = float(words[5])
                        if abs(QS_Mvar) > 1e-9:
                            shunt_name = f"SH{int(bus_name):02d}"
                            shunt = records.Shunt(
                                name=shunt_name, bus=b, Mvar_at_Vnom=QS_Mvar
                            )
                            sys.store_injector(shunt)

                # Import lines
                elif words[0] == "LINE":
                    # Read parameters
                    line_name = words[1]
                    from_bus = bus_objects[words[2]]
                    to_bus = bus_objects[words[3]]
                    R_ohm = float(words[4])
                    X_ohm = float(words[5])
                    B_total_mho = 2 * float(words[6]) / 1e6
                    Snom_MVA = float(words[7])

                    # Call constructor
                    transmission_line = sys.add_line(
                        from_bus=from_bus,
                        to_bus=to_bus,
                        X=X_ohm,
                        R=R_ohm,
                        total_B=B_total_mho,
                        pu=False,
                        Snom_MVA=Snom_MVA,
                        name=line_name,
                    )

                # Import transformers
                elif words[0] == "TRFO":
                    trfo_name = words[1]
                    # Yes, these lines were flipped intentionally:
                    from_bus = bus_objects[words[3]]
                    to_bus = bus_objects[words[2]]
                    # Determine offset due to presence or absence of controlled
                    # bus (cumbersome, I know)
                    offset = 1 if "'" in words else 0
                    # Read remaining parameters
                    Snom_MVA = float(words[9 + offset])
                    R_perc = float(words[5 + offset])
                    X_perc = float(words[6 + offset])
                    R_pu = utils.change_base(
                        quantity=R_perc / 100,
                        base_MVA_old=Snom_MVA,
                        base_MVA_new=sys.base_MVA,
                        type="Z",
                    )
                    X_pu = utils.change_base(
                        quantity=X_perc / 100,
                        base_MVA_old=Snom_MVA,
                        base_MVA_new=sys.base_MVA,
                        type="Z",
                    )
                    # Transformers rarely have B:
                    n_pu = float(words[8 + offset]) / 100.0
                    # Call constructor
                    transformer = sys.add_transformer(
                        from_bus=from_bus,
                        to_bus=to_bus,
                        X=X_pu,
                        R=R_pu,
                        n_pu=n_pu,
                        pu=True,
                        Snom_MVA=Snom_MVA,
                        name=trfo_name,
                    )

                    # Read OLTC-related parameters
                    n_first_pu = float(words[10 + offset]) / 100
                    n_last_pu = float(words[11 + offset]) / 100
                    nb_pos = float(words[12 + offset])
                    half_db_pu = float(words[13 + offset])
                    v_setpoint_pu = float(words[14 + offset])

                    # If an OLTC is present, add it as an object
                    if n_first_pu * 100 > 0.5:
                        step_pu = (n_last_pu - n_first_pu) / (nb_pos - 1)
                        positions_up = round((n_last_pu - 1) / step_pu)
                        positions_down = round((1 - n_first_pu) / step_pu)
                        transformer.add_OLTC(
                            positions_up=positions_up,
                            positions_down=positions_down,
                            step_pu=step_pu,
                            v_setpoint_pu=v_setpoint_pu,
                            half_db_pu=half_db_pu,
                        )

        return sys

    def get_bus(self, name: str) -> records.Bus:
        """
        Get object associated to named bus.
        """

        if not isinstance(name, str):
            raise TypeError("Bus name must be a string.")

        if name not in self.bus_dict:
            raise RuntimeError(f"Bus {name} does not exist.")

        return self.bus_dict[name]

    def get_line(self, name: str) -> records.Branch:
        """
        Get object associated to named line.
        """

        if name not in self.line_dict:
            raise RuntimeError(f"Line {name} does not exist.")

        return self.line_dict[name]

    def get_transformer(self, name: str) -> records.Branch:
        """
        Get object associated to named transformer.
        """

        if name not in self.transformer_dict:
            raise RuntimeError(f"Transformer {name} does not exist.")

        return self.transformer_dict[name]

    def get_branches_between(
        self, bus_name_1: str, bus_name_2: str, warn: bool = True
    ) -> list[records.Branch]:
        """
        Return branches between two named buses.
        """

        bus_1 = self.get_bus(name=bus_name_1)
        bus_2 = self.get_bus(name=bus_name_2)

        candidates = [
            b
            for b in self.branches
            if (b.from_bus == bus_1 and b.to_bus == bus_2)
            or (b.from_bus == bus_2 and b.to_bus == bus_1)
        ]

        if warn and len(candidates) == 0:
            warnings.warn(
                f"There are no branches between {bus_name_1} and {bus_name_2}."
            )

        return candidates

    def get_generator(self, name: str) -> records.Generator:
        """
        Get object associated to named generator.
        """

        if name not in self.gen_dict:
            raise RuntimeError(f"Generator {name} does not exist.")

        return self.gen_dict[name]

    def get_injector(self, name: str) -> records.Injector:
        """
        Get object associated to named injector.
        """

        if name not in self.inj_dict:
            raise RuntimeError(f"Injector {name} does not exist.")

        return self.inj_dict[name]

    def get_bus_load_MVA(
        self, bus: records.Bus, attr: str = "P", tol: float = 1e-6
    ) -> float:
        """
        Return total (appreciable) bus load, with injectors as negative loads.
        """

        # Add loads from PL_pu
        total_load = self.base_MVA * getattr(bus, f"{attr}L_pu")

        # Possibly substract power injected by injectors
        total_load -= sum(
            getattr(inj, f"get_{attr}")()
            for inj in self.bus_to_injectors[bus]
            if isinstance(inj, records.Load)
        )

        return total_load if abs(total_load) > tol else None

    def get_bus_generation_MVA(
        self, bus: records.Bus, attr: str = "P", tol: float = 1e-4
    ) -> float:
        """
        Return total (appreciable) bus generation.
        """

        if isinstance(bus, records.Slack):
            return self.base_MVA * getattr(bus, f"{attr}_to_network_pu")

        elif isinstance(bus, records.PV):
            # If asking for P, read it from the generators
            if attr == "P":
                # Recall that generator P already is in MW
                total_gen = sum(
                    gen.PG_MW for gen in self.bus_to_generators[bus]
                )
                return total_gen if abs(total_gen) > tol else None
            # If asking for Q, use the one that flows towards the network
            elif attr == "Q":
                return self.base_MVA * bus.Q_to_network_pu

        elif isinstance(bus, records.PQ):
            return None

    def get_sensitive_load_MW_Mvar(
        self, bus: records.Bus
    ) -> tuple[float, float]:
        """
        Measure sensitive load (P, Q) at a particular bus.
        """

        sensitive_P_load_MW, sensitive_Q_load_Mvar = 0, 0

        for inj in self.bus_to_injectors[bus]:
            if isinstance(inj, records.Load):
                # Negative because get_P() and get_Q() return injected powers
                sensitive_P_load_MW -= inj.get_P()
                sensitive_Q_load_Mvar -= inj.get_Q()

        return sensitive_P_load_MW, sensitive_Q_load_Mvar

    def get_P_and_G(
        self, boundary_bus: str, sending_buses: Sequence[str]
    ) -> tuple[float, float]:
        """
        Return (P, G) with G at, and P entering into, the boundary bus.

        The boundary bus and the sending buses are names.
        """

        # Map names to objects
        boundary_bus = self.get_bus(name=boundary_bus)
        sending_buses = [self.get_bus(name=bus) for bus in sending_buses]

        # Run power flow using last (present) state as initial condition
        self.run_pf(flat_start=False)

        # Fetch voltage of boundary bus, denoted i
        Vi = boundary_bus.get_phasor_V()

        # Compute current entering boundary bus i by adding contributions
        Ii = 0
        for line in self.lines:
            if line.in_operation:
                # Get phasor voltages
                V_from = line.from_bus.get_phasor_V()
                V_to = line.to_bus.get_phasor_V()
                # Add contributions
                if (
                    boundary_bus is line.from_bus
                    and line.to_bus in sending_buses
                ):
                    Ii += (V_to - V_from) / (
                        line.R_pu + 1j * line.X_pu
                    )  # entering the bus
                    Ii -= V_from * (line.from_Y_pu)  # leaving the bus
                elif (
                    boundary_bus is line.to_bus
                    and line.from_bus in sending_buses
                ):
                    Ii += (V_from - V_to) / (
                        line.R_pu + 1j * line.X_pu
                    )  # entering the bus
                    Ii -= V_to * (
                        line.to_Y_pu
                    )  # leaving the bus through the admittance

        # Compute P and G by definition
        Pi = np.real(Vi * np.conj(Ii))
        Gi = np.real(Ii / Vi)

        return Pi, Gi

    def get_min_NLI(
        self,
        corridor: tuple[str, Sequence[str]],
        transformer_names: Sequence[str],
    ) -> float:
        """
        Get minimum (worst) NLI after changing each tap ratio, one at a time.

        The corridor is a tuple with the boundary bus and the sending buses,
        i.e. it's possible to unpack it as follows:

            boundary_bus, sending_buses = corridor

        as required by get_P_and_G().

        At some point in the past, I had thought of using the average NLI, but
        experiments suggested that the worst (minimum) NLI is a better metric.
        """

        # Do everything with a deep copy to avoid messing up the original
        # system
        sys = copy.deepcopy(self)

        boundary_bus, sending_buses = corridor

        # Compute P0 and G0
        P0, G0 = sys.get_P_and_G(
            boundary_bus=boundary_bus, sending_buses=sending_buses
        )

        NLIs = []
        h = 1e-3  # perturbation of the tap ratio

        for transformer_name in transformer_names:
            # Get transformer
            transformer = sys.get_transformer(name=transformer_name)
            # Change tap ratio
            transformer.n_pu += h
            # Get P and G. Notice that get_P_and_G() calls run_pf().
            P1, G1 = sys.get_P_and_G(
                boundary_bus=boundary_bus, sending_buses=sending_buses
            )
            # Change the tap ratio back
            transformer.n_pu -= h
            # Compute the NLI using its static definition and then save it
            NLI = (P1 - P0) / (G1 - G0)
            NLIs.append(NLI)

        return min(NLIs)

    def generate_table(
        self,
        show_buses: bool = True,
        show_lines: bool = True,
        show_transformers: bool = True,
        show_injectors: bool = True,
    ) -> str:
        """
        Display system data in tabular form.

        The net load can vary depending on the method chosen for simulating
        capacitors.

        If they were considered shunt admittances, they will not
        contribute to the net load, because it's as if they were part of the
        network, not a device that is connected to the bus.

        If, instead, they were considered as injectors, they will contribute
        to the net load.

        In any case, this does not affect the voltages. It's only a matter of
        displaying results.
        """

        # Fetch bus data
        if show_buses:
            bus_data = [
                [
                    self.buses.index(bus),
                    bus.name,
                    bus.bus_type,
                    bus.base_kV,
                    bus.V_pu,
                    np.rad2deg(bus.theta_radians),
                    self.get_bus_load_MVA(bus=bus, attr="P"),
                    self.get_bus_load_MVA(bus=bus, attr="Q"),
                    self.get_bus_generation_MVA(bus=bus, attr="P"),
                    self.get_bus_generation_MVA(bus=bus, attr="Q"),
                ]
                for bus in self.buses
            ]

            # Define headers
            bus_headers = [
                "\n\nIndex",
                "\n\nName",
                "\n\nType",
                "Nominal\nvoltage\n(kV)",
                "\nVoltage\n(pu)",
                "\nPhase\n(degrees)",
                "\nLoad\n(MW)",
                "\nLoad\n(Mvar)",
                "\nGeneration\n(MW)",
                "\nGeneration\n(Mvar)",
            ]

            # Build bus table
            bus_precision = (
                0,
                0,
                0,
                ".1f",
                ".4f",
                ".2f",
                ".3f",
                ".3f",
                ".3f",
                ".3f",
            )
            bus_table = tabulate.tabulate(
                tabular_data=bus_data,
                headers=bus_headers,
                floatfmt=bus_precision,
            )

        if show_lines:
            line_data = [
                [
                    self.lines.index(line),
                    line.name,
                    line.from_bus.name,
                    line.to_bus.name,
                    line.R_pu,
                    line.X_pu,
                    line.from_Y_pu.imag,
                    line.to_Y_pu.imag,
                    line.Snom_MVA,
                    line.get_pu_flows()[0] * self.base_MVA,
                    line.get_pu_flows()[1] * self.base_MVA,
                    line.get_pu_flows()[2] * self.base_MVA,
                    line.get_pu_flows()[3] * self.base_MVA,
                    line.get_pu_flows()[4] * self.base_MVA,
                ]
                for line in self.lines
            ]

            # Define headers
            line_headers = [
                "\nIndex",
                "\nName",
                "\nFrom bus",
                "\nTo bus",
                "\nR (pu)",
                "\nX (pu)",
                "B from\n(pu)",
                "B to\n(pu)",
                "Rating\n(MVA)",
                "P from\n(MW)",
                "Q from\n(Mvar)",
                "P to\n(MW)",
                "Q to\n(Mvar)",
                "Losses\n(MW)",
            ]

            # Build line table
            line_precision = (
                0,
                0,
                0,
                0,
                ".4f",
                ".4f",
                ".4f",
                ".4f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
            )
            line_table = tabulate.tabulate(
                tabular_data=line_data,
                headers=line_headers,
                floatfmt=line_precision,
            )

        if show_transformers:

            def has_OLTC(transformer: records.Branch) -> str:
                return "Yes" if transformer.has_OLTC else None

            transformer_data = [
                [
                    self.transformers.index(transformer),
                    transformer.name,
                    transformer.n_pu,
                    has_OLTC(transformer),
                    transformer.R_pu,
                    transformer.X_pu,
                    transformer.Snom_MVA,
                    transformer.get_pu_flows()[0] * self.base_MVA,
                    transformer.get_pu_flows()[1] * self.base_MVA,
                    transformer.get_pu_flows()[2] * self.base_MVA,
                    transformer.get_pu_flows()[3] * self.base_MVA,
                ]
                for transformer in self.transformers
            ]

            # Define headers
            transformer_headers = [
                "\nIndex",
                "\nName",
                "Ratio\n(pu)",
                "\nOLTC?",
                "\nR (pu)",
                "\nX (pu)",
                "Rating\n(MVA)",
                "P from\n(MW)",
                "Q from\n(Mvar)",
                "P to\n(MW)",
                "Q to\n(Mvar)",
            ]

            # Build transformer table
            transformer_precision = (
                0,
                0,
                ".2f",
                0,
                ".4f",
                ".4f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
                ".1f",
            )
            transformer_table = tabulate.tabulate(
                tabular_data=transformer_data,
                headers=transformer_headers,
                floatfmt=transformer_precision,
            )

        if show_injectors:

            injector_data = [
                [
                    self.injectors.index(injector),
                    injector.name,
                    injector.bus.name,
                    injector.get_P() if not np.isclose(injector.get_P(), 0) else 0.0,
                    injector.get_Q() if not np.isclose(injector.get_Q(), 0) else 0.0,
                    injector.get_dP_dV() if not np.isclose(injector.get_dP_dV(), 0) else 0.0,
                    injector.get_dQ_dV() if not np.isclose(injector.get_dQ_dV(), 0) else 0.0
                ]
                for injector in self.injectors
            ]

            # Define headers
            injector_headers = [
                "Index",
                "Name",
                "Bus",
                "P (MW)",
                "Q (Mvar)",
                "dP/dV (MW/pu)",
                "dQ/dV (Mvar/pu)",
            ]

            # Build injector table
            injector_precision = (
                0,
                0,
                0,
                ".3f",
                ".3f",
                ".3f",
                ".3f",
            )
            injector_table = tabulate.tabulate(
                tabular_data=injector_data,
                headers=injector_headers,
                floatfmt=injector_precision,
            )

        # Possibly add a filler name for the system
        if self.name == "":
            display_name = str(len(self.buses)) + "-bus system"
        else:
            display_name = self.name

        # Report the status (including convergence and number of iterations)
        display_status = "Status: " + self.status

        # Build output string
        output_str = (
            f"\n{display_name}\n\n{display_status}\n\n"
            f"BUS DATA:\n\n{bus_table}\n"
        )

        if show_lines:
            output_str += f"\n\nLINE DATA:\n\n\n{line_table}\n"

        if show_transformers:
            output_str += f"\n\nTRANSFORMER DATA:\n\n\n{transformer_table}\n"

        if show_injectors:
            output_str += f"\n\nINJECTOR DATA:\n\n\n{injector_table}\n"

        return output_str

    def __str__(self) -> str:
        """
        Print only the bus data.
        """

        return self.generate_table(
            show_buses=True, show_lines=False, show_transformers=False
        )

    def build_G(self) -> None:
        """
        Build (multi)graph associated to the network.
        """

        # Initialize multigraph (can hold multiple edges between two nodes)
        self.G = nx.MultiGraph()

        # Create labels and store as attribute. These labels must be stored
        # in a dictionary keyed by node (bus object).
        self.bus_labels = {bus: bus.name for bus in self.buses}

        # Add nodes
        for bus in self.buses:
            self.G.add_node(bus)

        # Define edge colors
        def get_branch_color(branch: records.Branch) -> str:
            return "black" if branch.branch_type == "Line" else "red"

        # Add edges
        for branch in self.branches:
            # Adding only those branches that are in operation is critical to
            # detecting islanding (bus.is_connected == False).
            if branch.in_operation:
                self.G.add_edge(
                    u_for_edge=branch.from_bus,
                    v_for_edge=branch.to_bus,
                    data=branch,
                    key=branch,
                    color=get_branch_color(branch),
                )

    def color_nodes(self, attr: str) -> None:
        """
        Color nodes according to attribute (PL_pu, QL_pu, V_pu, theta_radians).
        """

        for bus in self.buses:
            if attr == "PL_pu":
                color = self.get_sensitive_load_MW_Mvar(bus=bus)[0]
            elif attr == "QL_pu":
                color = self.get_sensitive_load_MW_Mvar(bus=bus)[1]
            else:
                color = getattr(bus, attr)

            self.G.nodes[bus]["color"] = color

    def draw_network(
        self,
        parameter: str = None,
        display: bool = False,
        filename: str = None,
        title: str = None,
    ) -> None:
        """
        Display network data visually.
        """

        # Build graph
        self.build_G()

        # Initialize figure
        plt.figure()

        # Define options
        options = {
            "font_size": 5,
            "node_size": 200,
            "linewidths": 1,
            "width": 1,
            "with_labels": True,
        }

        # Define label (they're predefined)
        if parameter == "QL_pu":
            label = "Reactive load (Mvar)"
        elif parameter == "PL_pu":
            label = "Active load (MW)"
        elif parameter == "base_kV":
            label = "Nominal voltage (kV)"
        elif parameter == "theta_radians":
            label = "Phase (degrees)"
        else:
            label = "Voltage (pu)"

        # Possibly color the network
        if parameter is not None:
            # Color nodes
            self.color_nodes(attr=parameter)

            # Define colormap and node color list (should be exported...)
            cmap = matplotlib.pyplot.cm.coolwarm
            node_colors = [self.G.nodes[bus]["color"] for bus in self.buses]

            # Add remaining options
            options["node_color"] = node_colors
            options["cmap"] = cmap

            # Find meta min and max
            vmin = min(node_colors)
            vmax = max(node_colors)

            # Set colorbar
            norm = matplotlib.colors.Normalize(
                vmin=vmin, vmax=vmax, clip=False
            )
            matplotlib.pyplot.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                shrink=0.6,
                label=label,
            )

        # Draw network
        colors = nx.get_edge_attributes(self.G, "color").values()
        my_pos = nx.spring_layout(self.G, seed=100)
        nx.draw_networkx(
            self.G,
            pos=my_pos,
            edge_color=colors,
            labels=self.bus_labels,
            **options,
        )

        if title is not None:
            plt.title(title)

        if filename is not None:
            plt.savefig(filename, dpi=1200)

        if display:
            plt.show()

    def get_subsystem(self, buses: Container[records.Bus]) -> "StaticSystem":
        """
        Pack the specified buses and their connections into a StaticSystem.

        The idea of packing everything into a StaticSystem is to leverage
        draw_network(), but DO NOT attempt to run a power flow.

        This algorithm will fail if the subsystem contains no branches, in
        particular if it only contains one bus.
        """

        branches = [
            branch
            for branch in self.branches
            if branch.from_bus in buses and branch.to_bus in buses
        ]

        sys = StaticSystem(base_MVA=self.base_MVA)

        # Make a deep copy of the branches to avoid messing up the original
        # system
        sys.branches = copy.deepcopy(branches)
        # The new buses are the ones that are connected by the branches
        sys.buses = list(
            set(
                [b.from_bus for b in sys.branches]
                + [b.to_bus for b in sys.branches]
            )
        )

        return sys

    def isolate_buses_by_kV(
        self, starting_bus: records.Bus
    ) -> set[records.Bus]:
        """
        Isolate adjacent buses that share their nominal voltage.
        """

        # Build graph
        self.build_G()

        # Initialize
        buses = {starting_bus}  # set of buses with the same nominal voltage
        iters = 0
        max_iters = len(
            self.buses
        )  # expanding the search space beyond the number
        # of buses is pointless

        while iters < max_iters:
            # Expand
            neighbor_lists = [list(self.G[bus]) for bus in buses]
            all_neighbors = {
                bus for sublist in neighbor_lists for bus in sublist
            }
            neighbors_same_kV = {
                bus
                for bus in all_neighbors
                if bus.base_kV == starting_bus.base_kV
            }

            # Break if expansion was not fruitful
            if len(neighbors_same_kV) == 0 or neighbors_same_kV == buses:
                break

            # Add neighbors to set of buses
            buses.update(neighbors_same_kV)

            # Count iteration
            iters += 1

        return buses

    def isolate_by_kV(self, starting_bus: records.Bus) -> "StaticSystem":
        """
        Isolate adjacent that share their nominal voltage and return subsystem.

        This method forms a subsystem containing b = starting_bus and all other
        buses that can be reached from b without encountering a transformer.
        """

        adjacent_buses = self.isolate_buses_by_kV(starting_bus=starting_bus)

        return self.get_subsystem(buses=adjacent_buses)

    def isolate_buses_by_radius(
        self, starting_bus: records.Bus, r: int
    ) -> set[records.Bus]:
        """
        Isolate buses within a radius r from starting_bus.
        """

        # Build graph
        self.build_G()

        # Initialize
        buses = {starting_bus}

        # Isolate buses
        for _ in range(int(r)):
            # Find neighbors
            neighbor_lists = [list(self.G[bus]) for bus in buses]
            all_neighbors = {
                bus for sublist in neighbor_lists for bus in sublist
            }
            # Expand
            buses.update(all_neighbors)

        return buses

    def isolate_by_radius(
        self, starting_bus: records.Bus, r: int
    ) -> "StaticSystem":
        """
        Return a subsystem of radius r around starting_bus.
        """

        adjacent_buses = self.isolate_buses_by_radius(
            starting_bus=starting_bus, r=r
        )

        return self.get_subsystem(buses=adjacent_buses)

    def update_connectivity(self, reference_bus: records.Bus) -> None:
        """
        Update attribute is_connected of buses.

        This method checks if there is a path between each bus and the
        reference bus.
        """

        # Build G from scratch. This may not be optimal, but it makes sure that
        # the multigraph is compatible with the in_operation attribute.
        self.build_G()

        # Build set of connected nodes
        connected_nodes = nx.node_connected_component(
            G=self.G, n=reference_bus
        )

        # Update attributes
        for bus in self.buses:
            bus.is_connected = bus in connected_nodes

    def has_contingency(self) -> bool:
        """
        Is the system under a contingency (branches, generators, or shunts).
        """

        return (
            not all(bus.in_operation for bus in self.branches)
            or not all(gen.in_operation for gen in self.generators)
            or not all(
                inj.in_operation
                for inj in self.injectors
                if isinstance(inj, records.Shunt)
            )
        )

    def has_undervoltages(self) -> bool:
        """
        Check if system has undervoltages.
        """

        return any(bus.V_pu < bus.V_min_pu for bus in self.buses)

    def has_overvoltages(self) -> bool:
        """
        Check if system has overvoltages.
        """

        return any(bus.V_pu > bus.V_max_pu for bus in self.buses)

    def has_voltage_violations(self) -> bool:
        """
        Check if system has voltage violations.
        """

        return self.has_undervoltages() or self.has_overvoltages()

    def scale_powers(self, x: np.ndarray) -> None:
        """
        Scale allocated powers by the same amount (but P and Q independently).

        The array x is unpacked into two scalars, alpha and beta, which are
        used to scale all active and reactive powers, respectively.

        The allocated powers (allocated_P and allocated_Q) are not modified,
        as otherwise the load could increase exponentially with each iteration.
        """

        # Unpack scaling factors
        alpha, beta = x[:, 0]

        # Scale allocated load
        for bus in self.PQ_buses:
            bus.PL_pu = alpha * bus.allocated_PL_pu
            bus.QL_pu = beta * bus.allocated_QL_pu

        # Possibly scale load injectors as well
        for inj in self.injectors:
            if isinstance(inj, records.Load):
                inj.P0_MW = alpha * inj.allocated_P0_MW
                inj.Q0_Mvar = beta * inj.allocated_Q0_Mvar

    def get_S_slack_MVA(self) -> complex:
        """
        Return (fetch) the slack complex power in MVA without doing any
        scaling.
        """

        P_slack_MW = self.slack.P_to_network_pu * self.base_MVA
        Q_slack_Mvar = self.slack.Q_to_network_pu * self.base_MVA

        return P_slack_MW + 1j * Q_slack_Mvar

    def get_scaled_S_slack_MVA(self, x: np.ndarray) -> complex:
        """
        Return slack S (in MVA) when loads are scaled.

        The complex power S is assumed to be generated by the slack bus when
        all active loads were scaled by x[0] and all reactive loads were scaled
        by x[1].
        """

        # Scale powers and run power flow
        self.scale_powers(x=x)
        self.run_pf()

        # Fetch slack power in MVA
        return self.get_S_slack_MVA()

    def get_slack_mismatch_MW_Mvar(
        self, x: np.ndarray, P_desired_MW: float, Q_desired_Mvar: float
    ) -> np.ndarray:
        """
        Return mismatch between actual and desired slack S after scaling loads.

        This assumes, again, that P was scaled by x[0] and Q was scaled by
        x[1].
        """

        S_MVA = self.get_scaled_S_slack_MVA(x=x)

        return np.array(
            [[S_MVA.real - P_desired_MW], [S_MVA.imag - Q_desired_Mvar]]
        )

    def get_slack_mismatch_J(
        self, x: np.ndarray, P_desired_MW: float, Q_desired_Mvar: float
    ) -> np.ndarray:
        """
        Return numeric (approximate) Jacobian of get_slack_mismatch_MW_Mvar().

        Because only two evaluations are required, the derivatives are
        approximated using secant lines.

        The algorithm seems to be sensible to the step h. If NR of
        match_power() is not converging, one cause might be that the powers in
        the network are too small.
        """

        h = 1e-4
        x_0 = 1.0 * x  # multiplication by 1.0 creates a copy
        x_1 = 1.0 * x
        x_0[0, 0] += h
        x_1[1, 0] += h

        J11 = (
            1
            / h
            * (
                self.get_slack_mismatch_MW_Mvar(
                    x=x_0,
                    P_desired_MW=P_desired_MW,
                    Q_desired_Mvar=Q_desired_Mvar,
                )
                - self.get_slack_mismatch_MW_Mvar(
                    x=x,
                    P_desired_MW=P_desired_MW,
                    Q_desired_Mvar=Q_desired_Mvar,
                )
            )
        )
        J12 = (
            1
            / h
            * (
                self.get_slack_mismatch_MW_Mvar(
                    x=x_1,
                    P_desired_MW=P_desired_MW,
                    Q_desired_Mvar=Q_desired_Mvar,
                )
                - self.get_slack_mismatch_MW_Mvar(
                    x=x,
                    P_desired_MW=P_desired_MW,
                    Q_desired_Mvar=Q_desired_Mvar,
                )
            )
        )

        return np.hstack([J11, J12])

    def correct_voltages(self) -> bool:
        """
        Let the OLTCs act and correct all voltages (or hit their limits).
        """

        max_iters = sum(
            transformer.OLTC.positions_up + transformer.OLTC.positions_down + 1
            for transformer in self.transformers
            if transformer.has_OLTC
        )
        iters = 0

        while iters < max_iters:
            # Run initial power flow
            self.run_pf()
            # Let transformers act
            tap_actions = 0
            for transformer in self.transformers:
                if transformer.has_OLTC:
                    transformer_acted = transformer.OLTC.act()
                    if transformer_acted:
                        tap_actions += 1
            # Terminate if no transformer was able to act
            if tap_actions == 0:
                break
            # Count interations
            iters += 1
        # Return False if the voltages never converged
        else:
            return False

        # Return True if voltages converged inside the deadband of the OLTCs
        return True

    def match_power(
        self,
        P_desired_MW: float,
        Q_desired_Mvar: float,
        V_desired_pu: float,
        theta_desired_radians: float,
        tol: float = 1e-6,
        max_iters: float = 10,
        use_OLTCs: bool = True,
    ) -> None:
        """
        Modify loads of a system so it can be used to disaggregate a load.

        This method modifies power consumption at load buses so that the
        system's consumption matches specified P + jQ in MVA when its slack
        operates at voltage V/_theta.
        """

        # Modify slack
        # Phase difference is added later, as it bears no physical meaning
        self.slack.V_pu = V_desired_pu
        self.slack.theta_radians = 0

        # Initialize
        x0 = np.array([[1.0], [1.0]])

        # Cast mismatch and Jacobian as one-variable functions
        def f(x):
            return self.get_slack_mismatch_MW_Mvar(
                x=x, P_desired_MW=P_desired_MW, Q_desired_Mvar=Q_desired_Mvar
            )

        def jac(x):
            return self.get_slack_mismatch_J(
                x=x, P_desired_MW=P_desired_MW, Q_desired_Mvar=Q_desired_Mvar
            )

        # Ensure good voltages
        while True:
            # Initialize Newton-Raphson
            x = x0
            iters = 0
            # Run Newton-Raphson
            while np.linalg.norm(f(x), np.inf) > tol and iters < max_iters:
                x -= np.linalg.inv(jac(x)) @ f(x)
                iters += 1
            # Test for poor voltages
            if self.has_voltage_violations() and use_OLTCs:
                # Try to correct them using OLTCs
                tap_actions = 0
                for transformer in self.transformers:
                    if transformer.has_OLTC:
                        transformer_acted = transformer.OLTC.act()
                        if transformer_acted:
                            tap_actions += 1
                # If no OLTC acted, break, as there's nothing more to do
                if tap_actions < 1:
                    break
            # If voltages are correct or OLTCs cannot be used, break as well
            else:
                break

        # Remember to shift all angles by the same amount
        for bus in self.buses:
            bus.theta_radians += theta_desired_radians

    def append_sys(
        self,
        bus: records.Bus,
        new_sys: "StaticSystem",
        P_desired_MW: float = 0,
        Q_desired_Mvar: float = 0,
    ) -> None:
        """
        Append an additional StaticSystem to the specified bus.

        The method does not take into account that bus and new_sys.slack might
        have different voltages. This shouldn't make much of a difference.
        """

        # Change the base power of buses and branches
        base_kV_new = bus.base_kV
        base_MVA_new = self.base_MVA
        # Converting the branches first is important, so that buses still
        # preserve the old values
        for new_element in new_sys.branches + new_sys.buses:
            # Fetch old base kV
            if isinstance(new_element, records.Branch):
                base_kV_old = new_element.from_bus.base_kV
            elif isinstance(new_element, records.Bus):
                base_kV_old = new_element.base_kV
            # Fetch old base MVA
            base_MVA_old = new_sys.base_MVA
            # Change base
            new_element.change_base(
                base_MVA_old=base_MVA_old,
                base_MVA_new=base_MVA_new,
                base_kV_old=base_kV_old,
                base_kV_new=base_kV_new,
            )

        # Match power (up to 1e-9 pu)
        new_sys.match_power(
            P_desired_MW=P_desired_MW,
            Q_desired_Mvar=Q_desired_Mvar,
            V_desired_pu=bus.V_pu,
            theta_desired_radians=bus.theta_radians,
            tol=P_desired_MW / self.base_MVA / 1e9,
        )

        # Add all PQ and PV buses from new_sys. This should be done before
        # replacing the slack, as the replacement will probably not be a slack
        # and hence the method would try to add it a second time (which would
        # raise a RuntimeError in store_bus()).
        for new_bus in new_sys.non_slack_buses:
            self.store_bus(bus=new_bus)

        # Replace old slack by new bus. This takes care of the dictionaries,
        # the lists,
        new_sys.replace_bus(old_bus=new_sys.slack, new_bus=bus)

        # Change system that "owns" each branch
        for new_branch in new_sys.branches:
            new_branch.sys = self

        # The following methods make sure that the dictionaries are updated

        # Add all branches of new_sys
        for new_branch in new_sys.branches:
            self.store_branch(branch=new_branch)

        # Add all generators of new_sys
        for new_gen in new_sys.generators:
            if new_gen.bus is not new_sys.slack:
                self.store_generator(gen=new_gen)

        # Add all injectors of new_sys
        for new_inj in new_sys.injectors:
            self.store_injector(inj=new_inj)

    def remove_loads_at_bus(self, bus: records.Bus) -> tuple[float, float]:
        """
        Remove all loads at bus (both fixed and voltage-dependent).

        Returns the exponents alpha and beta of the last removed load.
        """

        # Remove fixed loads
        bus.PL_pu = 0
        bus.QL_pu = 0
        bus.allocated_PL_pu = 0
        bus.allocated_QL_pu = 0

        # Remove voltage-sensitive loads
        sensitive_loads = [
            inj
            for inj in self.bus_to_injectors[bus]
            if isinstance(inj, records.Load)
        ]
        for inj in sensitive_loads:
            self.injectors.remove(inj)
            self.bus_to_injectors[bus].remove(inj)

        if len(sensitive_loads) > 0:
            return inj.alpha, inj.beta
        else:
            return 0, 0

    def disaggregate_load(
        self, bus: records.Bus, systems: Sequence["StaticSystem"]
    ) -> None:
        """
        Disaggregate load at bus by replacing it with some (maybe all) systems.

        It's crucial that the subset is formed by making deep copies.
        """

        def complex2array(z: complex) -> np.ndarray:
            """
            Convert complex number to array.
            """

            return np.array([z.real, z.imag])

        # Prepare input for subset-sum solver (powers in MVA)
        powers_MVA = [complex2array(sys.get_S_slack_MVA()) for sys in systems]

        (
            sensitive_P_load_MW,
            sensitive_Q_load_Mvar,
        ) = self.get_sensitive_load_MW_Mvar(bus=bus)
        target_S_MVA = complex2array(
            bus.PL_pu * self.base_MVA
            + sensitive_P_load_MW
            + 1j * bus.QL_pu * self.base_MVA
            + 1j * sensitive_Q_load_Mvar
        )

        # Check if solution is infeasible
        if target_S_MVA[0] == 0 or target_S_MVA[1] == 0:
            return None

        # Solve subset sum
        _, indices, best_sum = subsum.subsetsum(
            powers=powers_MVA, S=target_S_MVA, precision=1e-2
        )

        # Create deep copies of the relevant systems
        subset = [copy.deepcopy(systems[i]) for i in indices]

        # Obtain scaling factors
        kP = target_S_MVA[0] / best_sum[0]
        kQ = target_S_MVA[1] / best_sum[1]

        # Rename buses in each component network
        for i, sys in enumerate(subset):
            for new_bus in sys.buses:
                # Define new name
                extended_name = f"{bus.name}_{i+1}_{new_bus.name}"
                # Update bus_dict
                if new_bus.name in sys.bus_dict:
                    sys.bus_dict[extended_name] = new_bus
                    del sys.bus_dict[new_bus.name]
                # The other dictionaries don't have to be updated because
                # they map buses (not names) to objects.
                # Change bus name
                new_bus.name = extended_name

            # Renaming branches, generators, and injectors does not require
            # changing any dictionary
            for new_branch in sys.branches:
                new_branch.name = f"{bus.name}_{i+1}_{new_branch.name}"
            for new_gen in sys.generators:
                new_gen.name = f"{bus.name}_{i+1}_{new_gen.name}"
            for new_inj in sys.injectors:
                new_inj.name = f"{bus.name}_{i+1}_{new_inj.name}"

        # Replace load by networks
        alpha, beta = self.remove_loads_at_bus(bus=bus)

        for sys in subset:
            S_slack_MVA = sys.get_S_slack_MVA()
            self.append_sys(
                bus=bus,
                new_sys=sys,
                P_desired_MW=kP * S_slack_MVA.real,
                Q_desired_Mvar=kQ * S_slack_MVA.imag,
            )
            # Make all loads at bus voltage-sensitive
            for inj in sys.injectors:
                if isinstance(inj, records.Load):
                    inj.make_voltage_sensitive(alpha=alpha, beta=beta)

        # Make sure that power flow continuity is given
        self.build_Y()
        self.build_F()

        return subset, np.linalg.norm(self.F, np.inf)
