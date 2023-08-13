"""
This file specifies the test systems from the master's thesis

    "Control of distributed energy resources and load tap changers to enhance
    the long-term voltage stability of transmission networks."

While the experiments should be reproducible by looking at this file and
experiments.py, these two programs rely on an extensive library of functions and
classes located elsewhere in this repository. These functions and classes handle
all aspects of the simulations, from initializing the power flow cases to
processing the results.

Francisco Escobar (fescobar@ieee.org)
August 2023, San José
"""

# \section{Preliminaries}
#
# The program begins by importing all the modules from \pyv{src}. They take care
# of most aspects of the simulation, from initializing the power-flow case to
# processing the results.
import sys

sys.path.append("src")
import pf_dynamic as pf
import records as rec
import utils

# Dependencies from the standard library are \pyv{os} for defining paths,
# \pyv{copy} for deep-copying instances of \pyv{pf_dynamic.System}, and
# \pyv{random} for introducing randomizations.
import os
import copy
import random

# Finally, other dependencies are \pyv{numpy} for miscellaneous calculations
# and \pyv{scipy.optimize} for the solution of an \gls{AC-OPF} problem when
# building the \glspl{DN}.
import numpy as np
# The seed is, again, arbitrary
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint


# \par
# To ensure that the simulated \gls{TN} preserves the power flows
# from~\cite{ieeetaskforceontestsystemsforvoltagestabilityandsecurityassessment2015},
# a helper function compares the slack complex power against the documented
# value of $\SI{2137.35}{\mega\watt} + j \SI{377.4}{\mega\var}$.%
#   \footnote{Unless they are too complex, functions are typeset as tightly
#            as possible and without \textit{docstrings}.}


def benchmark_nordic(nordic: pf.System) -> None:
    # For validation purposes this is enough (notice atol)
    nordic.run_pf(tol=1e-6)
    assert np.isclose(
        nordic.get_S_slack_MVA(), 2137.35 + 1j * 377.4, atol=1e-1
    ), "Slack power does not match the documented value."


# As shown by the type annotation, the \gls{TN} is handled as an instance of
# \pyv{pf.System}. This class is endowed with methods such as \pyv{run_pf()} and
# \pyv{get_bus()}, as well as attributes that store the system elements:
# \pyv{buses}, \pyv{lines}, \pyv{transformers}, and so on.


# \section{Construction of the \gls{TN}}
#
# The \gls{TN} is generated by a helper function, as this facilitates creating
# identical instances for each experiment.
def is_load(injector: rec.Injector) -> bool:
    return isinstance(injector, rec.Load)


def get_nordic() -> pf.System:
    # This function begins by importing the system from a \pyv{.dat} file,
    # assumed to conform with the format of \textsc{Artere}.%
    #   \footnote{An academic software developed at the University of Liège
    #             for the solution of power-flow scenarios.}
    # This file only contains static information.%
    #   \footnote{The leading white spaces in the following lines are,
    #             of course, very important in \python.}
    nordic = pf.System.import_ARTERE(
        filename=os.path.join(
            "src", "networks", "Nordic", "Nordic test system", "lf_A.dat"
        ),
        system_name="Nordic Test System - Case A - Sensitive loads",
        use_injectors=True,
    )
    # The argument \pyv{use_injectors=True} imports the loads and shunt
    # compensators as instances of an \pyv{Injector} class defined in
    # \pyv{src/records.py}. Importantly, this class makes it possible to model
    # voltage-dependent power sources (resp.\,sinks). If loads are not imported
    # this way, the power-flow runs needed to compute the sensitivity
    # matrix~$\sensitivitymatrix$ after the disturbance may not converge (see
    # \cref{fig:flowchart}).

    # That the importation was successfull is tested by looking at the slack power.
    nordic.run_pf()
    benchmark_nordic(nordic)

    # \par
    # The previous power-flow run assumed that all loads had exponents
    # $\alpha=0$ and~$\beta=0$ for active and reactive power, respectively,
    # as this forces the power flows to match those
    # of~\cite{ieeetaskforceontestsystemsforvoltagestabilityandsecurityassessment2015}.
    # To make the loads sensitive to voltage, the new values $\alpha=1$
    # and~$\alpha=2$ are passed to a method of the \pyv{Load} class:
    for load in filter(is_load, nordic.injectors):
        load.make_voltage_sensitive(alpha=1.0, beta=2.0)
    # The new active power consumed at an arbitrary voltage~$\voltage$ will be
    # $\activepower_0(\voltage/\voltage_0)^\alpha$, where~$0$ denotes the values
    # from the \pyv{run_pf()} call above. At this point of the program,
    # $\voltage = \voltage_0$, and hence the power flows should not have changed.
    benchmark_nordic(nordic)

    # \par
    # Having initialized the power-flow scenario, the function imports the
    # information required for dynamic simulations. This includes parameters of
    # synchronous machines, excitation systems, \gls{LTC}, and the like. The
    # information is read from another \pyv{.dat} file, but this time in
    # the format of \ramses.
    nordic.import_dynamic_data(
        filename=os.path.join(
            "src", "networks", "Nordic", "Nordic test system", "dyn_A.dat"
        )
    )

    # \par
    # At this point, the system has the features that are known to the
    # \gls{MPC}. Before returning the system, these features are stored in a
    # twin (deep copy) that will later be accesed by the controller.
    # This will be done in a deep copy, so don't worry.
    np.random.seed(0)
    nordic.generate_twin(
        parameter_randomizations=(branch_randomization, exponent_randomization)
    )
    return nordic


def randomize_normally(parameter: float) -> float:
    # In addition, the line parameters used to calculate the sensitivity matrix
    # were corrupted by a random error whose mean value is zero and standard
    # deviation is 10\% of the actual line parameter.
    return np.random.normal(parameter, 0.1 * parameter)


def branch_randomization(system: pf.System) -> None:
    for branch in system.branches:
        branch.R_pu = randomize_normally(branch.R_pu)
        branch.X_pu = randomize_normally(branch.X_pu)
        branch.from_Y_pu = randomize_normally(
            branch.from_Y_pu.real
        ) + 1j * randomize_normally(branch.from_Y_pu.imag)
        branch.to_Y_pu = randomize_normally(
            branch.to_Y_pu.real
        ) + 1j * randomize_normally(branch.to_Y_pu.imag)


def exponent_randomization(system: pf.System) -> None:
    for load in filter(is_load, system.injectors):
        load.alpha = randomize_normally(load.alpha)
        load.beta = randomize_normally(load.beta)


SIGNIFICANT_LOAD_MW = 0.1


def is_significant(load: rec.Load) -> bool:
    return abs(load.get_P()) >= SIGNIFICANT_LOAD_MW


# To populate the \gls{TN} with \glspl{DER}, it is convenient to define a
# function that returns an instance of \pyv{pf.System} based on a given
# \gls{DER} \emph{penetration} and a given \emph{frequency}. In a \gls{DN}, if
# the loads consume a total power~$\activepower_L$ and the \gls{DER} generate a
# total power~$\activepower_\text{\gls{DER}}$, the penetration is defined
# as~$\activepower_\text{\gls{DER}}/\activepower_L$. The frequency, on the
# other hand, is defined as the percentage of buses that host a \gls{DER}.
def place_DERs(
    nordic: pf.System, penetration: float, frequency: float, headroom: float
) -> pf.System:
    central_transformers = filter(
        lambda t: t.touches("CENTRAL"), nordic.transformers
    )

    random.seed(1)
    np.random.seed(1)

    def has_load(bus: rec.Bus) -> bool:
        return nordic.get_sensitive_load_MW_Mvar(bus)[0] > SIGNIFICANT_LOAD_MW

    for transformer in central_transformers:
        secondary = transformer.get_LV_bus()

        i = 0 if secondary is transformer.from_bus else 2
        P_fed_MW = -transformer.get_pu_flows()[i] * nordic.base_MVA
        P_DERs_MW = P_fed_MW * penetration / (1 - penetration)

        buses_downstream = nordic.isolate_buses_by_kV(starting_bus=secondary)

        load_buses = list(filter(has_load, buses_downstream))
        N_DER = int(np.ceil(frequency * len(load_buses)))

        DER_buses = random.sample(load_buses, N_DER)

        x = np.random.normal(1, 0.1, N_DER)
        DER_powers_MW = x / np.sum(x) * P_DERs_MW

        for bus, PG_MW in zip(DER_buses, DER_powers_MW):
            DER = rec.DERA(
                name=f"D{bus.name}",
                bus=bus,
                P0_MW=PG_MW,
                Q0_Mvar=0.0,
                Snom_MVA= PG_MW/(1 - headroom),
            )
            nordic.store_injector(DER)

            load = next(filter(is_load, nordic.bus_to_injectors[bus]))
            load.increment_P_by(PG_MW)

    benchmark_nordic(nordic)

    return nordic


def get_T_system(
    penetration: float, frequency: float, headroom: float
) -> pf.System:
    return place_DERs(get_nordic(), penetration, frequency, headroom)


# \section{Construction of the \glspl{DN}}


def get_UKGDS(V_slack_pu: float) -> pf.System:
    # From the Nordic
    BASE_MVA = 100.0
    BASE_kV = 20.0

    UKGDS = pf.System.import_ARTERE(
        filename=os.path.join("src", "networks", "UKGDS", "DN.dat"),
        system_name="UK Generic Distribution System",
        base_MVA=BASE_MVA,
        use_injectors=True,
    )

    step_down_1, step_down_2 = UKGDS.transformers
    UKGDS.remove_branch(step_down_1)
    UKGDS.remove_branch(step_down_2)

    UKGDS.remove_bus(UKGDS.slack)
    new_slack = rec.Slack(
        V_pu=V_slack_pu,
        theta_radians=0,
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
    UKGDS.replace_bus(old_bus=step_down_1.get_LV_bus(), new_bus=new_slack)

    # Reset loads
    for load in filter(is_load, UKGDS.injectors):
        load.set_P_to(0.0)
        load.set_Q_to(0.0)

    # Change base power
    for element in UKGDS.lines + UKGDS.buses:
        if isinstance(element, rec.Branch):
            base_kV_old = element.from_bus.base_kV
        elif isinstance(element, rec.Bus):
            base_kV_old = element.base_kV
        element.change_base(
            base_MVA_old=UKGDS.base_MVA,
            base_MVA_new=BASE_MVA,
            base_kV_old=base_kV_old,
            base_kV_new=BASE_kV,
        )
    UKGDS.base_MVA = BASE_MVA

    # Initialize the power flow
    UKGDS.run_pf(tol=1e-9)

    return UKGDS


def place_loads(
    system: pf.System,
    P_max_MW: float,
    PF_lagging: float,
    V_min_pu: float,
    V_max_pu: float,
) -> pf.System:

    # This one should be inside, so it's not called at every iteration of the optimizer.
    loads = list(filter(is_load, system.injectors))

    def run_pf_with_loads(x: np.ndarray) -> None:
        for PL_MW, load in zip(x, loads):
            load.set_P_to(PL_MW)
            QL_Mvar = np.sqrt((PL_MW / PF_lagging) ** 2 - PL_MW**2)
            load.set_Q_to(QL_Mvar)

        # This 1e-12 is very important, as it should be less than the optimizers tolerance.
        system.run_pf(tol=1e-12)

    def cost(x: np.ndarray) -> float:
        run_pf_with_loads(x)
        return -system.get_S_slack_MVA().real

    def constrained_voltages(x: np.ndarray) -> np.ndarray:
        run_pf_with_loads(x)
        return np.array([bus.V_pu for bus in system.buses])

    NLC = NonlinearConstraint(constrained_voltages, V_min_pu, V_max_pu)
    LC = LinearConstraint(np.eye(len(loads)), 0, P_max_MW)

    x0 = np.array([-load.get_P() for load in loads])

    res = minimize(fun=cost, x0=x0, constraints=(NLC, LC), tol=1.0)
    run_pf_with_loads(res.x)

    for load in loads:
        if not is_significant(load):
            system.remove_injector(load)

    return system


# \section{Disaggregation into a \gls{T-D} system}


def get_TD_system(
    penetration: float,
    frequency: float,
    headroom: float,
    N_monitored_buses: int,
    N_monitored_DERs: int,
) -> pf.System:

    nordic = get_nordic()

    def variant(P_max_MW: float) -> pf.System:
        return place_loads(
            system=get_UKGDS(V_slack_pu=1.0),
            P_max_MW=P_max_MW,
            PF_lagging=0.95,
            V_min_pu=0.965,
            V_max_pu=1.05,
        )

    pool = [
        copy.deepcopy(variant(P_max_MW)) for P_max_MW in (1.0, 2.5, 3.0, 4.0)
    ]

    central_transformers = filter(
        lambda t: t.touches("CENTRAL"), nordic.transformers
    )
    central_secondaries = [t.get_LV_bus() for t in central_transformers]

    # Nice about this function: the loads inherit the same exponents as the
    # big loads.
    for secondary in central_secondaries:
        _, mismatch = nordic.disaggregate_load(bus=secondary, systems=pool)
        assert mismatch < 1e-6, "Largest mismatch is unacceptable."

    benchmark_nordic(nordic)

    nordic_with_DERs = place_DERs(nordic, penetration, frequency, headroom)

    def is_DERA(injector: rec.Injector) -> bool:
        return isinstance(injector, rec.DERA)

    DERs = list(filter(is_DERA, nordic_with_DERs.injectors))
    central_buses = []
    for secondary in central_secondaries:
        central_buses += list(
            nordic_with_DERs.isolate_buses_by_kV(starting_bus=secondary)
        )

    # Get random sample of the DERs of size N
    if N_monitored_buses > len(central_buses) or N_monitored_DERs > len(DERs):
        raise ValueError("I'm not able to monitor that many elements.")

    random.seed(2)
    monitored_buses = random.sample(central_buses, N_monitored_buses)
    monitored_DERs = random.sample(DERs, N_monitored_DERs)

    for element in monitored_buses + monitored_DERs:
        element.is_monitored = True

    return nordic_with_DERs


# "Iterations of Newton's method." Always converged after 4 iterations, although
# there is no certificate for convergence (solutino of the system of equations).
# Overall, it takes about 3 minutes.

if __name__ == "__main__":
    penetration = 0.20
    frequency = 0.15
    headroom = 0.20

    with utils.Timer(name="T and T-D system generation"):
        T_system = get_T_system(penetration, frequency, headroom)
        with open("temp_T.txt", "w") as f:
            f.write(T_system.generate_table())

        TD_system = get_TD_system(
            penetration,
            frequency,
            headroom,
            N_monitored_buses=100,
            N_monitored_DERs=100,
        )
        with open("temp_TD.txt", "w") as f:
            f.write(TD_system.generate_table())
