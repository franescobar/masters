"""
A module providing classes for interacting with RAMSES: disturbances (applied
to the system) and observables (extracted from the system).
"""

# Modules from this repository
import records

# Modules from the standard library
from typing import Union

# Other modules
import numpy as np


class Disturbance:
    """
    Class for representing disturbances in RAMSES.

    Disturbances can be applied to buses, branches, injectors, DCTLs, and even
    the solver.

    A disturbance is defined by: the time at which it occurs, the object it
    acts on, the parameter it modifies, and the value it takes.
    """

    def __init__(
        self,
        ocurrence_time: float,
        object_acted_on: Union[
            records.Bus, records.Branch, records.Injector, records.DCTL, str
        ],
        par_name: str = None,
        par_value: Union[float, None, str] = None,
    ) -> None:
        """
        For buses, par_name should be either 'fault' or 'clearance', whereas
        par_value should be the fault reactance or None.

        For branches, par_name should be 'status', whereas par_value should be
        either 0 or 1.

        For the "solver", par_value should be a string with the solver
        settings.
        """

        if ocurrence_time < 0:
            raise RuntimeError("Disturbance time must be non-negative.")

        for record_type in [
            records.Bus,
            records.Branch,
            records.Injector,
            records.DCTL,
            str,
        ]:
            if isinstance(object_acted_on, record_type):
                break
        else:
            raise RuntimeError(
                f"Unknown object {object_acted_on} acted on by disturbance."
            )

        if isinstance(object_acted_on, records.Bus):
            if par_name not in ["fault", "clearance"]:
                raise RuntimeError(
                    f"Unknown parameter {par_name} "
                    f"for bus {object_acted_on.name}."
                )

            if not isinstance(par_value, float) and par_value is not None:
                raise RuntimeError(
                    f"Unknown value {par_value} for parameter {par_name} "
                    f"of bus {object_acted_on.name}."
                )

        if isinstance(object_acted_on, records.Branch):
            # Test parameter name
            if par_name != "status":
                raise RuntimeError(
                    f"Unknown parameter {par_name} "
                    f"for branch {object_acted_on.name}."
                )
            # Test that parameter value is either 0 or 1
            if not np.isclose(par_value, 0) and not np.isclose(par_value, 1):
                raise RuntimeError(
                    f"Unknown value {par_value} for parameter {par_name} "
                    f"of branch {object_acted_on.name}."
                )

        if object_acted_on == "solver" and not isinstance(par_value, str):
            raise RuntimeError(
                f"The solver's parameter value should be a string, "
                f"not {par_value}."
            )

        self.ocurrence_time = (
            ocurrence_time  # time at which the disturbance occurs
        )
        self.object_acted_on = object_acted_on  # object acted on
        self.par_name = par_name  # name of the parameter that is modified
        self.par_value = (
            par_value  # new value of the parameter that is modified
        )

    def __lt__(self, other: "Disturbance") -> bool:
        """
        Implement ordering of disturbances by their time of ocurrence.
        """

        return self.ocurrence_time < other.ocurrence_time

    def __str__(self) -> str:
        """
        Convert disturbance to RAMSES' format.

        Once in RAMSES' format a disturbance has the form

            <time of ocurrence> <descriptor>

        For instance, in the disturbance

            1.000 FAULT BUS SEVEN 0.1

        the time of ocurrence is 1.000 and the descriptor is
        FAULT BUS SEVEN 0.1. This method returns the descriptor.
        """

        if isinstance(self.object_acted_on, records.Bus):
            if self.par_name == "fault":
                # Cause short circuit (reactance in par_lambda)
                descriptor = (
                    f"FAULT BUS {self.object_acted_on.name} "
                    f"0.0 {self.par_value}"
                )
            elif self.par_name == "clearance":
                # Clear the short circuit (par_lambda == None)
                descriptor = f"CLEAR BUS {self.object_acted_on.name}"

        elif isinstance(self.object_acted_on, records.Branch):
            # Get new breaker status
            status = int(self.par_value)
            # Change state of branch
            descriptor = (
                f"BREAKER BRANCH {self.object_acted_on.name} {status} {status}"
            )

        elif isinstance(self.object_acted_on, records.Injector):
            # Change parameter of an injector
            descriptor = (
                f"CHGPRM INJ {self.object_acted_on.name} {self.par_name} "
                f"{self.par_value} SETP 0.0"
            )

        elif isinstance(self.object_acted_on, records.DCTL):
            # Change parameter of the DCTL
            descriptor = (
                f"CHGPRM DCTL {self.object_acted_on.name} {self.par_name} "
                f"{self.par_value} 0.0"
            )

        elif self.object_acted_on == "solver":
            # Change solver parameters
            descriptor = self.par_value

        return descriptor

    def __eq__(self, other: "Disturbance") -> bool:
        """
        Overload disturbance equality: same time and action (descriptor).
        """

        # We use np.isclose because the time is a float (1 ms does not matter).
        return np.isclose(
            self.ocurrence_time, other.ocurrence_time, atol=1e-3
        ) and str(self) == str(other)


class Observable:
    """
    Class for representing observables.

    When writing the observables.dat file, only the observed_object attribute
    should be paid attention to, not the observable name. This is used when
    extracting the observables.
    """

    def __init__(
        self,
        observed_object: Union[records.Record, records.EXC],
        obs_name: str,
    ) -> None:
        self.observed_object = observed_object
        self.obs_name = obs_name

        # Infer type: BUS, SYNC, INJEC, SHUNT, BRANCH, DCTL
        if isinstance(observed_object, records.Bus):
            self.element_type = "BUS"

        elif isinstance(observed_object, records.Branch):
            self.element_type = "BRANCH"

        elif isinstance(observed_object, records.Generator) or isinstance(
            observed_object, records.EXC
        ):
            self.element_type = "SYNC"

        elif isinstance(observed_object, records.Injector):
            if isinstance(observed_object, records.Shunt):
                self.element_type = "SHUNT"
            else:
                self.element_type = "INJEC"

        elif isinstance(observed_object, records.DCTL):
            self.element_type = "DCTL"

        else:
            raise RuntimeError(
                f"Observable {observed_object} has unknown type."
            )

    def __str__(self) -> str:
        """
        Return the observable as a string in the format of the .dat file.
        """

        return f"{self.element_type} {self.observed_object.name}"

    def __eq__(self, other: "Observable") -> bool:
        """
        Implement observable equality so that duplicates can be removed.
        """

        return str(self) == str(other)  # and self.obs_name == other.obs_name

    def __hash__(self) -> int:
        """
        Implement hash function so that observables can used as keys in sets.
        """

        return hash(f"{self} {self.obs_name}")

    def __lt__(self, other: "Observable") -> bool:
        """
        Implement ordering to sort observables alphabetically by type.
        """

        return self.element_type < other.element_type
