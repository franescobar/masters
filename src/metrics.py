"""
A module for defining performance metrics.
"""

# Modules from this repository
import pf_dynamic
import records
import nli

# Modules from the standard library

# Other modules
import numpy as np
import pyramses
from pyramses import simulator

class Metric:
    """
    A class for specifying performance metrics.
    """

    pass

class VoltageIntegral(Metric):
    """
    Compute the integral of voltage deviations from their initial value.

    Ideally, this metric would be zero.
    """

    name = "Discrete average of (time average of (V - V_initial))"
    units = "pu"

    def __init__(self, only_central: bool) -> None:
        """
        If only_central is True, then only the central buses are considered.
        """

        if not isinstance(only_central, bool):
            raise ValueError("only_central must be a boolean.")
        
        self.only_central = only_central

        if only_central:
            self.name += " - CENTRAL"

    def evaluate(self, 
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> float:
        
        # The computation of this metric only includes load buses, as it is
        # their voltages what one tries to keep within limits.
        if self.only_central:
            secondary_buses = [
                transformer.get_LV_bus()
                for transformer in system.transformers
                if transformer.touches(location="CENTRAL")
            ]
            load_buses = []
            for bus in secondary_buses:
                load_buses += list(system.isolate_buses_by_kV(starting_bus=bus))
        else:
            load_buses = [bus for bus in system.buses 
                        if any(inj.bus is bus for inj in system.injectors
                                if isinstance(inj, records.Load))]
            
        # It is useful that the indicator averages the deviation across several
        # buses, hence we consider the number of such buses.
        N = len(load_buses)

        delta_V_offset = 0
        for bus in load_buses:
            data = extractor.getBus(bus.name)
            time = data.mag.time
            voltage = data.mag.value
            delta_t = time[-1] - time[0]
            integrand = np.abs(voltage - voltage[0])
            delta_V_offset += np.trapz(x=time, y=integrand) / delta_t / N
        
        return delta_V_offset
        
class ReactiveMargin(Metric):
    """
    A measure of the reactive power that remains from synchronous machines.

    Ideally, this metric would be very positive.
    """

    name = "Discrete average of (final value of (IFLIM - IF))"
    units = "norm. wrt. iflim"

    def __init__(self, only_central: bool) -> None:
        """
        If only_central is True, then only the central generators are considered.
        """

        if not isinstance(only_central, bool):
            raise ValueError("only_central must be a boolean.")
        
        self.only_central = only_central

        if only_central:
            self.name += " - CENTRAL"

    def evaluate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> float:
        
        if self.only_central:
            generator_names = [
                "g6", 
                "g7", 
                "g13",
                "g14", 
                "g15",
                "g16",
            ]
            generators = [system.get_generator(name=generator_name)
                          for generator_name in generator_names]
        else:
            generators = system.generators

        N = len(generators)

        margin = 0
        for generator in generators:
            data = extractor.getExc(syncname=generator.name)
            data = getattr(data, "if")
            time = data.time
            field_current = data.value
            delta_t = time[-1] - time[0]
            # The following change returns this distance as normalized
            integrand = (float(generator.exciter.iflim) - field_current)/float(generator.exciter.iflim)
            margin += np.trapz(x=time, y=integrand) / delta_t / N

        return margin
    
class ControlEffort(Metric):
    """
    A measure of the effort made by DERAs.

    Ideally, this metric would be zero.
    """


    def __init__(self, power_type: str) -> None:
        """
        power should be either "P" or "Q".
        """

        if power_type not in {"P", "Q"}:
            raise ValueError(f"Unknown power type {power_type}.")

        self.power_type = power_type
        self.name = (
            f"Discrete average of (time average of ("
            f"{power_type} - {power_type}_initial))"
        )
        self.units = "MW" if power_type == "P" else "Mvar"

    def evaluate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> float:
        
        DERAs = [inj 
                 for inj in system.injectors 
                 if isinstance(inj, records.DERA)]
        N = len(DERAs)
        
        effort = 0

        for DERA in DERAs:
            if abs(DERA.get_P()) > 0.1 or abs(DERA.get_Q()) > 0.1:
                if self.power_type == "P":
                    data = extractor.getInj(DERA.name).Pgen
                elif self.power_type == "Q":
                    data = extractor.getInj(DERA.name).Qgen
                time = data.time
                power_MVA = data.value * DERA.Snom_MVA
                integrand = power_MVA - power_MVA[0]
                delta_t = time[-1] - time[0]
                effort += np.trapz(x=time, y=integrand) / delta_t / N
        
        return effort


class TapMovements(Metric):
    """
    Measure the number of tap movements up, down, and total.
    """

    units = "movements"

    def __init__(self, only_central: bool) -> None:
        """
        If only_central is True, then only the central transformers are considered.
        """

        if not isinstance(only_central, bool):
            raise ValueError("only_central must be a boolean.")
        
        self.only_central = only_central
        self.name = "Total number of (remaining tap movements)"
        if only_central:
            self.name += " - CENTRAL"

    def evaluate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> float:
        
        # DCTLs = filter(lambda r: isinstance(r, records.DCTL), system.records)

        if self.only_central:
            DCTLs = []
            for transformer in system.transformers:
                if transformer.touches(location="CENTRAL"):
                    DCTLs.append(transformer.OLTC.OLTC_controller)
        else:
            DCTLs = filter(lambda r: isinstance(r, records.DCTL), 
                           system.records)

        remaining_movements = 0
        for DCTL in DCTLs:
            data = extractor.getDctl(dctlname=DCTL.name).ratio
            last_ratio = data.value[-1]
            movements = (
                (last_ratio - DCTL.OLTC.nmin_pu)
                /
                (DCTL.OLTC.nmax_pu - DCTL.OLTC.nmin_pu)
                *
                (DCTL.OLTC.positions_up + DCTL.OLTC.positions_down)
            )
            remaining_movements += movements
        
        return remaining_movements 


class NLI(Metric):
    """
    Measure the average value of the NLI.
    """

    name = "Discrete average of (time average of (NLI))"
    units = ""

    def evaluate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> float:
        
        # Identify NLI detectors
        NLI_detectors = [
            d for d in system.detectors
            if isinstance(d, nli.NLI)
        ]

        N = len(NLI_detectors)

        # The following will propagate as np.nan if there are no NLI
        # measurements, which is OK.
        metric = 0
        for detector in NLI_detectors:
            values = [NLI_bar for tk, NLI_bar in detector.boundary_bus.NLI_bar]
            metric += np.mean(values) / N

        return metric
    
class PowerReserve(Metric):
    """
    Measure the power reserve in the DERs.
    """

    name = "Discrete average of (final value of (SNOM - S))"
    units = "MVA"

    def evaluate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> float:
        
        DERAs = [inj 
                 for inj in system.injectors 
                 if isinstance(inj, records.DERA)]
        N = len(DERAs)
        
        reserve_MVA = 0

        for DERA in DERAs:
            if abs(DERA.get_P()) > 0.1 or abs(DERA.get_Q()) > 0.1:
                P_MW = extractor.getInj(DERA.name).Pgen.value * DERA.Snom_MVA
                Q_Mvar = extractor.getInj(DERA.name).Qgen.value * DERA.Snom_MVA
                last_S_MVA = np.sqrt(
                    P_MW[-1]**2 + Q_Mvar[-1]**2
                )
                reserve_MVA += (DERA.Snom_MVA - last_S_MVA) / N
        
        return reserve_MVA
    
class ActivatedOELs(Metric):
    """
    Measure the number of activated OELs (possibly in the central area).
    """

    units = "generators"

    def __init__(self, only_central: bool) -> None:
        """
        If only_central is True, then only the central transformers are considered.
        """

        if not isinstance(only_central, bool):
            raise ValueError("only_central must be a boolean.")
        
        self.only_central = only_central
        self.name = "Total number of (limited generators)"
        if only_central:
            self.name += " - CENTRAL"

    def evaluate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> float:

        if self.only_central:
            generator_names = [
                "g6", 
                "g7", 
                "g13",
                "g14", 
                "g15",
                "g16",
            ]
            generators = [system.get_generator(name=generator_name)
                          for generator_name in generator_names]
        else:
            generators = system.generators

        count = 0
        for generator in generators:
            data = extractor.getExc(syncname=generator.name).zswitch
            final_state = data.value[-1]
            if final_state > 0.5:
                count += 1

        return count
        