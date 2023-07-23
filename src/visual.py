"""
    Visualization ideas:
    - overloading of the lines

    Nice idea:
    - create
    - example

    Really nice idea:
    - a visualization can be a table. It must not necessarily be a picture.
      I could have subclasses picture and table, to know how to wrap them.

     More ideas for visualizations:
     - showing the trajectory on the sigma plane for different substations
     - maybe illustrate the mapping I did? Not really
"""

from collections.abc import Sequence
# import control
import nli
import pyramses
import pf_dynamic
import matplotlib.pyplot as plt

class Visualization:
    pass


class NLI_plots(Visualization):
    """
    A test visualization class.
    """

    def __init__(self, receiving_buses: Sequence[str]) -> None:

        self.receiving_buses = receiving_buses

    def generate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> None:

        NLI_detectors = (d for d in system.detectors if isinstance(d, nli.NLI))

        plt.figure()
        for detector in NLI_detectors:
            if (
                not detector.boundary_bus.NLI_bar \
                or detector.boundary_bus.name not in self.receiving_buses
            ):
                continue

            x, y = list(zip(*detector.boundary_bus.NLI_bar))
            plt.plot(x, y,
                    label=f"Bus {detector.boundary_bus.name}")
        plt.show()

class CentralVoltages(Visualization):

    def generate(self, 
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor) -> None:
        
        central_buses = (bus for bus in system.buses if bus.location == "CENTRAL")

        plt.figure()
        for bus in central_buses:
            data = extractor.getBus(bus.name)
            time = data.mag.time
            voltage = data.mag.value
            plt.plot(time, voltage, label=bus.name)
        plt.legend()
        plt.xlabel("Time (s)") 
        plt.ylabel("Voltage (pu)")
        plt.title("Central voltages")
        plt.show()


