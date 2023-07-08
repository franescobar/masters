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
                 system: pf_dynamic.system,
                 extractor: pyramses.extractor) -> None:

        NLI_detectors = (d for d in system.detectors if isinstance(d, nli.NLI))

        plt.figure()
        for detector in NLI_detectors:
            if detector.boundary_bus.name in self.receiving_buses:
                x, y = list(zip(*detector.boundary_bus.NLI_bar))
                plt.plot(x, y,
                         label=f"Bus {detector.boundary_bus.name}")
        plt.show()


