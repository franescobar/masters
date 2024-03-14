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
import utils
import shutil
import records

# Other modules
import numpy as np

# Modules from the standard library
import os

class Visualization:

    def get_filepath(self, vis_dir: str, filename: str) -> str:

        return os.path.join(vis_dir, self.name, filename)
    
    def save_data(self, vis_dir: str, filename: str, header: str, *columns) -> None:

        np.savetxt(
            fname=self.get_filepath(vis_dir=vis_dir, filename=filename),
            X=np.hstack(
                [column.reshape(column.shape[0], 1) for column in columns]
            ),
            header=header
        )

    def generate_dir(self, vis_dir: str) -> None:

        new_dir = os.path.join(vis_dir, self.name)

        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        else:
            shutil.rmtree(new_dir)
            os.mkdir(new_dir)

    def save_figure(self, vis_dir: str, filename: str) -> None:

        plt.savefig(
            self.get_filepath(vis_dir=vis_dir, filename=filename)
        )
        plt.close()
    
    def generate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor,
                 vis_dir: str) -> None:
        """
        A method that should be present in all visualizations.

        This method generates a PDF with the visualization and also prints
        the required data to .txt files, to be included then in LaTeX.
        """

        pass


class NLIPlots(Visualization):
    """
    A test visualization class.
    """

    name = "time_NLI"

    def __init__(self, receiving_buses: Sequence[str]) -> None:

        self.receiving_buses = receiving_buses

    def generate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor,
                 vis_dir: str) -> None:
        
        NLI_detectors = (d for d in system.detectors if isinstance(d, nli.NLI))

        plt.figure()
        for detector in NLI_detectors:
            # Skip this detector if it has no measurements
            if (
                not detector.boundary_bus.NLI_bar \
                or detector.boundary_bus.name not in self.receiving_buses
            ):
                continue
            # Extract the x and y coordinates
            x, y = list(zip(*detector.boundary_bus.NLI_bar))
            # Reduce them
            x, y = utils.reduce(
                x=x,
                y=y,
                step=10, # ensure samples every 10 s
                integral_tol=1e-3,
            )
            # Add them to the plot
            plt.plot(x, y,
                    label=f"Bus {detector.boundary_bus.name}")
            # Save them to files
            self.save_data(
                vis_dir,
                f"time_NLI_{detector.boundary_bus.name}.txt",
                "Time (s), NLI (no units)",
                x,
                y
            )
        # Having added the plots, decorate them
        plt.xlabel("Time (s)")
        plt.ylabel("NLI")
        plt.title("NLI at boundary buses of the Central Area")
        plt.legend()

        # Finally, save the figure without displaying it and close it
        self.save_figure(vis_dir=vis_dir, filename="time_NLI.pdf")

class CentralVoltages(Visualization):

    name = "time_voltage"

    def generate(self, 
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor,
                 vis_dir: str) -> None:
        
        central_buses = (bus for bus in system.buses if bus.location == "CENTRAL")
        # central_buses = [b for b in system.buses if hasattr(b, "is_monitored")]

        plt.figure()
        for bus in central_buses:
            # Extract the x and y coordinates
            data = extractor.getBus(bus.name)
            time = data.mag.time
            voltage = data.mag.value
            # Reduce them
            x, y = utils.reduce(
                x=time,
                y=voltage,
                step=10,
                integral_tol=1e-1, # would be 0.0001 pu every 1000 seconds
            )
            # Add them to the plot
            plt.plot(time, voltage, label=bus.name)
            # Save them to files
            self.save_data(
                vis_dir,
                f"time_voltage_{bus.name}.txt",
                "Time (s), Voltage (pu)",
                x,
                y,
            )
        # Having added the plots, decorate the figure
        plt.legend()
        plt.xlabel("Time (s)") 
        plt.ylabel("Voltage (pu)")
        plt.title("Distribution (load) voltages from the Central Area")
        self.save_figure(vis_dir=vis_dir, filename="time_voltage.pdf")

class FieldCurrents(Visualization):

    name = "time_fieldcurrentnorm"

    def generate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor,
                 vis_dir: str) -> None:
        
        # Initialize figure
        plt.figure()

        for generator in system.generators:
            # Extract data
            data = extractor.getExc(syncname=generator.name)
            data = getattr(data, "if")
            time = data.time
            normalized_fieldcurrent = data.value/float(generator.exciter.iflim)
            # Reduce them
            x, y = utils.reduce(
                x=time,
                y=normalized_fieldcurrent,
                step=10,
                integral_tol=1e0, # gives 0.001 for every 1000 s
            )
            # Add them to the plot
            plt.plot(x, y, label=generator.name)
            # Save them to file
            self.save_data(
                vis_dir,
                f"time_fieldcurrentnorm_{generator.name}.txt",
                "Time (s), Field current (norm.)",
                x,
                y,
            )
        # Having added the plots, decorate them
        plt.xlabel("Time (s)")
        plt.ylabel("Field current (norm.)")
        plt.title("Field currents normalized w.r.t. thermal limit")
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        # Finally, save figure
        self.save_figure(vis_dir=vis_dir, filename="time_fieldcurrentnorm.pdf")

class DERAPowers(Visualization):

    name = "time_DERApower"

    def generate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor,
                 vis_dir: str) -> None:
        
        # Locate DERAs
        # DERAs = (inj for inj in system.injectors if isinstance(inj, records.DERA))
        DERAs = [inj for inj in system.injectors
                 if isinstance(inj, records.DERA) and hasattr(inj, "is_monitored")]

        # Draw one picture per DERA
        for DERA in DERAs:
            # Extract powers in SI units
            data_P = extractor.getInj(DERA.name).Pgen
            data_Q = extractor.getInj(DERA.name).Qgen
            # Extract time from either dataset
            time = data_P.time
            # Extract values
            P = data_P.value * float(DERA.Snom_MVA)
            Q = data_Q.value * float(DERA.Snom_MVA)
            # Reduce them
            x_P, y_P = utils.reduce(
                x=time,
                y=P,
                step=10,
                integral_tol=1e-1 # gives 0.0001 for 100 seconds
            )
            x_Q, y_Q = utils.reduce(
                x=time,
                y=Q,
                step=10,
                integral_tol=1e-1 # gives 0.0001 for 100 seconds
            )
            # Add them to the plot
            plt.figure()
            plt.plot(x_P, y_P, label="P (MW)")
            plt.plot(x_Q, y_Q, label="Q (Mvar)")
            plt.legend()
            plt.xlabel("Time (s)")
            plt.ylabel("Power")
            name = DERA.bus.name
            plt.title(f"Generated powers at bus {name} in SI units")
            self.save_figure(vis_dir=vis_dir, filename=f"time_DERApower_{name}.pdf")
            # Save the data to files
            self.save_data(
                vis_dir,
                f"time_DERApowerP_{name}.txt",
                "Time (s), P (MW)",
                x_P,
                y_P,
            )
            self.save_data(
                vis_dir,
                f"time_DERApowerQ_{name}.txt",
                "Time (s), Q (Mvar)",
                x_Q,
                y_Q,
            )

class VoltageTrajectories(Visualization):

    name = "VT_VD"

    def generate(self,
                 system: pf_dynamic.System,
                 extractor: pyramses.extractor,
                 vis_dir: str) -> None:
        
        # Get transformers from Central Area
        central_transformers = (
            transformer for transformer in system.transformers
            if transformer.touches(location="CENTRAL")
        )

        # For each one, draw the trajectory
        for transformer in central_transformers:
            # Extract data
            HV_bus_name = transformer.get_HV_bus().name
            LV_bus_name = transformer.get_LV_bus().name
            data_VT = extractor.getBus(HV_bus_name)
            data_VD = extractor.getBus(LV_bus_name)
            voltage_VT = data_VT.mag.value
            time_VT = data_VT.mag.time
            voltage_VD = data_VD.mag.value
            time_VD = data_VD.mag.time
            # Down sample every 1 s
            _, VT = utils.downsample_every(x=time_VT, y=voltage_VT, delta_x=1)
            _, VD = utils.downsample_every(x=time_VD, y=voltage_VD, delta_x=1)
            # Create the plot
            plt.figure()
            plt.plot(VT, VD)
            plt.xlabel(f"Transmission voltage, bus {HV_bus_name} (pu)")
            plt.ylabel(f"Distribution voltage, bus {LV_bus_name} (pu)")
            plt.title(f"Voltage trajectory at transformer {transformer.name}")
            self.save_figure(vis_dir=vis_dir, filename=f"VT_VD_{transformer.name}.pdf")
            # Having created the plot, save the data
            self.save_data(
                vis_dir,
                f"VT_VD_{transformer.name}.txt",
                "Transmission voltage (pu), Distribution voltage (pu)",
                VT,
                VT
            )

