from .DiagramBuilder import DiagramBuilder
# from .SignalPlotter import SignalPlotter
from .ComplexPlane import ComplexPlane
# from .DiscreteSignalPlotter import DiscreteSignalPlotter

# __all__ = ["DiagramBuilder", "SignalPlotter", "DiscreteSignalPlotter", "ComplexPlane"]

from .ContinuousSignalPlotter import ContinuousSignalPlotter
from .SignalPlotter import SignalPlotter

from .DiscreteSignalPlotter import DiscreteSignalPlotter
from .DiscreteSignalPlotter_old import DiscreteSignalPlotterOld

DiscreteSignalPlotterNew = DiscreteSignalPlotter
DiscreteSignalPlotterRef = DiscreteSignalPlotterOld
ContinuousSignalPlotterNew = ContinuousSignalPlotter
ContinuousSignalPlotterRef = SignalPlotter

__all__ = [
    "DiagramBuilder",
    "ComplexPlane",
    "ContinuousSignalPlotterNew",
    "ContinuousSignalPlotterRef",
    "DiscreteSignalPlotterNew",
    "DiscreteSignalPlotterRef",
    # otros módulos o utilidades
]