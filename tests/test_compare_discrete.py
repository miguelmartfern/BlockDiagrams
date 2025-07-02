import numpy as np
import pytest
from signalblocks import DiscreteSignalPlotterRef, DiscreteSignalPlotterNew

tests = [
    ("x[n] = delta[n]"),
    ("x[n] = u[n]"),
    ("x[n] = rect[n]"),
    ("x[n] = tri[n]"),
    ("x[n] = delta[n] + u[n]"),
]

@pytest.mark.parametrize("expr_str", tests)
def test_discrete_plotter_equivalence(expr_str):
    show = False  # Cambiar a True para visualizar las señales

    dsp_ref = DiscreteSignalPlotterRef(horiz_range=(-5, 5), show_plot=False)
    dsp_new = DiscreteSignalPlotterNew(horiz_range=(-5, 5), show_plot=False)

    dsp_ref.add_signal(expr_str)
    dsp_new.add_signal(expr_str)

    dsp_ref.plot("x")
    dsp_new.plot("x")

    n_vals = np.arange(-5, 6)

    # Evaluar punto a punto para evitar problemas con funciones no vectorizables
    y_ref = np.array([dsp_ref.funcs["x"](int(n)) for n in n_vals], dtype=float)
    y_new = np.array([dsp_new.funcs["x"](int(n)) for n in n_vals], dtype=float)

    assert np.allclose(y_ref, y_new, atol=1e-8), (
        f"Mismatch between Ref and New for {expr_str}:\nRef: {y_ref}\nNew: {y_new}"
    )
