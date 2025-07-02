import numpy as np
import pytest

from signalblocks import ContinuousSignalPlotterRef, ContinuousSignalPlotterNew

# Señales de prueba: (x_expr, h_expr, rango de tiempo)
tests = [
    ("x(t)=rect(t)", "h(t)=tri(t)", (-5, 5)),
    ("x(t)=tri(t)", "h(t)=tri(t)", (-5, 5)),
    ("x(t)=sin(2*pi*t)*u(t)", "h(t)=exp(-t)*u(t)", (0, 10)),
]


@pytest.mark.parametrize("expr_x, expr_h, horiz_range", tests)
def test_convolution_continuous_equivalence(expr_x, expr_h, horiz_range):
    """Test numerical convolution equivalence between ContinuousSignalPlotterRef and ContinuousSignalPlotterNew."""

    # --- Visualización opcional ---
    show = False

    # Configurar plotters
    sp_old = ContinuousSignalPlotterRef(horiz_range=horiz_range, show_plot=show)
    sp_new = ContinuousSignalPlotterNew(horiz_range=horiz_range, show_plot=show)

    # Añadir señales
    sp_old.add_signal(expr_x)
    sp_old.add_signal(expr_h)
    sp_new.add_signal(expr_x)
    sp_new.add_signal(expr_h)

    # Realizar convolución (se añade internamente como y(t))
    sp_old.plot_convolution("x", "h", show=True)
    sp_new.plot_convolution("x", "h", show=True)

    # Evaluar la convolución en ambas versiones sobre la misma rejilla
    t_vals = np.linspace(*horiz_range, 1000)
    y_old = sp_old._evaluate_signal(t_vals)
    y_new = sp_new._evaluate_signal(t_vals)

    # Comparar con tolerancia estricta
    assert np.allclose(y_old, y_new, atol=1e-8), (
        f"Convolution mismatch between versions for {expr_x} * {expr_h}: "
        f"max diff = {np.max(np.abs(y_old - y_new))}"
    )

    # --- Visualización opcional ---
    # if False:
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 3))
        # plt.plot(t_vals, y_old, label="ContinuousSignalPlotterRef", color="blue")
        # plt.plot(t_vals, y_new, "--", label="ContinuousSignalPlotterNew", color="red")
        # plt.title(f"Convolution: {expr_x} * {expr_h}")
        # plt.xlabel("t")
        # plt.ylabel("y(t)")
        # plt.legend()
        # plt.grid(True, ls=":")
        # plt.tight_layout()
        # plt.show()