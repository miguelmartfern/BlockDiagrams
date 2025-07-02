import numpy as np
import pytest

from signalblocks import ContinuousSignalPlotterRef, ContinuousSignalPlotterNew

# Señales de prueba: lista de (expresión, rango)
tests = [
    ("x(t)=rect(t)", (-2, 2)),
    ("x(t)=tri(t)", (-3, 3)),
    ("x(t)=sin(2*pi*t)*u(t)", (0, 5)),
    # ("x(t)=delta(t)", (-1, 1)),
    ("x(t)=u(t)-u(t-1)", (-1, 2))
]

@pytest.mark.parametrize("expr_str,horiz_range", tests)
def test_continuous_plotter_equivalence(expr_str, horiz_range):
    """Test that ContinuousSignalPlotter produces the same signal as SignalPlotter."""

    # --- Visualización opcional ---
    show = False
    
    # Configurar plotters
    sp_old = ContinuousSignalPlotterRef(horiz_range=horiz_range, show_plot=True)
    sp_new = ContinuousSignalPlotterNew(horiz_range=horiz_range, show_plot=True)

    # ✅ Extraer nombre de la señal
    name = expr_str.split('=')[0].strip()
    name = name.split('(')[0].strip()  # ✅ quitar (t)

    # ✅ Añadir la señal antes de evaluar
    sp_old.add_signal(expr_str)
    sp_new.add_signal(expr_str)

    # Evaluar sobre la misma rejilla
    t_vals = np.linspace(*horiz_range, 1000)

    # ✅ Llamar sin 'name=', usando el flujo normal de SignalBlocks
    y_old = sp_old._evaluate_signal(t_vals)
    y_new = sp_new._evaluate_signal(t_vals)

    # Comparar con tolerancia estricta
    assert np.allclose(y_old, y_new, atol=1e-8), (
        f"Test failed for {expr_str}: max difference = {np.max(np.abs(y_old - y_new))}"
    )

    if show:  # Cambia a False si no quieres visualizar
        print(f"\nMostrando comparación gráfica para: {expr_str}")

        # Mostrar la señal en la versión antigua
        print("Mostrando SignalBlocks (Ref)")
        sp_old.plot(name)

        # Mostrar la señal en la versión nueva
        print("Mostrando SignalBlocks (New)")
        sp_new.plot(name)