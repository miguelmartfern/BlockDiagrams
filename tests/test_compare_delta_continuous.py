import numpy as np
import pytest
import sympy as sp

from signalblocks import ContinuousSignalPlotterRef, ContinuousSignalPlotterNew

def test_continuous_plotter_impulse_equivalence():
    """Test that ContinuousSignalPlotter and SignalPlotter detect and handle delta(t) identically."""

    # --- Visualización opcional ---
    show = False
    
    expr_str = "x(t)=delta(t-0.5)"
    horiz_range = (-1, 1)

    # Configurar ambas versiones
    sp_old = ContinuousSignalPlotterRef(horiz_range=horiz_range, show_plot=True)
    sp_new = ContinuousSignalPlotterNew(horiz_range=horiz_range, show_plot=True)

    sp_old.add_signal(expr_str)
    sp_new.add_signal(expr_str)

    # --- Extract impulses ---
    impulses_old = sp_old._extract_impulses()
    impulses_new = sp_new._extract_impulses()

    # Comprobar que ambos detectan impulsos
    assert impulses_old is not None, "Old plotter did not detect impulses"
    assert impulses_new is not None, "New plotter did not detect impulses"

    positions_old, amplitudes_old = impulses_old
    positions_new, amplitudes_new = impulses_new

    # Deben detectar el mismo número de impulsos
    assert len(positions_old) == len(positions_new), "Different number of impulses detected"

    # Deben detectar el mismo impulso en la misma posición
    assert np.allclose(positions_old, positions_new, atol=1e-12), (
        f"Impulse positions differ: {positions_old} vs {positions_new}"
    )

    # Deben tener la misma amplitud
    assert np.allclose(amplitudes_old, amplitudes_new, atol=1e-12), (
        f"Impulse amplitudes differ: {amplitudes_old} vs {amplitudes_new}"
    )

    # --- Opcional: visualización manual ---
    if show:
        sp_old.plot("x")
        sp_new.plot("x")
