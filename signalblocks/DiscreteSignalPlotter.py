# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Miguel Á. Martín <miguelmartfern@github>

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import re
import warnings

class DiscreteSignalPlotter:
    """
    Discrete signal plotting class for symbolic discrete-time signals x[n].
    
    Supports:
    - Dirac impulses (Kronecker delta)
    - Unit step
    - Rectangular, triangular and sinc primitives
    - Symbolic manipulation using SymPy
    - Multiple signals stored and plotted
    
    Args:
        expr_str (str, optional): Optional initial signal to add on creation.
        n_range (tuple, optional): Range of n values (start, end) for plotting.
        figsize (tuple, optional): Figure size.
        color (str, optional): Default plot color.
        
    Example:
        >>> dsp = DiscreteSignalPlotter(n_range=(-5, 10))
        >>> dsp.add_signal("x[n] = delta(n) + delta(n-2) + u(n-3)")
        >>> dsp.plot("x")
    """
    def __init__(self, expr_str=None, n_range=(-10, 10), figsize=(8, 3), color='black'):
        self.signal_defs = {}
        self.custom_labels = {}
        self.figsize = figsize
        self.color = color
        self.n_range = n_range
        self._define_local_dict()
        if expr_str:
            self.add_signal(expr_str)

    def _define_local_dict(self):
        """Define the primitives available in discrete-time domain"""
        self.n = sp.Symbol('n', integer=True)
        self.local_dict = {
            'n': self.n,
            'delta': lambda n: sp.KroneckerDelta(n, 0),
            'u': lambda n: sp.Piecewise((1, n >= 0), (0, True)),
            'rect': lambda n: sp.Piecewise((1, abs(n) <= 1), (0, True)),
            'tri': lambda n: sp.Piecewise((1 - abs(n) / 3, abs(n) <= 3), (0, True)),
            'sinc': lambda n: sp.sinc(n),
            'KroneckerDelta': sp.KroneckerDelta,
            'Piecewise': sp.Piecewise
        }

    def add_signal(self, expr_str, label=None):
        """
        Adds a discrete signal definition in the form "name[n] = expression".
        
        Args:
            expr_str (str): Example: "x[n] = delta(n) + u(n-3)".
            label (str, optional): Custom label for vertical axis when plotting.
            
        Examples:
            >>> dsp.add_signal("x[n] = delta(n) + delta(n-2)")
            >>> dsp.add_signal("y[n] = x(n-1) + u(n)")
        """
        m = re.match(r"^\s*(?P<name>[^\W\d_]+)\[\s*n\s*\]\s*=\s*(?P<expr>.+)$", expr_str)
        if not m:
            raise ValueError("Expression must have the form: name[n] = ...")
        name, expr_body = m.group('name'), m.group('expr')

        # Allow reference to previously defined signals
        local_dict = self.local_dict.copy()
        for k, v in self.signal_defs.items():
            local_dict[k] = v

        expr = sp.parse_expr(expr_body, local_dict=local_dict, transformations='all')
        self.signal_defs[name] = expr
        if label:
            self.custom_labels[name] = label

    def plot(self, name):
        """
        Plot the specified signal.
        
        Args:
            name (str): Name of the signal previously added.
            
        Examples:
            >>> dsp.plot("x")
        """
        if name not in self.signal_defs:
            raise ValueError(f"Signal '{name}' not defined.")
        
        expr = self.signal_defs[name]
        f_lamb = sp.lambdify(self.n, expr, modules=["numpy"])

        n_vals = np.arange(self.n_range[0], self.n_range[1] + 1)
        y_vals = np.array([f_lamb(int(n_i)) for n_i in n_vals], dtype=float)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.stem(n_vals, y_vals, linefmt=self.color, markerfmt=f"{self.color}", basefmt="k")
        ax.axhline(0, color='black', lw=1)
        ax.set_xlabel(r"$n$")
        ylabel = self.custom_labels.get(name, f"${name}[n]$")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        plt.show()
