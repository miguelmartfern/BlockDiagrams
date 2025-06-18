# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Miguel Á. Martín <miguelmartfern@github>

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import re

class DiscreteSignalPlotter:
    """
    Class for symbolic definition and plotting of discrete-time signals.
    """

    def __init__(self, expr_str=None, n_range=(-10, 10), figsize=(8, 4), color='black'):
        self.n = sp.Symbol('n', integer=True)
        self.signal_defs = {}
        self.custom_labels = {}
        self.figsize = figsize
        self.color = color
        self.n_range = n_range
        self._build_local_dict()

        if expr_str:
            self.add_signal(expr_str)

    def _build_local_dict(self):
        """Defines the discrete primitives."""
        self.local_dict = {
            'delta': lambda n: sp.KroneckerDelta(n, 0),
            'u': lambda n: sp.Piecewise((1, n >= 0), (0, True)),
            'rect': lambda n: sp.Piecewise((1, abs(n) <= 1), (0, True)),
            'tri': lambda n: sp.Piecewise((1 - abs(n)/3, abs(n) <= 3), (0, True)),
            'sinc': lambda n: sp.sinc(n),
            'KroneckerDelta': sp.KroneckerDelta,
            'Piecewise': sp.Piecewise
        }

    def add_signal(self, expr_str, label=None):
        """
        Adds a new discrete-time signal to the internal dictionary.

        Args:
            expr_str (str): Definition like "x[n] = delta[n-2] + u[n]".
            label (str, optional): Custom y-axis label for this signal.
        """
        m = re.match(r"(?P<name>\w+)\s*\[\s*(?P<var>\w+)\s*\]\s*=\s*(?P<expr>.+)", expr_str)
        if not m:
            raise ValueError("Expression must be in form: name[n] = ...")

        name, expr_body = m.group('name'), m.group('expr')
        expr_body = re.sub(r"delta\s*\[\s*(.+?)\s*\]", r"KroneckerDelta(\1, 0)", expr_body)
        expr_body = re.sub(r"u\s*\[\s*(.+?)\s*\]", r"Piecewise((1, \1 >= 0), (0, True))", expr_body)

        for other_name in self.signal_defs:
            expr_body = re.sub(
                rf"{other_name}\s*\[\s*(.+?)\s*\]",
                rf"{other_name}(\1)",
                expr_body
            )

        local_dict = self.local_dict.copy()
        for other_name, other_expr in self.signal_defs.items():
            local_dict[other_name] = sp.Lambda(self.n, other_expr.subs(self.n, self.n))

        expr = sp.parse_expr(expr_body, local_dict=local_dict, transformations="all")
        self.signal_defs[name] = expr

        if label:
            self.custom_labels[name] = label

    def _evaluate_signal(self, expr):
        n_vals = np.arange(self.n_range[0], self.n_range[1]+1)
        f_lamb = sp.lambdify(self.n, expr, modules=["numpy", self.local_dict])
        y_vals = np.array([f_lamb(int(n_i)) for n_i in n_vals], dtype=float)
        return n_vals, y_vals

    def setup_axes(self, horiz_range=None):
        """
        Configures the plot axes: hides borders, sets limits, and draws arrow-like axes.
        This method is typically called after `_prepare_plot()` to finalize the plot appearance.
        This method is usually called internally from `plot()`, but can also be used manually.

        This method:
        - Hides the default box (spines) around the plot.
        - Clears all default ticks.
        - Sets the horizontal and vertical limits based on the signal range.
        - Adds margin space around the plotted data to improve visual clarity.
        - Draws custom x- and y-axis arrows using `annotate`.
        - Calls `tight_layout()` to prevent label clipping.

        Notes:
        - The horizontal axis includes a 20% margin on both sides.
        - The vertical axis includes 30% below and 60% above the data range.
        - The vertical range must be computed beforehand via `_prepare_plot()`.

        Args:
            horiz_range (tuple, optional): If provided, overrides the default horizontal range.

        Examples:
            >>> self._prepare_plot()
            >>> self.setup_axes()
        """
        # Hide all axis spines (borders)
        for spine in self.ax.spines.values():
            spine.set_color('none')

        # Remove default ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        if horiz_range is None:
            horiz_range = self.horiz_range

        # Compute horizontal range and margin
        x0, x1 = horiz_range
        x_range = x1 - x0
        # You can adjust this value if needed
        x_margin = 0.2 * x_range

        # Use vertical range computed in _prepare_plot
        y_min, y_max = self.y_min, self.y_max
        y_range = y_max - y_min

        # Add a 30% margin below and 60% margin above the signal range
        if y_range <= 0:
            # In degenerate cases, ensure a minimum visible height
            y_margin = 1.0
        else:
            y_margin = 0.3 * y_range

        self.ax.set_xlim(horiz_range[0] - x_margin, horiz_range[1] + x_margin)
        self.ax.set_ylim(self.y_min - y_margin, self.y_max + 1.6 * y_margin)

        # Draw x-axis arrow
        self.ax.annotate('', xy=(self.ax.get_xlim()[1], 0), xytext=(self.ax.get_xlim()[0], 0),
                         arrowprops=dict(arrowstyle='-|>', linewidth=1.5, color='black',
                                         mutation_scale=16, mutation_aspect=0.8, fc='black'))

        # Draw x-axis arrow
        self.ax.annotate('', xy=(0, self.ax.get_ylim()[1]), xytext=(0, self.ax.get_ylim()[0]),
                         arrowprops=dict(arrowstyle='-|>', linewidth=1.5, color='black',
                                         mutation_scale=12, mutation_aspect=2, fc='black'))
        
        # Prevent labels from being clipped
        self.fig.tight_layout()
        
    def plot(self, name):
        """
        Plots the given discrete-time signal by name.

        Args:
            name (str): Name of signal to plot (must have been defined).
        """
        if name not in self.signal_defs:
            raise ValueError(f"Signal '{name}' is not defined.")

        expr = self.signal_defs[name]
        n_vals, y_vals = self._evaluate_signal(expr)

        fig, ax = plt.subplots(figsize=self.figsize)
        
        markerline, stemlines, baseline = ax.stem(n_vals, y_vals)
        markerline.set_color(self.color)
        stemlines.set_color(self.color)
        markerline.set_marker('o')

        plt.setp(markerline, markersize=6)
        plt.setp(stemlines, linewidth=1.8)

        self.setup_axes()

        # ax.axhline(0, color='black', linewidth=1, zorder=0)
        # ax.axvline(0, color='black', linewidth=1, zorder=0)
        # ax.grid(True, linestyle=':', alpha=0.5)
        ax.set_xticks(n_vals)
        # ax.set_xlabel("$n$")
        # ax.set_ylabel(self.custom_labels.get(name, f"${name}[n]$"))
        plt.show()
