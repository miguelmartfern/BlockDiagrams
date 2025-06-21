# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2025 Miguel Á. Martín <miguelmartfern@github>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# SignalPlotter.py

import re
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from fractions import Fraction
import warnings
from scipy import integrate

class SignalPlotter:
    """
    A helper class to plot signals y a minimalistic way. It has predefined
    typical signals, like rect, tri, Heaviside, delta, sinc, ...
    It allow to do operations with signals, like time shifts and inversions,
    sums, products, convolutions, ...
    
    Args:
        expr_str (str, optional): A string expression defining the signal, e.g. "x(t)=sin(t)*u(t)".
        horiz_range (tuple, optional): Tuple (t_min, t_max) specifying the horizontal plotting range.
        vert_range (tuple, optional): Tuple (y_min, y_max) specifying the vertical range. Auto-scaled if None.
        period (float, optional): If provided, the signal is treated as periodic with this period.
        num_points (int, optional): Number of points used to discretize the time axis.
        figsize (tuple, optional): Size of the figure in centimeters (width, height).
        tick_size_px (int, optional): Size of axis tick marks in pixels.
        xticks (list or 'auto' or None): Positions of x-axis ticks. If 'auto', they are generated automatically.
        yticks (list or 'auto' or None): Same for y-axis.
        xtick_labels (list of str, optional): Labels for xticks. Must match xticks in length.
        ytick_labels (list of str, optional): Labels for yticks. Must match yticks in length.
        pi_mode (bool, optional): If True, x and y tick labels are shown as fractionary multiples of π if possible.
        fraction_ticks (bool, optional): If True, tick labels are shown as rational fractions.
        xticks_delta (float, optional): If provided, generates xticks at this interval (when xticks='auto').
        yticks_delta (float, optional): Same for yticks.
        save_path (str, optional): If provided, saves the plot to the given path instead of displaying.
        show_plot (bool, optional): Whether to show the plot window (if False and save_path is given, it only saves).
        color (str, optional): Color for the plot line and impulses.
        alpha (float, optional): Transparency for background label boxes (between 0 and 1).

    Examples:
        >>> from signalblocks import SignalPlotter
        >>> sp = SignalPlotter("x(t)=rect(t)", horiz_range=(-2, 2), pi_mode=True).plot()
        >>> SignalPlotter("x(t)=delta(t/2-1) + 3*delta(t + 2)", color='blue', figsize=(8,4)).plot('x')
        >>> signal1 = SignalPlotter("x(t)=cos(4 pi t)*tri(t/2)", alpha=0.7, horiz_range=[-3, 3], xticks=np.linspace(-2, 2, 9), color='blue', figsize=(12,4))
        >>> signal1.plot()
        >>> SignalPlotter("x(t)=pw((t**2, (t>-1) & (t<0)), (-t, (t>=0) & (t<1)), (0, True))", horiz_range=[-2.5, 2.5], xticks=np.linspace(-2, 2, 9), color='blue', period=2)
    """
    def __init__(
        self,
        expr_str=None, 
        horiz_range=(-5, 5),
        vert_range=None,
        period=None,
        num_points=1000,
        figsize=(8, 3), 
        tick_size_px=5,
        xticks='auto',
        yticks='auto',
        xtick_labels=None,
        ytick_labels=None,
        xticks_delta=None,
        yticks_delta=None,
        pi_mode=False,
        fraction_ticks=False,
        save_path=None, 
        show_plot=True,
        color='black', 
        alpha=0.5 
    ):
        """
        (Private) Creator of the SignalPlotter class.
        """
        self.signal_defs = {}
        self.var_symbols = {}
        self.custom_labels = {}
        self.signal_periods = {}
        self.current_name = None
        self.horiz_range = horiz_range
        self.vert_range = vert_range
        self.num_points = num_points
        self.figsize = figsize
        self.tick_size_px = tick_size_px
        self.color = color
        self.alpha = alpha
        self.period = period
        self.save_path = save_path
        self.show_plot = show_plot

        self.fraction_ticks = fraction_ticks

        # Preserve original tick arguments to differentiate None / [] / 'auto'
        self.init_xticks_arg = xticks
        self.init_yticks_arg = yticks

        if isinstance(xticks, (list, tuple, np.ndarray)) and len(xticks) > 0:
            self.xticks = np.array(xticks)
        else:
            self.xticks = None
        if isinstance(yticks, (list, tuple, np.ndarray)) and len(yticks) > 0:
            self.yticks = np.array(yticks)
        else:
            self.yticks = None
        
        self.pi_mode = pi_mode

        self.xtick_labels = xtick_labels
        self.ytick_labels = ytick_labels

        if self.xtick_labels is not None:
            if self.xticks is None:
                raise ValueError("xtick_labels provided without xticks positions")
            if len(self.xtick_labels) != len(self.xticks):
                raise ValueError("xtick_labels and xticks must have the same length")
        if self.ytick_labels is not None:
            if self.yticks is None:
                raise ValueError("ytick_labels provided without yticks positions")
            if len(self.ytick_labels) != len(self.yticks):
                raise ValueError("ytick_labels and yticks must have the same length")

        self.xticks_delta = xticks_delta
        self.yticks_delta = yticks_delta

        self.expr_str_pending = expr_str  # Expression to initialize later if plot() is called first

        self.transformations = (standard_transformations + (implicit_multiplication_application,))


    def _get_local_dict(self):
        """
        (Private) Returns a local dictionary of predefined symbols and functions used
        during symbolic parsing and evaluation of signal expressions.

        This includes:
        - Common signal processing functions such as:
            - u(t): Heaviside step (centered at 0.5)
            - rect(t), tri(t), sinc(t), ramp(t), delta(t)
        - Piecewise functions:
            - Piecewise, pw
        - Mathematical functions and constants:
            - sin, cos, exp, pi, abs, arg, re, im, conj
        - Symbols used in frequency/time analysis: t, ω, Ω, τ, λ
        - Support for complex signals: i, j, re, im, conj, abs, arg 
        - Previously defined signal names in the format "name(variable)"

        Returns:
            dict: A dictionary mapping names to SymPy expressions or functions.

        Examples:
            >>> local_dict = self._get_local_dict()
            >>> expr = parse_expr("x(t) + rect(t)", local_dict=local_dict)
        """
        d = {
            'u':            lambda t: sp.Heaviside(t, 0.5),
            'rect':         lambda t: sp.Piecewise((1, sp.And(t >= -0.5, t <= 0.5)), (0, True)),
            'tri':          lambda t: (1 - abs(t)) * sp.Heaviside(1 - abs(t), 0),   # 0 explícito en bordes de triángulo
            'ramp':         lambda t: sp.Heaviside(t, 0) * t,
            'sinc':         lambda t: sp.sin(sp.pi * t) / (sp.pi * t),
            'delta':        sp.DiracDelta,
            'DiracDelta':   sp.DiracDelta,
            'Heaviside':    lambda t: sp.Heaviside(t, 0.5),
            'pi':           sp.pi,
            'sin':          sp.sin,
            'cos':          sp.cos,
            'exp':          sp.exp,
            'Piecewise':    sp.Piecewise,
            'pw':           sp.Piecewise,
            're':           sp.re,
            'im':           sp.im,
            'conj':         sp.conjugate,
            'abs':          lambda x: np.abs(x),
            'arg':          sp.arg,
            'i':            sp.I,
            'j':            sp.I,
            't':            sp.Symbol('t'),
            'omega':        sp.Symbol('omega'),
            'Omega':        sp.Symbol('Omega'),
            'tau':          sp.Symbol('tau'),
            'lambda':       sp.Symbol('lambda'),
        }
        d.update(self.var_symbols)
        for name, expr in self.signal_defs.items():
            for var in self.var_symbols.values():
                d[f"{name}({var})"] = expr.subs(var, var)
        return d

    # def _initialize_expression(self, expr_str):
    #     m = re.match(r"^(?P<fn>[^\W\d_]+)\((?P<vr>[^)]+)\)\s*=\s*(?P<ex>.+)$", expr_str)
    #     if m:
    #         self.func_name = m.group('fn')
    #         var_name = m.group('vr')
    #         expr_body = m.group('ex')
    #     else:
    #         self.func_name = 'x'
    #         var_name = 't'
    #         expr_body = expr_str

    #     replacements = {'\\omega': 'ω', '\\tau': 'τ'}
    #     for latex_var, unicode_var in replacements.items():
    #         var_name = var_name.replace(latex_var, unicode_var)
    #         expr_body = expr_body.replace(latex_var, unicode_var)

    #     self.expr_str = expr_body
    #     self.var = sp.Symbol(var_name)
    #     self.xlabel = var_name
    #     self.ylabel = self.func_name + '(' + var_name + ')'

    #     self.local_dict = self._get_local_dict()

    #     transformations = standard_transformations + (implicit_multiplication_application,)
    #     self.expr = parse_expr(expr_body, local_dict=self.local_dict, transformations=transformations)

    #     self.expr_cont = self._remove_dirac_terms()
    #     self.impulse_locs, self.impulse_areas = self._extract_impulses()

    #     t0, t1 = self.horiz_range
    #     self.t_vals = np.linspace(t0, t1, self.num_points)
    #     if self.period is not None:
    #         T = self.period
    #         self.t_vals = ((self.t_vals + T/2) % T) - T/2
    #         self.t_vals.sort()

    #     self.func = sp.lambdify(self.var, self.expr_cont, modules=["numpy", self.local_dict])
    #     self.fig, self.ax = plt.subplots(figsize=self.figsize)
    #     self._prepare_plot()

    def add_signal(self, expr_str, label=None, period=None):
        r"""
        Adds a new signal to the internal dictionary for later plotting.
        - The expression is parsed symbolically using SymPy.
        - If other signals are referenced, their definitions are recursively substituted.
        - If `period` is set, the signal will be expanded as a sum of time-shifted versions over the full horizontal range.

        Args:
            expr_str (str): Signal definition in the form "name(var) = expression", e.g. "x(t) = rect(t) + delta(t-1)". The expression may include previously defined signals.
            label (str, optional): Custom label for the vertical axis when plotting this signal.
            period (float, optional): If provided, the signal will be treated as periodic with this period.

        Examples:
            >>> sp = SignalPlotter(horiz_range=(-1, 1), fraction_ticks=True, figsize=(12, 4))
            >>> sp.add_signal("x1(t) = tri(t)")
            >>> sp.add_signal("x2(t) = delta(t)", period=0.2)
            >>> sp.add_signal("x3(t) = x1(t) * (1 + x2(t))", label="x_3(t)")
            >>> sp.plot("x3")
            >>> sp.add_signal("x4(t) = exp((-2+j*8*pi)*t)*u(t)")
            >>> sp.add_signal("x5(t) = re(x4(t))", label="\Re\{x_4(t)\}")
            >>> sp.plot('x5')
        """

        replacements = {'\\omega': 'ω', '\\tau': 'τ'}
        for latex_var, unicode_var in replacements.items():
            expr_str = expr_str.replace(latex_var, unicode_var)
        m = re.match(r"^(?P<name>\w+)\((?P<var>\w+)\)\s*=\s*(?P<expr>.+)$", expr_str)

        name = m.group('name')
        var = m.group('var')
        body = m.group('expr')

        if var not in self.var_symbols:
            self.var_symbols[var] = sp.Symbol(var)
        var_sym = self.var_symbols[var]

        local_dict = self._get_local_dict()
        for other_name in self.signal_defs:
            local_dict[other_name] = sp.Function(other_name)

        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed_expr = parse_expr(body, local_dict=local_dict, transformations=transformations)

        # Perform recursive substitution of previously defined signals.
        # This enables expressions to reference earlier signals (e.g., x(t) = z(t-2) + delta(t))
        # by replacing every function call (e.g., z(t-2)) with the corresponding shifted expression.
        for other_name, other_expr in self.signal_defs.items():
            f = sp.Function(other_name)
            matches = parsed_expr.find(f)
            for call in matches:
                if isinstance(call, sp.Function):
                    arg = call.args[0]
                    replaced = other_expr.subs(var_sym, arg)
                    parsed_expr = parsed_expr.subs(call, replaced)

        self.signal_defs[name] = parsed_expr
        self.var_symbols[name] = var_sym

        # Assign custom label if argument is given
        self.custom_labels[name] = label

        if period is not None:
            # Assign period of signal if given
            self.signal_periods[name] = period

            # Expand signal as sum of shifts within range
            horiz_min, horiz_max = self.horiz_range
            num_periods = int(np.ceil((horiz_max - horiz_min) / period))
            k_range = range(-num_periods - 2, num_periods + 3)  # márgenes extra

            # Expanded as sum of shifted expressions (in SymPy)
            expanded_expr = sum(parsed_expr.subs(var_sym, var_sym - period * k) for k in k_range)

            self.signal_defs[name] = expanded_expr
        else:
            self.signal_defs[name] = parsed_expr


    def _prepare_plot(self):
        """
        (Private) Determines the vertical plotting range (y-axis limits) based on the signal values. 
        Called from `plot()`.

        This method:
        - Evaluates the continuous part of the signal over `self.t_vals`.
        - Identifies Dirac delta impulses located within the horizontal plotting range.
        - Computes the minimum and maximum of both continuous and impulsive parts.
        - Ensures a minimal vertical span to avoid flat-looking plots.
        - Uses `self.vert_range` if explicitly provided.

        Sets:
        - self.y_min, self.y_max: Vertical limits for plotting.

        Notes:
        - If evaluation fails (e.g. undefined expression), defaults to [-1, 1].
        - Delta impulses are included only if they fall within the horizontal range.

        Examples:
            >>> self._prepare_plot()
            >>> print(self.y_min, self.y_max)  # → computed bounds
        """
        try:
            # Evaluate continuous expression
            y_vals = self.func(self.t_vals)
            y_vals = np.array(y_vals, dtype=np.float64)
            y_vals = y_vals[np.isfinite(y_vals)]

            if y_vals.size > 0:
                cont_min = np.min(y_vals)
                cont_max = np.max(y_vals)
            else:
                cont_min = 0.0
                cont_max = 0.0
           
            # Visible deltas in hoirzontal range
            if self.impulse_locs and self.impulse_areas:
                t_min, t_max = self.horiz_range
                filtered_areas = [
                    area for loc, area in zip(self.impulse_locs, self.impulse_areas)
                    if t_min <= loc <= t_max
                ]
                if filtered_areas:
                    imp_min = min(filtered_areas)
                    imp_max = max(filtered_areas)
                    overall_min = min(cont_min, imp_min, 0.0)
                    overall_max = max(cont_max, imp_max, 0.0)
                else:
                    overall_min = min(cont_min, 0.0)
                    overall_max = max(cont_max, 0.0)
            else:
                overall_min = min(cont_min, 0.0)
                overall_max = max(cont_max, 0.0)

            # Fit in case range is too narrow
            if abs(overall_max - overall_min) < 1e-2:
                overall_min -= 1.0
                overall_max += 1.0

            # Apply vertical range if provided
            if self.vert_range:
                self.y_min, self.y_max = self.vert_range
            else:
                self.y_min, self.y_max = overall_min, overall_max

        except Exception:
            self.y_min, self.y_max = -1, 1


    def _evaluate_signal(self, t):
        """
        (Private) Evaluates the continuous (non-impulsive) part of the signal at the given time values.
        Notes:
        - This method assumes `self.func` is a callable created with `lambdify(...)`.
        - Ensures consistent array output for plotting, regardless of scalar/vector behavior.

        Args:
            t (array-like or scalar): Time values at which to evaluate the signal function.

        Returns:
            (Numpy.NDArray): A NumPy array of evaluated values with the same shape as `t`.
            If the result is scalar (i.e., constant function), it is broadcast across all `t`.

        Examples:
            >>> t_vals = np.linspace(-1, 1, 100)
            >>> y_vals = self._evaluate_signal(t_vals)
        """
        y = self.func(t)
        return np.full_like(t, y, dtype=float) if np.isscalar(y) else np.array(y, dtype=float)

    def _extract_impulses(self):
        """
        (Private) Extracts the locations and amplitudes of Dirac delta impulses from the signal expression.

        This method:
        - Expands the symbolic expression `self.expr` into additive terms.
        - Identifies all DiracDelta terms and their arguments.
        - Solves each delta argument for its root(s) to determine impulse location(s).
        - Computes the effective amplitude of each impulse, accounting for time scaling:
            For δ(a·t + b), amplitude is scaled by 1/|a|.
        - Merges nearby impulses numerically (within a tolerance of 1e-8) to avoid duplicates.
        - Ignores impulses with near-zero amplitude (threshold: 1e-6).

        Returns:
            impulse_locs (list of float): Time positions of Dirac delta impulses.
            impulse_areas (list of float): Corresponding amplitudes (areas) of each impulse.

        Example:
            >>> self.expr = sp.DiracDelta(t - 1) + 2*sp.DiracDelta(2*t)
            >>> locs, areas = self._extract_impulses()
            >>> print(locs)   # → [1.0, 0.0]
            >>> print(areas)  # → [1.0, 1.0]  (2*δ(2t) has area 1 due to 1/|2| scaling)
        """
        impulse_map = {}

        # Expandir y descomponer en términos
        expr_terms = self.expr.expand().as_ordered_terms()

        for term in expr_terms:
            deltas = term.atoms(sp.DiracDelta)
            for delta in deltas:
                arg = delta.args[0]
                roots = sp.solve(arg, self.var)
                amp = term.coeff(delta)
                d_arg = sp.diff(arg, self.var)
                scale = sp.Abs(d_arg)

                for r in roots:
                    try:
                        scale_val = float(scale.subs(self.var, r))
                        amp_eval = amp.subs(self.var, r).doit().evalf()
                        effective_amp = float(amp_eval) / scale_val if scale_val != 0 else 0.0
                        if abs(effective_amp) > 1e-6:
                            loc = float(r)
                            # Buscar ubicación cercana ya existente (tolerancia)
                            found = False
                            for known_loc in impulse_map:
                                if abs(known_loc - loc) < 1e-8:
                                    impulse_map[known_loc] += effective_amp
                                    found = True
                                    break
                            if not found:
                                impulse_map[loc] = effective_amp
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue

        # Filtrar deltas resultantes ≠ 0 tras sumar contribuciones
        impulse_locs = []
        impulse_areas = []
        for loc, area in impulse_map.items():
            if abs(area) > 1e-6:
                impulse_locs.append(loc)
                impulse_areas.append(area)

        return impulse_locs, impulse_areas



    def _remove_dirac_terms(self):
        """
        Removes all Dirac delta (impulse) terms from the symbolic expression.

        This method:
        - Scans `self.expr` for any subexpressions containing `DiracDelta(...)`.
        - Replaces each occurrence with 0, effectively isolating the continuous part
        of the signal (excluding impulses).

        Returns:
            sympy.Expr: A new expression identical to `self.expr` but with all DiracDelta terms removed.

        Example:
            >>> self.expr = delta(t) + sp.sin(t)
            >>> self._remove_dirac_terms()
            sin(t)
        """
        return self.expr.replace(lambda expr: expr.has(sp.DiracDelta), lambda _: 0)


    def draw_function(self, horiz_range=None):
        """
        Plots the continuous part of the signal over the specified horizontal range.
        This method is typically called after `setup_axes()`.
        This method is usually called internally from `plot()`, but can also be used manually.

        This method:
        - Evaluates the function defined in `self.func` across `self.t_vals`.
        - Plots the result as a smooth curve using the configured color and linewidth.
        - Automatically detects and adds ellipsis ("⋯") on the left/right ends if:
            - The signal is marked as periodic, or
            - Significant energy exists just outside the plotting range.

        Notes:
        - This method does not draw delta impulses. Use `draw_impulses()` for that.
        - Ellipsis are drawn at 5% beyond the plot edges when appropriate.

        Args:
            horiz_range (tuple, optional): Tuple (t_min, t_max) to override the default horizontal range. If None, uses `self.horiz_range`.

        Examples:
            >>> self.draw_function()
            >>> self.draw_impulses()  # to add deltas on top of the curve
        """

        if horiz_range is None:
            horiz_range = self.horiz_range

        t0, t1 = horiz_range
        t_plot = self.t_vals
        y_plot = self._evaluate_signal(t_plot)

        # Assure arrays and format
        t_plot = np.array(t_plot)
        y_plot = np.array(y_plot)
        if y_plot.ndim == 0:
            y_plot = np.full_like(t_plot, y_plot, dtype=float)

        # Plot curve
        self.ax.plot(t_plot, y_plot, color=self.color, linewidth=2.5, zorder=5)

        # Decide whether to draw ellipsis
        delta = (t1 - t0) * 0.05
        tol = 1e-3
        span = t1 - t0
        draw_left = draw_right = False

        # Show always if periodic
        if hasattr(self, 'signal_periods') and self.current_name in self.signal_periods:
            draw_left = draw_right = True
        else:
            N = max(10, int(0.05 * self.num_points))
            xs_left = np.linspace(t0 - 0.05 * span, t0, N)
            ys_left = np.abs(self._evaluate_signal(xs_left))
            if np.trapz(ys_left, xs_left) > tol:
                draw_left = True

            xs_right = np.linspace(t1, t1 + 0.05 * span, N)
            ys_right = np.abs(self._evaluate_signal(xs_right))
            if np.trapz(ys_right, xs_right) > tol:
                draw_right = True

        # Draw ellipsis if needed
        y_mid = (self.y_min + 2 * self.y_max) / 3
        if draw_left:
            self.ax.text(t0 - delta, y_mid, r'$\cdots$', ha='left', va='center',
                        color=self.color, fontsize=14, zorder=10)
        if draw_right:
            self.ax.text(t1 + delta, y_mid, r'$\cdots$', ha='right', va='center',
                        color=self.color, fontsize=14, zorder=10)


    def draw_impulses(self):
        """
        Draws Dirac delta impulses at the extracted positions and amplitudes.
        This method is typically called after `draw_functions()`.
        This method is usually called internally from `plot()`, but can also be used manually.

        This method:
        - Iterates over the list of impulse locations (`self.impulse_locs`)
        and their corresponding amplitudes (`self.impulse_areas`).
        - Calls `_draw_single_impulse()` for each impulse located within
        the current horizontal plotting range (`self.horiz_range`).

        Notes:
        - This method only draws impulses that are visible within the plotting window.
        - Periodicity is not assumed. Use `add_signal(..., period=...)` to manually expand periodic impulses.
        - The drawing includes both a vertical arrow and a bold label showing the impulse area.

        Examples:
            >>> self.draw_function()
            >>> self.draw_impulses()
        """
        t_min, t_max = self.horiz_range
        for t0, amp in zip(self.impulse_locs, self.impulse_areas):
            if t_min <= t0 <= t_max:
                self._draw_single_impulse(t0, amp)

    def _draw_single_impulse(self, t0, amp):
        """
        Draws a single Dirac delta impulse at the specified location and amplitude.

        This method:
        - Draws a vertical arrow starting from (t0, 0) up to (t0, amp).
        - Places a bold numerical label near the tip of the arrow indicating the amplitude.
        - Slightly offsets the label horizontally if the impulse is located at or near t = 0,
        to avoid overlapping with the vertical axis.

        Notes:
        - Arrow and label use the color specified in `self.color`.
        - Label placement is adjusted to avoid axis clutter at t = 0.

        Args:
            t0 (float): The location of the impulse along the time axis.
            amp (float): The area of the impulse. Determines arrow height and label.

        Examples:
            >>> self._draw_single_impulse(1.0, 2.5)  # draws 2.5·δ(t − 1)
        """
        # Arrow from (t0,0) to (t0, amp)
        self.ax.annotate(
            '', xy=(t0, amp + 0.01 * (self.y_max - self.y_min)), xytext=(t0, 0),
            arrowprops=dict(
                arrowstyle='-|>',
                linewidth=2.5,
                color=self.color,
                mutation_scale=16
            ),
            zorder=10
        )

        # Calculate horizontal offset for the label if t0 ≈ 0
        x_min, x_max = self.ax.get_xlim()
        x_range = x_max - x_min
        # Threshold to consider that t0 is 'almost' zero
        tol = 1e-6 * max(1.0, abs(x_range))
        if abs(t0) < tol:
            # Shift label a 2% of horizontal range to the left
            x_offset = -0.01 * x_range
            ha = 'right'
        else:
            x_offset = 0.0
            ha = 'center'

        # Algin label above continuous curve if necessary
        arrow_headroom = 0.05 * (self.y_max - self.y_min)
        x_text = t0 + x_offset
        y_text = amp + arrow_headroom

        self.ax.text(
            x_text, y_text,
            f'{amp:g}',
            ha=ha,
            va='bottom' if amp > 0 else 'top',
            fontsize=12,
            color=self.color,
            fontweight='bold',
            zorder=10
        )


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

        # Draw y-axis arrow
        self.ax.annotate('', xy=(0, self.ax.get_ylim()[1]), xytext=(0, self.ax.get_ylim()[0]),
                         arrowprops=dict(arrowstyle='-|>', linewidth=1.5, color='black',
                                         mutation_scale=12, mutation_aspect=2, fc='black'))
        
        # Prevent labels from being clipped
        self.fig.tight_layout()

    def draw_ticks(self,
                tick_size_px=None,
                xticks=None,
                yticks=None,
                xtick_labels='auto',
                ytick_labels='auto'):
        """
        Draws tick marks and labels on both x- and y-axes, using automatic or manual configurations.
        This method is typically called after `draw_impulses()`.
        This method is usually called internally from `plot()`, but can also be used manually.
        
        Features:
        - Adds tick marks and LaTeX-formatted labels.
        - Integrates impulse positions into xticks automatically, unless explicitly overridden.
        - Supports:
            - `pi_mode`: labels as multiples of π.
            - `fraction_ticks`: labels as rational fractions.
            - Hiding y=0 label if x=0 tick is shown (to avoid overlapping at the origin).
        - Avoids duplicate tick values based on numerical tolerance.

        Notes:
        - If `xticks_delta` or `yticks_delta` are provided (in constructor), evenly spaced ticks are placed at multiples.
        - If custom labels are passed and their count does not match the tick list, raises ValueError.
        - Labels are drawn on white rounded boxes for visibility over plots.

        Args:
            tick_size_px (int, optional): Length of tick marks in pixels. If None, uses self.tick_size_px.
            xticks (list or 'auto' or None): X-axis tick positions.
                - 'auto': automatically computed ticks (even spacing or using xticks_delta).
                - list: manually specified tick positions.
                - None: ticks are shown only at Dirac impulse positions.
            yticks (list or 'auto' or None): Y-axis tick positions.
                - Same behavior as `xticks`.
            xtick_labels (list or 'auto' or None): Custom labels for xticks (must match length of xticks).
            ytick_labels (list or 'auto' or None): Same for yticks.

        Examples:
            >>> self.draw_ticks(xticks='auto', yticks=[-1, 0, 1], ytick_labels=['-1', '0', '1'])
        """

        # Helper: filter duplicate values with tolerance
        def unique_sorted(values, tol):
            unique = []
            for v in values:
                if not any(abs(v - u) <= tol for u in unique):
                    unique.append(v)
            return sorted(unique)
        
        # Helper: get impulse locations and amplitudes within range (with periodic extension if needed)
        def get_impulse_positions_and_areas(t_min, t_max, tol):
            impulse_positions = []
            impulse_positions_areas = []
            if self.impulse_locs:
                if self.period is None:
                    # Non-periodic case: keep impulses in visible range
                    for base_loc, base_area in zip(self.impulse_locs, self.impulse_areas):
                        if t_min - tol <= base_loc <= t_max + tol:
                            impulse_positions.append(base_loc)
                            impulse_positions_areas.append(base_area)
                else:
                    # Periodic case: replicate impulses across periods
                    T = self.period
                    for base_loc, base_area in zip(self.impulse_locs, self.impulse_areas):
                        k_min = int(np.floor((t_min - base_loc) / T))
                        k_max = int(np.ceil((t_max - base_loc) / T))
                        for k in range(k_min, k_max + 1):
                            t_k = base_loc + k * T
                            if t_min - tol <= t_k <= t_max + tol:
                                impulse_positions.append(t_k)
                                impulse_positions_areas.append(base_area)
            # Eliminate duplicates within tolerance
            unique_pos = []
            unique_area = []
            for loc, area in zip(impulse_positions, impulse_positions_areas):
                if not any(abs(loc - u) <= tol for u in unique_pos):
                    unique_pos.append(loc)
                    unique_area.append(area)
            if unique_pos:
                idx_sort = np.argsort(unique_pos)
                impulse_positions = [unique_pos[i] for i in idx_sort]
                impulse_positions_areas = [unique_area[i] for i in idx_sort]
            else:
                impulse_positions, impulse_positions_areas = [], []
            return impulse_positions, impulse_positions_areas

        # Helper: validate tick list
        def has_valid_ticks(ticks):
            if ticks is None:
                return False
            try:
                arr = np.array(ticks)
                return arr.ndim >= 1 and arr.size >= 1
            except Exception:
                return False

        # Helper: generate xticks and labels
        def generate_xticks(effective_xticks, impulse_positions, tol, t_min, t_max):
            raw_xticks = []
            manual_xticks = []
            manual_xlabels = []

            has_init_xticks = has_valid_ticks(getattr(self, 'xticks', None))
            xticks_delta = getattr(self, 'xticks_delta', None)  # Nuevo atributo opcional

            if isinstance(effective_xticks, str) and effective_xticks == 'auto':
                if has_init_xticks:
                    raw_xticks = list(self.xticks)
                    if self.xtick_labels is not None:
                        if len(self.xticks) != len(self.xtick_labels):
                            raise ValueError("xtick_labels and xticks from init must have the same length")
                        manual_xticks = list(self.xticks)
                        manual_xlabels = list(self.xtick_labels)
                else:
                    if xticks_delta is not None:
                        n_left = int(np.floor((0 - t_min) / xticks_delta))
                        n_right = int(np.floor((t_max - 0) / xticks_delta))
                        base_ticks = [k * xticks_delta for k in range(-n_left, n_right + 1)]
                    else:
                        base_ticks = list(np.linspace(t_min, t_max, 5))
                    raw_xticks = base_ticks.copy()

                # Add impulses
                for loc in impulse_positions:
                    if t_min - tol <= loc <= t_max + tol and not any(abs(loc - x0) <= tol for x0 in raw_xticks):
                        raw_xticks.append(loc)

            else:
                if xticks_delta is not None:
                    warnings.warn("xticks_delta will be ignored because xticks not in 'auto' mode", stacklevel=2)
                raw_xticks = list(effective_xticks)
                if xtick_labels not in (None, 'auto'):
                    if len(raw_xticks) != len(xtick_labels):
                        raise ValueError("xtick_labels and xticks must have the same length")
                    manual_xticks = list(raw_xticks)
                    manual_xlabels = list(xtick_labels)
                elif self.xtick_labels is not None:
                    if len(raw_xticks) != len(self.xtick_labels):
                        raise ValueError("xtick_labels and xticks from init must have the same length")
                    manual_xticks = list(raw_xticks)
                    manual_xlabels = list(self.xtick_labels)

                for loc in impulse_positions:
                    if t_min - tol <= loc <= t_max + tol and not any(abs(loc - x0) <= tol for x0 in raw_xticks):
                        raw_xticks.append(loc)

            raw_xticks = unique_sorted(raw_xticks, tol)

            # Gnerate labels
            xlabels = []
            for x in raw_xticks:
                label = None
                for xm, lbl in zip(manual_xticks, manual_xlabels):
                    if abs(xm - x) <= tol:
                        label = lbl
                        break
                if label is None:
                    if getattr(self, 'pi_mode', False):
                        f = Fraction(x / np.pi).limit_denominator(24)
                        if abs(float(f) * np.pi - x) > tol:
                            label = f'{x:g}'
                        elif f == 0:
                            label = '0'
                        elif f == 1:
                            label = r'\pi'
                        elif f == -1:
                            label = r'-\pi'
                        else:
                            num = f.numerator
                            denom = f.denominator
                            prefix = '-' if num * denom < 0 else ''
                            num, denom = abs(num), abs(denom)
                            if denom == 1:
                                label = rf"{prefix}{num}\pi"
                            elif num == 1:
                                label = rf"{prefix}\frac{{\pi}}{{{denom}}}"
                            else:
                                label = rf"{prefix}\frac{{{num}\pi}}{{{denom}}}"
                    elif getattr(self, 'fraction_ticks', False):
                        f = Fraction(x).limit_denominator(24)
                        label = f"{f.numerator}/{f.denominator}" if f.denominator != 1 else f"{f.numerator}"
                    else:
                        label = f'{x:g}'
                xlabels.append(label)

            return raw_xticks, xlabels

        # Helper: generate yticks and labels
        def generate_yticks(effective_yticks, tol):
            raw_yticks = []
            manual_yticks = []
            manual_ylabels = []

            has_init_yticks = has_valid_ticks(getattr(self, 'yticks', None))
            ytick_labels = getattr(self, 'ytick_labels', None)
            ydelta = getattr(self, 'yticks_delta', None)

            if effective_yticks is None:
                raw_yticks = []
            elif isinstance(effective_yticks, str) and effective_yticks == 'auto':
                if has_init_yticks:
                    raw_yticks = list(self.yticks)
                    if self.ytick_labels is not None:
                        if len(self.yticks) != len(self.ytick_labels):
                            raise ValueError("ytick_labels and yticks from init must have the same length")
                        manual_yticks = list(self.yticks)
                        manual_ylabels = list(self.ytick_labels)
                    if ydelta is not None:
                        warnings.warn("yticks_delta ignored because yticks where specified at init")
                else:
                    if ydelta is not None and ydelta > 0:
                        y_start = np.ceil(self.y_min / ydelta)
                        y_end = np.floor(self.y_max / ydelta)
                        raw_yticks = [k * ydelta for k in range(int(y_start), int(y_end) + 1)]
                    else:
                        y0 = np.floor(self.y_min)
                        y1 = np.ceil(self.y_max)
                        if abs(y1 - y0) < 1e-6:
                            raw_yticks = [y0 - 1, y0, y0 + 1]
                        else:
                            raw_yticks = list(np.linspace(y0, y1, 3))
            else:
                raw_yticks = list(effective_yticks)
                if ydelta is not None:
                    warnings.warn("yticks_delta ignored because yticks is not in 'auto' mode")

                if ytick_labels not in (None, 'auto'):
                    if len(raw_yticks) != len(ytick_labels):
                        raise ValueError("ytick_labels and yticks must have the same length")
                    manual_yticks = list(raw_yticks)
                    manual_ylabels = list(ytick_labels)
                elif self.ytick_labels is not None:
                    if len(raw_yticks) != len(self.ytick_labels):
                        raise ValueError("ytick_labels and yticks from init must have the same length")
                    manual_yticks = list(raw_yticks)
                    manual_ylabels = list(self.ytick_labels)

            raw_yticks = unique_sorted(raw_yticks, tol)

            ylabels = []
            for y in raw_yticks:
                label = None
                for ym, lbl in zip(manual_yticks, manual_ylabels):
                    if abs(ym - y) <= tol:
                        label = lbl
                        break
                if label is None:
                    if self.pi_mode:
                        f = Fraction(y / np.pi).limit_denominator(24)
                        if abs(float(f) * np.pi - y) > tol:
                            label = f'{y:.3g}'
                        elif f == 0:
                            label = '0'
                        elif f == 1:
                            label = r'\pi'
                        elif f == -1:
                            label = r'-\pi'
                        else:
                            num = f.numerator
                            denom = f.denominator
                            prefix = '-' if num * denom < 0 else ''
                            num, denom = abs(num), abs(denom)
                            if denom == 1:
                                label = rf"{prefix}{num}\pi"
                            elif num == 1:
                                label = rf"{prefix}\frac{{\pi}}{{{denom}}}"
                            else:
                                label = rf"{prefix}\frac{{{num}\pi}}{{{denom}}}"
                    elif self.fraction_ticks:
                        f = Fraction(y).limit_denominator(24)
                        label = f"{f.numerator}/{f.denominator}" if f.denominator != 1 else f"{f.numerator}"
                    else:
                        label = f'{y:.3g}'
                ylabels.append(label)

            return raw_yticks, ylabels
        
        # Helper: hide y=0 label if x=0 tick exists
        def filter_yticks(raw_yticks, ylabels, raw_xticks, tol):
            has_xtick_zero = any(abs(x) <= tol for x in raw_xticks)
            if has_xtick_zero:
                filtered_yticks = []
                filtered_ylabels = []
                for y, lbl in zip(raw_yticks, ylabels):
                    if abs(y) <= tol:
                        continue
                    filtered_yticks.append(y)
                    filtered_ylabels.append(lbl)
                return filtered_yticks, filtered_ylabels
            else:
                return raw_yticks, ylabels

        # Helper: convert pixel length to data coordinates
        def px_to_data_length(tick_px):
            origin_disp = self.ax.transData.transform((0, 0))
            up_disp = origin_disp + np.array([0, tick_px])
            right_disp = origin_disp + np.array([tick_px, 0])
            origin_data = np.array(self.ax.transData.inverted().transform(origin_disp))
            up_data = np.array(self.ax.transData.inverted().transform(up_disp))
            right_data = np.array(self.ax.transData.inverted().transform(right_disp))
            dy = up_data[1] - origin_data[1]
            dx = right_data[0] - origin_data[0]
            return dx, dy

        # Helper: draw xticks and labels
        def draw_xticks(raw_xticks, xlabels, impulse_positions, impulse_positions_areas, dx, dy, tol):
            xlim = self.ax.get_xlim()
            for x, lbl in zip(raw_xticks, xlabels):
                if xlim[0] <= x <= xlim[1]:
                    self.ax.plot([x, x], [0 - dy/2, 0 + dy/2], transform=self.ax.transData,
                                color='black', linewidth=1.2, clip_on=False)
                    area = None
                    for loc, a in zip(impulse_positions, impulse_positions_areas):
                        if abs(loc - x) <= tol:
                            area = a
                            break
                    y_off = +8 if (area is not None and area < 0) else -8
                    offset = (-8, y_off) if abs(x) < tol else (0, y_off)
                    va = 'bottom' if y_off > 0 else 'top'
                    self.ax.annotate(rf'${lbl}$', xy=(x, 0), xycoords='data',
                                    textcoords='offset points', xytext=offset,
                                    ha='center', va=va, fontsize=12, zorder=10,
                                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                            edgecolor='none', alpha=self.alpha))

        # Helper: draw yticks and labels
        def draw_yticks(raw_yticks, ylabels, dx, dy):
            ylim = self.ax.get_ylim()
            for y, lbl in zip(raw_yticks, ylabels):
                if ylim[0] <= y <= ylim[1]:
                    self.ax.plot([0 - dx/2, 0 + dx/2], [y, y], transform=self.ax.transData,
                                color='black', linewidth=1.2, clip_on=False)
                    offset = (-4, -16) if abs(y) < 1e-10 else (-4, 0)
                    self.ax.annotate(rf'${lbl}$', xy=(0, y), xycoords='data',
                                    textcoords='offset points', xytext=offset,
                                    ha='right', va='center', fontsize=12, zorder=10,
                                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                            edgecolor='none', alpha=self.alpha))

         # === MAIN LOGIC ===

        # 0. Use constructor defaults if nothing passed explicitly
        effective_xticks = xticks if xticks is not None else getattr(self, 'init_xticks_arg', None)
        effective_yticks = yticks if yticks is not None else getattr(self, 'init_yticks_arg', None)

        # 1. Determine tick size in pixels
        tick_px = tick_size_px if tick_size_px is not None else self.tick_size_px

        # 2. Get plotting range and numeric tolerance
        t_min, t_max = self.horiz_range
        tol = 1e-8 * max(1.0, abs(t_max - t_min))

        # 3. Get impulse positions in the current range
        impulse_positions, impulse_positions_areas = get_impulse_positions_and_areas(t_min, t_max, tol)

        # 4. Generate x ticks and labels
        raw_xticks, xlabels = generate_xticks(effective_xticks, impulse_positions, tol, t_min, t_max)

        # 5. Generate y ticks and labels
        raw_yticks, ylabels = generate_yticks(effective_yticks, tol)

        # 6. Remove y=0 label if x=0 tick exists
        raw_yticks, ylabels = filter_yticks(raw_yticks, ylabels, raw_xticks, tol)

        # 7. Convert tick length in px to data coordinates
        dx, dy = px_to_data_length(tick_px)

        # 8. Draw x-axis ticks and labels
        draw_xticks(raw_xticks, xlabels, impulse_positions, impulse_positions_areas, dx, dy, tol)

        # 9. Draw y-axis ticks and labels
        draw_yticks(raw_yticks, ylabels, dx, dy)


    def draw_labels(self):
        """
        Adds axis labels to the x and y axes using the current signal variable and name.
        This method is typically called after `draw_ticks()`.
        This method is usually called internally from `plot()`, but can also be used manually.

        This method:
        - Retrieves the current axis limits.
        - Places the x-axis label slightly to the right of the horizontal axis arrow.
        - Places the y-axis label slightly below the top of the vertical axis arrow.
        - Uses LaTeX formatting for the labels (e.g., "x(t)", "y(\\tau)", etc.).
        - The labels use the values of `self.xlabel` and `self.ylabel`.

        Notes:
        - This method is called automatically in `plot()` after drawing ticks and arrows.
        - Rotation for y-axis label is disabled to keep it horizontal.

        Examples:
            >>> self.draw_labels()
        """
        # Get the current x and y axis limits
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()

        # X-axis label: slightly to the right of the rightmost x limit
        x_pos = x_lim[1] - 0.01 * (x_lim[1] - x_lim[0])
        y_pos = 0.02 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, rf'${self.xlabel}$', fontsize=16, ha='right', va='bottom')

        # Y-axis label: slightly below the top y limit (still inside the figure)
        x_pos = 0.01 * (x_lim[1] - x_lim[0])
        y_pos = y_lim[1] - 0.1 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, rf'${self.ylabel}$', fontsize=16, ha='left', va='bottom', rotation=0)

    def _update_expression_and_func(self, expr, var):
        """
        Internal helper to update self.expr, self.expr_cont and self.func
        for plotting, safely removing any DiracDelta terms.
        """
        self.expr = expr
        self.var = var
        self.expr_cont = self._remove_dirac_terms()
        self.func = sp.lambdify(self.var, self.expr_cont, modules=["numpy", self._get_local_dict()])
        self.impulse_locs, self.impulse_areas = self._extract_impulses()

    def show(self):
        """
        Displays or saves the final plot, depending on configuration.
        This method is typically called after `draw_labels()`.
        This method is usually called internally from `plot()`, but can also be used manually.

        This method:
        - Disables the background grid.
        - Applies tight layout to reduce unnecessary whitespace.
        - If `self.save_path` is set, saves the figure to the given file (PNG, PDF, etc.).
        - If `self.show_plot` is True, opens a plot window (interactive view).
        - Finally, closes the figure to free up memory (especially important in batch plotting).

        Notes:
        - `self.save_path` and `self.show_plot` are set in the constructor.
        - If both are enabled, the plot is shown and saved.
        - The output file format is inferred from the file extension.

        Examples:
            >>> self.show()  # Typically called at the end of plot()
        """
        self.ax.grid(False)
        plt.tight_layout()
        if self.save_path:
            self.fig.savefig(self.save_path, dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        plt.close(self.fig)

    def plot(self, name=None):
        """
        Plots the signal specified by name (or the default if defined via expr_str in constructor).

        This method:
        - Initializes the expression from `expr_str_pending` (if any), only once.
        - Looks up the signal in the internal dictionary (`self.signal_defs`) using its name.
        - Sets up symbolic and numeric representations of the signal.
        - Removes DiracDelta terms from the continuous part.
        - Extracts impulses and prepares data for plotting.
        - Calls the full sequence: axis setup, function drawing, impulses, ticks, labels, and final display/save.

        Args:
            name (str, optional): Name of the signal to plot, e.g., "x1". If None and `expr_str` was given at init,
                        it uses the last-added expression.

        Raises:
            ValueError: If the signal is not defined or its variable cannot be determined.

        Examples:
            >>> SignalPlotter("x(t)=rect(t-1)").plot()
            >>> sp1 = SignalPlotter("x(t)=rect(t-1)", period=2)
            >>> sp1.plot("x")
            >>> sp2 = SignalPlotter()
            >>> sp2.add_signal("x(t) = rect(t)")
            >>> sp2.plot("x")
        """
        # Initialize from expr_str (only once), if provided at construction
        if (hasattr(self, 'expr_str_pending') and 
            self.expr_str_pending is not None and 
            isinstance(self.expr_str_pending, str) and 
            not getattr(self, '_initialized_from_expr', False)):
            expr_str = self.expr_str_pending
            self._initialized_from_expr = True
            self.add_signal(expr_str, period=self.period)
            name = list(self.signal_defs.keys())[-1]

        if name:
            if name not in self.signal_defs:
                raise ValueError(f"Signal '{name}' is not defined.")
            self.current_name = name
            self.func_name = name

            # Use declared variable or infer it
            expr = self.signal_defs[name]
            var = self.var_symbols.get(name, None)
            if var is None:
                free_vars = list(expr.free_symbols)
                if not free_vars:
                    raise ValueError(f"Could not determine the variable for signal '{name}'.")
                var = free_vars[0]

            # Update expression and lambdified function, remove Dirac terms, extract impulses
            self._update_expression_and_func(expr, var)

            # Use declared variable or infer it
            expr = self.signal_defs[name]
            var = self.var_symbols.get(name, None)
            if var is None:
                free_vars = list(expr.free_symbols)
                if not free_vars:
                    raise ValueError(f"Could not determine the variable for signal '{name}'.")
                var = free_vars[0]

            # Update expression and lambdified function, remove Dirac terms, extract impulses
            self._update_expression_and_func(expr, var)

            # Set axis labels
            self.xlabel = str(var)
            self.ylabel = f"{self.func_name}({self.xlabel})"

            if hasattr(self, 'custom_labels') and self.func_name in self.custom_labels:
                if self.custom_labels[self.func_name] is not None:
                    self.ylabel = self.custom_labels[self.func_name]

            # Time discretization for plotting
            self.t_vals = np.linspace(*self.horiz_range, self.num_points)

            # Create figure and compute y-range
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self._prepare_plot()

        # Draw all components of the plot
        self.setup_axes()
        self.draw_function()
        self.draw_impulses()
        self.ax.relim()
        self.ax.autoscale_view()
        self.draw_ticks()
        self.draw_labels()
        self.show()

    # Definir las transformaciones a nivel global para uso posterior
    # def _get_transformations(self):
    #     return self.transformations

    ## Convolution-specific methods

    def _setup_figure(self):
        """
        Initializes the Matplotlib figure and axes for plotting.

        This method:
        - Creates a new figure and axis using the configured `figsize`.
        - Calls `_prepare_plot()` to compute vertical bounds for plotting based on the signal.
        - Applies padding to the layout using `subplots_adjust` to avoid clipping of labels and arrows.

        Notes:
        - This method is typically used in convolution plotting routines where a clean figure is needed.
        - For standard plotting, `plot()` uses its own setup sequence and may not rely on this method.

        Examples:
            >>> self._setup_figure()
            >>> self.draw_function()
            >>> self.show()
        """
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._prepare_plot()
        self.fig.subplots_adjust(right=0.9, top=0.85, bottom=0.15)


    def plot_convolution_view(self, expr_str, t_val, label=None, tau=None, t=None):
        """
        Plots an intermediate signal in the convolution process, such as x(t−τ), h(τ+t), etc.

        This method:
        - Substitutes the convolution variable t with a fixed value `t_val` in a symbolic expression.
        - Evaluates the resulting signal in terms of τ.
        - Optionally adjusts x-axis direction and labels if the expression has a form like (t − τ) or (t + τ).
        - Automatically handles periodic xtick reversal or shift based on convolution expression.
        - Renders the function using existing plot methods (function, impulses, ticks, etc.).

        Args:
            expr_str (str): A symbolic expression involving τ and t, e.g. "x(t - tau)" or "h(t + tau)".
                        The base signal must already be defined with `add_signal(...)`.
            t_val (float): The value of the time variable `t` at which the expression is evaluated.
            label (str, optional): Custom y-axis label to display (default is derived from the expression).
            tau (sympy.Symbol or str, optional): Symbol to use as integration variable (default: 'tau').
            t (sympy.Symbol or str, optional): Symbol used in shifting (default: 't').

        Raises:
            ValueError: If the base signal is not defined or the expression format is invalid.

        Examples:
            >>> sp = SignalPlotter(xticks=[-1, 0, 3], num_points=200, fraction_ticks=True)
            >>> sp.add_signal("x(t)=exp(-2t)*u(t)")
            >>> sp.add_signal("h(t)=u(t)")
            >>> sp.plot_convolution_view("x(t - tau)", t_val=1)
            >>> sp.plot_convolution_view("h(t + tau)", t_val=2, tau='lambda', t='omega')
        """
        import re
        local_dict = self._get_local_dict()

        # Define symbolic variables for τ and t
        if tau is None:
            tau = local_dict.get('tau')
        elif isinstance(tau, str):
            tau = sp.Symbol(tau)
        if t is None:
            t = local_dict.get('t')
        elif isinstance(t, str):
            t = sp.Symbol(t)

        local_dict.update({'tau': tau, 't': t, str(tau): tau, str(t): t})

        # Extract base signal name and ensure it's defined
        if "(" in expr_str:
            name = expr_str.split("(")[0].strip()
            if name not in self.signal_defs:
                raise ValueError(f"Signal '{name}' is not defined.")
            expr_base = self.signal_defs[name]
            var_base = self.var_symbols.get(name, t)
        else:
            raise ValueError("Invalid expression: expected something like 'x(t - tau)'.")

        # Parse expression and apply to base
        parsed_expr = parse_expr(expr_str.replace(name, "", 1), local_dict)
        expr = expr_base.subs(var_base, parsed_expr)

        # Analyze structure to adapt axis
        xticks = self.init_xticks_arg
        horiz_range = self.horiz_range
        xticks_custom = None
        xtick_labels_custom = None

        if isinstance(parsed_expr, sp.Expr):
            diff1 = parsed_expr - tau
            if diff1.has(t):
                diff2 = parsed_expr - t
                coef = diff2.coeff(tau)
                if coef == -1:
                    # Case t - tau ⇒ reverse x-axis
                    if xticks == 'auto':
                        xticks_custom = [t_val]
                        xtick_labels_custom = [sp.latex(t)]
                    elif isinstance(xticks, (list, tuple)):
                        xticks_custom = [t_val - v for v in xticks][::-1]
                        xtick_labels_custom = [
                            f"{sp.latex(t)}" if v == 0 else f"{sp.latex(t)}{'-' if v > 0 else '+'}{abs(v)}"
                            for v in xticks
                        ][::-1]
                    horiz_range = (t_val - np.array(self.horiz_range)[::-1]).tolist()
                elif coef == 1:
                    # Case t + tau ⇒ shift axis
                    if xticks == 'auto':
                        xticks_custom = [- t_val]
                        xtick_labels_custom = [sp.latex(t)]
                    elif isinstance(xticks, (list, tuple)):
                        xticks_custom = [- t_val + v for v in xticks]
                        xtick_labels_custom = [
                            f"-{sp.latex(t)}" if v == 0 else f"-{sp.latex(t)}{'+' if v > 0 else '-'}{abs(v)}"
                            for v in xticks
                        ]
                    horiz_range = (np.array(self.horiz_range) - t_val).tolist()

        # Evaluate the expression at t = t_val
        expr_evaluated = expr.subs(t, t_val)

        # Update expression and lambdified function
        self._update_expression_and_func(expr_evaluated, tau)

        # Axis labels
        self.xlabel = sp.latex(tau)
        tau_str = sp.latex(tau)
        t_str = sp.latex(t)
        self.ylabel = label if label else expr_str.replace("tau", tau_str).replace("t", t_str)

        # Discretize time
        self.t_vals = np.linspace(*horiz_range, self.num_points)

        # Prepare and render plot
        self._setup_figure()
        self.setup_axes(horiz_range)
        self.draw_function(horiz_range)
        self.draw_impulses()
        self.draw_ticks(xticks=xticks_custom, xtick_labels=xtick_labels_custom)
        self.draw_labels()
        self.show()



    def plot_convolution_steps(self, x_name, h_name, t_val, tau=None, t=None):
        """
        Plots four key signals involved in a convolution step at a fixed time `t_val`:
        x(tau), x(t-tau), h(tau), h(t-tau)., all in terms of τ.

        This method is particularly useful for visualizing the time-reversed and shifted
        versions of the input signals used in the convolution integral.

        Notes:
        - The horizontal axis is adjusted for time-reversed signals (e.g., t−τ),
        and tick labels are shifted accordingly.
        - Four separate plots are generated in sequence, with labels and axes automatically set.

        Args:
            x_name (str): Name of the signal x, previously defined with `add_signal(...)`.
            h_name (str): Name of the signal h, previously defined with `add_signal(...)`.
            t_val (float): The fixed time t at which the convolution step is evaluated.
            tau (sympy.Symbol or str, optional): Symbol for the integration variable (default: 'tau').
            t (sympy.Symbol or str, optional): Symbol for the time variable (default: 't').

        Examples:
            >>> sp = SignalPlotter()
            >>> sp.add_signal("x(t)=sinc(t)")
            >>> sp.add_signal("h(t)=sinc(t/2)")
            >>> sp.plot_convolution_steps("x", "h", t_val=1)
        """
        local_dict = self._get_local_dict()

        # Use default symbols if not provided
        if tau is None:
            tau = local_dict.get('tau')
        elif isinstance(tau, str):
            tau = sp.Symbol(tau)

        if t is None:
            t = local_dict.get('t')
        elif isinstance(t, str):
            t = sp.Symbol(t)

        # Evaluate x(τ) and h(τ) using their respective symbolic variable
        x_expr = self.signal_defs[x_name].subs(self.var_symbols[x_name], tau)
        h_expr = self.signal_defs[h_name].subs(self.var_symbols[h_name], tau)

        # Compute x(t−τ) and h(t−τ), and substitute t = t_val
        x_shift = x_expr.subs(tau, t - tau).subs(t, t_val)
        h_shift = h_expr.subs(tau, t - tau).subs(t, t_val)

        # Convert to LaTeX strings for labeling
        tau_str = sp.latex(tau)
        t_str = sp.latex(t)

        # Generate custom xticks and labels for the shifted signals
        xticks = self.init_xticks_arg
        if xticks == 'auto':
            xticks_shifted = [t_val]
            xtick_labels_shifted = [f"{t_str}"]
        elif isinstance(xticks, (list, tuple)):
            xticks_shifted = [t_val - v for v in xticks]
            xtick_labels_shifted = []
            for v in xticks:
                delta = - v
                if delta == 0:
                    label = fr"{t_str}"
                elif delta > 0:
                    label = fr"{t_str}+{delta}"
                else:
                    label = fr"{t_str}{delta}"  # delta is already negative
                xtick_labels_shifted.append(label)
        else:
            xticks_shifted = None
            xtick_labels_shifted = None

        horiz_range = self.horiz_range
        # Compute reversed horizontal range for time-reversed signals
        horiz_range_shifted = t_val - np.array(horiz_range)[::-1]

        # Define all 4 signals to be plotted with labels and optional custom ticks
        items = [
            (x_expr, fr"{x_name}({tau_str})", None, None, horiz_range),
            (h_expr, fr"{h_name}({tau_str})", None, None, horiz_range),
            (x_shift, fr"{x_name}({t_str}-{tau_str})", xticks_shifted, xtick_labels_shifted, horiz_range_shifted),
            (h_shift, fr"{h_name}({t_str}-{tau_str})", xticks_shifted, xtick_labels_shifted, horiz_range_shifted),
        ]

        for expr, label, xticks_custom, xtick_labels_custom, horiz_range_custom in items:
            # Prepare expression and plot configuration
            self._update_expression_and_func(expr, tau)
            self.xlabel = fr"\{tau}"
            self.ylabel = label
            self.t_vals = np.linspace(*horiz_range_custom, self.num_points)

            self._setup_figure()
            self.setup_axes(horiz_range_custom)
            self.draw_function(horiz_range_custom)
            self.draw_impulses()
            self.draw_ticks(xticks=xticks_custom, xtick_labels=xtick_labels_custom)
            self.draw_labels()
            self.show()

    def _estimate_numerical_support(self, expr, var, horiz_range, margin_multiplier=1.0):
        """
        Estimate the true support of a signal numerically by evaluating the function
        over an extended range and detecting where it is significantly nonzero.

        Parameters
        ----------
        expr : sympy.Expr
            The symbolic expression of the signal.
        var : sympy.Symbol
            The variable of the expression (t or tau or others).
        horiz_range : tuple
            The current horizontal range.
        margin_multiplier : float
            Factor to extend the range for safety.

        Returns
        -------
        tuple :
            Estimated (min, max) support where the signal is not negligible.
        """
        margin = margin_multiplier * (horiz_range[1] - horiz_range[0])
        search_range = (horiz_range[0] - margin, horiz_range[1] + margin)
        num_points = 1000  # resolution

        t_vals = np.linspace(*search_range, num_points)
        func = sp.lambdify(var, expr, modules=["numpy"])
        y_vals = np.abs(func(t_vals))

        threshold = np.max(y_vals) * 1e-3  # relative threshold
        nonzero_indices = np.where(y_vals > threshold)[0]

        if len(nonzero_indices) == 0:
            return (0.0, 0.0)

        min_index = nonzero_indices[0]
        max_index = nonzero_indices[-1]
        return (t_vals[min_index], t_vals[max_index])


    def plot_convolution_result(self, x_name, h_name, num_points=None, show_expr=False):
        """
        Compute and plot the convolution result y(t) = (x * h)(t) between two signals x(t) and h(t).

        The function automatically detects impulses (DiracDelta) and applies analytical formulas.
        For general signals, it supports two numerical integration methods:

        Parameters
        ----------
        x_name : str
            Name of the first signal (x).
        h_name : str
            Name of the second signal (h).
        num_points : int, optional
            Number of time points for evaluation (default: self.num_points).
        method : {'fast', 'precise'}, optional
            Integration method for general convolution (default 'fast'):
            - 'fast': numpy trapezoidal rule (np.trapz)
            - 'precise': adaptive quadrature (scipy.integrate.quad)
        show_expr : bool, optional
            Reserved for future use.

        Examples
        --------
        >>> sp.plot_convolution_result("x", "h", method='fast')
        >>> sp.plot_convolution_result("x", "h", method='precise')
        """

        t = sp.Symbol('t')
        tau = sp.Symbol('tau')

        if num_points is None:
            num_points = self.num_points

        x_expr = self.signal_defs[x_name]
        h_expr = self.signal_defs[h_name]
        var_x = self.var_symbols[x_name]
        var_h = self.var_symbols[h_name]

        x_tau_expr = x_expr.subs(var_x, tau)
        h_t_tau_expr = h_expr.subs(var_h, t - tau)

        local_dict = self._get_local_dict()
        t_vals = np.linspace(*self.horiz_range, num_points)
        y_vals = []

        # Case 1: Dirac in x(t)
        if x_tau_expr.has(sp.DiracDelta):
            y_expr = 0
            for term in x_tau_expr.as_ordered_terms():
                if term.has(sp.DiracDelta):
                    args = term.args if term.func == sp.Mul else [term]
                    scale = 1
                    for a in args:
                        if a.func == sp.DiracDelta:
                            delta_arg = a.args[0]
                        else:
                            scale *= a
                    shift = sp.solve(delta_arg, tau)
                    if shift:
                        y_expr += scale * h_expr.subs(var_h, t - shift[0])

            # Extract impulses from y_expr
            impulse_locs = []
            impulse_areas = []
            for term in y_expr.as_ordered_terms():
                if term.has(sp.DiracDelta):
                    args = term.args if term.func == sp.Mul else [term]
                    area = 1
                    shift = 0
                    for a in args:
                        if a.func == sp.DiracDelta:
                            sol = sp.solve(a.args[0], t)
                            if sol:
                                shift = float(sol[0])
                        else:
                            area *= a
                    impulse_locs.append(shift)
                    impulse_areas.append(float(area))

            self._update_expression_and_func(y_expr, t)
            self.impulse_locs = impulse_locs
            self.impulse_areas = impulse_areas

        # Case 2: Dirac in h(t)
        elif h_t_tau_expr.has(sp.DiracDelta):
            y_expr = 0
            for term in h_t_tau_expr.as_ordered_terms():
                if term.has(sp.DiracDelta):
                    args = term.args if term.func == sp.Mul else [term]
                    scale = 1
                    for a in args:
                        if a.func == sp.DiracDelta:
                            delta_arg = a.args[0]
                        else:
                            scale *= a
                    shift = sp.solve(delta_arg, tau)
                    if shift:
                        y_expr += scale * x_tau_expr.subs(tau, shift[0])

            impulse_locs = []
            impulse_areas = []
            for term in y_expr.as_ordered_terms():
                if term.has(sp.DiracDelta):
                    args = term.args if term.func == sp.Mul else [term]
                    area = 1
                    shift = 0
                    for a in args:
                        if a.func == sp.DiracDelta:
                            sol = sp.solve(a.args[0], t)
                            if sol:
                                shift = float(sol[0])
                        else:
                            area *= a
                    impulse_locs.append(shift)
                    impulse_areas.append(float(area))

            self._update_expression_and_func(y_expr, t)
            self.impulse_locs = impulse_locs
            self.impulse_areas = impulse_areas

        # Case 3: General convolution via numerical integration
        else:
            # Estimate supports
            support_x = self._estimate_numerical_support(x_expr, var_x, self.horiz_range, margin_multiplier=1.0)
            support_h = self._estimate_numerical_support(h_expr, var_h, self.horiz_range, margin_multiplier=1.0)

            # Prepare full symbolic integrand once
            integrand_expr = x_expr.subs(var_x, tau) * h_expr.subs(var_h, t - tau)
            integrand_func = sp.lambdify((tau, t), integrand_expr, modules=["numpy", local_dict])

            tau_grid = np.linspace(
                min(support_x[0], t_vals[0] - support_h[1]),
                max(support_x[1], t_vals[-1] - support_h[0]),
                num_points
            )

            y_vals = []

            for t_val in t_vals:
                a = max(support_x[0], t_val - support_h[1])
                b = min(support_x[1], t_val - support_h[0])
                if a >= b:
                    y_vals.append(0)
                    continue
                integrand = lambda tau_val: integrand_func(tau_val, t_val)
                try:
                    val, _ = integrate.quad(integrand, a, b)
                except Exception:
                    val = 0
                y_vals.append(val)

            # Create interpolation function for plotting and future evaluations
            # We computed y_vals only at discrete t_vals, but the drawing functions 
            # may request evaluations at any t ∈ horiz_range.
            # This avoids recomputing integrals during plot rendering:
            self.func = lambda t_: np.interp(t_, t_vals, y_vals)

            # Store the t values used for this computation.
            # This may be useful for plot ticks, axis limits, or any further processing.
            self.t_vals = t_vals

            # No closed-form expression exists for the convolution result (purely numeric case).
            # We assign None to indicate no symbolic expression is available.
            self.expr = None

            # Clear any impulse information: no Dirac impulses exist in the general case.
            self.impulse_locs = []
            self.impulse_areas = []


        # Final settings and plot
        self.t_vals = t_vals
        self.xlabel = "t"
        self.ylabel = r"y(t)"

        self._setup_figure()
        self.setup_axes()
        self.draw_function()
        self.draw_impulses()
        self.draw_ticks()
        self.draw_labels()
        self.show()



        # ✅ Restaurar periodo original al final del todo
        # if name:
        #     self.period = prev_period

    # def animate_signal(self, interval=20, save_path=None):
    #     """
    #     Anima la señal mostrando cómo se dibuja progresivamente desde el inicio al final del rango horizontal.
    #     Parámetros:
    #     - interval: tiempo en ms entre frames
    #     - save_path: si se proporciona, guarda la animación en ese archivo (mp4 o gif)
    #     """
    #     # Preparar datos base para la animación (igual que en draw_function, con periodicidad)
    #     t0, t1 = self.horiz_range
    #     if self.period is None:
    #         t_plot = self.t_vals
    #         y_plot = self._evaluate_signal(t_plot)
    #     else:
    #         T = self.period
    #         base_t = np.linspace(-T/2, T/2, self.num_points)
    #         base_y = self.func(base_t)
    #         k_min = int(np.floor((t0 + T/2)/T))
    #         k_max = int(np.floor((t1 + T/2)/T))
    #         segs_t = []
    #         segs_y = []
    #         for k in range(k_min, k_max+1):
    #             t_seg = base_t + k*T
    #             mask = (t_seg >= t0) & (t_seg <= t1)
    #             segs_t.append(t_seg[mask])
    #             segs_y.append(np.array(base_y)[mask])
    #         t_plot = np.concatenate(segs_t)
    #         y_plot = np.concatenate(segs_y)
    #         order = np.argsort(t_plot)
    #         t_plot = t_plot[order]
    #         y_plot = y_plot[order]

    #     # Asegurar numpy arrays y formato adecuado
    #     t_plot = np.array(t_plot)
    #     y_plot = np.array(y_plot)
    #     if y_plot.ndim == 0:
    #         y_plot = np.full_like(t_plot, y_plot, dtype=float)

    #     # Crear la figura y ejes igual que en self.fig, self.ax
    #     fig, ax = self.fig, self.ax

    #     # Limpiar ejes para animar
    #     ax.clear()
    #     self.setup_axes()  # configura ejes y flechas
    #     self.draw_impulses()  # dibuja impulsos fijos

    #     # Línea para la animación (vacía al principio)
    #     line, = ax.plot([], [], color=self.color, linewidth=2.5, zorder=5)

    #     # Función de inicialización para FuncAnimation
    #     def init():
    #         line.set_data([], [])
    #         return (line,)

    #     # Función de actualización para cada frame
    #     def update(frame):
    #         # frame es índice del punto final a dibujar
    #         i = frame
    #         if i == 0:
    #             xdata = []
    #             ydata = []
    #         else:
    #             xdata = t_plot[:i]
    #             ydata = y_plot[:i]
    #         line.set_data(xdata, ydata)
    #         return (line,)

    #     # Número de frames es la cantidad de puntos
    #     frames = len(t_plot)

    #     # Crear animación
    #     ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
    #                                 blit=True, interval=interval, repeat=False)

    #     # Guardar si se indica
    #     if save_path is not None:
    #         ext = save_path.split('.')[-1].lower()
    #         if ext in ['mp4', 'avi', 'mov']:
    #             ani.save(save_path, writer='ffmpeg', fps=1000/interval)
    #         elif ext in ['gif']:
    #             ani.save(save_path, writer='pillow', fps=1000/interval)
    #         else:
    #             raise ValueError("Extensión de archivo no soportada para guardar la animación")

    #     # Mostrar animación en notebook si no se guarda y está en entorno interactivo
    #     try:
    #         from IPython.display import HTML
    #         return HTML(ani.to_jshtml())
    #     except ImportError:
    #         plt.show()
