import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
import warnings
from matplotlib.widgets import Slider

class DiscreteSignalPlotter:
    """
    Class for symbolic definition and plotting of discrete-time signals.
    """
    def __init__(
        self,
        expr_str=None, 
        horiz_range=(-10, 10),
        vert_range=None,
        period=None,
        # num_points=1000,
        figsize=(8, 3), 
        tick_size_px=5,
        xticks='auto',
        yticks='auto',
        xtick_labels=None,
        ytick_labels=None,
        xticks_delta=None,
        yticks_delta=None,
        # pi_mode=False,
        fraction_ticks=False,
        save_path=None, 
        show_plot=True,
        color='black', 
        alpha=0.5 
    ):
        """
        (Private) Creator of the DiscreteSignalPlotter class.
        """

        def is_natural(x, tol=1e-12):
            if x is None:
                return False
            return (abs(x - round(x)) < tol) and (x >= 0)

        if xticks_delta is not None:
            if not is_natural(xticks_delta):
                raise ValueError("xticks_delta must be natural (integer >= 0)")
 
        if yticks_delta is not None:
            if yticks_delta <= 0:
                raise ValueError("yticks_delta must be positive")
 
        # Global defaults
        self.default_xticks = xticks
        self.default_yticks = yticks
        self.default_xtick_labels = xtick_labels
        self.default_ytick_labels = ytick_labels
        self.default_xticks_delta = xticks_delta
        self.default_yticks_delta = yticks_delta

        # Per signal storage
        self.signal_defs = {}
        self.funcs = {}
        self.var_symbols = {}
        self.signal_xticks = {}
        self.signal_yticks = {}
        self.signal_xtick_labels = {}
        self.signal_ytick_labels = {}
        self.signal_xticks_delta = {}
        self.signal_yticks_delta = {}

        self.custom_labels = {}
        self.signal_periods = {}

        self.current_name = None
        self.horiz_range = horiz_range
        self.vert_range = vert_range
        # self.num_points = num_points
        self.figsize = figsize
        self.tick_size_px = tick_size_px
        self.color = color
        self.alpha = alpha
        self.period = period
        self.save_path = save_path
        self.show_plot = show_plot

        self.fraction_ticks = fraction_ticks


        # self.expr_str_pending = expr_str  # Expression to initialize later if plot() is called first

        self.transformations = (standard_transformations + (implicit_multiplication_application,))

        # This will be used for numerical lambdify
        # self.lambdify_numeric_dict = self._build_numeric_dict()

    # def _get_local_dict(self):
    #     """
    #     (Private) Returns a local dictionary of predefined symbols and functions used
    #     during symbolic parsing and evaluation of signal expressions.

    #     This includes:
    #     - Common signal processing functions such as:
    #         - u(t): Heaviside step (centered at 0.5)
    #         - rect(t), tri(t), sinc(t), ramp(t), delta(t)
    #     - Piecewise functions:
    #         - Piecewise, pw
    #     - Mathematical functions and constants:
    #         - sin, cos, exp, pi, abs, arg, re, im, conj
    #     - Symbols used in frequency/time analysis: t, ω, Ω, τ, λ
    #     - Support for complex signals: i, j, re, im, conj, abs, arg 
    #     - Previously defined signal names in the format "name(variable)"

    #     Returns:
    #         dict: A dictionary mapping names to SymPy expressions or functions.

    #     Examples:
    #         >>> local_dict = self._get_local_dict()
    #         >>> expr = parse_expr("x(t) + rect(t)", local_dict=local_dict)
    #     """
    #     d = {
    #         'delta':        lambda n: np.where(n == 0, 1, 0),
    #         'u':            lambda n: np.where(n >= 0, 1, 0),
    #         'rect':         lambda n: np.where(abs(n) <= 1, 1, 0),
    #         'tri':          lambda n: np.where(abs(n) <= 3, 1 - abs(n)/3, 0),
    #         'ramp':         lambda n: np.where(n >= 0, n, 0),
    #         'sinc':         lambda n: np.sinc(n),
    #         'KroneckerDelta': sp.KroneckerDelta,
    #         'pi':           sp.pi,
    #         'sin':          sp.sin,
    #         'cos':          sp.cos,
    #         'exp':          sp.exp,
    #         'Piecewise':    sp.Piecewise,
    #         'pw':           sp.Piecewise,
    #         're':           sp.re,
    #         'im':           sp.im,
    #         'conj':         sp.conjugate,
    #         'abs':          lambda x: np.abs(x),
    #         'arg':          sp.arg,
    #         'i':            sp.I,
    #         'j':            sp.I,
    #         'n':            sp.Symbol('n', integer=True),
    #         'omega':        sp.Symbol('omega'),
    #         'Omega':        sp.Symbol('Omega'),
    #         'k':            sp.Symbol('k'),
    #         'm':            sp.Symbol('m'),
    #     }
    #     d.update(self.var_symbols)
    #     for name, expr in self.signal_defs.items():
    #         for var in self.var_symbols.values():
    #             d[f"{name}({var})"] = expr.subs(var, var)
    #     return d


    def _get_local_dict(self):
        """
        Local dictionary for parsing expressions (pure symbolic version for SymPy).
        """
        n = sp.Symbol('n', integer=True)

        d = {
            # 'KroneckerDelta': sp.KroneckerDelta,
            # 'delta':        lambda n: sp.Piecewise((1, n == 0), (0, True)),
            # 'KroneckerDelta': sp.KroneckerDelta,
            'delta':        sp.Function('delta'),
            # 'u':            lambda n: sp.Piecewise((1, n >= 0), (0, True)),
            'u':            sp.Function('u'),
            # 'rect':         lambda n: sp.Piecewise((1, abs(n) <= 1), (0, True)),
            'rect':         sp.Function('rect'),
            'tri':          sp.Function('tri'),
            # 'tri':          lambda n: sp.Piecewise((1 - abs(n)/3, abs(n) <= 3), (0, True)),
            'ramp':         sp.Function('ramp'),
            # 'ramp':         lambda n: sp.Piecewise((n, n >= 0), (0, True)),
            'Piecewise':    sp.Piecewise,
            'pw':           sp.Piecewise,
            'sin':          sp.sin,
            'cos':          sp.cos,
            # 'sinc':         lambda n: sp.sinc(n),
            'sinc':         sp.sinc,
            'exp':          sp.exp,
            're':           sp.re,
            'im':           sp.im,
            'conj':         sp.conjugate,
            'abs':          sp.Abs,
            'arg':          sp.arg,
            'i':            sp.I,
            'j':            sp.I,
            'pi':           sp.pi,
        }
        d.update(self.var_symbols)
        return d

    def _get_numeric_dict(self):
        d = {
            # 'KroneckerDelta': lambda n, m: np.where(np.equal(n, m), 1, 0),
            'delta': lambda n: np.where(n == 0, 1, 0),
            'u': lambda n: np.where(n >= 0, 1, 0),
            'rect': lambda n: np.where(abs(n) <= 1, 1, 0),
            'tri': lambda n: np.where(abs(n) <= 3, 1 - abs(n)/3, 0),
            'ramp': lambda n: np.where(n >= 0, n, 0),
            'sinc': lambda n: np.sinc(n),
            'abs': np.abs,
            'sin': np.sin,
            'cos': np.cos,
            'exp': np.exp,
            're': np.real,
            'im': np.imag,
            'conj': np.conj,
            'arg': np.angle,
            'pi': np.pi
        }
        return d
    
   
    def add_signal(self, expr_str, label=None, period=None, 
                   xticks=None, yticks=None, xtick_labels=None, ytick_labels=None,
                   xticks_delta=None, yticks_delta=None):
        """
        Adds a new discrete signal to the system.

        Supports both symbolic definitions and numerical convolutions, with fully customizable 
        tick management for plotting.

        Parameters
        ----------
        expr_str : str
            String defining the signal. Must follow one of these formats:
            - "x[n] = expression"  (for symbolic expressions)
            - "y[n] = conv(x[n], h[n])"  (for convolution between existing signals)

        label : str, optional
            Custom label for the y-axis (overrides automatic signal label).

        period : int or float, optional
            If provided, creates periodic extension of the signal over the defined horizontal range.

        xticks : 'auto', list, or None, optional
            Positions of the xticks:
            - 'auto' (default): automatic ticks on non-zero values or spaced by `xticks_delta` if defined.
            - list: manually specified integer positions.
            - None: no xticks.

        yticks : 'auto', 'fit', list, or None, optional
            Positions of the yticks:
            - 'auto' (default): automatic range-based ticks or spaced by `yticks_delta` if defined.
            - 'fit': places a tick for each unique value in the signal.
            - list: manually specified positions.
            - None: no yticks.

        xtick_labels : list or None, optional
            Custom labels for the xticks. If provided, must match the length of `xticks`.

        ytick_labels : list or None, optional
            Custom labels for the yticks. If provided, must match the length of `yticks`.

        xticks_delta : int or float, optional
            Step size for automatic xtick spacing (used only if `xticks='auto'`).

        yticks_delta : int or float, optional
            Step size for automatic ytick spacing (used only if `yticks='auto'`).

        Behavior Summary
        ----------------
        - If tick arguments are not provided for a signal, they inherit the global defaults 
        defined at object creation (__init__).
        - If no defaults exist, fallback behavior applies:
            * xticks → 'auto' (ticks on non-zero positions or spaced by xticks_delta).
            * yticks → 'auto' (range-based or spaced by yticks_delta).
        - yticks mode 'fit' allows plotting all distinct output values.
        - Tick label lists must match their respective tick lists in length.

        """
        # Detectar si es una convolución numérica
        conv_match = re.match(r"^(\w+)\[n\]\s*=\s*conv\((\w+)\[n\],\s*(\w+)\[n\]\)$", expr_str.strip())
        if conv_match:
            output_name, x_name, h_name = conv_match.groups()
            n_vals, y_vals = self.convolution(x_name, h_name)
            self.signal_defs[output_name] = None  # No symbolic expr
            conv_dict = dict(zip(n_vals, y_vals))
            self.funcs[output_name] = lambda n: np.array([conv_dict.get(k, 0.0) for k in np.atleast_1d(n)])
            self.var_symbols[output_name] = sp.Symbol('n', integer=True)
            self.signal_xticks[output_name] = xticks if xticks is not None else self.default_xticks
            self.signal_yticks[output_name] = yticks if yticks is not None else self.default_yticks
            self.signal_xticks_delta[output_name] = xticks_delta if xticks_delta is not None else self.default_xticks_delta
            self.signal_yticks_delta[output_name] = yticks_delta if yticks_delta is not None else self.default_yticks_delta
            self.signal_xtick_labels[output_name] = xtick_labels if xtick_labels is not None else self.default_xtick_labels
            self.signal_ytick_labels[output_name] = ytick_labels if ytick_labels is not None else self.default_ytick_labels
            return  # Termina aquí porque ya hemos registrado la convolución

        m = re.match(r"(?P<name>\w+)\s*\[\s*(?P<var>\w+)\s*\]\s*=\s*(?P<expr>.+)", expr_str)
        if not m:
            raise ValueError("Expression must be in form: name[n] = ...")

        name, var, expr_body = m.group('name'), m.group('var'), m.group('expr')

        if var not in self.var_symbols:
            self.var_symbols[var] = sp.Symbol(var)
        var_sym = self.var_symbols[var]

        self._update_expression_and_func(name, expr_body, var_sym)
        self.var_symbols[name] = var_sym

        if label is not None:
            self.custom_labels[name] = label

        if period is not None:
            self.signal_periods[name] = period

            horiz_min, horiz_max = self.horiz_range
            num_periods = int(np.ceil((horiz_max - horiz_min) / period))
            k_range = range(-num_periods - 2, num_periods + 3)

            expanded_expr = sum(self.signal_defs[name].subs(var_sym, var_sym - period * k) for k in k_range)
            self.signal_defs[name] = expanded_expr
            self._update_expression_and_func(name, str(expanded_expr), var_sym)

        # Store ticks with proper inheritance
        self.signal_xticks[name] = xticks if xticks is not None else self.default_xticks
        self.signal_yticks[name] = yticks if yticks is not None else self.default_yticks
        self.signal_xtick_labels[name] = xtick_labels if xtick_labels is not None else self.default_xtick_labels
        self.signal_ytick_labels[name] = ytick_labels if ytick_labels is not None else self.default_ytick_labels
        self.signal_xticks_delta[name] = xticks_delta if xticks_delta is not None else self.default_xticks_delta
        self.signal_yticks_delta[name] = yticks_delta if yticks_delta is not None else self.default_yticks_delta

    def convolution(self, x_name, h_name, horiz_range=None, margin_multiplier=1.0):
        var_x = self.var_symbols[x_name]
        var_h = self.var_symbols[h_name]
        expr_x = self.signal_defs[x_name]
        expr_h = self.signal_defs[h_name]

        support_x = self._estimate_discrete_support(expr_x, var_x, self.horiz_range, margin_multiplier)
        support_h = self._estimate_discrete_support(expr_h, var_h, self.horiz_range, margin_multiplier)

        n_min_x, n_max_x = support_x
        n_min_h, n_max_h = support_h
        n_vals_x = np.arange(n_min_x, n_max_x + 1)
        n_vals_h = np.arange(n_min_h, n_max_h + 1)

        x_vals = self.funcs[x_name](n_vals_x)
        h_vals = self.funcs[h_name](n_vals_h)
        y_vals_full = np.convolve(x_vals, h_vals, mode='full')

        n_start = n_min_x + n_min_h
        n_end = n_max_x + n_max_h
        n_vals_conv = np.arange(n_start, n_end + 1)

        return n_vals_conv, y_vals_full

    def draw_function(self, marker='o', stem_color=None, marker_size=6, line_width=3):
        """
        Draws the discrete-time signal as a stem plot.

        Args:
            marker (str): Marker symbol (default 'o').
            stem_color (str): Color of stem and markers. If None, uses self.color.
            marker_size (float): Size of the marker.
            line_width (float): Width of the stem lines.

        Examples:
            >>> self.draw_function()
        """
        if stem_color is None:
            stem_color = self.color

        markerline, stemlines, baseline = self.ax.stem(
            self.n_vals, self.y_vals, basefmt='k'
        )
        markerline.set_color(stem_color)
        stemlines.set_color(stem_color)
        markerline.set_marker(marker)
        plt.setp(markerline, markersize=marker_size)
        plt.setp(stemlines, linewidth=line_width)

        n0, n1 = self.horiz_range
        delta = (n1 - n0) * 0.1

        expr = self.signal_defs[self.current_name]
        var = self.var_symbols[self.current_name]
        func = self.funcs[self.current_name]
        tol = 1e-12

        # Evaluate small neighborhoods outside range numerically
        n_left = np.arange(n0 - 10, n0)
        n_right = np.arange(n1 + 1, n1 + 11)
        
        try:
            y_left = np.array(func(n_left), dtype=float)
            y_right = np.array(func(n_right), dtype=float)
        except Exception:
            # fallback in case function can't evaluate
            y_left = np.zeros_like(n_left, dtype=float)
            y_right = np.zeros_like(n_right, dtype=float)

        draw_left = np.sum(np.abs(y_left)) > tol
        draw_right = np.sum(np.abs(y_right)) > tol

        y_mid = (self.y_min + self.y_max) / 2
        if draw_left:
            self.ax.text(n0 - delta, y_mid, r'$\cdots$', ha='left', va='center',
                         color=self.color, fontsize=14, zorder=10)
        if draw_right:
            self.ax.text(n1 + delta, y_mid, r'$\cdots$', ha='right', va='center',
                         color=self.color, fontsize=14, zorder=10)
        
        return markerline, stemlines


    def _prepare_plot(self, y_vals):
        """
        (Private) Determines the vertical plotting range (y-axis limits) based on the signal values. 
        Called from `plot()`.
        """

        try:
            y_min = min(np.min(y_vals), 0)
            y_max = max(np.max(y_vals), 0)

            # Fit in case range is too narrow
            if abs(y_max - y_min) < 1e-2:
                ymin -= 1
                ymax += 1

            # Apply vertical range if provided
            if self.vert_range:
                self.y_min, self.y_max = self.vert_range
            else:
                self.y_min, self.y_max = y_min, y_max
        
        except Exception:
            self.y_min, self.y_max = -1, 1

    # def _evaluate_signal(self, n_vals):
    #     y = self.func(n_vals)
    #     return np.full_like(n_vals, y, dtype=float) if np.isscalar(y) else np.array(y, dtype=float)

    def _evaluate_signal(self, expr):
        """
        Lambdify the symbolic expression and evaluate over self.n_vals.
        """
        n_vals = np.arange(self.n_range[0], self.n_range[1] + 1)
        f_lamb = sp.lambdify(self.n, expr, modules=["numpy", self.local_dict])
        try:
            y_vals = np.array(f_lamb(n_vals), dtype=float)
        except:
            # Fallback to scalar evaluation (safe for non-vectorizable cases)
            y_vals = np.array([float(f_lamb(int(n_i))) for n_i in n_vals])
        return n_vals, y_vals

    def setup_axes(self, horiz_range=None):
        """
        Heredar de BasePlotter
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
        # self.fig.tight_layout()

    def draw_ticks(self, xticks=None, yticks=None, xtick_labels=None, ytick_labels=None, 
                tick_size_px=None, tol=1e-12, fontsize=12):
        """
        Draws tick marks and labels for discrete signals:
        - X-axis: optional labels at non-zero positions or user-defined.
        - Y-axis: tick marks and labels.
        Full inheritance of per-signal and global tick parameters is applied.

        Args:
            xticks (list or 'auto' or None): X-axis tick positions.
            yticks (list or 'auto' or 'fit' or None): Y-axis tick positions.
            xtick_labels (list or None): Custom labels for xticks.
            ytick_labels (list or None): Custom labels for yticks.
            tick_size_px (int): Length of tick marks for yticks in pixels.
            tol (float): Tolerance for detecting zero values.
            fontsize (int): Font size for tick labels.
        """
            # --- Inheritance logic ---
        name = self.current_name
        if xticks is None:
            xticks = self.signal_xticks.get(name, self.default_xticks)
        if yticks is None:
            yticks = self.signal_yticks.get(name, self.default_yticks)
        if xtick_labels is None:
            xtick_labels = self.signal_xtick_labels.get(name, self.default_xtick_labels)
        if ytick_labels is None:
            ytick_labels = self.signal_ytick_labels.get(name, self.default_ytick_labels)

        xticks_delta = self.signal_xticks_delta.get(name, self.default_xticks_delta)
        yticks_delta = self.signal_yticks_delta.get(name, self.default_yticks_delta)

        def get_nonzero_positions(name=None, n_min=None, n_max=None, tol=1e-12):
            if name is None:
                if not self.current_name:
                    raise ValueError("No signal selected.")
                name = self.current_name
            if name not in self.signal_defs:
                raise ValueError(f"Signal '{name}' is not defined.")
            if not hasattr(self, 'n_vals') or not hasattr(self, 'y_vals') or self.current_name != name:
                self.plot(name)
                plt.close(self.fig)
            positions = []
            for n, y in zip(self.n_vals, self.y_vals):
                if n_min is not None and n < n_min:
                    continue
                if n_max is not None and n > n_max:
                    continue
                if abs(y) > tol:
                    positions.append(int(n))
            return positions

        def format_fraction_label(value, tol=1e-12):
            from fractions import Fraction
            f = Fraction(value).limit_denominator(24)
            if abs(float(f) - value) > tol:
                return f"{value:.3g}"
            if f.denominator == 1:
                return f"{f.numerator}"
            else:
                return rf"\frac{{{f.numerator}}}{{{f.denominator}}}"

        def get_unique_yvalues(tol=1e-12):
            unique = []
            for y in self.y_vals:
                if not any(abs(y - u) <= tol for u in unique):
                    unique.append(y)
            return sorted(unique)
        
        def validate_tick_list(ticks, axis):
            if ticks is None:
                return []
            if isinstance(ticks, str) and ticks == 'auto':
                return 'auto'
            # Asegurarse que es iterable (aunque sea escalar único)
            arr = np.atleast_1d(ticks)
            if axis == 'x' and not np.all(np.equal(np.mod(arr, 1), 0)):
                raise ValueError("All xticks must be integers")
            return list(arr)

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

        # --- Determine effective xticks and yticks ---
        # effective_xticks = xticks if xticks is not None else getattr(self, 'init_xticks_arg', 'auto')
        # effective_yticks = yticks if yticks is not None else getattr(self, 'init_yticks_arg', 'auto')

        tick_px = tick_size_px if tick_size_px is not None else self.tick_size_px
        dx, dy = px_to_data_length(tick_px)

        # -------- Y-AXIS --------
        if yticks == 'auto':
            y0 = np.floor(self.y_min)
            y1 = np.ceil(self.y_max)
            if yticks_delta:
                k_min = int(np.floor(y0 / yticks_delta))
                k_max = int(np.ceil(y1 / yticks_delta))
                raw_yticks = [k * yticks_delta for k in range(k_min, k_max + 1)]
            else:
                raw_yticks = np.linspace(y0, y1, 3)
        elif yticks == 'fit':
            raw_yticks = get_unique_yvalues()
        elif yticks is None:
            raw_yticks = []
        else:
            raw_yticks = validate_tick_list(yticks, axis='y')

        if ytick_labels:
            if len(ytick_labels) != len(raw_yticks):
                raise ValueError("ytick_labels and yticks must have the same length")
            ylabels = ytick_labels
        else:
            ylabels = []
            for y in raw_yticks:
                if self.fraction_ticks:
                    ylabels.append(format_fraction_label(y, tol))
                else:
                    ylabels.append(f"{y:.3g}")

        ylim = self.ax.get_ylim()
        for y, lbl in zip(raw_yticks, ylabels):
            if ylim[0] <= y <= ylim[1]:
                self.ax.plot([0 - dx/2, 0 + dx/2], [y, y], transform=self.ax.transData,
                            color='black', linewidth=1.2, clip_on=False)
                offset = (-4, -8) if abs(y) < tol else (-4, 0)
                self.ax.annotate(rf'${lbl}$', xy=(0, y), xycoords='data',
                                textcoords='offset points', xytext=offset,
                                ha='right', va='center', fontsize=fontsize, zorder=10,
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                        edgecolor='none', alpha=self.alpha))

        # -------- X-AXIS --------
        if xticks == 'auto':
            n_min, n_max = self.horiz_range
            if xticks_delta:
                k_min = int(np.ceil(n_min / xticks_delta))
                k_max = int(np.floor(n_max / xticks_delta))
                raw_xticks = [k * xticks_delta for k in range(k_min, k_max + 1)]
            else:
                raw_xticks = [n for n in get_nonzero_positions(name, n_min, n_max, tol=0) if n != 0]
        elif xticks is None:
            raw_xticks = []
        else:
            raw_xticks = validate_tick_list(xticks, axis='x')

        if xtick_labels:
            if len(xtick_labels) != len(raw_xticks):
                raise ValueError("xtick_labels and xticks must have the same length")
            xlabels = xtick_labels
        else:
            xlabels = [f"{int(n)}" for n in raw_xticks]

        xlim = self.ax.get_xlim()
        for n, lbl in zip(raw_xticks, xlabels):
            if xlim[0] <= n <= xlim[1] and n != 0:
                idx = np.where(self.n_vals == n)[0]
                y_val = self.y_vals[idx[0]] if idx.size > 0 else 0
                offset_y = 8 if y_val < 0 else -8
                self.ax.annotate(rf'${lbl}$', xy=(n, 0), xycoords='data',
                                 textcoords='offset points', xytext=(0, offset_y),
                                 ha='center', va='bottom' if offset_y > 0 else 'top',
                                 fontsize=fontsize, zorder=10,
                                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                           edgecolor='none', alpha=self.alpha))
                
    def draw_labels(self, fontsize=16):
        """
        Heredar de BasePlotter
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
        self.ax.text(x_pos, y_pos, rf'${self.xlabel}$', fontsize=fontsize, ha='right', va='bottom')

        # Y-axis label: slightly below the top y limit (still inside the figure)
        x_pos = 0.01 * (x_lim[1] - x_lim[0])
        y_pos = y_lim[1] - 0.1 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, rf'${self.ylabel}$', fontsize=fontsize, ha='left', va='bottom', rotation=0)

    def discrete_upsampling_wrapper(self, func, factor, offset=0):
        """
        Wraps the lambdified function to simulate upsampling behavior with optional offset.
        If (n - offset) is multiple of factor, evaluates func((n - offset)//factor), else returns 0.
        """
        def wrapped(n):
            n = np.asarray(n)
            result = np.zeros_like(n, dtype=float)
            mask = ((n - offset) % factor == 0)
            valid_n = ((n - offset) // factor)[mask]
            result[mask] = func(valid_n)
            return result
        return wrapped
    
    def _update_expression_and_func(self, name, expr_body, var_sym):
        """
        Parses and updates the symbolic expression and numeric function
        for the given signal name.

        - Converts array-style access (e.g. z[n-2]) into functional form (z(n-2))
        - Substitutes previously defined signals recursively
        - Lambdifies the final expression for numerical evaluation
        - Detects upsampling patterns and wraps evaluation if needed
        """

        # Step 1: preprocess to replace [] by () for function-like calls
        expr_body_preprocessed = re.sub(r"(\w+)\s*\[\s*(.+?)\s*\]", r"\1(\2)", expr_body)

        # Build local dictionary with primitives as symbolic functions
        local_dict = self._get_local_dict()
        for other_name in self.signal_defs:
            local_dict[other_name] = sp.Function(other_name)

        # Parse expression
        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed_expr = parse_expr(expr_body_preprocessed, local_dict=local_dict, transformations=transformations)

        # Save symbolic expression before substitution
        original_expr = parsed_expr

        # Perform recursive substitution of previous signals
        for other_name, other_expr in self.signal_defs.items():
            f = sp.Function(other_name)
            matches = parsed_expr.find(f)
            for call in matches:
                if isinstance(call, sp.Function):
                    arg = call.args[0]
                    replaced = other_expr.subs(var_sym, arg)
                    parsed_expr = parsed_expr.subs(call, replaced)

        # Save final symbolic expression
        self.signal_defs[name] = parsed_expr

        # Create numerical function using numeric_dict
        # func = sp.lambdify(var_sym, parsed_expr, modules=["numpy", self._get_numeric_dict()])
        func = sp.lambdify(var_sym, parsed_expr, modules=[self._get_numeric_dict()])

        # Analyze upsampling pattern from original (non-expanded) expression
        factor = None
        offset = 0
        a, b = sp.Wild('a'), sp.Wild('b')
        pattern = a * var_sym + b

        # Define which functions are discrete-only (subject to upsampling)
        discrete_functions = {'delta', 'u', 'rect', 'tri', 'ramp'}

        for f in original_expr.atoms(sp.Function):
            # Allow primitives and user-defined functions
            if f.func.__name__ in discrete_functions or f.func.__name__ in self.signal_defs:
                arg = f.args[0]
                match = arg.match(pattern)
                if match and match[a].is_Rational and match[a].q > 1:
                    factor = match[a].q
                    offset = int(match[b] * factor)
                    break

        if factor:
            func = self.discrete_upsampling_wrapper(func, factor, offset)

        self.funcs[name] = func


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
        Plots the discrete signal specified by name.

        This method:
        - Retrieves the previously parsed expression and lambdified function.
        - Evaluates it over the specified integer range.
        - Sets up the axes, ticks, and renders the stem plot.

        Args:
            name (str, optional): Name of the signal to plot.
                                If None, uses the last-added signal.

        Raises:
            ValueError: If the signal name is not defined.

        Examples:
            >>> dsp = DiscreteSignalPlotter("x[n]=delta(n)+u(n-1)", n_range=(-5, 5))
            >>> dsp.plot("x")
        """
        # Select signal
        if name is None:
            if not self.signal_defs:
                raise ValueError("No signals defined to plot.")
            name = list(self.signal_defs.keys())[-1]

        if name not in self.signal_defs:
            raise ValueError(f"Signal '{name}' is not defined.")

        self.current_name = name
        # self.func_name = name

        expr = self.signal_defs[name]
        var = self.var_symbols[name]
        self.xlabel = str(var)
        self.ylabel = f"{name}[{self.xlabel}]"
        if name in self.custom_labels:
            self.ylabel = self.custom_labels[name]

        # Generate discrete points for evaluation
        self.n_vals = np.arange(self.horiz_range[0], self.horiz_range[1] + 1)

        # Use precomputed lambdified function
        func = self.funcs[name]

        # Evaluate function over n_vals
        try:
            y_raw = func(self.n_vals)
            y_vals = np.array(y_raw, dtype=float)
        except Exception:
            y_vals = np.zeros_like(self.n_vals, dtype=float)

        self.y_vals = y_vals

        # Prepare vertical range
        self._prepare_plot(self.y_vals)

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.4)

        self.setup_axes()
        self.draw_function()

        xticks = self.signal_xticks.get(name, 'auto')
        yticks = self.signal_yticks.get(name, 'auto')
        xtick_labels = self.signal_xtick_labels.get(name, 'auto')
        ytick_labels = self.signal_ytick_labels.get(name, 'auto')
        self.draw_ticks(xticks=xticks, yticks=yticks, xtick_labels=xtick_labels, ytick_labels=ytick_labels)

        self.draw_labels()
        self.show()

    def _estimate_discrete_support(self, expr, var, horiz_range, margin_multiplier=1.0):
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
        search_range = (horiz_range[0] - margin, horiz_range[1] + margin + 1)

        n_vals = np.arange(*search_range)
        func = sp.lambdify(var, expr, modules=[self._get_numeric_dict()])
        y_vals = np.abs(func(n_vals))

        threshold = np.max(y_vals) * 1e-3
        nonzero_indices = np.where(y_vals > threshold)[0]

        if len(nonzero_indices) == 0:
            return (0, 0)
        
        min_index = nonzero_indices[0]
        max_index = nonzero_indices[-1]
        return (n_vals[min_index], n_vals[max_index])
    
    def plot_convolution(self, x_name, h_name, output_name=None, horiz_range=None, margin_multiplier=1.0):
        """
        Computes and plots the discrete convolution y[n] = (x * h)[n] between two signals (purely numeric version).

        Parameters:
        x_name : str
            Name of the first signal.
        h_name : str
            Name of the second signal.
        output_name : str, optional
            Name to assign to the output convolution signal.
        horiz_range : tuple, optional
            Range of n values to compute the convolution.
        """
        import numpy as np

        # Retrieve variables and expressions
        var_x = self.var_symbols[x_name]
        var_h = self.var_symbols[h_name]
        expr_x = self.signal_defs[x_name]
        expr_h = self.signal_defs[h_name]

        # Estimate supports
        support_x = self._estimate_discrete_support(expr_x, var_x, self.horiz_range, margin_multiplier)
        support_h = self._estimate_discrete_support(expr_h, var_h, self.horiz_range, margin_multiplier)

        n_min_x, n_max_x = support_x
        n_min_h, n_max_h = support_h

        n_vals_x = np.arange(n_min_x, n_max_x + 1)
        n_vals_h = np.arange(n_min_h, n_max_h + 1)
    
        # Evaluate signals numerically
        x_vals = self.funcs[x_name](n_vals_x)
        h_vals = self.funcs[h_name](n_vals_h)

        # Perform full discrete convolution
        y_vals_full = np.convolve(x_vals, h_vals, mode='full')

        # Determine new n range after convolution
        n_start = n_min_x + n_min_h
        n_end = n_max_x + n_max_h
        n_vals_conv = np.arange(n_start, n_end + 1)

        # Register result as new signal (only numeric, no symbolic expression)
        if output_name is None:
            output_name = "y"
        else:
            output_name = re.sub(r"\[.*?\]", "", output_name)
          
        self.signal_defs[output_name] = None    # No symbolic expression
        conv_dict = dict(zip(n_vals_conv, y_vals_full))
        self.funcs[output_name] = lambda n: np.array([conv_dict.get(k, 0.0) for k in np.atleast_1d(n)])
        self.var_symbols[output_name] = sp.Symbol('n', integer=True)
        self.current_name = output_name # Set as current active signal

        # Use requested plot range
        if horiz_range is None:
            horiz_range = self.horiz_range

        self.signal_xticks[output_name] = 'auto'
        self.signal_yticks[output_name] = 'auto'
        self.signal_xtick_labels[output_name] = None
        self.signal_ytick_labels[output_name] = None

        self.n_vals = np.arange(horiz_range[0], horiz_range[1] + 1)
        self.y_vals = self.funcs[output_name](self.n_vals)
        self._prepare_plot(self.y_vals)

        self.xlabel = "n"
        self.ylabel = rf"{output_name}[n]"

        # Ensure current_name is fully consistent for downstream plotting
        if hasattr(self, 'init_xticks_arg'):
            self.init_xticks_arg = None  # Disable old cached ticks for new signal

        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self.fig.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.4)

        self.setup_axes()
        self.draw_function()
        self.draw_ticks()
        self.draw_labels()
        self.show()

    def plot_convolution_steps(self, x_name, h_name, n_actual):
        n_min, n_max = self.horiz_range
        k_vals = np.arange(n_min, n_max + 1)

        x_vals = self.funcs[x_name](k_vals)
        h_vals = self.funcs[h_name](k_vals)
        h_shifted_vals = self.funcs[h_name](n_actual - k_vals)
        product_vals = x_vals * h_shifted_vals

        # Compute full convolution result
        n_conv_raw, y_conv_raw = self.convolution(x_name, h_name)

        # Build full convolution over horiz_range
        n_conv = np.arange(n_min, n_max + 1)
        conv_dict = dict(zip(n_conv_raw, y_conv_raw))
        y_conv = np.array([conv_dict.get(n, 0.0) for n in n_conv])

        signals = [
            (k_vals, h_vals, rf"{h_name}[k]", h_name, False, False),
            (k_vals, x_vals, rf"{x_name}[k]", x_name, False, False),
            (k_vals, h_shifted_vals, rf"{h_name}[n-k]", h_name, True, True),
            (k_vals, product_vals, rf"{x_name}[k]{h_name}[n-k]", None, True, True),
            (n_conv, y_conv, rf"{x_name}*{h_name}", None, False, False)
        ]

        for k, y, label, ref_name, shifted, annotate in signals:
            self.n_vals = k
            self.y_vals = y
            self.xlabel = "k" if label != rf"{x_name}*{h_name}" else "n"
            self.ylabel = label

            self._prepare_plot(self.y_vals)
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.fig.tight_layout()
            plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.4)

            self.current_name = '__temp_convolution__'
            self.signal_defs[self.current_name] = None
            self.funcs[self.current_name] = lambda n: np.interp(n, self.n_vals, self.y_vals, left=0, right=0)
            self.var_symbols[self.current_name] = sp.Symbol('n', integer=True)

            self.setup_axes()
            self.draw_function()

            # Draw continuation dots if values exist beyond horiz_range
            tol = 1e-12
            delta = (n_max - n_min) * 0.1
            y_left = self.funcs[self.current_name](np.arange(n_min - 10, n_min))
            y_right = self.funcs[self.current_name](np.arange(n_max + 1, n_max + 11))
            y_mid = (self.y_min + self.y_max) / 2
            if np.any(np.abs(y_left) > tol):
                self.ax.text(n_min - delta, y_mid, r'$\cdots$', ha='left', va='center',
                            color=self.color, fontsize=14, zorder=10)
            if np.any(np.abs(y_right) > tol):
                self.ax.text(n_max + delta, y_mid, r'$\cdots$', ha='right', va='center',
                            color=self.color, fontsize=14, zorder=10)

            if ref_name and ref_name in self.signal_xticks:
                xticks_def = self.signal_xticks[ref_name]
                xtick_labels_def = self.signal_xtick_labels.get(ref_name, None)
                if shifted:
                    if xticks_def is None or xticks_def == 'auto':
                        self.draw_ticks(xticks=[n_actual], xtick_labels=['n'])
                    elif isinstance(xticks_def, (list, tuple)):
                        shifted_xticks = [n_actual - v for v in xticks_def]
                        shifted_labels = []
                        for i, v in enumerate(xticks_def):
                            delta_shift = n_actual - v
                            if delta_shift == 0:
                                label_str = "n"
                            elif delta_shift > 0:
                                label_str = f"n+{delta_shift}"
                            else:
                                label_str = f"n{delta_shift}"
                            shifted_labels.append(label_str)
                        self.draw_ticks(xticks=shifted_xticks, xtick_labels=shifted_labels)
                    else:
                        self.draw_ticks(xticks=[n_actual], xtick_labels=['n'])
                else:
                    self.draw_ticks(xticks=xticks_def, xtick_labels=xtick_labels_def)
            else:
                self.draw_ticks()

            self.draw_labels()
            
            if annotate:
                x_lim = self.ax.get_xlim()
                y_lim = self.ax.get_ylim()
                self.ax.text(x_lim[1] * 0.95, y_lim[1] * 0.95, f"n={n_actual}", fontsize=12, ha='right', va='top')
            
            # # Mark n_actual point on convolution plot
            # if label == rf"{x_name}*{h_name}":
            #     if n_actual in k:
            #         idx = np.where(k == n_actual)[0][0]
            #         y_val = y[idx]
            #         self.ax.plot(n_actual, y_val, 'o', color='blue', markersize=10)

            self.show()

            # conv_signal_name = f"temp_conv_{x_name}_{h_name}"
            self.add_signal(f"temp[n]=conv({x_name}[n], {h_name}[n])", label='y')
            self.plot('temp')

            # del self.signal_defs[y]
            # del self.funcs[y]
            # del self.var_symbols[y]


    def convolution_anim(self, x_name, h_name):
        n_min, n_max = self.horiz_range
        k_vals = np.arange(n_min, n_max + 1)
        x_vals = self.funcs[x_name](k_vals)
        h_vals = self.funcs[h_name](k_vals)
        h_shifted_vals_init = self.funcs[h_name](n_min - k_vals)
        h_shifted_forward_init = self.funcs[h_name](k_vals + n_min)
        product_vals_init = x_vals * h_shifted_vals_init

        # Calcular convolución completa
        n_conv, y_conv = self.convolution(x_name, h_name)

        signals = [
            (k_vals, x_vals, rf"{x_name}[k]"),
            (k_vals, h_vals, rf"{h_name}[k]"),
            (k_vals, h_shifted_vals_init, rf"{h_name}[n-k]"),
            (k_vals, h_shifted_forward_init, rf"{h_name}[k+n]"),
            (k_vals, product_vals_init, rf"{x_name}[k]{h_name}[n-k]"),
            (n_conv, y_conv, rf"{x_name}*{h_name}")
        ]

        figs, axs = plt.subplots(3, 2, figsize=(self.figsize[0]*2, self.figsize[1]*3))
        plt.subplots_adjust(bottom=0.15, top=0.95, hspace=0.2, wspace=0.15)

        positions = [(0,0), (0,1), (1,0), (1,1), (2,0), (2,1)]
        markers = []
        for (row, col), (k, y, label) in zip(positions, signals):
            ax = axs[row, col]
            self.ax = ax
            self.n_vals = k
            self.y_vals = y
            self.xlabel = "k"
            self.ylabel = label

            self._prepare_plot(self.y_vals)
            self.current_name = '__temp_convolution__'
            self.signal_defs[self.current_name] = None
            self.funcs[self.current_name] = lambda n: np.interp(n, self.n_vals, self.y_vals, left=0, right=0)
            self.var_symbols[self.current_name] = sp.Symbol('n', integer=True)

            self.setup_axes()
            markerline, stemlines = self.draw_function()
            self.draw_ticks(fontsize=10)
            self.draw_labels(fontsize=12)
            markers.append((ax, markerline, stemlines, k))

        # Añadir marcador azul en la convolución
        ax_conv = axs[2, 1]
        point, = ax_conv.plot([n_min], [y_conv[0]], 'o', color='blue', markersize=10)

        ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])
        slider = Slider(ax_slider, 'n', n_min, n_max, valinit=n_min, valstep=1)

        def update(n_actual):
            h_shifted_vals = self.funcs[h_name](n_actual - k_vals)
            h_shifted_forward = self.funcs[h_name](k_vals + n_actual)
            product_vals = x_vals * h_shifted_vals
            updates = [x_vals, h_vals, h_shifted_vals, h_shifted_forward, product_vals, y_conv]
            for (ax, markerline, stemlines, n_vals_i), new_y in zip(markers, updates):
                markerline.set_data(n_vals_i, new_y)
                segments = [ [(n, 0), (n, y)] for n, y in zip(n_vals_i, new_y) ]
                stemlines.set_segments(segments)
            # Actualizar marcador azul
            if n_actual in n_conv:
                idx = np.where(n_conv == n_actual)[0][0]
                point.set_data(n_actual, y_conv[idx])
            else:
                point.set_data([], [])
            figs.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()
