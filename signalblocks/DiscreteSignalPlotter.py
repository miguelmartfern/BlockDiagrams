import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re
import warnings

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
        # xticks_delta=None,
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
        self.signal_defs = {}
        self.var_symbols = {}
        self.funcs = {} 
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
        
        # self.pi_mode = pi_mode

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

        # self.xticks_delta = xticks_delta
        self.yticks_delta = yticks_delta

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
            'u':            lambda n: sp.Piecewise((1, n >= 0), (0, True)),
            'rect':         lambda n: sp.Piecewise((1, abs(n) <= 1), (0, True)),
            'tri':          lambda n: sp.Piecewise((1 - abs(n)/3, abs(n) <= 3), (0, True)),
            'ramp':         lambda n: sp.Piecewise((n, n >= 0), (0, True)),
            'sinc':         lambda n: sp.sinc(n),
            'Piecewise':    sp.Piecewise,
            'pw':           sp.Piecewise,
            'pi':           sp.pi,
            'sin':          sp.sin,
            'cos':          sp.cos,
            'exp':          sp.exp,
            're':           sp.re,
            'im':           sp.im,
            'conj':         sp.conjugate,
            'abs':          sp.Abs,
            'arg':          sp.arg,
            'i':            sp.I,
            'j':            sp.I,
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
    
    # def add_signal(self, expr_str, label=None, period=None):
    #     r"""
    #     Adds a new signal to the internal dictionary for later plotting.
    #     - The expression is parsed symbolically using SymPy.
    #     - If other signals are referenced, their definitions are recursively substituted.
    #     - If `period` is set, the signal will be expanded as a sum of time-shifted versions over the full horizontal range.

    #     Args:
    #         expr_str (str): Signal definition in the form "name(var) = expression", e.g. "x(t) = rect(t) + delta(t-1)". The expression may include previously defined signals.
    #         label (str, optional): Custom label for the vertical axis when plotting this signal.
    #         period (float, optional): If provided, the signal will be treated as periodic with this period.

    #     Examples:
    #         >>> sp = SignalPlotter(horiz_range=(-1, 1), fraction_ticks=True, figsize=(12, 4))
    #         >>> sp.add_signal("x1(t) = tri(t)")
    #         >>> sp.add_signal("x2(t) = delta(t)", period=0.2)
    #         >>> sp.add_signal("x3(t) = x1(t) * (1 + x2(t))", label="x_3(t)")
    #         >>> sp.plot("x3")
    #         >>> sp.add_signal("x4(t) = exp((-2+j*8*pi)*t)*u(t)")
    #         >>> sp.add_signal("x5(t) = re(x4(t))", label="\Re\{x_4(t)\}")
    #         >>> sp.plot('x5')
    #     """
    #     m = re.match(r"(?P<name>\w+)\s*\[\s*(?P<var>\w+)\s*\]\s*=\s*(?P<expr>.+)", expr_str)
    #     if not m:
    #         raise ValueError("Expression must be in form: name[n] = ...")

    #     name, var, expr_body = m.group('name'), m.group('var'), m.group('expr')

    #     # Paso 1: preprocesar el cuerpo reemplazando [] por ()  → convierte s[n-2] → s(n-2)
    #     expr_body_preprocessed = re.sub(r"(\w+)\s*\[\s*(.+?)\s*\]", r"\1(\2)", expr_body)

    #     if var not in self.var_symbols:
    #         self.var_symbols[var] = sp.Symbol(var)
    #     var_sym = self.var_symbols[var]

    #     local_dict = self._get_local_dict()
    #     for other_name in self.signal_defs:
    #         local_dict[other_name] = sp.Function(other_name)

    #     transformations = standard_transformations + (implicit_multiplication_application,)
    #     parsed_expr = parse_expr(expr_body_preprocessed, local_dict=local_dict, transformations=transformations)

    #     # Perform recursive substitution of previously defined signals.
    #     # This enables expressions to reference earlier signals (e.g., x[n] = z[n-2] + delta(n))
    #     # by replacing every function call (e.g., z(n-2)) with the corresponding shifted expression.
    #     for other_name, other_expr in self.signal_defs.items():
    #         f = sp.Function(other_name)
    #         matches = parsed_expr.find(f)
    #         for call in matches:
    #             if isinstance(call, sp.Function):
    #                 arg = call.args[0]
    #                 replaced = other_expr.subs(var_sym, arg)
    #                 parsed_expr = parsed_expr.subs(call, replaced)

    #     self.signal_defs[name] = parsed_expr
    #     self.var_symbols[name] = var_sym

    #     if label is not None:
    #         if not hasattr(self, 'custom_labels'):
    #             self.custom_labels = {}
    #         self.custom_labels[name] = label

    #     if period is not None:
    #         if not hasattr(self, 'signal_periods'):
    #             self.signal_periods = {}
    #         self.signal_periods[name] = period

    #         # Expand signal as sum of shifts within range
    #         horiz_min, horiz_max = self.horiz_range
    #         num_periods = int(np.ceil((horiz_max - horiz_min) / period))
    #         k_range = range(-num_periods - 2, num_periods + 3)  # márgenes extra

    #         # Expanded as sum of shifted expressions (in SymPy)
    #         expanded_expr = sum(parsed_expr.subs(var_sym, var_sym - period * k) for k in k_range)

    #         self.signal_defs[name] = expanded_expr
    #     else:
    #         self.signal_defs[name] = parsed_expr

    def add_signal(self, expr_str, label=None, period=None):
        r"""
        Adds a new signal definition to the dictionary.

        Args:
            expr_str (str): Expression like "x[n] = delta(n) + u(n-2)"
            label (str, optional): Custom label for plotting
            period (float, optional): Periodicity (optional)

        Examples:
            >>> dsp.add_signal("x[n] = delta(n) + delta(n-2) + u(n-3)")
        """
        m = re.match(r"(?P<name>\w+)\s*\[\s*(?P<var>\w+)\s*\]\s*=\s*(?P<expr>.+)", expr_str)
        if not m:
            raise ValueError("Expression must be in form: name[n] = ...")

        name, var, expr_body = m.group('name'), m.group('var'), m.group('expr')

        if var not in self.var_symbols:
            self.var_symbols[var] = sp.Symbol(var)
        var_sym = self.var_symbols[var]

        # Initial expression update (this includes parsing, substituting previous signals, and lambdify)
        self._update_expression_and_func(name, expr_body, var_sym)
        self.var_symbols[name] = var_sym

        # Store custom label
        if label is not None:
            self.custom_labels[name] = label

        # If periodic, expand as periodic sum
        if period is not None:
            self.signal_periods[name] = period

            horiz_min, horiz_max = self.horiz_range
            num_periods = int(np.ceil((horiz_max - horiz_min) / period))
            k_range = range(-num_periods - 2, num_periods + 3)

            expanded_expr = sum(self.signal_defs[name].subs(var_sym, var_sym - period * k) for k in k_range)

            # Update expression again after expansion
            self.signal_defs[name] = expanded_expr
            self._update_expression_and_func(name, str(expanded_expr), var_sym)

    def draw_function(self, marker='o', stem_color=None, marker_size=6, line_width=1.8):
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
        print('plotting')
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
        self.fig.tight_layout()

    def draw_ticks(self, xticks=None, yticks=None, xtick_labels='auto', ytick_labels='auto', tick_size_px=None, tol=1e-12):
        """
        Draws tick marks and labels for discrete signals:
        - X-axis: optional labels at non-zero positions or user-defined.
        - Y-axis: tick marks and labels.

        Args:
            xticks (list or 'auto' or None): X-axis tick positions.
                - 'auto': ticks on non-zero values.
                - list: manually specified integer positions.
                - None: no xticks.
            yticks (list or 'auto' or None): Y-axis tick positions.
                - 'auto': automatic range.
                - list: manually specified values.
                - None: no yticks.
            xtick_labels (list or 'auto' or None): Custom labels for xticks.
            ytick_labels (list or 'auto' or None): Custom labels for yticks.
            tick_size_px (int, optional): Length of tick marks for yticks in pixels.
            tol (float): Tolerance for detecting zero values.
        """
        def get_nonzero_positions(name=None, n_min=None, n_max=None, tol=1e-12):
            if name is None:
                if not self.current_name:
                    raise ValueError("No signal selected.")
                name = self.current_name
            if name not in self.signal_defs:
                raise ValueError(f"Signal '{name}' is not defined.")
            if not hasattr(self, 'n_vals') or not hasattr(self, 'y_vals') or self.func_name != name:
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

        def validate_tick_list(ticks, axis):
            if ticks is None:
                return []
            if isinstance(ticks, str) and ticks == 'auto':
                return 'auto'
            arr = np.array(ticks)
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
        effective_xticks = xticks if xticks is not None else getattr(self, 'init_xticks_arg', 'auto')
        effective_yticks = yticks if yticks is not None else getattr(self, 'init_yticks_arg', 'auto')

        tick_px = tick_size_px if tick_size_px is not None else self.tick_size_px
        dx, dy = px_to_data_length(tick_px)

        # -------- Y-AXIS --------
        if isinstance(effective_yticks, str) and effective_yticks == 'auto':
            y0 = np.floor(self.y_min)
            y1 = np.ceil(self.y_max)
            raw_yticks = list(np.linspace(y0, y1, 3))
        elif effective_yticks is None:
            raw_yticks = []
        else:
            raw_yticks = validate_tick_list(effective_yticks, 'y')

        ylabels = []
        if isinstance(ytick_labels, list):
            if len(ytick_labels) != len(raw_yticks):
                raise ValueError("ytick_labels and yticks must have the same length")
            ylabels = ytick_labels
        else:
            ylabels = [f"{y:.3g}" for y in raw_yticks]

        ylim = self.ax.get_ylim()
        for y, lbl in zip(raw_yticks, ylabels):
            if ylim[0] <= y <= ylim[1]:
                self.ax.plot([0 - dx/2, 0 + dx/2], [y, y], transform=self.ax.transData,
                             color='black', linewidth=1.2, clip_on=False)
                offset = (-4, -16) if abs(y) < tol else (-4, 0)
                self.ax.annotate(rf'${lbl}$', xy=(0, y), xycoords='data',
                                 textcoords='offset points', xytext=offset,
                                 ha='right', va='center', fontsize=12, zorder=10,
                                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                           edgecolor='none', alpha=self.alpha))

        # -------- X-AXIS --------
        if isinstance(effective_xticks, str) and effective_xticks == 'auto':
            raw_xticks = get_nonzero_positions(name=self.current_name, tol=tol)
        elif effective_xticks is None:
            raw_xticks = []
        else:
            raw_xticks = validate_tick_list(effective_xticks, 'x')

        xlabels = []
        if isinstance(xtick_labels, list):
            if len(xtick_labels) != len(raw_xticks):
                raise ValueError("xtick_labels and xticks must have the same length")
            xlabels = xtick_labels
        else:
            xlabels = [f"{int(n)}" for n in raw_xticks]

        xlim = self.ax.get_xlim()
        for n, lbl in zip(raw_xticks, xlabels):
            if xlim[0] <= n <= xlim[1]:
                idx = np.where(self.n_vals == n)[0]
                y_val = self.y_vals[idx[0]] if idx.size > 0 else 0
                offset_y = 8 if y_val < 0 else -8
                self.ax.annotate(rf'${lbl}$', xy=(n, 0), xycoords='data',
                                 textcoords='offset points', xytext=(0, offset_y),
                                 ha='center', va='bottom' if offset_y > 0 else 'top',
                                 fontsize=12, zorder=10,
                                 bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                                           edgecolor='none', alpha=self.alpha))
    def draw_labels(self):
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
        self.ax.text(x_pos, y_pos, rf'${self.xlabel}$', fontsize=16, ha='right', va='bottom')

        # Y-axis label: slightly below the top y limit (still inside the figure)
        x_pos = 0.01 * (x_lim[1] - x_lim[0])
        y_pos = y_lim[1] - 0.1 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, rf'${self.ylabel}$', fontsize=16, ha='left', va='bottom', rotation=0)

    def _update_expression_and_func(self, name, expr_body, var_sym):
        """
        Parses and updates the symbolic expression and numeric function
        for the given signal name.

        - Converts array-style access (e.g. z[n-2]) into functional form (z(n-2))
        - Substitutes previously defined signals recursively
        - Lambdifies the final expression for numerical evaluation
        """

        # Step 1: preprocess to replace [] by () for function-like calls
        expr_body_preprocessed = re.sub(r"(\w+)\s*\[\s*(.+?)\s*\]", r"\1(\2)", expr_body)
        print('1:', expr_body_preprocessed)
        

        # Build local dictionary with primitives
        local_dict = self._get_local_dict()
        for other_name in self.signal_defs:
            local_dict[other_name] = sp.Function(other_name)

        # Parse expression
        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed_expr = parse_expr(expr_body_preprocessed, local_dict=local_dict, transformations=transformations)
        print('2:', parsed_expr)

        # Robust substitution of delta(expr)
        # delta_func = sp.Function('delta')
        # for expr_delta in parsed_expr.atoms(delta_func):
        #     argumento = expr_delta.args[0]
        #     parsed_expr = parsed_expr.subs(expr_delta, sp.KroneckerDelta(argumento, 0))
        # print('3:', parsed_expr)

        # # Después, reordenar manualmente los KroneckerDelta si sympy ha cambiado el orden:
        # parsed_expr = parsed_expr.replace(
        #     lambda expr: isinstance(expr, sp.KroneckerDelta) and expr.args[0] == 0,
        #     lambda expr: sp.KroneckerDelta(expr.args[1], 0)
        # )
        # print('4:', parsed_expr)


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

        # Create numerical function
        self.funcs[name] = sp.lambdify(
            var_sym, parsed_expr, 
            modules=["numpy", self._get_numeric_dict()]
            )


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

    # def plot(self, name=None):
    #     """
    #     Plots the discrete signal specified by name.

    #     This method:
    #     - Looks up the signal from internal definitions.
    #     - Evaluates it over the specified integer range.
    #     - Sets up the axes, ticks, and renders the stem plot.
        
    #     Args:
    #         name (str, optional): Name of the signal to plot.
    #                             If None, uses the last-added signal.

    #     Raises:
    #         ValueError: If the signal name is not defined.

    #     Examples:
    #         >>> dsp = DiscreteSignalPlotter("x[n]=delta(n)+u(n-1)", n_range=(-5, 5))
    #         >>> dsp.plot("x")
    #     """
    #     # If no name provided, use the most recently added signal
    #     if name is None:
    #         if not self.signal_defs:
    #             raise ValueError("No signals defined to plot.")
    #         name = list(self.signal_defs.keys())[-1]

    #     if name not in self.signal_defs:
    #         raise ValueError(f"Signal '{name}' is not defined.")
        
    #     expr = self.signal_defs[name]
    #     var = self.var_symbols[name]
    #     self.xlabel = str(var)
    #     self.ylabel = f"{name}[{self.xlabel}]"
    #     if name in self.custom_labels:
    #         self.ylabel = self.custom_labels[name]

    #     # Create evaluation grid
    #     self.n_vals = np.arange(self.n_range[0], self.n_range[1]+1)
        
    #     # Lambdify expression
    #     func = sp.lambdify(var, expr, modules=["numpy", self.local_dict])
        
    #     # Evaluate
    #     y_raw = [func(int(n)) for n in self.n_vals]
    #     self.y_vals = np.array(y_raw, dtype=float)

    #     # Create figure and compute y-range
    #     self.fig, self.ax = plt.subplots(figsize=self.figsize)
    #     self._prepare_plot(self.y_vals)

    #     # Draw all components of the plot
    #     self.setup_axes()
    #     # Draw stem plot
    #     markerline, stemlines, baseline = self.ax.stem(
    #         self.n_vals, self.y_vals)
    #     markerline.set_color(self.color)
    #     stemlines.set_color(self.color)
    #     markerline.set_marker('o')
    #     plt.setp(markerline, markersize=6)
    #     plt.setp(stemlines, linewidth=1.8)

    #     self.draw_ticks()
    #     self.draw_labels()
    #     self.show()

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
        self.func_name = name

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
        self.setup_axes()
        self.draw_function()

        self.draw_ticks()
        self.draw_labels()
        self.show()