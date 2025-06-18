import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import re

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
            'delta':        lambda n: np.where(n == 0, 1, 0),
            'u':            lambda n: np.where(n >= 0, 1, 0),
            'rect':         lambda n: np.where(abs(n) <= 1, 1, 0),
            'tri':          lambda n: np.where(abs(n) <= 3, 1 - abs(n)/3, 0),
            'ramp':         lambda n: np.where(n >= 0, n, 0),
            'sinc':         lambda n: np.sinc(n)
            'KroneckerDelta': sp.KroneckerDelta,
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
            'n':            sp.Symbol('n', integer=True),
            'omega':        sp.Symbol('omega'),
            'Omega':        sp.Symbol('Omega'),
            'k':            sp.Symbol('k'),
            'm':            sp.Symbol('m'),
        }
        d.update(self.var_symbols)
        for name, expr in self.signal_defs.items():
            for var in self.var_symbols.values():
                d[f"{name}({var})"] = expr.subs(var, var)
        return d

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
        m = re.match(r"(?P<name>\w+)\s*\[\s*(?P<var>\w+)\s*\]\s*=\s*(?P<expr>.+)", expr_str)
        if not m:
            raise ValueError("Expression must be in form: name[n] = ...")

        name, var, expr_body = m.group('name'), m.group('var'), m.group('expr')

        # Paso 1: preprocesar el cuerpo reemplazando [] por ()  → convierte s[n-2] → s(n-2)
        expr_body_preprocessed = re.sub(r"(\w+)\s*\[\s*(.+?)\s*\]", r"\1(\2)", expr_body)

        if var not in self.var_symbols:
            self.var_symbols[var] = sp.Symbol(var)
        var_sym = self.var_symbols[var]

        local_dict = self._get_local_dict()
        for other_name in self.signal_defs:
            local_dict[other_name] = sp.Function(other_name)

        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed_expr = parse_expr(expr_body_preprocessed, local_dict=local_dict, transformations=transformations)

        # Perform recursive substitution of previously defined signals.
        # This enables expressions to reference earlier signals (e.g., x[n] = z[n-2] + delta(n))
        # by replacing every function call (e.g., z(n-2)) with the corresponding shifted expression.
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

        if label is not None:
            if not hasattr(self, 'custom_labels'):
                self.custom_labels = {}
            self.custom_labels[name] = label

        if period is not None:
            if not hasattr(self, 'signal_periods'):
                self.signal_periods = {}
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

        Falta

    def _evaluate_signal(self, n_vals):
        y = self.func(n_vals)
        return np.full_like(n_vals, y, dtype=float) if np.isscalar(y) else np.array(y, dtype=float)
