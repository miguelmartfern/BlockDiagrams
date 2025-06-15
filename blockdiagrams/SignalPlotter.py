# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Miguel Á. Martín (miguelmartfern@github)

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
    def __init__(
        self,
        expr_str=None, 
        horiz_range=(-5, 5),
        vert_range=None,
        periodo=None,
        num_points=1000,
        figsize=(8, 3), 
        tick_size_px=5,
        xticks='auto',
        yticks='auto',
        xtick_labels=None,
        ytick_labels=None,
        pi_mode=False,
        fraction_ticks=False,
        save_path=None, 
        show_plot=True,
        color='black', 
        alpha=0.5, 
        xticks_delta=None,
        yticks_delta=None
    ):
        self.signal_defs = {}
        self.var_symbols = {}
        self.current_name = None
        self.horiz_range = horiz_range
        self.vert_range = vert_range
        self.num_points = num_points
        self.figsize = figsize
        self.tick_size_px = tick_size_px
        self.color = color
        self.alpha = alpha
        self.periodo = periodo
        self.save_path = save_path
        self.show_plot = show_plot

        self.fraction_ticks = fraction_ticks

        # Guardar argumento original para diferenciar None / [] / 'auto'
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

        self.expr_str_pending = expr_str  # Se inicializa más adelante

        self.transformations = (standard_transformations + (implicit_multiplication_application,))


    def _get_local_dict(self):
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
            're':           sp.re,            # Parte real
            'im':           sp.im,            # Parte imaginaria
            'conj':         sp.conjugate,    # Conjugado complejo
            'abs':          lambda x: np.abs(x),           # Magnitud / módulo
            'arg':          sp.arg,           # Argumento / fase
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

    def _initialize_expression(self, expr_str):
        m = re.match(r"^(?P<fn>[^\W\d_]+)\((?P<vr>[^)]+)\)\s*=\s*(?P<ex>.+)$", expr_str)
        if m:
            self.func_name = m.group('fn')
            var_name = m.group('vr')
            expr_body = m.group('ex')
        else:
            self.func_name = 'x'
            var_name = 't'
            expr_body = expr_str

        replacements = {'\\omega': 'ω', '\\tau': 'τ'}
        for latex_var, unicode_var in replacements.items():
            var_name = var_name.replace(latex_var, unicode_var)
            expr_body = expr_body.replace(latex_var, unicode_var)

        self.expr_str = expr_body
        self.var = sp.Symbol(var_name)
        self.xlabel = var_name
        self.ylabel = self.func_name + '(' + var_name + ')'

        self.local_dict = self._get_local_dict()

        transformations = standard_transformations + (implicit_multiplication_application,)
        self.expr = parse_expr(expr_body, local_dict=self.local_dict, transformations=transformations)

        self.expr_cont = self._remove_dirac_terms()
        self.impulse_locs, self.impulse_areas = self._extract_impulses()

        t0, t1 = self.horiz_range
        self.t_vals = np.linspace(t0, t1, self.num_points)
        if self.periodo is not None:
            T = self.periodo
            self.t_vals = ((self.t_vals + T/2) % T) - T/2
            self.t_vals.sort()

        self.func = sp.lambdify(self.var, self.expr_cont, modules=["numpy", self.local_dict])
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._prepare_plot()

    def add_signal(self, expr_str, label=None, period=None):
        m = re.match(r"^(?P<fn>\w+)\((?P<vr>\w+)\)\s*=\s*(?P<ex>.+)$", expr_str)

        replacements = {'\\omega': 'ω', '\\tau': 'τ'}
        for latex_var, unicode_var in replacements.items():
            expr_str = expr_str.replace(latex_var, unicode_var)
        m = re.match(r"^(?P<fn>\w+)\((?P<vr>\w+)\)\s*=\s*(?P<ex>.+)$", expr_str)

        name = m.group('fn')
        var = m.group('vr')
        body = m.group('ex')

        if var not in self.var_symbols:
            self.var_symbols[var] = sp.Symbol(var)
        var_sym = self.var_symbols[var]

        local_dict = self._get_local_dict()
        for other_name in self.signal_defs:
            local_dict[other_name] = sp.Function(other_name)

        transformations = standard_transformations + (implicit_multiplication_application,)
        parsed_expr = parse_expr(body, local_dict=local_dict, transformations=transformations)

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

            # Expandir la señal como suma de traslaciones dentro del rango
            horiz_min, horiz_max = self.horiz_range
            num_periods = int(np.ceil((horiz_max - horiz_min) / period))
            k_range = range(-num_periods - 2, num_periods + 3)  # márgenes extra

            # expandido como suma de expresiones desplazadas (en SymPy)
            expanded_expr = sum(parsed_expr.subs(var_sym, var_sym - period * k) for k in k_range)

            self.signal_defs[name] = expanded_expr
        else:
            self.signal_defs[name] = parsed_expr


    def _prepare_plot(self):
        try:
            # Evaluar señal continua
            y_vals = self.func(self.t_vals)
            y_vals = np.array(y_vals, dtype=np.float64)
            y_vals = y_vals[np.isfinite(y_vals)]

            if y_vals.size > 0:
                cont_min = np.min(y_vals)
                cont_max = np.max(y_vals)
            else:
                cont_min = 0.0
                cont_max = 0.0
           
            # Impulsos visibles en el rango horizontal
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


            # Ajuste si el rango es demasiado pequeño
            if abs(overall_max - overall_min) < 1e-2:
                overall_min -= 1.0
                overall_max += 1.0

            # Aplicar rango vertical explícito si se indicó
            if self.vert_range:
                self.y_min, self.y_max = self.vert_range
            else:
                self.y_min, self.y_max = overall_min, overall_max

        except Exception:
            self.y_min, self.y_max = -1, 1



    def _eval_func_array(self, t):
        y = self.func(t)
        return np.full_like(t, y, dtype=float) if np.isscalar(y) else np.array(y, dtype=float)

    def _extract_impulses(self):
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
        return self.expr.replace(lambda expr: expr.has(sp.DiracDelta), lambda _: 0)

    def draw_function(self, horiz_range=None):
        """
        Dibuja la señal continua sin asumir periodicidad.
        Añade puntos suspensivos si hay valores significativos fuera del rango,
        o siempre si la señal fue definida como periódica.
        """

        if horiz_range is None:
            horiz_range = self.horiz_range

        t0, t1 = horiz_range
        t_plot = self.t_vals
        y_plot = self._eval_func_array(t_plot)

        # Asegurar arrays y formato
        t_plot = np.array(t_plot)
        y_plot = np.array(y_plot)
        if y_plot.ndim == 0:
            y_plot = np.full_like(t_plot, y_plot, dtype=float)

        # Dibujar curva
        self.ax.plot(t_plot, y_plot, color=self.color, linewidth=2.5, zorder=5)

        # Decidir puntos suspensivos
        delta = (t1 - t0) * 0.05
        tol = 1e-3
        span = t1 - t0
        draw_left = draw_right = False

        # Mostrar siempre si la señal es periódica
        if hasattr(self, 'signal_periods') and self.current_name in self.signal_periods:
            draw_left = draw_right = True
        else:
            N = max(10, int(0.05 * self.num_points))
            xs_left = np.linspace(t0 - 0.05 * span, t0, N)
            ys_left = np.abs(self._eval_func_array(xs_left))
            if np.trapz(ys_left, xs_left) > tol:
                draw_left = True

            xs_right = np.linspace(t1, t1 + 0.05 * span, N)
            ys_right = np.abs(self._eval_func_array(xs_right))
            if np.trapz(ys_right, xs_right) > tol:
                draw_right = True

        # Dibujar puntos suspensivos
        y_mid = (self.y_min + 2 * self.y_max) / 3
        if draw_left:
            self.ax.text(t0 - delta, y_mid, r'$\cdots$', ha='left', va='center',
                        color=self.color, fontsize=14, zorder=10)
        if draw_right:
            self.ax.text(t1 + delta, y_mid, r'$\cdots$', ha='right', va='center',
                        color=self.color, fontsize=14, zorder=10)


    def draw_impulses(self):
        """
        Dibuja los impulsos (Dirac) en las ubicaciones extraídas de la expresión.
        No asume periodicidad.
        """
        t_min, t_max = self.horiz_range
        for t0, amp in zip(self.impulse_locs, self.impulse_areas):
            if t_min <= t0 <= t_max:
                self._draw_single_impulse(t0, amp)

    def _draw_single_impulse(self, t0, amp):
        """
        Dibuja un único impulso en t0 con amplitud amp.
        Si t0 está cerca de 0, desplaza la etiqueta ligeramente a la izquierda
        para que no quede sobre el eje vertical.
        """
        # Flecha desde (t0,0) hasta (t0, amp)
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

        # Calcular desplazamiento horizontal de la etiqueta si t0 ≈ 0
        x_min, x_max = self.ax.get_xlim()
        x_range = x_max - x_min
        # Umbral para considerar que t0 es “casi” cero:
        tol = 1e-6 * max(1.0, abs(x_range))
        if abs(t0) < tol:
            # Desplazar la etiqueta un 2% del rango horizontal hacia la izquierda
            x_offset = -0.01 * x_range
            ha = 'right'
        else:
            x_offset = 0.0
            ha = 'center'

        # Alinear etiqueta más arriba de la curva continua si es necesario
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
        for spine in self.ax.spines.values():
            spine.set_color('none')

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        if horiz_range is None:
            horiz_range = self.horiz_range

        # Rango horizontal
        x0, x1 = horiz_range
        x_range = x1 - x0
        x_margin = 0.2 * x_range  # puedes ajustar si quieres cambiar la extensión horizontal

        # Rango vertical original calculado en _prepare_plot: self.y_min, self.y_max
        y_min, y_max = self.y_min, self.y_max
        y_range = y_max - y_min

        # Extensión de un 10% adicional por cada lado => total 1.2×
        if y_range <= 0:
            # En caso degenerate, mantener un rango mínimo
            y_margin = 1.0
        else:
            y_margin = 0.3 * y_range

        self.ax.set_xlim(horiz_range[0] - x_margin, horiz_range[1] + x_margin)
        self.ax.set_ylim(self.y_min - y_margin, self.y_max + 1.6 * y_margin)

        self.ax.annotate('', xy=(self.ax.get_xlim()[1], 0), xytext=(self.ax.get_xlim()[0], 0),
                         arrowprops=dict(arrowstyle='-|>', linewidth=1.5, color='black',
                                         mutation_scale=16, mutation_aspect=0.8, fc='black'))

        self.ax.annotate('', xy=(0, self.ax.get_ylim()[1]), xytext=(0, self.ax.get_ylim()[0]),
                         arrowprops=dict(arrowstyle='-|>', linewidth=1.5, color='black',
                                         mutation_scale=12, mutation_aspect=2, fc='black'))
        
        self.fig.tight_layout()

    def draw_ticks(self,
                tick_size_px=None,
                xticks=None,
                yticks=None,
                xtick_labels='auto',
                ytick_labels='auto'):
        """
        Draws tick marks and labels on the x and y axes, according to the current configuration.

        Supported behaviors:

        X AXIS (xticks):

        - `xticks=None`:
            Only the positions of delta impulses (if any) are shown.

        - `xticks='auto'`:
            Ticks are generated automatically:
                - If `xticks` were set during initialization, they are used.
                - Otherwise, ticks are placed evenly across the horizontal range.
                If `xticks_delta` is provided, ticks are placed every that interval from the origin (both positive and negative).
                Additional ticks at delta locations are added if not already present (according to tolerance `tol`).

        - `xticks=[... list of positions ...]`:
            These positions are used as manual ticks. If `xtick_labels` is provided, those are used as labels;
            otherwise, labels are generated automatically.

        X-axis tick labels:
        - If `pi_mode=True`, ticks are shown as multiples of π (e.g., `-\\pi/2`, `3\\pi/4`, `0`, etc.).
        - If `fraction_ticks=True` and `pi_mode=False`, ticks are formatted as rational fractions (e.g., `1/2`, `-3/4`).
        - If both are active, labels appear as rational multiples of π (e.g., `-\\pi/2`, `3\\pi/4`).
        - If neither is active, simple decimal labels are used (`1.5`, `-2`, etc.).
        - Manual labels (via `xtick_labels`) always take precedence.

        Warning:
        - If `xticks_delta` is provided and `xticks` is not `'auto'`, a `UserWarning` is issued to indicate it will be ignored.

        Y AXIS (yticks):

        - `yticks=None`:
            No ticks are shown on the y-axis.

        - `yticks='auto'`:
            Ticks are generated automatically:
                - If `yticks` were set during initialization, they are used.
                - Otherwise, 3 evenly spaced ticks are placed within the vertical bounds.
                If `yticks_delta` is provided, ticks are placed every that interval from the origin (positive and negative).

        - `yticks=[... list of positions ...]`:
            These positions are used as manual ticks. If `ytick_labels` is provided, those are used as labels;
            otherwise, labels are generated automatically.

        Y-axis tick labels:
        - If `fraction_ticks=True`, ticks are shown as rational fractions (`1/2`, `-3/4`, etc.).
        - If not, simple decimal labels are used (`0.2`, `1`, etc.).
        - Manual labels (via `ytick_labels`) always take precedence.

        Warning:
        - If `yticks_delta` is provided and `yticks` is not `'auto'`, a `UserWarning` is issued to indicate it will be ignored.

        Notes:
        - All automatically generated labels avoid near-duplicates using the `tol` parameter.
        - Returned ticks are sorted and cleaned of redundant values.
        """

        def unique_sorted(values, tol):
            unique = []
            for v in values:
                if not any(abs(v - u) <= tol for u in unique):
                    unique.append(v)
            return sorted(unique)

        def get_impulse_positions_and_areas(t_min, t_max, tol):
            impulse_positions = []
            impulse_positions_areas = []
            if self.impulse_locs:
                if self.periodo is None:
                    for base_loc, base_area in zip(self.impulse_locs, self.impulse_areas):
                        if t_min - tol <= base_loc <= t_max + tol:
                            impulse_positions.append(base_loc)
                            impulse_positions_areas.append(base_area)
                else:
                    T = self.periodo
                    for base_loc, base_area in zip(self.impulse_locs, self.impulse_areas):
                        k_min = int(np.floor((t_min - base_loc) / T))
                        k_max = int(np.ceil((t_max - base_loc) / T))
                        for k in range(k_min, k_max + 1):
                            t_k = base_loc + k * T
                            if t_min - tol <= t_k <= t_max + tol:
                                impulse_positions.append(t_k)
                                impulse_positions_areas.append(base_area)
            # eliminar duplicados
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

        def has_valid_ticks(ticks):
            if ticks is None:
                return False
            try:
                arr = np.array(ticks)
                return arr.ndim >= 1 and arr.size >= 1
            except Exception:
                return False

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
                            raise ValueError("xtick_labels y xticks de init deben tener la misma longitud")
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

                # Añadir impulsos
                for loc in impulse_positions:
                    if t_min - tol <= loc <= t_max + tol and not any(abs(loc - x0) <= tol for x0 in raw_xticks):
                        raw_xticks.append(loc)

            else:
                if xticks_delta is not None:
                    warnings.warn("xticks_delta será ignorado porque xticks no está en modo 'auto'.", stacklevel=2)
                raw_xticks = list(effective_xticks)
                if xtick_labels not in (None, 'auto'):
                    if len(raw_xticks) != len(xtick_labels):
                        raise ValueError("xtick_labels y xticks deben tener la misma longitud")
                    manual_xticks = list(raw_xticks)
                    manual_xlabels = list(xtick_labels)
                elif self.xtick_labels is not None:
                    if len(raw_xticks) != len(self.xtick_labels):
                        raise ValueError("xtick_labels y xticks deben tener la misma longitud (init)")
                    manual_xticks = list(raw_xticks)
                    manual_xlabels = list(self.xtick_labels)

                for loc in impulse_positions:
                    if t_min - tol <= loc <= t_max + tol and not any(abs(loc - x0) <= tol for x0 in raw_xticks):
                        raw_xticks.append(loc)

            raw_xticks = unique_sorted(raw_xticks, tol)

            # Generar etiquetas
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
                            raise ValueError("ytick_labels y yticks de init deben tener la misma longitud")
                        manual_yticks = list(self.yticks)
                        manual_ylabels = list(self.ytick_labels)
                    if ydelta is not None:
                        warnings.warn("yticks_delta ignorado porque ya se especificaron yticks en el init")
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
                    warnings.warn("yticks_delta ignorado porque yticks no está en modo 'auto'")

                if ytick_labels not in (None, 'auto'):
                    if len(raw_yticks) != len(ytick_labels):
                        raise ValueError("ytick_labels y yticks deben tener la misma longitud")
                    manual_yticks = list(raw_yticks)
                    manual_ylabels = list(ytick_labels)
                elif self.ytick_labels is not None:
                    if len(raw_yticks) != len(self.ytick_labels):
                        raise ValueError("ytick_labels y yticks deben tener la misma longitud (init)")
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

        # -- Código principal draw_ticks --

        # 0. Determinar effective ticks
        effective_xticks = xticks if xticks is not None else getattr(self, 'init_xticks_arg', None)
        effective_yticks = yticks if yticks is not None else getattr(self, 'init_yticks_arg', None)

        # 1. Tamaño del tick en px
        tick_px = tick_size_px if tick_size_px is not None else self.tick_size_px

        # 2. Rango y tolerancia
        t_min, t_max = self.horiz_range
        tol = 1e-8 * max(1.0, abs(t_max - t_min))

        # 3. Impulsos periódicos
        impulse_positions, impulse_positions_areas = get_impulse_positions_and_areas(t_min, t_max, tol)

        # 4 y 5. Procesar ticks X + etiquetas
        raw_xticks, xlabels = generate_xticks(effective_xticks, impulse_positions, tol, t_min, t_max)

        # 6 y 7. Procesar ticks Y + etiquetas
        raw_yticks, ylabels = generate_yticks(effective_yticks, tol)

        # 8. Filtrar ytick=0 si xtick=0 existe
        raw_yticks, ylabels = filter_yticks(raw_yticks, ylabels, raw_xticks, tol)

        # 9. Conversión px -> datos
        dx, dy = px_to_data_length(tick_px)

        # 10 Dibujar marcas xticks + etiquetas
        draw_xticks(raw_xticks, xlabels, impulse_positions, impulse_positions_areas, dx, dy, tol)

        # 11. Dibujar marcas yticks + etiquetas
        draw_yticks(raw_yticks, ylabels, dx, dy)


    def draw_labels(self):
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()

        # Etiqueta eje X: un poco a la derecha del límite derecho del eje x
        x_pos = x_lim[1] - 0.01 * (x_lim[1] - x_lim[0])
        y_pos = 0.02 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, rf'${self.xlabel}$', fontsize=16, ha='right', va='bottom')

        # Etiqueta eje Y: un poco por debajo del límite superior del eje y (pero dentro)
        x_pos = 0.01 * (x_lim[1] - x_lim[0])
        y_pos = y_lim[1] - 0.1 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, rf'${self.ylabel}$', fontsize=16, ha='left', va='bottom', rotation=0)

    def show(self):
        self.ax.grid(False)
        plt.tight_layout()
        if self.save_path:
            self.fig.savefig(self.save_path, dpi=300, bbox_inches='tight')
        if self.show_plot:
            plt.show()
        plt.close(self.fig)


    def plot(self, name=None):
        # Procesar expr_str_pending si existe, es una cadena y aún no se ha inicializado
        if (hasattr(self, 'expr_str_pending') and 
            self.expr_str_pending is not None and 
            isinstance(self.expr_str_pending, str) and 
            not getattr(self, '_initialized_from_expr', False)):
            expr_str = self.expr_str_pending
            self._initialized_from_expr = True
            self.add_signal(expr_str, period=self.periodo)
            name = list(self.signal_defs.keys())[-1]

        if name:
            if name not in self.signal_defs:
                raise ValueError(f"Señal '{name}' no definida")
            self.current_name = name
            self.func_name = name
            self.expr = self.signal_defs[name]

            # Determinar la variable independiente
            free_vars = list(self.expr.free_symbols)
            if free_vars:
                self.var = free_vars[0]
            elif name in self.var_symbols:
                self.var = self.var_symbols[name]
            else:
                raise ValueError(f"No se pudo determinar la variable independiente para la señal '{name}'")

            # Asignar etiquetas para los ejes
            self.xlabel = str(self.var)
            self.ylabel = f"{self.func_name}({self.xlabel})"
            if hasattr(self, 'custom_labels') and self.func_name in self.custom_labels:
                self.ylabel = self.custom_labels[self.func_name]

            # Preparar señal continua e impulsos
            self.expr_cont = self._remove_dirac_terms()
            self.impulse_locs, self.impulse_areas = self._extract_impulses()
            self.var = next(iter(self.var_symbols.values()))
            self.func = sp.lambdify(self.var, self.expr_cont, modules=["numpy", self._get_local_dict()])
            self.t_vals = np.linspace(*self.horiz_range, self.num_points)
            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self._prepare_plot()

        self.setup_axes()
        self.draw_function()
        self.draw_impulses()
        self.ax.relim()
        self.ax.autoscale_view()

        self.draw_ticks()
        self.draw_labels()
        self.show()

    # Definir las transformaciones a nivel global para uso posterior
    def _get_transformations(self):
        return self.transformations

    def _setup_figure(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._prepare_plot()
        self.fig.subplots_adjust(right=0.9, top=0.85, bottom=0.15)

    def plot_convolution_view(self, expr_str, t_val, label=None, tau=None, t=None):
        import re
        local_dict = self._get_local_dict()

        # Definir variables simbólicas con nombres personalizados o por defecto
        if tau is None:
            tau = local_dict.get('tau')
        elif isinstance(tau, str):
            tau = sp.Symbol(tau)
        if t is None:
            t = local_dict.get('t')
        elif isinstance(t, str):
            t = sp.Symbol(t)

        local_dict.update({'tau': tau, 't': t, str(tau): tau, str(t): t})

        # Identificar el nombre de la señal
        if "(" in expr_str:
            name = expr_str.split("(")[0].strip()
            if name not in self.signal_defs:
                raise ValueError(f"La señal '{name}' no está definida.")
            expr_base = self.signal_defs[name]
            var_base = self.var_symbols.get(name, t)
        else:
            raise ValueError("Expresión inválida: se esperaba algo como 'x(t-tau)'.")

        parsed_expr = parse_expr(expr_str.replace(name, "", 1), local_dict)
        expr = expr_base.subs(var_base, parsed_expr)

        # Analizar forma de la variable: t - tau o t + tau
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
                    # Caso t - tau => invertir ejes y etiquetas
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
                    # Caso t + tau => solo desplazamiento
                    if xticks == 'auto':
                        xticks_custom = [t_val]
                        xtick_labels_custom = [sp.latex(t)]
                    elif isinstance(xticks, (list, tuple)):
                        xticks_custom = [v - t_val for v in xticks]
                        xtick_labels_custom = [
                            f"-{sp.latex(t)}" if v == 0 else f"-{sp.latex(t)}{'+' if v > 0 else '-'}{abs(v)}"
                            for v in xticks
                        ]
                    horiz_range = (np.array(self.horiz_range) - t_val).tolist()

        expr_evaluated = expr.subs(t, t_val)

        # Preparar etiquetas
        self.expr = expr_evaluated
        self.var = tau
        self.xlabel = sp.latex(tau)
        tau_str = sp.latex(tau)
        t_str = sp.latex(t)
        if label:
            self.ylabel = label
        else:
            self.ylabel = expr_str.replace("tau", tau_str).replace("t", t_str)

        # Dibujar
        self.func = sp.lambdify(self.var, self.expr, modules=["numpy"])
        self.t_vals = np.linspace(*horiz_range, self.num_points)

        self.impulse_locs, self.impulse_areas = self._extract_impulses()
        self._setup_figure()
        # self._prepare_plot()
        self.setup_axes(horiz_range)
        self.draw_function(horiz_range)
        self.draw_impulses()
        self.draw_ticks(xticks=xticks_custom, xtick_labels=xtick_labels_custom)
        self.draw_labels()
        self.show()


    def plot_convolution_steps(self, x_name, h_name, t_val, tau=None, t=None):
        local_dict = self._get_local_dict()

        # Usar símbolos de 'local_dict' si están definidos
        if tau is None:
            tau = local_dict.get('tau')
        elif isinstance(tau, str):
            tau = sp.Symbol(tau)

        if t is None:
            t = local_dict.get('t')
        elif isinstance(t, str):
            t = sp.Symbol(t)

        x_expr = self.signal_defs[x_name].subs(self.var_symbols[x_name], tau)
        h_expr = self.signal_defs[h_name].subs(self.var_symbols[h_name], tau)
        x_shift = x_expr.subs(tau, t - tau).subs(t, t_val)
        h_shift = h_expr.subs(tau, t - tau).subs(t, t_val)

        tau_str = sp.latex(tau)
        t_str = sp.latex(t)

        # xticks personalizados solo para x_shift y h_shift
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
                    label = fr"{t_str}{delta}"  # delta es negativo, así que el signo ya va incluido
                xtick_labels_shifted.append(label)
        else:
            xticks_shifted = None
            xtick_labels_shifted = None

        horiz_range = self.horiz_range
        # I get the inverted-shifted range for the horizontal axis (reverse order)
        horiz_range_shifted = t_val - np.array(horiz_range)[::-1]

        items = [
            (x_expr, fr"{x_name}({tau_str})", None, None, horiz_range),
            (h_expr, fr"{h_name}({tau_str})", None, None, horiz_range),
            (x_shift, fr"{x_name}({t_str}-{tau_str})", xticks_shifted, xtick_labels_shifted, horiz_range_shifted),
            (h_shift, fr"{h_name}({t_str}-{tau_str})", xticks_shifted, xtick_labels_shifted, horiz_range_shifted),
        ]

        for expr, label, xticks_custom, xtick_labels_custom, horiz_range_custom in items:
            self.expr = expr
            self.var = tau
            self.xlabel = fr"\{tau}"
            self.ylabel = label
            self.func = sp.lambdify(self.var, self.expr, modules=["numpy"])
            self.t_vals = np.linspace(*horiz_range_custom, self.num_points)

            self.impulse_locs, self.impulse_areas = self._extract_impulses()
            self._setup_figure()
            # self._prepare_plot()
            self.setup_axes(horiz_range_custom)
            self.draw_function(horiz_range_custom)
            self.draw_impulses()
            self.draw_ticks(xticks=xticks_custom, xtick_labels=xtick_labels_custom)
            self.draw_labels()
            self.show()





    def plot_convolution_result(self, x_name, h_name, num_points=None, show_expr=False):
        import numpy as np
        import sympy as sp
        from scipy import integrate

        t = sp.Symbol('t')
        tau = sp.Symbol('tau')

        if num_points is None:
            num_points = self.num_points

        # Obtener definiciones y variables
        x_expr = self.signal_defs[x_name]
        h_expr = self.signal_defs[h_name]
        var_x = self.var_symbols[x_name]
        var_h = self.var_symbols[h_name]

        x_tau_expr = x_expr.subs(var_x, tau)
        h_t_tau_expr = h_expr.subs(var_h, t - tau)

        local_dict = self._get_local_dict()
        t_vals = np.linspace(*self.horiz_range, num_points)
        y_vals = []

        # Caso 1: x contiene solo deltas => usar propiedad directa
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

            # Extraer impulsos del resultado
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

            self.impulse_locs = impulse_locs
            self.impulse_areas = impulse_areas
            self.func = None

        # Caso 2: h contiene solo deltas => usar propiedad directa
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

            # Extraer impulsos del resultado
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

            self.impulse_locs = impulse_locs
            self.impulse_areas = impulse_areas
            self.func = None

        # Caso general: integrar numéricamente
        else:
            x_func_tau = sp.lambdify(tau, x_tau_expr, modules=["numpy"])
            def h_func_tau_shifted(tau_val, t_val):
                h_t_tau = h_t_tau_expr.subs(t, t_val)
                h_func = sp.lambdify(tau, h_t_tau, modules=["numpy"])
                return h_func(tau_val)

            support_x = (self.horiz_range[0], self.horiz_range[1])
            support_h = (self.horiz_range[0], self.horiz_range[1])

            for t_val in t_vals:
                a = max(support_x[0], t_val - support_h[1])
                b = min(support_x[1], t_val - support_h[0])
                if a >= b:
                    y_vals.append(0)
                    continue
                integrand = lambda tau_val: x_func_tau(tau_val) * h_func_tau_shifted(tau_val, t_val)
                try:
                    val, _ = integrate.quad(integrand, a, b)
                except Exception:
                    val = 0
                y_vals.append(val)

            self.func = lambda t_: np.interp(t_, t_vals, y_vals)
            self.impulse_locs = []
            self.impulse_areas = []

        self.expr = None
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






# Asociar los métodos a la clase SignalPlotter
# setattr(SignalPlotter, "_setup_figure", _setup_figure)
# setattr(SignalPlotter, "_get_transformations", _get_transformations)
# etattr(SignalPlotter, "_get_signal_support", _get_signal_support)
# setattr(SignalPlotter, "plot_convolution_view", plot_convolution_view)
# setattr(SignalPlotter, "plot_convolution_steps", plot_convolution_steps)
# setattr(SignalPlotter, "plot_convolution_result", plot_convolution_result)



# Añadir métodos a la clase SignalPlotter
# setattr(SignalPlotter, "plot_convolution_view", plot_convolution_view)
# setattr(SignalPlotter, "plot_convolution_steps", plot_convolution_steps)
# setattr(SignalPlotter, "plot_convolution_result", plot_convolution_result)

        # ✅ Restaurar periodo original al final del todo
        # if name:
        #     self.periodo = prev_periodo

    # def animate_signal(self, interval=20, save_path=None):
    #     """
    #     Anima la señal mostrando cómo se dibuja progresivamente desde el inicio al final del rango horizontal.
    #     Parámetros:
    #     - interval: tiempo en ms entre frames
    #     - save_path: si se proporciona, guarda la animación en ese archivo (mp4 o gif)
    #     """
    #     # Preparar datos base para la animación (igual que en draw_function, con periodicidad)
    #     t0, t1 = self.horiz_range
    #     if self.periodo is None:
    #         t_plot = self.t_vals
    #         y_plot = self._eval_func_array(t_plot)
    #     else:
    #         T = self.periodo
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
