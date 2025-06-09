# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Miguel Á. Martín (miguelmartfern@github)

# SignalPlotter.py

import re
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
import matplotlib.animation as animation
from IPython.display import HTML

class SignalPlotter:
    def __init__(self, expr_str, horiz_range=(-5, 5), vert_range=None, num_points=1000,
                 figsize=(8, 3), tick_size_px=5,
                 xticks='auto', yticks='auto', xtick_labels=None, ytick_labels=None,
                 save_path=None, show_plot=True, color='black', alpha=0.5, periodo=None):
        # parse expression of form "f(v)=..." or default
        m = re.match(r"^(?P<fn>[^\W\d_]+)\((?P<vr>[^)]+)\)\s*=\s*(?P<ex>.+)$", expr_str)
        if m:
            self.func_name = m.group('fn')
            var_name = m.group('vr')
            expr_body = m.group('ex')
        else:
            self.func_name = 'x'
            var_name = 't'
            expr_body = expr_str

        # Transparence for labels
        self.alpha = alpha

        # Replace LaTeX-like variable names
        replacements = {'\\omega': 'ω', '\\tau': 'τ'}
        for latex_var, unicode_var in replacements.items():
            var_name = var_name.replace(latex_var, unicode_var)
            expr_body = expr_body.replace(latex_var, unicode_var)

        self.expr_str = expr_body
        self.var = sp.Symbol(var_name)

        # store label names
        self.xlabel = var_name
        self.ylabel = self.func_name + '(' + var_name + ')'

        self.horiz_range = horiz_range
        self.vert_range = vert_range
        self.num_points = num_points
        self.figsize = figsize
        self.tick_size_px = tick_size_px
        self.color = color
        self.periodo = periodo

        # Guardar argumento original para diferenciar None / [] / 'auto'
        self.init_xticks_arg = xticks
        self.init_yticks_arg = yticks

        # Luego convertir solo si es lista/tupla/array no vacío
        if isinstance(xticks, (list, tuple, np.ndarray)) and len(xticks) > 0:
            self.xticks = np.array(xticks)
        else:
            self.xticks = None
        if isinstance(yticks, (list, tuple, np.ndarray)) and len(yticks) > 0:
            self.yticks = np.array(yticks)
        else:
            self.yticks = None
        self.xtick_labels = xtick_labels
        self.ytick_labels = ytick_labels

        # validate labels
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

        self.save_path = save_path
        self.show_plot = show_plot

        # Local dictionary (for parsing functions)
        self.local_dict = self._get_local_dict()

        # Parse expression
        transformations = standard_transformations + (implicit_multiplication_application,)
        self.expr = parse_expr(expr_body, local_dict=self.local_dict, transformations=transformations)

        # Extract components
        self.expr_cont = self._remove_dirac_terms()
        self.impulse_locs, self.impulse_areas = self._extract_impulses()

        # Prepare t_vals with periodicity
        t0, t1 = self.horiz_range
        self.t_vals = np.linspace(t0, t1, self.num_points)
        if self.periodo is not None:
            T = self.periodo
            self.t_vals = ((self.t_vals + T/2) % T) - T/2
            self.t_vals.sort()

        # create numeric function
        self.func = sp.lambdify(self.var, self.expr_cont, modules=["numpy", self.local_dict])
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        self._prepare_plot()

    def _get_local_dict(self):
        return {
            'u':            sp.Heaviside,
            'rect':         lambda t: sp.Heaviside(t + 0.5) - sp.Heaviside(t - 0.5),
            'tri':          lambda t: (1 - abs(t)) * sp.Heaviside(1 - abs(t)),
            'sinc':         lambda t: sp.sin(sp.pi * t) / (sp.pi * t),
            'delta':        sp.DiracDelta,
            'DiracDelta':   sp.DiracDelta,
            'Heaviside':    sp.Heaviside,
            'pi':           sp.pi,
            self.var.name:  self.var,
            'sin':          sp.sin,
            'cos':          sp.cos,
            'exp':          sp.exp,
            'Piecewise':    sp.Piecewise,
            'pw':           sp.Piecewise
        }


    def _prepare_plot(self):
        # recompute y-range
        try:
            y_vals = self.func(self.t_vals)
            y_vals = np.array(y_vals, dtype=np.float64)
            y_vals = y_vals[np.isfinite(y_vals)]
            if self.vert_range:
                self.y_min, self.y_max = self.vert_range
            else:
                # Valores continuos
                if y_vals.size > 0:
                    cont_min = np.min(y_vals)
                    cont_max = np.max(y_vals)
                else:
                    cont_min = 0.0
                    cont_max = 0.0

                # Impulsos
                if self.impulse_areas:
                    imp_min = min(self.impulse_areas)
                    imp_max = max(self.impulse_areas)
                    # El rango debe abarcar ambos:
                    overall_min = min(cont_min, imp_min, 0.0)
                    overall_max = max(cont_max, imp_max, 0.0)
                else:
                    overall_min = min(cont_min, 0.0)
                    overall_max = max(cont_max, 0.0)

                # Si el rango es muy pequeño, expandir un poco para visibilidad
                if abs(overall_max - overall_min) < 1e-2:
                    overall_min -= 1.0
                    overall_max += 1.0

                self.y_min, self.y_max = overall_min, overall_max

            # Segunda comprobación para evitar rango nulo o casi nulo
            if abs(self.y_max - self.y_min) < 1e-2:
                self.y_min -= 1.0
                self.y_max += 1.0
        except Exception:
            self.y_min, self.y_max = -1, 1


    def _eval_func_array(self, t):
        y = self.func(t)
        return np.full_like(t, y, dtype=float) if np.isscalar(y) else np.array(y, dtype=float)

    def _extract_impulses(self):
        impulse_locs = []
        impulse_areas = []
        expr_terms = self.expr.as_ordered_terms()

        for term in expr_terms:
            deltas = term.atoms(sp.DiracDelta)
            for delta in deltas:
                arg = delta.args[0]
                roots = sp.solve(arg, self.var)
                amp = term.coeff(delta)

                # derivar factor de escalado de la delta
                d_arg = sp.diff(arg, self.var)
                scale = sp.Abs(d_arg)

                for r in roots:
                    # Evaluar el valor de escala en el punto r
                    try:
                        scale_val = float(scale.subs(self.var, r))
                    except Exception:
                        scale_val = 1.0  # fallback si no evaluable

                    effective_amp = float(amp) / scale_val if scale_val != 0 else 0.0
                    impulse_locs.append(float(r))
                    impulse_areas.append(effective_amp)

        return impulse_locs, impulse_areas

    def _remove_dirac_terms(self):
        return self.expr.replace(lambda expr: expr.has(sp.DiracDelta), lambda _: 0)

    def draw_function(self):
        """
        Dibuja la señal continua. Si 'periodo' está definido, repite la forma de onda en todo el rango horizontal.
        Añade puntos suspensivos en los extremos si la señal está truncada.
        """
        # preparar datos de trazado
        t0, t1 = self.horiz_range
        if self.periodo is None:
            t_plot = self.t_vals
            y_plot = self._eval_func_array(t_plot)
        else:
            # señal periódica
            T = self.periodo
            base_t = np.linspace(-T/2, T/2, self.num_points)
            base_y = self._eval_func_array(base_t)
            k_min = int(np.floor((t0 + T/2)/T))
            k_max = int(np.floor((t1 + T/2)/T))
            segs = []
            for k in range(k_min, k_max+1):
                t_seg = base_t + k*T
                mask = (t_seg >= t0) & (t_seg <= t1)
                segs.append((t_seg[mask], np.array(base_y)[mask]))
            # concatenar y ordenar
            t_plot = np.concatenate([seg[0] for seg in segs])
            y_plot = np.concatenate([seg[1] for seg in segs])
            order = np.argsort(t_plot)
            t_plot = t_plot[order]
            y_plot = y_plot[order]
        # asegurar arrays
        t_plot = np.array(t_plot)
        y_plot = np.array(y_plot)
        if y_plot.ndim == 0:
            y_plot = np.full_like(t_plot, y_plot, dtype=float)
        # dibujar la curva
        self.ax.plot(t_plot, y_plot, color=self.color, linewidth=2.5, zorder=5)

        # decidir puntos suspensivos mediante integración si no es periódica
        delta = (t1 - t0) * 0.05
        tol = 1e-3
        span = t1 - t0
        draw_left = draw_right = False
        if self.periodo is not None:
            draw_left = draw_right = True
        else:
            N = max(10, int(0.05*self.num_points))
            xs_left = np.linspace(t0 - 0.05*span, t0, N)
            ys_left = np.abs(self._eval_func_array(xs_left))
            if np.trapz(ys_left, xs_left) > tol:
                draw_left = True
            xs_right = np.linspace(t1, t1 + 0.05*span, N)
            ys_right = np.abs(self._eval_func_array(xs_right))
            if np.trapz(ys_right, xs_right) > tol:
                draw_right = True
        y_mid = (self.y_min + 2 * self.y_max)/3
        if draw_left:
            self.ax.text(t0 - delta, y_mid, r'$\cdots$', ha='left', va='center', color=self.color, fontsize=14, zorder=10)
        if draw_right:
            self.ax.text(t1 + delta, y_mid, r'$\cdots$', ha='right', va='center', color=self.color,fontsize=14, zorder=10)

    def draw_impulses(self):
        """
        Dibuja los impulsos (Dirac) en las ubicaciones extraídas de la expresión.
        Si 'periodo' está definido, repite cada impulso en t0 + k*T dentro del rango horizontal.
        """
        # Rango horizontal
        t_min, t_max = self.horiz_range
        # Período
        T = self.periodo

        for base_t0, amp in zip(self.impulse_locs, self.impulse_areas):
            if T is None:
                # Sin periodicidad: dibujar solo en la ubicación base
                if t_min <= base_t0 <= t_max:
                    self._draw_single_impulse(base_t0, amp)
            else:
                # Señal periódica: calcular k_min y k_max de forma que t0 + kT esté en [t_min, t_max]
                # Queremos k tales que: t_min <= base_t0 + k*T <= t_max
                # => (t_min - base_t0)/T <= k <= (t_max - base_t0)/T
                # Se toman los enteros entre floor(...) a ceil(...).
                k_min = int(np.floor((t_min - base_t0) / T))
                k_max = int(np.ceil((t_max - base_t0) / T))
                for k in range(k_min, k_max + 1):
                    t_k = base_t0 + k * T
                    if t_min <= t_k <= t_max:
                        self._draw_single_impulse(t_k, amp)

    def _draw_single_impulse(self, t0, amp):
        """
        Dibuja un único impulso en t0 con amplitud amp.
        Si t0 está cerca de 0, desplaza la etiqueta ligeramente a la izquierda
        para que no quede sobre el eje vertical.
        """
        # Flecha desde (t0,0) hasta (t0, amp)
        self.ax.annotate(
            '', xy=(t0, amp), xytext=(t0, 0),
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

        # Desplazamiento vertical normal: ligeramente por encima (o debajo) de amp
        y_offset = 0.03 * np.sign(amp)
        # Ubicación final de la etiqueta
        x_text = t0 + x_offset
        y_text = amp + y_offset

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


    def setup_axes(self):
        for spine in self.ax.spines.values():
            spine.set_color('none')

        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Rango horizontal
        x0, x1 = self.horiz_range
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
            y_margin = 0.2 * y_range

        self.ax.set_xlim(self.horiz_range[0] - x_margin, self.horiz_range[1] + x_margin)
        self.ax.set_ylim(self.y_min - y_margin, self.y_max + 1.3 * y_margin)

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
        Dibuja las marcas de los ejes centradas en las líneas de los ejes.
        Parámetros:
        - tick_size_px: tamaño del tick en píxeles (por defecto self.tick_size_px).
        - xticks, yticks: None, 'auto', lista de posiciones.
            * Si xticks es None en esta llamada, usa lo que se pasó en init (self.init_xticks_arg).
            * Si xticks es 'auto', comportamiento automático:
                - En X: si en init hubo lista no vacía, úsala como base y combina con impulsos periódicos;
                si no hubo pero existen impulsos, muestra solo posiciones periódicas;
                si no hay ni init ni impulsos, genera 5 equiespaciados.
                - En Y: si en init hubo lista no vacía, úsala; si no, genera 3 equiespaciados entre floor(y_min) y ceil(y_max).
            * Si xticks es lista en esta llamada, la usa como manual y en X la combina con impulsos; en Y, solo usa la lista.
        - xtick_labels, ytick_labels: 'auto' (etiquetas por defecto) o lista de mismas longitudes que xticks/yticks manuales.
        Comportamiento extra:
        - Los impulsos periódicos afectan solo al eje X: se repiten según self.periodo.
        - El offset vertical de la etiqueta de un xtick: si coincide con impulso de área<0 → etiqueta encima del eje; área>=0 o no impulso → debajo.
        - Si se dibuja un xtick en 0, no mostrar el ytick en 0 (filtrado posterior).
        - Se eliminan duplicados con tolerancia relativa y se ordenan.
        """
        # 0. Determinar “effective_xticks”: si xticks param es None, usar init_xticks_arg; si no, usar xticks
        effective_xticks = xticks if xticks is not None else getattr(self, 'init_xticks_arg', None)
        effective_yticks = yticks if yticks is not None else getattr(self, 'init_yticks_arg', None)

        # 1. Configuración inicial de tamaño del tick en px
        tick_px = tick_size_px if tick_size_px is not None else self.tick_size_px

        # 2. Rango horizontal y tolerancia relativa para floats
        t_min, t_max = self.horiz_range
        tol = 1e-8 * max(1.0, abs(t_max - t_min))

        # 3. Calcular posiciones periódicas de impulsos en X dentro de [t_min, t_max]
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
        # Eliminar duplicados en posiciones de impulsos
        unique_impulse_positions = []
        unique_impulse_areas = []
        for loc, area in zip(impulse_positions, impulse_positions_areas):
            if not any(abs(loc - loc0) <= tol for loc0 in unique_impulse_positions):
                unique_impulse_positions.append(loc)
                unique_impulse_areas.append(area)
        if unique_impulse_positions:
            idx_sort = np.argsort(unique_impulse_positions)
            impulse_positions = [unique_impulse_positions[i] for i in idx_sort]
            impulse_positions_areas = [unique_impulse_areas[i] for i in idx_sort]
        else:
            impulse_positions = []
            impulse_positions_areas = []

        # 4. Procesar eje X: determinar raw_xticks según effective_xticks
        raw_xticks = []
        manual_xticks = []
        manual_xlabels = []
        # Detectar si en init hubo ticks válidos (lista no vacía)
        has_init_xticks = False
        init_xt = getattr(self, 'xticks', None)
        if init_xt is not None:
            try:
                arr = np.array(init_xt)
                if arr.ndim >= 1 and arr.size >= 1:
                    has_init_xticks = True
            except Exception:
                has_init_xticks = False

        if effective_xticks is None:
            # No mostrar ningún tick en X
            raw_xticks = []
        elif isinstance(effective_xticks, str) and effective_xticks == 'auto':
            if has_init_xticks:
                # Usar ticks pasados en init como base manual
                raw_xticks = list(self.xticks)
                if self.xtick_labels is not None:
                    if len(self.xticks) != len(self.xtick_labels):
                        raise ValueError("xtick_labels y xticks de init deben tener la misma longitud")
                    manual_xticks = list(self.xticks)
                    manual_xlabels = list(self.xtick_labels)
                # Combinar con posiciones periódicas de impulsos
                for loc in impulse_positions:
                    if t_min - tol <= loc <= t_max + tol and not any(abs(loc - x0) <= tol for x0 in raw_xticks):
                        raw_xticks.append(loc)
            else:
                # Sin init ticks: si hay impulsos, mostrar solo sus posiciones periódicas; si no, generar equiespaciados
                if impulse_positions:
                    raw_xticks = list(impulse_positions)
                else:
                    raw_xticks = list(np.linspace(t_min, t_max, 5))
        else:
            # effective_xticks es lista o array explícita
            raw_xticks = list(effective_xticks)

            # Si no se pasaron etiquetas en la llamada pero se dieron en init, usarlas
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

            # Combinar con posiciones periódicas de impulsos
            for loc in impulse_positions:
                if t_min - tol <= loc <= t_max + tol and not any(abs(loc - x0) <= tol for x0 in raw_xticks):
                    raw_xticks.append(loc)
        # Eliminar duplicados y ordenar
        unique_xticks = []
        for x in raw_xticks:
            if not any(abs(x - x0) <= tol for x0 in unique_xticks):
                unique_xticks.append(x)
        raw_xticks = sorted(unique_xticks)

        # 5. Generar etiquetas X: usar manuales cuando coincida, o f'{x:g}'
        xlabels = []
        for x in raw_xticks:
            label = None
            for xm, lbl in zip(manual_xticks, manual_xlabels):
                if abs(xm - x) <= tol:
                    label = lbl
                    break
            if label is None:
                label = f'{x:g}'
            xlabels.append(label)

        # 6. Procesar eje Y: determinar raw_yticks según effective_yticks
        raw_yticks = []
        manual_yticks = []
        manual_ylabels = []
        # Detectar si en init hubo yticks válidos
        has_init_yticks = False
        init_yt = getattr(self, 'yticks', None)
        if init_yt is not None:
            try:
                arr_y = np.array(init_yt)
                if arr_y.ndim >= 1 and arr_y.size >= 1:
                    has_init_yticks = True
            except Exception:
                has_init_yticks = False

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
            else:
                # Generar 3 equiespaciados entre floor(y_min) y ceil(y_max)
                y0 = np.floor(self.y_min)
                y1 = np.ceil(self.y_max)
                if abs(y1 - y0) < 1e-6:
                    raw_yticks = [y0 - 1, y0, y0 + 1]
                else:
                    raw_yticks = list(np.linspace(y0, y1, 3))
        else:
            # Lista explícita para Y
            raw_yticks = list(effective_yticks)

            # Si se pasaron etiquetas en la llamada
            if ytick_labels not in (None, 'auto'):
                if len(raw_yticks) != len(ytick_labels):
                    raise ValueError("ytick_labels y yticks deben tener la misma longitud")
                manual_yticks = list(raw_yticks)
                manual_ylabels = list(ytick_labels)
            # O si se pasaron al inicializar
            elif self.ytick_labels is not None:
                if len(raw_yticks) != len(self.ytick_labels):
                    raise ValueError("ytick_labels y yticks deben tener la misma longitud (init)")
                manual_yticks = list(raw_yticks)
                manual_ylabels = list(self.ytick_labels)
        # Eliminar duplicados y ordenar
        unique_yticks = []
        for y in raw_yticks:
            if not any(abs(y - y0) <= tol for y0 in unique_yticks):
                unique_yticks.append(y)
        raw_yticks = sorted(unique_yticks)

        # 7. Generar etiquetas Y: usar manuales cuando coincida, o f'{y:g}'
        ylabels = []
        for y in raw_yticks:
            label = None
            for ym, lbl in zip(manual_yticks, manual_ylabels):
                if abs(ym - y) <= tol:
                    label = lbl
                    break
            if label is None:
                label = f'{y:g}'
            ylabels.append(label)

        # 8. Filtrar ytick=0 si existe xtick=0
        has_xtick_zero = any(abs(x) <= tol for x in raw_xticks)
        if has_xtick_zero:
            filtered_yticks = []
            filtered_ylabels = []
            for y, lbl in zip(raw_yticks, ylabels):
                if abs(y) <= tol:
                    continue
                filtered_yticks.append(y)
                filtered_ylabels.append(lbl)
            raw_yticks = filtered_yticks
            ylabels = filtered_ylabels

        # 9. Conversión px -> data para longitud de ticks
        origin_disp = self.ax.transData.transform((0, 0))
        up_disp = origin_disp + np.array([0, tick_px])
        right_disp = origin_disp + np.array([tick_px, 0])
        origin_data = np.array(self.ax.transData.inverted().transform(origin_disp))
        up_data = np.array(self.ax.transData.inverted().transform(up_disp))
        right_data = np.array(self.ax.transData.inverted().transform(right_disp))
        dy = up_data[1] - origin_data[1]
        dx = right_data[0] - origin_data[0]

        # 10. Dibujar ticks X
        xlim = self.ax.get_xlim()
        for x, lbl in zip(raw_xticks, xlabels):
            if xlim[0] <= x <= xlim[1]:
                self.ax.plot([x, x], [0 - dy/2, 0 + dy/2], transform=self.ax.transData,
                            color='black', linewidth=1.2, clip_on=False)
                # Offset vertical según área de impulso
                area = None
                for loc, a in zip(impulse_positions, impulse_positions_areas):
                    if abs(loc - x) <= tol:
                        area = a
                        break
                if area is not None and area < 0:
                    y_off = +8
                else:
                    y_off = -8
                # Offset horizontal si x≈0
                if abs(x) < tol:
                    offset = (-8, y_off)
                else:
                    offset = (0, y_off)
                va = 'bottom' if y_off > 0 else 'top'
                self.ax.annotate(rf'${lbl}$', xy=(x, 0), xycoords='data',
                                textcoords='offset points', xytext=offset,
                                ha='center', va=va, fontsize=12, zorder=10,
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=self.alpha))

        # 11. Dibujar ticks Y
        ylim = self.ax.get_ylim()
        for y, lbl in zip(raw_yticks, ylabels):
            if ylim[0] <= y <= ylim[1]:
                self.ax.plot([0 - dx/2, 0 + dx/2], [y, y], transform=self.ax.transData,
                            color='black', linewidth=1.2, clip_on=False)
                if abs(y) < 1e-10:
                    offset = (-8, -8)
                else:
                    offset = (-8, 0)
                self.ax.annotate(rf'${lbl}$', xy=(0, y), xycoords='data',
                                textcoords='offset points', xytext=offset,
                                ha='right', va='center', fontsize=12, zorder=10,
                                bbox=dict(boxstyle='round,pad=0.1', facecolor='white', edgecolor='none', alpha=self.alpha))


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

    def plot(self):
        self.setup_axes()
        self.draw_function()
        self.draw_impulses()
        self.draw_ticks()
        self.draw_labels()
        self.show()

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
