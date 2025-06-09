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
                 xticks=None, yticks=None, xtick_labels=None, ytick_labels=None,
                 save_path=None, show_plot=True, color='black', periodo=None):
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

        # tick positions and labels
        self.xticks = np.array(xticks) if xticks is not None else None
        self.yticks = np.array(yticks) if yticks is not None else None
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
                min_y = np.min(y_vals) if y_vals.size > 0 else 0
                max_y = np.max(y_vals) if y_vals.size > 0 else 0
                # incluir amplitudes (áreas) de deltas
                if self.impulse_areas:
                    min_y = min(min_y, 0)  # asegurar que el eje base esté visible
                    max_y = max(max_y, max(self.impulse_areas))
                if abs(max_y - min_y) < 1e-2:
                    min_y -= 1
                    max_y += 1
                self.y_min, self.y_max = min_y, max_y
            if abs(self.y_max - self.y_min) < 1e-2:
                self.y_min -= 1; self.y_max += 1
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
            base_y = self.func(base_t)
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
        for t0, amp in zip(self.impulse_locs, self.impulse_areas):
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
            self.ax.text(
                t0, amp + 0.05 * np.sign(amp),
                f'{amp:g}',
                ha='center',
                va='bottom' if amp > 0 else 'top',
                fontsize=12,
                color=self.color,
                zorder=10
            )

    def setup_axes(self):
        for spine in self.ax.spines.values():
            spine.set_color('none')

        self.ax.set_xticks([])
        self.ax.set_yticks([])

        x_margin = 0.2 * (self.horiz_range[1] - self.horiz_range[0])
        y_margin = 0.2 * (self.y_max - self.y_min)
        self.ax.set_xlim(self.horiz_range[0] - x_margin, self.horiz_range[1] + x_margin)
        self.ax.set_ylim(self.y_min - y_margin, self.y_max + y_margin)

        self.ax.annotate('', xy=(self.ax.get_xlim()[1], 0), xytext=(self.ax.get_xlim()[0], 0),
                         arrowprops=dict(arrowstyle='-|>', linewidth=1.5, color='black',
                                         mutation_scale=16, mutation_aspect=0.8, fc='black'))

        self.ax.annotate('', xy=(0, self.ax.get_ylim()[1]), xytext=(0, self.ax.get_ylim()[0]),
                         arrowprops=dict(arrowstyle='-|>', linewidth=1.5, color='black',
                                         mutation_scale=12, mutation_aspect=2, fc='black'))
        
        self.fig.tight_layout()

    def draw_ticks(self,
                   tick_size_px=None,
                   xticks='auto',
                   yticks='auto',
                   xtick_labels='auto',
                   ytick_labels='auto'):
        """
        Dibuja las marcas de los ejes centradas en las líneas de los ejes.
        Parámetros:
         - tick_size_px: tamaño del tick en píxeles (por defecto self.tick_size_px)
         - xticks, yticks: 'auto' (por defecto de init o cálculo), lista de posiciones, o None (sin ticks)
         - xtick_labels, ytick_labels: 'auto' (por defecto de init o numérico), lista de etiquetas o None
        """
        # tamaño de tick en px
        tick_px = tick_size_px if tick_size_px is not None else self.tick_size_px

        # ---- POSICIONES RAW ----
        if xticks is None:
            raw_xticks = []
        elif xticks == 'auto':
            raw_xticks = list(self.xticks) if self.xticks is not None else list(np.linspace(*self.horiz_range, 5))
        else:
            raw_xticks = list(xticks)

        if yticks is None:
            raw_yticks = []
        elif yticks == 'auto':
            raw_yticks = list(self.yticks) if self.yticks is not None else list(np.linspace(np.floor(self.y_min), np.ceil(self.y_max), 3))
        else:
            raw_yticks = list(yticks)

        # ---- VALIDACIÓN ETIQUETAS ----
        if xtick_labels not in (None, 'auto'):
            if not raw_xticks:
                raise ValueError("Se han proporcionado xtick_labels pero no hay xticks")
            if len(xtick_labels) != len(raw_xticks):
                raise ValueError("xtick_labels y xticks deben tener la misma longitud")
        if ytick_labels not in (None, 'auto'):
            if not raw_yticks:
                raise ValueError("Se han proporcionado ytick_labels pero no hay yticks")
            if len(ytick_labels) != len(raw_yticks):
                raise ValueError("ytick_labels y yticks deben tener la misma longitud")

        # ---- FILTRADO SOLO AUTO PARA Y ----
        xticks_final = raw_xticks
        if yticks is None or (yticks == 'auto' and self.yticks is None):
            yticks_final = [y for y in raw_yticks if abs(y) > 1e-6]
        else:
            yticks_final = raw_yticks

        # ---- ETIQUETAS ----
        if xticks_final:
            if xtick_labels not in (None, 'auto'):
                xlabels = xtick_labels
            elif self.xtick_labels is not None:
                xlabels = self.xtick_labels
            else:
                xlabels = [f'{x:g}' for x in xticks_final]
        else:
            xlabels = []
        if yticks_final:
            if ytick_labels not in (None, 'auto'):
                ylabels = ytick_labels
            elif self.ytick_labels is not None:
                ylabels = self.ytick_labels
            else:
                ylabels = [f'{y:g}' for y in yticks_final]
        else:
            ylabels = []

        # ---- CONVERSIÓN px -> data ----
        # origen en data
        origin_disp = self.ax.transData.transform((0, 0))
        # desplazamiento en px hacia arriba
        up_disp = origin_disp + np.array([0, tick_px])
        right_disp = origin_disp + np.array([tick_px, 0])
        # transformar de vuelta a data
        origin_data = np.array(self.ax.transData.inverted().transform(origin_disp))
        up_data = np.array(self.ax.transData.inverted().transform(up_disp))
        right_data = np.array(self.ax.transData.inverted().transform(right_disp))
        dy = up_data[1] - origin_data[1]
        dx = right_data[0] - origin_data[0]

        # dibujar ticks X (centrados en y=0)
        for x, lbl in zip(xticks_final, xlabels):
            self.ax.plot([x, x], [0 - dy/2, 0 + dy/2], transform=self.ax.transData,
                         color='black', linewidth=1.2, clip_on=False)
            if abs(x) < 1e-10:  # etiqueta sobre el eje vertical
                offset = (-8, -8)  # desplazar más hacia abajo
            else:
                offset = (0, -8)
            self.ax.annotate(rf'${lbl}$', xy=(x, 0), xycoords='data', textcoords='offset points',
                             xytext=offset, ha='center', va='top', fontsize=12, zorder=10)

        # dibujar ticks Y (centrados en x=0)
        for y, lbl in zip(yticks_final, ylabels):
            self.ax.plot([0 - dx/2, 0 + dx/2], [y, y], transform=self.ax.transData,
                         color='black', linewidth=1.2, clip_on=False)
            if abs(y) < 1e-10:  # etiqueta sobre el eje horizontal
                offset = (-8, -8)  # desplazar más a la derecha
            else:
                offset = (-8, 0)
            self.ax.annotate(rf'${lbl}$', xy=(0, y), xycoords='data', textcoords='offset points',
                             xytext=offset, ha='right', va='center', fontsize=12)

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
