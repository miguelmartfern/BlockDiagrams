# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Miguel Á. Martín (miguelmartfern@github)

# SignalPlotter.py

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

class SignalPlotter:
    def __init__(self, expr_str, var='t', horiz_range=(-5, 5), vert_range=None, num_points=1000,
                 figsize=(8, 3), tick_size_px=5,
                 xticks=None, yticks=None, xtick_labels=None, ytick_labels=None,
                 save_path=None, show_plot=True, color='black'):
        self.expr_str = expr_str
        self.var = sp.Symbol(var)
        self.horiz_range = horiz_range
        self.vert_range = vert_range
        self.num_points = num_points
        self.figsize = figsize
        self.tick_size_px = tick_size_px

        self.color =color

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
        self.expr = parse_expr(expr_str, local_dict=self.local_dict, transformations=transformations)

        # Extract components
        self.expr_cont = self._remove_dirac_terms()
        self.impulse_locs, self.impulse_amps = self._extract_impulses()

        # For plotting
        self.t_vals = np.linspace(*self.horiz_range, self.num_points)
        self.func = sp.lambdify(self.var, self.expr_cont, modules=["numpy", self.local_dict])
        self.fig, self.ax = plt.subplots(figsize=self.figsize)

        self._prepare_plot()

    def _get_local_dict(self):
        return {
            'u':            sp.Heaviside,
            'rect':         lambda t: sp.Heaviside(t + 0.5) - sp.Heaviside(t - 0.5),
            'tri':          lambda t: (1 - abs(t)) * sp.Heaviside(1 - abs(t)),
            'delta':        sp.DiracDelta,
            'DiracDelta':   sp.DiracDelta,
            'Heaviside':    sp.Heaviside,
            'pi':           sp.pi,
            str(self.var):  self.var,
            'sin':          sp.sin,
            'cos':          sp.cos,
            'exp':          sp.exp,
            'Piecewise':    sp.Piecewise,
            'pw':           sp.Piecewise,  # opcional, alias más corto
        }
    
    def _prepare_plot(self):
        self.t_vals = np.linspace(self.horiz_range[0], self.horiz_range[1], self.num_points)

        # Extraer impulsos y expresión continua
        self.impulse_locs, self.impulse_amps = self._extract_impulses()
        self.expr_cont = self._remove_dirac_terms()
        self.func = sp.lambdify(self.var, self.expr_cont, modules=["numpy"])

        try:
            y_vals = self.func(self.t_vals)
            y_vals = np.array(y_vals, dtype=np.float64)
            y_vals = y_vals[np.isfinite(y_vals)]
            if self.vert_range:
                self.y_min, self.y_max = self.vert_range
            else:
                self.y_min, self.y_max = np.min(y_vals), np.max(y_vals)
            if abs(self.y_max - self.y_min) < 1e-2:
                self.y_min -= 1
                self.y_max += 1
        except Exception:
            self.y_min, self.y_max = -1, 1

    def _extract_impulses(self):
        impulse_locs = []
        impulse_amps = []
        expr_terms = self.expr.as_ordered_terms()

        for term in expr_terms:
            deltas = term.atoms(sp.DiracDelta)
            for d in deltas:
                arg = d.args[0]
                roots = sp.solve(arg, self.var)
                amp = term.coeff(d)
                for r in roots:
                    impulse_locs.append(float(r))
                    impulse_amps.append(float(amp))
        return impulse_locs, impulse_amps

    def _remove_dirac_terms(self):
        return self.expr.replace(lambda expr: expr.has(sp.DiracDelta), lambda _: 0)

    def draw_function(self):
        if self.expr_cont == 0:
            y_plot = np.zeros_like(self.t_vals)
        else:
            y_plot = self.func(self.t_vals)
            if np.isscalar(y_plot):
                y_plot = np.full_like(self.t_vals, y_plot, dtype=float)
        self.ax.plot(self.t_vals, y_plot, color=self.color, linewidth=2.5, zorder=5)

    def draw_impulses(self):
        for t0, amp in zip(self.impulse_locs, self.impulse_amps):
            self.ax.annotate('', xy=(t0, amp), xytext=(t0, 0),
                            arrowprops=dict(facecolor=self.color, shrink=0.05, width=1.5, headwidth=8))
            self.ax.text(t0, amp + 0.1 * np.sign(amp), f'{amp:g}', ha='center',
                        va='bottom' if amp > 0 else 'top', fontsize=11)

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
            self.ax.annotate(f'${lbl}$', xy=(x, 0), xycoords='data', textcoords='offset points',
                             xytext=offset, ha='center', va='top', fontsize=12, zorder=10)

        # dibujar ticks Y (centrados en x=0)
        for y, lbl in zip(yticks_final, ylabels):
            self.ax.plot([0 - dx/2, 0 + dx/2], [y, y], transform=self.ax.transData,
                         color='black', linewidth=1.2, clip_on=False)
            if abs(y) < 1e-10:  # etiqueta sobre el eje horizontal
                offset = (-8, -8)  # desplazar más a la derecha
            else:
                offset = (-8, 0)
            self.ax.annotate(f'${lbl}$', xy=(0, y), xycoords='data', textcoords='offset points',
                             xytext=offset, ha='right', va='center', fontsize=12)

    def draw_labels(self):
        x_lim = self.ax.get_xlim()
        y_lim = self.ax.get_ylim()

        # Etiqueta eje X: un poco a la derecha del límite derecho del eje x
        x_pos = x_lim[1] - 0.01 * (x_lim[1] - x_lim[0])
        y_pos = 0.02 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, f'${self.var}$', fontsize=16, ha='right', va='bottom')

        # Etiqueta eje Y: un poco por debajo del límite superior del eje y (pero dentro)
        x_pos = 0.01 * (x_lim[1] - x_lim[0])
        y_pos = y_lim[1] - 0.1 * (y_lim[1] - y_lim[0])
        self.ax.text(x_pos, y_pos, rf'$x({self.var})$', fontsize=16, ha='left', va='bottom', rotation=0)

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
