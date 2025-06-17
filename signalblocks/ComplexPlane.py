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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import Counter


class ComplexPlane:
    """
    Helper class for creating complex plane representations step by step.
    
    Keeps track of horizontal position and allows adding components
    like blocks, arrows, multipliers, and input/output signals in order.
    """
    def __init__(self, figsize=(6, 6), xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), facecolor='white', fontsize=18):
    
        self.fig, self.ax = plt.subplots(figsize=figsize, facecolor='white')

        self.fontsize = fontsize
        self.xlim = xlim
        self.ylim = ylim

        self.poles = []
        self.zeros = []

        # circle = Circle((0, 0), 1, edgecolor='black', facecolor='none', linestyle='--', linewidth=1.2, zorder=6)
        # self.ax.add_patch(circle)
        # self.ax.plot([1], [0], marker='|', color='black', markersize=10, zorder=7)
        # self.ax.text(1, 0, "$1$", ha='left', va='bottom', fontsize=14, zorder=7)

    # --- Helper functions ---

    def __get_bbox__(self):
        return self.ax.dataLim
    
    # --- Calculus functions ---

    def max_pole_modulus(self, poles=None):
        """
        Returns the maximum modulus among a list of poles.
        Each pole can be given in Cartesian (complex number)
        or polar form (r, θ) as a tuple.
        """

        if poles is None:
            poles = self.poles
        if not isinstance(poles, list):
            raise ValueError("poles must be a list of complex numbers or (magnitude, phase) tuples.")
        
        moduli = []
        for p in poles:
            if isinstance(p, tuple):  # Polar form
                r, theta = p
                z = r * np.exp(1j * theta)
            else:
                z = p  # Cartesian form (complex number)
            moduli.append(abs(z))
        return max(moduli) if moduli else None
    
    def min_pole_modulus(self, poles=None):
        """
        Returns the minimum modulus among a list of poles.
        Each pole can be given in Cartesian (complex number)
        or polar form (r, θ) as a tuple.
        """

        if poles is None:
            poles = self.poles
        if not isinstance(poles, list):
            raise ValueError("poles must be a list of complex numbers or (magnitude, phase) tuples.")
        
        moduli = []
        for p in poles:
            if isinstance(p, tuple):  # Polar form
                r, theta = p
                z = r * np.exp(1j * theta)
            else:
                z = p  # Cartesian form (complex number)
            moduli.append(abs(z))
        return min(moduli) if moduli else None
    
    # --- Drawing functions ---

    def _process_points(self, points):
        """Convert all points to complex numbers, accepting polar or cartesian input."""
        result = []
        for p in points:
            if isinstance(p, complex):
                result.append(p)
            elif (isinstance(p, tuple) or isinstance(p, list)) and len(p) == 2:
                r, theta = p
                z = r * np.exp(1j * theta)
                result.append(z)
            else:
                raise ValueError(f"Invalid format: {p}")
        return result

    def _round_complex(self, z, decimals=6):
        """Round complex numbers to avoid floating-point noise."""
        return complex(round(z.real, decimals), round(z.imag, decimals))

    def draw_poles_and_zeros(self, poles=None, zeros=None):
        """
        Draw poles and zeros on the complex plane.

        Parameters:
        - poles: list of complex numbers or (magnitude, phase) tuples
        - zeros: list of complex numbers or (magnitude, phase) tuples
        """
        if poles:
            new_poles = self._process_points(poles)
            self.poles.extend(new_poles)
        if zeros:
            new_zeros = self._process_points(zeros)
            self.zeros.extend(new_zeros)

        # Count multiplicities
        poles_counter = Counter(self._round_complex(z) for z in self.poles)
        zeros_counter = Counter(self._round_complex(z) for z in self.zeros)

        # Plot poles
        for pole, count in poles_counter.items():
            self.ax.scatter(pole.real, pole.imag,
                       marker='x',
                       color='red',
                       s=100 + 20 * (count - 1),
                       alpha=min(0.4 + 0.15 * count, 1.0),
                       label=f'Pole x{count}' if count > 1 else 'Pole',
                       linewidths=3,
                       zorder=10)

        # Plot zeros
        for zero, count in zeros_counter.items():
            self.ax.scatter(zero.real, zero.imag,
                       marker='o',
                       edgecolors='blue',
                       facecolors='none',
                       linewidths=2,
                       s=100 + 20 * (count - 1),
                       alpha=min(0.4 + 0.15 * count, 1.0),
                       label=f'Zero x{count}' if count > 1 else 'Zero',
                       zorder=10)

    def draw_ROC(self, condition, color='orange', alpha=0.3, label="ROC"):
        """
        Draws the Region of Convergence (ROC) on the complex plane.

        Parameters
        ----------
        condition : str
            ROC condition as a string. Valid formats are:
            - "|z|<a"
            - "|z|>a"
            - "a<|z|<b"
        R_max : float, optional
            Maximum radius used to draw the outer boundary when |z| > a.
            Defaults to 2.5 * max(|xlim|, |ylim|).
        color : str, optional
            Fill and edge color of the ROC. Default is 'orange'.
        alpha : float, optional
            Transparency of the ROC area. Default is 0.3.
        label : str, optional
            Label to use in the legend for the ROC. Default is "ROC".
        """
        edge_color = color

        R_max = 2.5 * max(abs(self.xlim[0]), abs(self.xlim[1]), abs(self.ylim[0]), abs(self.ylim[1]))

        condition = condition.replace(" ", "")
        if condition.startswith("|z|<"):
            a = eval(condition[4:])
            region_type = 'less'
        elif condition.startswith("|z|>"):
            a = eval(condition[4:])
            region_type = 'greater'
        elif '<|z|<' in condition:
            a_str, b_str = condition.split('<|z|<')
            a = eval(a_str)
            b = eval(b_str)
            region_type = 'between'
        else:
            raise ValueError("Invalid condition. Use formats like '|z|<a', '|z|>a' or 'a<|z|<b'.")

        # === Draw ROC ===
        if region_type == 'less':
            patch = Circle((0, 0), a, facecolor=color, alpha=alpha, edgecolor=edge_color, linewidth=2, zorder=0, label=label)
            self.ax.add_patch(patch)
        elif region_type == 'greater':
            patch_outer = Circle((0, 0), R_max, facecolor=color, alpha=alpha, edgecolor=edge_color, linewidth=2, zorder=0)
            patch_inner = Circle((0, 0), a, facecolor='white', edgecolor='none', zorder=0)
            patch_border = Circle((0, 0), a, facecolor='none', edgecolor=edge_color, linewidth=2, zorder=2, label=label)
            self.ax.add_patch(patch_outer)
            self.ax.add_patch(patch_inner)
            self.ax.add_patch(patch_border)
        elif region_type == 'between':
            patch_outer = Circle((0, 0), b, facecolor=color, alpha=alpha, edgecolor=edge_color, linewidth=2, zorder=0)
            patch_inner = Circle((0, 0), a, facecolor='white', edgecolor='none', zorder=0)
            patch_border_outer = Circle((0, 0), b, facecolor='none', edgecolor=edge_color, linewidth=2, zorder=1)
            patch_border_inner = Circle((0, 0), a, facecolor='none', edgecolor=edge_color, linewidth=2, zorder=1, label=label)
            self.ax.add_patch(patch_outer)
            self.ax.add_patch(patch_inner)
            self.ax.add_patch(patch_border_outer)
            self.ax.add_patch(patch_border_inner)

    def draw_radial_guides(self, labels, radii, angles=None, circles=None,
                        avoid_overlap=True, delta_angle=np.pi/24, offset_angle=np.pi/30):
        """
        Draws radial lines from origin with optional labels and dashed circles.
        Avoids placing radios exactly at 0 and pi by offsetting them.

        Parameters:
            labels (list of str): List of labels to show on each radial.
            radii (list of float): Radii at which to draw each radial.
            angles (list of float or None): Optional list of angles in radians.
            circles (list of bool): Whether to draw a dashed circle at each radius.
            avoid_overlap (bool): Avoid pole/zero overlap if angles auto-generated.
            delta_angle (float): Step angle to try when searching free angles.
            offset_angle (float): Minimal angular offset to avoid 0 and π exactly.
        """
        ax = self.ax

        if circles is None:
            circles = [False] * len(radii)

        n = len(radii)

        if angles is None:
            angles = []

            # Collect forbidden angles if necessary
            forbidden = set()
            if avoid_overlap:
                def extract_angles(items):
                    result = []
                    for item in items:
                        if isinstance(item, tuple):
                            r, theta = item
                        else:
                            theta = np.angle(item)
                        result.append(round(theta % (2 * np.pi), 4))
                    return result

                forbidden.update(extract_angles(getattr(self, 'poles', [])))
                forbidden.update(extract_angles(getattr(self, 'zeros', [])))

            # Prepare candidate base angles avoiding exactly 0 and π
            base_angles = [offset_angle, np.pi - offset_angle,
                        np.pi + offset_angle, 2 * np.pi - offset_angle]

            # Fill angles list trying these first, then try offsets further away if needed
            i = 0
            while len(angles) < n:
                for base in base_angles:
                    candidate = (base + i * delta_angle) % (2 * np.pi)
                    if round(candidate, 4) not in forbidden and candidate not in angles:
                        angles.append(candidate)
                        if len(angles) == n:
                            break
                i += 1

            # If still not enough angles (rare), fill uniformly excluding forbidden
            while len(angles) < n:
                candidate = (angles[-1] + delta_angle) % (2 * np.pi)
                if round(candidate, 4) not in forbidden and candidate not in angles:
                    angles.append(candidate)

        # === Draw radial guides ===
        for label, r, ang, circ in zip(labels, radii, angles, circles):
            x = r * np.cos(ang)
            y = r * np.sin(ang)

            ax.plot([0, x], [0, y], color='blue', linewidth=2, zorder=8)

            if label:
                angle_deg = np.degrees(ang)
                if 90 < angle_deg < 270:
                    angle_deg -= 180
                elif angle_deg > 270:
                    angle_deg -= 360

                ax.text(x * 0.5, y * 0.5 + 0.05, f"${label}$", fontsize=self.fontsize,
                        ha='center', va='bottom', rotation=angle_deg,
                        rotation_mode='anchor', zorder=10)

            if circ:
                ax.add_patch(Circle((0, 0), r, edgecolor='blue', facecolor='none',
                                    linestyle='--', linewidth=1.2, zorder=7))
    
    def label_positions(self, positions, labels, offset=0.08):
        """
        Place labels on given positions, and draw a small black circle at each.

        Parameters:
        -----------
        positions : list
            List of positions, either:
            - complex numbers (x + jy), or
            - tuples (r, theta) in polar coordinates.
        labels : list of str
            Text labels to place at each position.
        offset : float
            Vertical offset for the text label to avoid overlapping the marker.
        """
        points_cartesian = self._process_points(positions)

        for pos, label in zip(points_cartesian, labels):
            x, y = pos.real, pos.imag

            # Small black circle
            self.ax.plot(x, y, 'o', color='black', markersize=3, zorder=9)

            # Label text slightly above the marker
            self.ax.text(x, y + offset, label, fontsize=12, ha='center', va='bottom', zorder=10)

    # === Show and save ===

    def show(self, savepath=None):
        self.ax.set_aspect('equal')
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)

        # Remove numeric ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Remove frame
        for spine in ['top', 'right', 'bottom', 'left']:
            self.ax.spines[spine].set_visible(False)

        # === Axis ===
        self.ax.axhline(0, color='black', linewidth=1, zorder=15)
        self.ax.axvline(0, color='black', linewidth=1, zorder=15)

        # # Axis lables
        # self.ax.axis('off')
        # self.ax.text(self.xlim[1], 0.05, r'$\Re\{z\}$', 
        #              fontsize=self.fontsize, ha='right', va='bottom')
        # self.ax.text(0.05, self.ylim[1], r'$\Im\{z\}$', 
        #              fontsize=self.fontsize, ha='left', va='top')

        # Axis labels (just outside the plot area)
        x_label_pos = self.xlim[1] + 0.03 * (self.xlim[1] - self.xlim[0])
        y_label_pos = self.ylim[1] + 0.03 * (self.ylim[1] - self.ylim[0])

        self.ax.text(x_label_pos, 0,
                    r'$\Re\{z\}$',
                    fontsize=self.fontsize,
                    ha='left', va='center',
                    zorder=15)

        self.ax.text(0, y_label_pos,
                    r'$\Im\{z\}$',
                    fontsize=self.fontsize,
                    ha='center', va='bottom',
                    zorder=15)
        
        # Deduplicate legend
        handles, labels = self.ax.get_legend_handles_labels()
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
        if unique:
            self.ax.legend(*zip(*unique), loc='upper left', bbox_to_anchor=(1.05, 1))

        if savepath:
            self.fig.savefig(savepath, bbox_inches='tight', dpi=self.fig.dpi, transparent=False, facecolor='white')
            print(f"Saved in: {savepath}")
        else:
            plt.show()

