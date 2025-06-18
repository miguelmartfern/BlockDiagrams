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
    Helper class to visualize poles, zeros and regions of convergence (ROC) 
    on the complex plane of the Z Transform. It supports both cartesian and polar input for coordinates.

    Examples:
        >>> cp = ComplexPlane()
        >>> cp.draw_poles_and_zeros(poles=[0.5+0.5j, (0.7, np.pi/4)], zeros=[-0.5+0j, (1, np.pi)])
        >>> cp.draw_ROC("|z|>0.7")
        >>> cp.draw_unit_circle()
        >>> cp.show()
    """
    def __init__(self, figsize=(6, 6), xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), facecolor='white', fontsize=18):
    
        self.fig, self.ax = plt.subplots(figsize=figsize, facecolor='white')

        self.fontsize = fontsize
        self.xlim = xlim
        self.ylim = ylim

        self.poles = []
        self.zeros = []

    # --- Helper functions ---

    def __get_bbox__(self):
        return self.ax.dataLim
    
    # --- Calculus functions ---

    def max_pole_modulus(self, poles=None):
        """
        Compute maximum modulus of poles.

        Args:
            poles (list, optional): List of poles (complex or (r, θ) tuples). Defaults to self.poles.

        Returns:
            (float): Maximum modulus.

        Examples:
            >>> cp.max_pole_modulus()
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
        Compute minimum modulus of poles.

        Args:
            poles (list, optional): List of poles (complex or (r, θ) tuples). Defaults to self.poles.

        Returns:
            (float): Minimum modulus.

        Examples:
            >>> cp.min_pole_modulus()
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
        Draw poles (red crosses) and zeros (blue circles).

        Args:
            poles (list, optional): Complex or polar coordinates.
            zeros (list, optional): Complex or polar coordinates.

        Examples:
            >>> cp.draw_poles_and_zeros(poles=[0.5+0.5j], zeros=[-0.5])
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
        Draw region of convergence (ROC).

        Args:
            condition (str): One of "|z|<a", "|z|>a", "a<|z|<b".
            color (str, optional): ROC fill color.
            alpha (float, optional): Transparency.
            label (str, optional): Label for legend.

        Examples:
            >>> cp.draw_ROC("|z|>0.7")
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
                        avoid_overlap=True, delta_angle=np.pi/24, offset_angle=np.pi/30, color='blue'):
        """
        Draws radial lines from origin with optional labels and dashed circles.
        Avoids placing radios exactly at 0 and pi by offsetting them.

        Args:
            labels (list): Labels for each radial.
            radii (list): Radii.
            angles (list, optional): Angles (if None, auto-generated).
            circles (list, optional): If True, draw dashed circle at each radius.
            avoid_overlap (bool): Avoid conflict with poles/zeros.
            delta_angle (float): Angular increment if searching free angles.
            offset_angle (float): Offset from 0 and π.
            color (str): Color of lines.

        Examples:
            >>> cp.draw_radial_guides(labels=["a", "b"], radii=[0.5, 1.2])
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

            ax.plot([0, x], [0, y], color=color, linewidth=2, zorder=8)

            if label:
                angle_deg = np.degrees(ang)
                if 90 < angle_deg < 270:
                    angle_deg -= 180
                elif angle_deg > 270:
                    angle_deg -= 360

                ax.text(x * 0.5, y * 0.5 + 0.05, f"${label}$", fontsize=self.fontsize,
                        ha='center', va='bottom', rotation=angle_deg,
                        rotation_mode='anchor', color=color, zorder=10)

            if circ:
                ax.add_patch(Circle((0, 0), r, edgecolor=color, facecolor='none',
                                    linestyle='--', linewidth=1.2, zorder=7))

    def draw_unit_circle(self, linestyle='--', color='black', linewidth=1.2):
        """
        Draw unit circle.

        Args:
            linestyle (str): Line style.
            color (str): Color.
            linewidth (float): Line width.

        Examples:
            >>> cp.draw_unit_circle()
        """
        ax = self.ax

        pos_x = 1 + self.xlim[1] / 25
        pos_y = self.ylim[1] / 25
        ax.text(pos_x, pos_y, f"${1}$", fontsize=12,
                ha='center', va='bottom', color=color,
                rotation_mode='anchor', zorder=10)

        ax.add_patch(Circle((0, 0), 1, edgecolor=color, facecolor='none',
                            linestyle=linestyle, linewidth=linewidth, label=f'$|z|=1$', zorder=7))

    def label_positions(self, positions, labels, offset=0.08):
        """
        Add text labels at given positions.

        Args:
            positions (list): Complex or polar coordinates.
            labels (list): Text labels.
            offset (float): Vertical offset.

        Examples:
            >>> cp.label_positions([0.5+0.5j], ["A"])
        """
        points_cartesian = self._process_points(positions)

        for pos, label in zip(points_cartesian, labels):
            x, y = pos.real, pos.imag

            # Small black circle
            self.ax.plot(x, y, 'o', color='black', markersize=3, zorder=9)
            print(label)
            # Label text slightly above the marker
            self.ax.text(x, y + offset, f"${label}$", fontsize=12, ha='center', va='bottom', zorder=10)

    # === Show and save ===

    def show(self, savepath=None):
        """
        Display or save the figure.

        Args:
            savepath (str, optional): Path to save figure as file. If None, shows the plot.

        Examples:
            >>> cp.show()
            >>> cp.show("myplot.png")
        """
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

