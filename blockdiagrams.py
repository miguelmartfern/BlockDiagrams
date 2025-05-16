# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Miguel Á. Martín (miguelmartfern@github)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.transforms import Bbox


# --- DiagramBuilder class ---

class DiagramBuilder:
    """
    Helper class for creating signal processing diagrams step by step.
    
    Keeps track of horizontal position and allows adding components
    like blocks, arrows, multipliers, and input/output signals in order.
    """
    def __init__(self, block_length=1.0, fontsize=20):
        self.fig, self.ax = plt.subplots()
        # self.ax.set_xlim(*xlim)
        # self.ax.set_ylim(*ylim)
        self.ax.axis('off')  # Hide axes
        self.fontsize = fontsize
        self.block_length = block_length
        # self.spacing = spacing
        self.thread_positions = {}
        self.thread_positions['main'] = [0, 0]
    
    def __print_threads__(self):
        for thread in self.thread_positions:
            print(thread, ": ", self.thread_positions[thread])


    # --- Drawing functions ---

    def __draw_block__(self, initial_position, text, length=1.5, height=1, fontsize=14):
        """
        Draws a rectangular block with centered text.

        Parameters:
        - ax: matplotlib Axes object where the diagram is drawn.
        - position: (x, y) coordinates of the center of the left edge of the block.
        - text: label to display inside the block.
        - length: horizontal length of the block.
        - height: vertical height of the block.
        - fontsize: font size of the text inside the block.
        """
        x0, y = initial_position
        x1 = x0 + length

        self.ax.add_patch(Rectangle((x0, y - height / 2), length, height, edgecolor='black', facecolor='white'))
        self.ax.text(x0 + length / 2, y, f"${text}$", ha='center', va='center', fontsize=fontsize)

        return [x1, y]

    def __draw_block_uparrow__(self, initial_position, text, input_bottom_text, length=1.5, height=1, text_offset=0.1, fontsize=14):
        """
        Draws a rectangular block with an additional vertical arrow input from below.

        Parameters:
        - ax: matplotlib Axes object where the diagram is drawn.
        - position: (x, y) coordinates of the center of the left edge of the block.
        - text: label inside the block.
        - input_bottom_text: label for the vertical arrow entering from below.
        - length: horizontal length of the block.
        - height: vertical height of the block.
        - text_offset: vertical offset for the bottom input label.
        - fontsize: font size of the labels.
        """
        x0, y = initial_position
        x1 = x0 + length
        cx = (x0 + x1) / 2

        self.ax.add_patch(Rectangle((x0, y - height / 2), length, height, edgecolor='black', facecolor='white'))
        self.ax.text(x0 + length / 2, y, f"${text}$", ha='center', va='center', fontsize=fontsize)

        self.ax.add_patch(FancyArrow(cx, y - 1.25 * length, 0, 1.25 * length - height / 2, width=0.01,
                                length_includes_head=True, head_width=0.15, color='black'))
        if input_bottom_text:
            self.ax.text(cx, y - 1.25 * length - text_offset, f"${input_bottom_text}$",
                    ha='center', va='top', fontsize=fontsize)
        
        return [x1, y]

    def __draw_arrow__(self, initial_position, length, text=None, arrow = True, text_offset=(0, 0.2), fontsize=14):
        """
        Draws a horizontal arrow with optional label.

        Parameters:
        - ax: matplotlib Axes object where the diagram is drawn.
        - position: (x, y) coordinates of the arrow starting point.
        - length: horizontal length of the arrow.
        - text: optional label above the arrow.
        - text_offset: (x, y) offset for positioning the label relative to the arrow.
        - fontsize: font size of the label.
        """
        # end = (initial_position[0] + length, initial_position[1])
        head_width = 0.15 if arrow else 0

        x0, y = initial_position
        x1 = x0 + length

        self.ax.add_patch(FancyArrow(x0, y, length, 0, width=0.01,
                                length_includes_head=True, head_width=head_width, color='black'))
        if text:
            tx = x0 + length / 2 + text_offset[0]
            ty = y + text_offset[1]
            self.ax.text(tx, ty, f"${text}$", ha='center', fontsize=fontsize)
        
        return [x1, y]

    def __draw_io_arrow__(self, initial_position, length=1, text="", io='input', text_offset=0.3, fontsize=14):
        """
        Draws an input/output horizontal arrow with a label.

        Parameters:
        - ax: matplotlib Axes object where the diagram is drawn.
        - position: (x, y) coordinates where the arrow starts.
        - length: horizontal length of the arrow.
        - text: label for the arrow.
        - io: 'input' or 'output' to place the label at the beginning or end.
        - text_offset: relative x offset for the label position between (0,1).
        - fontsize: font size of the label.
        """
        x0, y = initial_position
        x1 = x0 + length
        
        absolute_text_offset = text_offset * length
        self.ax.add_patch(FancyArrow(x0, y, length, 0, width=0.01,
                                length_includes_head=True, head_width=0.15, color='black'))
        if text:
            tx = x0 - absolute_text_offset if io == 'input' else x0 + length + absolute_text_offset
            ty = y
            self.ax.text(tx, ty, f"${text}$", ha='center', va='center', fontsize=fontsize)
        
        return [x1, y]

    def __draw_combiner__(self, initial_position, height=1,
                        input_text=None, operation='mult', side='bottom', text_offset=0.1, fontsize=14):
        """
        Draws a combiner block: a circle with a multiplication sign (×), sum sign (+) 
        or substraction sign (-) inside.

        Parameters:
        - self: matplotlib Axes object where the diagram is drawn.
        - intial_position: (x, y) coordinates of the starting point.
        - length: total horizontal length (diameter of the circle).
        - input_text: label for the bottom input arrow (below or above the arrow).
        - operation: 'mult' for multiplication sign (×), 'sum' for addition sign (+), 'dif' for substraction sign (-).
        - text_offset: vertical offset for input/output labels.
        - fontsize: font size of the labels.
        """
        x0, y = initial_position
        radius = height / 4
        x1 = x0 + 2 * radius
        cx = (x0 + x1) / 2
        
        circle = plt.Circle((cx, y), radius, edgecolor='black', facecolor='white', zorder=2)
        self.ax.add_patch(circle)

        rel_size = 0.7
        if operation == 'mult':
            # Líneas diagonales (forma de "X") dentro del círculo
            dx = radius * rel_size * np.cos(np.pi / 4)  # Escalamos un poco para que quepa dentro del círculo
            dy = radius * rel_size * np.sin(np.pi / 4)

            # Línea de 45°
            self.ax.plot([cx - dx, cx + dx], [y - dy, y + dy], color='black', linewidth=2, zorder=3)

            # Línea de 135°
            self.ax.plot([cx - dx, cx + dx], [y + dy, y - dy], color='black', linewidth=2, zorder=3)
        elif operation == 'sum':
            # Líneas horizontales y verticales (forma de "+") dentro del círculo
            self.ax.plot([cx - radius * rel_size, cx + radius * rel_size], [y, y], color='black', linewidth=2, zorder=3)
            self.ax.plot([cx, cx], [y - radius * rel_size, y + radius * rel_size], color='black', linewidth=2, zorder=3)
        elif operation == 'dif':
            # Línea horizontal (forma de "-") dentro del círculo
            self.ax.plot([cx - radius * rel_size, cx + radius * rel_size], [y, y], color='black', linewidth=2, zorder=3)
        else:
            raise ValueError(f"Unknown operation: {operation}. 'operation' must be 'mult', 'sum' or 'dif'.")

        # Side input
        if side == 'bottom':
            y_init = y - height - radius
            y_height = height
            y_text_pos = y_init - text_offset
            va = 'top'
        elif side == 'top':
            y_init = y + height + radius
            y_height = - height
            y_text_pos = y_init + text_offset
            va = 'bottom'
        else:
            raise ValueError(f"Unknown side: {side}. 'side' must be 'bottom' or 'top'.")
        
        self.ax.add_patch(FancyArrow(cx, y_init, 0, y_height, width=0.01,
                                length_includes_head=True, head_width=0.15, color='black'))
        if input_text:
            self.ax.text(cx, y_text_pos, f"${input_text}$",
                    ha='center', va=va, fontsize=fontsize)
        
        return [x1, y]

    def __draw_mult_combiner__(self, initial_position, length, inputs, output_text=None, operation='sum', side='bottom', fontsize=14):
        """
        Dibuja un sumador o multiplicador con múltiples entradas distribuidas desde pi/2 a 3*pi/2
        a lo largo del borde izquierdo de un círculo. Las entradas pueden tener signo.

        Parámetros:
        - inputs: lista de tuplas (x, y) o (x, y, signo)
        - output_text: etiqueta en la flecha de salida (a la derecha)
        - position: centro del sumador (x0, y0)
        - operation: 'sum' para suma, 'mult' para producto

        """
        # Get head positions of input threads
        thread_input_pos = np.array([self.thread_positions[key] for key in inputs])
        
        # If position is 'auto', obtain head position
        if isinstance(initial_position, str) and initial_position == 'auto':
            x0 = np.max(thread_input_pos[:, 0])
            y0 = np.mean(thread_input_pos[:,1])
        # If position is given, use it
        else:
            x0, y0 = initial_position
        
        x1 = x0 + length
        cx = (x0 + x1) / 2
        cy = y0
        radius = length / 10
        
        # Círculo
        circle = plt.Circle((cx, cy), radius, edgecolor='black', facecolor='white', linewidth=1.5, zorder=2)
        self.ax.add_patch(circle)

        # Dibujar símbolo dentro del círculo según operación
        rel_size = 0.7
        if operation == 'mult':
            dx = radius * rel_size * np.cos(np.pi / 4)
            dy = radius * rel_size * np.sin(np.pi / 4)
            self.ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color='black', linewidth=2, zorder=3)
            self.ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color='black', linewidth=2, zorder=3)
        elif operation == 'sum':
            self.ax.plot([cx - radius * rel_size, cx + radius * rel_size], [cy, cy], color='black', linewidth=2, zorder=3)
            self.ax.plot([cx, cx], [cy - radius * rel_size, cy + radius * rel_size], color='black', linewidth=2, zorder=3)
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'sum' or 'mult'.")

        n = len(thread_input_pos)
        angles = np.linspace(5* np.pi / 8, 11 * np.pi / 8, n)

        arrow_width = 0.01
        arrow_head_width = 0.15

        # Flechas de entrada
        for i, inp in enumerate(thread_input_pos):
            xi, yi = inp[:2]

            x_edge = cx + radius * np.cos(angles[i])
            y_edge = cy + radius * np.sin(angles[i])

            dx = x_edge - xi
            dy = y_edge - yi

            self.ax.add_patch(FancyArrow(
                xi, yi, dx, dy,
                width=arrow_width,
                length_includes_head=True,
                head_width=arrow_head_width,
                color='black',
                zorder=1
            ))

        # Flecha de salida
        x1 = cx + radius
        x2 = x0 + length

        self.ax.add_patch(FancyArrow(
            x1, cy, x2 - x1, 0,
            width=arrow_width,
            length_includes_head=True,
            head_width=arrow_head_width,
            color='black',
            zorder=1
        ))

        # Texto de salida
        if output_text:
            self.ax.text(
                (x1 + x2) / 2, cy + arrow_head_width / 2,
                f"${output_text}$",
                ha='center', va='bottom', fontsize=fontsize
            )

        return [x2, y0]

    def add(self,name, kind='block', thread='main', position=None, debug=False, **kwargs):
        """
        Adds a diagram element to the current position and advances the x coordinate.
        `kind` can be: input, output, arrow, block, block_uparrow, mult.
        """
        # print(thread)
        # print(self.current_x)
        # print(self.positions)
        # if thread not in self.current_x:
        #     self.current_x[thread] = 0
        
        # if thread == 'main':
        #     self.current_x['main'] = max(self.current_x.values())

        height = kwargs.get('height', self.block_length)

        # If position is 'auto' (draw_mult_combiner), position is calculated inside that method
        if isinstance(position, str) and position == 'auto':
            initial_pos = 'auto'
        # If input argument position is given and not 'auto', element position is asigned to position argument value
        elif position is not None:
            initial_pos = list(position)
        # If not given
        else:
            # If thread already exists, element position is asigned from thread head
            if thread in self.thread_positions:
                initial_pos = self.thread_positions[thread]
            # If doesn't exist
            else:
                initial_pos = [0, 0]

        if kind == 'input':
            length=kwargs.get('length', self.block_length)
            final_pos = self.__draw_io_arrow__(initial_pos, length=length, text=kwargs.get('text', name),
                          io='input', fontsize=self.fontsize)

        elif kind == 'arrow':
            length=kwargs.get('length', self.block_length)
            final_pos = self.__draw_arrow__(initial_pos, length=length,
                       text=kwargs.get('text', name), arrow = True, fontsize=self.fontsize)

        elif kind == 'line':
            length=kwargs.get('length', self.block_length)
            final_pos = self.__draw_arrow__(initial_pos, length=length,
                       text=kwargs.get('text', name), arrow = False, fontsize=self.fontsize)

        elif kind == 'block':
            length=kwargs.get('length', self.block_length)
            final_pos = self.__draw_block__(initial_pos, text=kwargs.get('text', name),
                       length=length, height=height, fontsize=self.fontsize)

        elif kind == 'block_uparrow':
            length=kwargs.get('length', self.block_length)
            final_pos = self.__draw_block_uparrow__(initial_pos, text=kwargs.get('text', name),
                               input_bottom_text=kwargs.get('input_bottom_text'),
                               length=length, height=height, fontsize=self.fontsize)

        elif kind == 'combiner':
            length=kwargs.get('length', self.block_length)
            final_pos = self.__draw_combiner__(initial_pos, height=length,
                            input_text=kwargs.get('input_text'),
                            operation=kwargs.get('operation'), 
                            side=kwargs.get('side'), fontsize=self.fontsize)

        elif kind == 'mult_combiner':
            length=kwargs.get('length', self.block_length*2.5)
            final_pos = self.__draw_mult_combiner__(initial_pos, length=length, inputs=kwargs.get('inputs'),
                            output_text=kwargs.get('text', name), 
                            operation=kwargs.get('operation'), fontsize=self.fontsize)

        elif kind == 'output':
            length=kwargs.get('length', self.block_length)
            final_pos = self.__draw_io_arrow__(initial_pos, length=length, text=kwargs.get('text', name),
                          io='output', fontsize=self.fontsize)

        else:
            raise ValueError(f"Unknown block type: {kind}")

        # Update head position of thread
        self.thread_positions[thread] = final_pos

        if debug:
            self.__print_threads__()


    def get_bbox(self):
        return self.ax.dataLim
        
    def show(self, margin=0.5, scale=1.0, savepath=None):
        """
        Muestra o guarda el diagrama ajustando los límites al contenido.

        Parámetros:
        - margin: margen alrededor del contenido.
        - scale: factor de escala aplicado al tamaño de la imagen.
        - savepath: si se proporciona, guarda la figura en lugar de mostrarla.
        """
        bbox = self.get_bbox()
        if bbox is None:
            plt.show()
            return

        x0 = bbox.x0 - margin
        x1 = bbox.x1 + margin
        y0 = bbox.y0 - margin
        y1 = bbox.y1 + margin

        width = x1 - x0
        height = y1 - y0

        # Tamaño de figura en pulgadas: 1 unidad = scale pulgadas
        fig_width = width * scale
        fig_height = height * scale
        self.fig.set_size_inches(fig_width, fig_height)

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_position([0, 0, 1, 1])  # usa toda la figura
        self.ax.axis("off")  # opcional

        if savepath:
            self.fig.savefig(savepath, bbox_inches='tight', dpi=self.fig.dpi, transparent=False, facecolor='white')
            print(f"Saved in: {savepath}")
        else:
            plt.show()

