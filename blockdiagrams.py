# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Miguel Á. Martín (miguelmartfern@github)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow
# from matplotlib.transforms import Bbox
from matplotlib import transforms
from matplotlib.text import Text

# --- DiagramBuilder class ---

class DiagramBuilder:
    """
    Helper class for creating signal processing diagrams step by step.
    
    Keeps track of horizontal position and allows adding components
    like blocks, arrows, multipliers, and input/output signals in order.
    """
    def __init__(self, block_length=1.0, block_height=1.0, fontsize=20):
        self.fig, self.ax = plt.subplots()
        # self.ax.set_xlim(*xlim)
        # self.ax.set_ylim(*ylim)
        self.ax.axis('off')  # Hide axes
        self.fontsize = fontsize
        self.block_length = block_length
        self.block_height = block_height
        # self.spacing = spacing
        self.thread_positions = {}
        self.thread_positions['main'] = [0, 0]
    
    def __print_threads__(self):
        for thread in self.thread_positions:
            print(thread, ": ", self.thread_positions[thread])


    # --- Helper functions ---

    def __get_bbox__(self):
        return self.ax.dataLim
    
    def __get_output_pos__(self, init_pos, outvector, angle):
        """
        Compute rotated output point

        Parameters:
        - init_pos: initial position of the block
        - outvector: output vector before rotation (respecto to init_pos)
        - angle: rotation angle in degrees
        """

        # Output point respect to input point (before rotation)
        out_vector = np.array(outvector)
        # Rotation matrix (without translation)
        rotation_matrix = transforms.Affine2D().rotate_deg(angle).get_matrix()[:2, :2]
        # Apply rotation to the output vector
        dx, dy = rotation_matrix @ out_vector
        # Add the rotated output vector to the initial position
        return [init_pos[0] + dx, init_pos[1] + dy]

    # --- Drawing functions ---

    def __draw_rotated_text__(self, anchor_point, text, angle, 
                      ha='center', va='center', fontsize=16, offset=(0, 0)):
        """
        Draws text rotated around the anchor point (x, y) with optional offset.

        Parameters:
        - ax: matplotlib Axes object.
        - anchor_point: coordinates of the anchor point.
        - text: string to display.
        - angle: rotation angle in degrees.
        - ha, va: horizontal and vertical alignment.
        - fontsize: font size.
        - offset: tuple (dx, dy) in data coordinates, before rotation.
        """
        # Apply rotation to the offset vector
        dx, dy = offset
        offset_vec = np.array([dx, dy])
        rot_matrix = transforms.Affine2D().rotate_deg(angle).get_matrix()[:2, :2]
        dx_rot, dy_rot = rot_matrix @ offset_vec

        # Compute final position
        tx = anchor_point[0] + dx_rot
        ty = anchor_point[1] + dy_rot

        # Draw text with angle, rotating around anchor point
        self.ax.text(tx, ty, f"${text}$", ha=ha, va=va, fontsize=fontsize,
                rotation=angle, rotation_mode='anchor', transform=self.ax.transData)


    def __draw_block__(self, initial_position, text=None, text_below=None, 
                       text_above=None, text_offset=0.1, input_text=None, 
                       input_side=None, length=1.5, height=1, fontsize=14, 
                       linestyle='-', orientation='horizontal'):
        """
        Draws a rectangular block with centered text.

        Parameters:
        - ax: matplotlib Axes object where the diagram is drawn.
        - initial_position: (x, y) coordinates of the center of the left edge of the block.
        - text: label to display in the block.
        - text_below: position of the text below the block
        - text_above: position of the text above the block
        - text_offset: vertical offset for the text position.
        - input_text: label for the input arrow (below or above the arrow).
        - input_side: 'bottom' or 'top' to place the input arrow.
        - length: horizontal length of the block.
        - height: vertical height of the block.
        - fontsize: font size of the text inside the block.
        - linesyle: linestyle of the block edge: '-, '--, ':', '-.'.
        - orientation: direction of the block: 'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle.
        """
        # Parameters validation
        if input_side not in (None, 'top', 'bottom'):
            raise ValueError(f"Invalid input_side: {input_side}. Use 'top' or 'bottom'.")
        if orientation not in (None, 'horizontal', 'vertical', 'up', 'down', 'left', 'right'):
            if isinstance(orientation, (int, float)):
                pass
            else:
                raise ValueError(f"Invalid orientation: {orientation}. Use 'horizontal', 'vertical', 'up', 'down', 'left', or 'right'.")
        if linestyle not in (None, '-', '--', ':', '-.', 'solid', 'dashed', 'dotted', 'dashdot'):
            raise ValueError(f"Invalid linestyle: {linestyle}. Use '-', '--', ':', '-.', 'solid', 'dashed', 'dotted', or 'dashdot'.")
        if not isinstance(length, (int, float)) or length <= 0:
            raise ValueError(f"Invalid length: {length}. Length must be a positive number.")
        if not isinstance(height, (int, float)) or height <= 0:
            raise ValueError(f"Invalid height: {height}. Height must be a positive number.")
        if not isinstance(text_offset, (int, float)):
            raise ValueError(f"Invalid text_offset: {text_offset}. Text offset must be a number.")
        if not isinstance(fontsize, (int, float)):
            raise ValueError(f"Invalid fontsize: {fontsize}. Font size must be a number.")
        
        
        angle = 0
        # Determine rotation angle based on orientation
        if orientation in ['horizontal', 'right']:
            angle = 0
        elif orientation == 'left':
            angle = 180
        elif orientation in ['vertical', 'down']:
            angle = -90
        elif orientation == 'up':
            angle = 90
        elif isinstance(orientation, (int, float)):
            angle = orientation
        else:
            angle = 0
  
        x_in, y_in = initial_position

        # Bottom-left corner of the block (before rotation)
        x0 = x_in
        y0 = y_in - height / 2

        # Center of the block (before rotation)
        cx = x_in + length / 2
        cy = y_in

        # Apply rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData   

        self.ax.add_patch(Rectangle((x0, y0), length, height, 
                                    edgecolor='black', facecolor='none', 
                                    linestyle=linestyle, transform=trans))

        # Draw text inside the block
        if text is not None:
            offset_vector = np.array([length / 2, 0])
            self.__draw_rotated_text__(initial_position, text, angle=angle,
                  ha='center', va='center', fontsize=fontsize, offset=offset_vector)
            
        # Draw text above the block
        if text_above is not None:
            offset_vector = np.array([length / 2, height / 2 + text_offset])
            self.__draw_rotated_text__(initial_position, text_above, angle=angle,
                  ha='center', va='bottom', fontsize=fontsize, offset=offset_vector)
            
        # Draw text below the block
        if text_below is not None:
            offset_vector = np.array([length / 2, - height / 2 - text_offset])
            self.__draw_rotated_text__(initial_position, text_below, angle=angle,
                  ha='center', va='top', fontsize=fontsize, offset=offset_vector)

        if input_side is not None:
            if input_side == 'bottom':
                arrow_height = 0.75 * height
                y_init = y0 - arrow_height
                offset_vector = np.array([length / 2, - height /2 - arrow_height - text_offset])
                va = 'top'
            elif input_side == 'top':
                arrow_height = - 0.75 * height
                y_init = y0 + height - arrow_height
                offset_vector = np.array([length / 2, height /2 - arrow_height + text_offset])
                va = 'bottom'
            else:
                raise ValueError(f"Unknown input side: {input_side}. Use 'bottom' or 'top'.")   

            self.ax.add_patch(FancyArrow(cx, y_init, 0, arrow_height, width=0.01,
                                    length_includes_head=True, head_width=0.15, 
                                    color='black', transform=trans))
            if input_text is not None:
                self.__draw_rotated_text__(initial_position, input_text, angle=angle,
                    ha='center', va=va, fontsize=fontsize, offset=offset_vector)

        # Compute rotated output point
        x_out, y_out = self.__get_output_pos__(initial_position, [length, 0], angle)
        return [x_out, y_out]

    def __draw_arrow__(self, initial_position, length, text=None, 
                       text_position = 'above', arrow = True, text_offset=0.2, 
                       fontsize=14, orientation='horizontal'):
        """
        Draws a horizontal arrow with optional label.

        Parameters:
        - ax: matplotlib Axes object where the diagram is drawn.
        - position: (x, y) coordinates of the arrow starting point.
        - length: horizontal length of the arrow.
        - text: optional label above the arrow.
        - text_offset: (x, y) offset for positioning the label relative to the arrow.
        - fontsize: font size of the label.
        - orientation: direction of the block: 'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle.
        """
        # end = (initial_position[0] + length, initial_position[1])
        head_width = 0.15 if arrow else 0

        angle = 0
        # Determine rotation angle based on orientation
        if orientation in ['horizontal', 'right']:
            angle = 0
        elif orientation == 'left':
            angle = 180
        elif orientation in ['vertical', 'down']:
            angle = -90
        elif orientation == 'up':
            angle = 90
        elif isinstance(orientation, (int, float)):
            angle = orientation
        else:
            angle = 0

        x_in, y_in = initial_position

        # Center of the block (before rotation)
        cx = x_in + length / 2
        cy = y_in

        # x0, y = initial_position
        # x1 = x0 + length

        # Apply rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData   


        self.ax.add_patch(FancyArrow(x_in, y_in, length, 0, width=0.01,
                                length_includes_head=True, head_width=head_width, 
                                color='black', transform=trans))
        if text:
            # Offset en coordenadas no rotadas
            if text_position == 'before':
                ha, va = 'right', 'center'
                offset_vector = np.array([-text_offset, 0])
            elif text_position == 'after':
                ha, va = 'left', 'center'
                offset_vector = np.array([length + text_offset, 0])
            elif text_position == 'above':
                ha, va = 'center', 'bottom'
                offset_vector = np.array([length / 2, text_offset])
            else:
                raise ValueError(f"Unknown text_position: {text_position}")

            self.__draw_rotated_text__(initial_position, text, angle=angle,
                  ha=ha, va=va, offset=offset_vector)
        
        # Compute rotated output point
        x_out, y_out = self.__get_output_pos__(initial_position, [length, 0], angle)
        return [x_out, y_out]

    def __draw_combiner__(self, initial_position, height=1,
                        input_text=None, operation='mult', input_side='bottom', 
                        text_offset=0.1, fontsize=14, orientation='horizontal'):
        """
        Draws a combiner block: a circle with a multiplication sign (×), sum sign (+) 
        or substraction sign (-) inside.

        Parameters:
        - intial_position: (x, y) coordinates of the starting point.
        - length: total horizontal length (diameter of the circle).
        - input_text: label for the bottom input arrow (below or above the arrow).
        - operation: 'mult' for multiplication sign (×), 'sum' for addition sign (+), 'dif' for substraction sign (-).
        - text_offset: vertical offset for input/output labels.
        - fontsize: font size of the labels.
        - orientation: direction of the block: 'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle.
        """
        angle = 0
        # Determine rotation angle based on orientation
        if orientation in ['horizontal', 'right']:
            angle = 0
        elif orientation == 'left':
            angle = 180
        elif orientation in ['vertical', 'down']:
            angle = -90
        elif orientation == 'up':
            angle = 90
        elif isinstance(orientation, (int, float)):
            angle = orientation
        else:
            angle = 0

        x_in, y_in = initial_position

        radius = height / 4
        # Center of the block (before rotation)
        cx = x_in + radius
        cy = y_in

        # Apply rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData  

        circle = plt.Circle((cx, cy), radius, edgecolor='black', 
                            facecolor='white', transform=trans, zorder=2)
        self.ax.add_patch(circle)

        rel_size = 0.7
        if operation == 'mult':
            # Líneas diagonales (forma de "X") dentro del círculo
            dx = radius * rel_size * np.cos(np.pi / 4)  # Escalamos un poco para que quepa dentro del círculo
            dy = radius * rel_size * np.sin(np.pi / 4)
            # Línea de 45°
            self.ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
            # Línea de 135°
            self.ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)

        elif operation == 'sum':
            dx = radius * rel_size
            dy = radius * rel_size
            # Líneas horizontales y verticales (forma de "+") dentro del círculo
            self.ax.plot([cx - dx, cx + dx], [cy, cy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
            self.ax.plot([cx, cx], [cy - dy, cy + dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
        elif operation == 'dif':
            dx = radius * rel_size
            # Línea horizontal (forma de "-") dentro del círculo
            self.ax.plot([cx - dx, cx + dx], [cy, cy], color='black', 
                         linewidth=2, transforms=trans, zorder=3)
        else:
            raise ValueError(f"Unknown operation: {operation}. 'operation' must be 'mult', 'sum' or 'dif'.")

        # Side input
        if input_side == 'bottom':
            arrow_height = height - radius
            y_init = y_in - radius - arrow_height
            offset_vector = np.array([radius, - (height + text_offset)])
            va = 'top'
        elif input_side == 'top':
            arrow_height = - (height - radius)
            y_init = y_in + radius - arrow_height
            offset_vector = np.array([radius, height + text_offset])
            va = 'bottom'
        else:
            raise ValueError(f"Unknown input_side: {input_side}. 'input_side' must be 'bottom' or 'top'.")

        self.ax.add_patch(FancyArrow(cx, y_init, 0, arrow_height, width=0.01,
                                length_includes_head=True, head_width=0.15, 
                                color='black', transform=trans))
        if input_text is not None:
            self.__draw_rotated_text__(initial_position, input_text, angle=angle,
                    ha='center', va=va, fontsize=fontsize, offset=offset_vector)
        
        # Compute rotated output point
        x_out, y_out = self.__get_output_pos__(initial_position, [2 * cx, 0], angle)
        return [x_out, y_out]

    def __draw_mult_combiner__(self, initial_position, length, inputs, 
                               output_text=None, operation='sum', side='bottom', 
                               text_offset=0.1, fontsize=14, orientation='horizontal'):
        """
        Draws a summation or multiplication block with multiple inputs distributed 
        along the left edge of a circle, from pi/2 to 3*pi/2. Inputs can have a sign.

        Parameters:
        - initial_position: (x, y) coordinates of the starting point.
        - length: total horizontal length (diameter of the circle).
        - inputs: list of tuples (x, y) or (x, y, sign)
        - output_text: label on the output arrow (on the right)
        - position: center of the summation block (x0, y0)
        - operation: 'sum' for addition, 'mult' for multiplication
        - text_offset: vertical offset for input/output labels.
        - fontsize: font size of the labels.
        - orientation: direction of the block: 'horizontal', 'vertical', 'up', 'down', 'left', 'right', or angle.
        """
        angle = 0
        # Determine rotation angle based on orientation
        if orientation in ['horizontal', 'right']:
            angle = 0
        elif orientation == 'left':
            angle = 180
        elif orientation in ['vertical', 'down']:
            angle = -90
        elif orientation == 'up':
            angle = 90
        elif isinstance(orientation, (int, float)):
            angle = orientation
        else:
            angle = 0

        # If position is 'auto', obtain head position
        if isinstance(initial_position, str) and initial_position == 'auto':
            # Get head positions of input threads
            thread_input_pos = np.array([self.thread_positions[key] for key in inputs])
            x_in = np.max(thread_input_pos[:, 0])
            y_in = np.mean(thread_input_pos[:,1])
        # If position is given, use it
        else:
            x_in, y_in = initial_position
        
        radius = length / 10
        x_out = x_in + length
        cx = (x_in + x_out) / 2
        cy = y_in

        # Apply rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData  

        # Circle
        circle = plt.Circle((cx, cy), radius, edgecolor='black', 
                            facecolor='white', transform=trans, zorder=2)
        self.ax.add_patch(circle)

        # Dibujar símbolo dentro del círculo según operación
        rel_size = 0.7
        if operation == 'mult':
            # Líneas diagonales (forma de "X") dentro del círculo
            dx = radius * rel_size * np.cos(np.pi / 4)  # Escalamos un poco para que quepa dentro del círculo
            dy = radius * rel_size * np.sin(np.pi / 4)
            # Línea de 45°
            self.ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
            # Línea de 135°
            self.ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
        elif operation == 'sum':
            dx = radius * rel_size
            dy = radius * rel_size
            # Líneas horizontales y verticales (forma de "+") dentro del círculo
            self.ax.plot([cx - dx, cx + dx], [cy, cy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
            self.ax.plot([cx, cx], [cy - dy, cy + dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
        else:
            raise ValueError(f"Unknown operation: {operation}. Use 'sum' or 'mult'.")

        # Get rotation matrix
        rot_matrix = transforms.Affine2D().rotate_deg(angle).get_matrix()[:2, :2]

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
            offset_vec = [dx, dy]

            dx_rot, dy_rot = rot_matrix @ offset_vec

            self.ax.add_patch(FancyArrow(
                xi, yi, dx_rot, dy_rot,
                width=arrow_width,
                length_includes_head=True,
                head_width=arrow_head_width,
                color='black', transform=self.ax.transData, zorder=1
            ))

        # Flecha de salida
        x1 = cx + radius
        x2 = x_in + length

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

        return [x2, y_in]

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

        if kind == 'arrow':
            # Default arguments
            default_kwargs = {
                'text': name,
                'text_position': 'above',
                'arrow': True,
                'text_offset': 0.1,
                'length': self.block_length,
                'fontsize': self.fontsize,
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_arrow__(initial_pos, **block_args)

        elif kind == 'input':
            # Default arguments
            default_kwargs = {
                'text': name,
                'text_position': 'before',
                'arrow': True,
                'text_offset': 0.1,
                'length': self.block_length,
                'fontsize': self.fontsize,
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_arrow__(initial_pos, **block_args)

        elif kind == 'output':
            # Default arguments
            default_kwargs = {
                'text': name,
                'text_position': 'after',
                'arrow': True,
                'text_offset': 0.1,
                'length': self.block_length,
                'fontsize': self.fontsize,
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_arrow__(initial_pos, **block_args)

        elif kind == 'line':
            # Default arguments
            default_kwargs = {
                'text': name,
                'text_position': 'above',
                'arrow': False,
                'text_offset': 0.1,
                'length': self.block_length,
                'fontsize': self.fontsize,
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_arrow__(initial_pos, **block_args)

        elif kind == 'block':
            # Default arguments
            default_kwargs = {
                'text': name,
                'text_above': None,
                'text_below': None,
                'text_offset': 0.1,
                'input_text': None,
                'input_side': None,
                'length': self.block_length,
                'height': self.block_height,
                'fontsize': self.fontsize,
                'linestyle': '-',
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_block__(initial_pos, **block_args)

        elif kind == 'combiner':
            # Default arguments
            default_kwargs = {
                'height': self.block_height,
                'fontsize': self.fontsize,
                'operation': 'mult',
                'input_side': 'bottom',
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_combiner__(initial_pos, **block_args)

        elif kind == 'mult_combiner':
            # Default arguments
            default_kwargs = {
                'length': self.block_length*2.5,
                'fontsize': self.fontsize,
                'operation': 'mult',
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_mult_combiner__(initial_pos, **block_args)


            # length=kwargs.get('length', self.block_length*2.5)
            # final_pos = self.__draw_mult_combiner__(initial_pos, length=length, inputs=kwargs.get('inputs'),
            #                 output_text=kwargs.get('text', name), 
            #                 operation=kwargs.get('operation'), fontsize=self.fontsize)

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

    def get_position(self, thread='main'):
        """
        Returns the current position of the specified thread.
        """
        if thread in self.thread_positions:
            return self.thread_positions[thread]
        else:
            raise ValueError(f"Thread '{thread}' not found.")

        
    def show(self, margin=0.5, scale=1.0, savepath=None):
        """
        Muestra o guarda el diagrama ajustando los límites al contenido.

        Parámetros:
        - margin: margen alrededor del contenido.
        - scale: factor de escala aplicado al tamaño de la imagen.
        - savepath: si se proporciona, guarda la figura en lugar de mostrarla.
        """
        bbox = self.__get_bbox__()
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

