# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Miguel Á. Martín (miguelmartfern@github)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib import transforms
from dataclasses import dataclass
from typing import Tuple

# --- DiagramBuilder class ---

@dataclass
class ElementPosition:
    input_pos: Tuple[float, float]
    output_pos: Tuple[float, float]
    feedback_pos: Tuple[float, float]
class DiagramBuilder:
    """
    A helper class for incrementally building signal processing diagrams using Matplotlib.

    This class provides high-level methods to add standard diagram components like blocks, arrows,
    combiners, and input/output labels, keeping track of layout and threading.

    Args:
        block_length (float, optional): Default horizontal size of blocks.
        block_height (float, optional): Default vertical size of blocks.
        fontsize (int, optional): Default font size for all text.
    
    Returns:
        (DiagramBuilder): created object.
        
    Examples:
        >>> from blockdiagrams import DiagramBuilder
        >>> db1 = DiagramBuilder()
        >>> db2 = DiagramBuilder(block_length=2, fontsize=16)
    """
    def __init__(self, block_length=1.0, block_height=1.0, fontsize=20):
        """
        (Private) Creator of the DiagramBuilder class.
        """
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')  # Hide axes
        self.fontsize = fontsize
        self.block_length = block_length
        self.block_height = block_height
        self.thread_positions = {}
        self.thread_positions['main'] = [0, 0]
        # Dictionary to store element positions: input_pos, output_pos, feedback_pos
        self.element_positions = {}
        # Counter for current element
        self.current_element = -1
    
    def print_threads(self):
        """
        Prints name of each thread in diagram and actual position.
    
        Examples:
            >>> from blockdiagrams import DiagramBuilder
            >>> db = DiagramBuilder(block_length=1, fontsize=16)
            >>> # Upper thread
            >>> db.add("x_1(t)", kind="input", thread='upper', position=(0, 1))
            >>> db.add("mult", kind="combiner", thread='upper', input_text="e^{-j\\omega_0 t}", input_side='top', operation='mult')
            >>> db.add("", kind="line", thread='upper')
            >>> # Lower thread
            >>> db.add("x_2(t)", kind="input", thread='lower', position=(0, -1))
            >>> db.add("mult", kind="combiner", input_text="e^{j\\omega_0 t}", input_side='bottom', operation='mult', thread='lower')
            >>> db.add("", kind="line", thread='lower')
            >>> input_threads = ['upper', 'lower']
            >>> # Adder
            >>> db.add("", kind="mult_combiner", inputs=input_threads, position="auto", operation='sum')
            >>> # Rest of the diagram (main thread)
            >>> db.add("x(t)", kind="output")
            >>> db.show()
            >>> db.print_threads()
        """
        for thread in self.thread_positions:
            print(thread, ": ", self.thread_positions[thread])


    # --- Helper functions ---

    def __get_bbox__(self):
        return self.ax.dataLim
    
    def __get_rotated_pos__(self, init_pos, outvector, angle):
        """
        Inner method.
        Compute rotated point init_pos + outvector.

        Args:
            init_pos (Numpy.NDArray or list): Initial position of the block (relative origin of coordinates).
            outvector (Numpy.NDArray or list): Output vector before rotation (relative position with respect to init_pos).
            angle (float): Rotation angle in degrees.

        Returns:
            (Numpy.NDArray): Rotated position of vector init_pos + outvector.
        """

        # Output point respect to input point (before rotation)
        out_vector = np.array(outvector)
        # Rotation matrix (without translation)
        rotation_matrix = transforms.Affine2D().rotate_deg(angle).get_matrix()[:2, :2]
        # Apply rotation to the output vector
        dx, dy = rotation_matrix @ out_vector
        # Add the rotated output vector to the initial position
        return np.array([init_pos[0] + dx, init_pos[1] + dy])

    def __add_element_position__(self, input_pos: Tuple[float, float], 
                                 output_pos: Tuple[float, float], 
                                 feedback_pos: Tuple[float, float]):
        """
        Inner method.
        Adds a new element with the given input, output and feedback positions.

        Args:
            input_pos (Numpy.NDArray or list): Input position of the block.
            output_pos (Numpy.NDArray or list): Output position of the block.
            feedback_pos (Numpy.NDArray or list): Feedback port position of the block.
        """
        self.current_element += 1
            
        self.element_positions[self.current_element] = ElementPosition(
            input_pos=input_pos,
            output_pos=output_pos,
            feedback_pos=feedback_pos
        )

    # --- Drawing functions ---

    def __draw_rotated_text__(self, anchor_point, text, angle, rotate_text = True,
                      ha='center', va='center', fontsize=16, offset=(0, 0)):
        """
        Inner method.
        Draws text rotated around the anchor point with optional offset. 
        Text position: rotation(anchor_point + offset)

        Args:
            anchor_point (Numpy.NDArray or list): Coordinates of the anchor point.
            text (string): String to display. LaTeX math accepted (without $...$).
            angle (float): Rotation angle in degrees.
            rotate_text (bool, optional): Indicates if text must be rotated or not.
            ha (string, optional): Horizontal alignment: {'center', 'left', 'right'}.
            va (string, optional): Vertical alignment: {'center', 'bottom', 'top'}.
            fontsize (int, optional): Font size.
            offset (Numpy.NDArray or list): Coordinates of texr position respect to anchor point, before rotation.
        """
        # Apply rotation to the offset vector
        dx, dy = offset
        offset_vec = np.array([dx, dy])
        rot_matrix = transforms.Affine2D().rotate_deg(angle).get_matrix()[:2, :2]
        dx_rot, dy_rot = rot_matrix @ offset_vec

        # Compute final position
        tx = anchor_point[0] + dx_rot
        ty = anchor_point[1] + dy_rot

        if rotate_text is False:
            text_angle = 0
        else:
            text_angle = angle
        
        # Draw text with angle, rotating around anchor point
        self.ax.text(tx, ty, f"${text}$", ha=ha, va=va, fontsize=fontsize,
                rotation=text_angle, rotation_mode='anchor', transform=self.ax.transData)


    def __draw_block__(self, initial_position, text=None, text_below=None, 
                       text_above=None, text_offset=0.1, input_text=None, 
                       input_side=None, length=1.5, height=1, fontsize=14, 
                       linestyle='-', orientation='horizontal'):
        """
        Inner method.
        Draws a rectangular block with centered text, optional texts below and/or above and optional input arrow with text.

        Args:
            initial_position (Numpy.NDarray or list): Coordinates of the center position of the input edge of the block.
            text (string, optional): Label to display in the block.
            text_below (string, optional): Label to display below the block.
            text_above (string, optional): Label to display above the block.
            text_offset (float, optional): Vertical offset for the text position.
            input_text (string, optional): Label for the optional input arrow (below or above the block).
            input_side (string, optional): Side to place the input arrow: {'bottom', 'top', None}
            length (float, optional): Horizontal length of the block. If not entered, default `block_length` is used.
            height (float, optional): Vertical height of the block. If not entered, default `block_height` is used.
            fontsize (int, optional): font size of the text inside the block. If not entered, default `fontsize` is used.
            linestyle (string, optional): linestyle of the block edge: {'-, '--, ':', '-.'}.
            orientation (string or float, optional): Direction of the block: {'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle}.

        Returns:
            (Numpy.NDArray): Coordinates of the center position of the output edge of the block.
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

        # Apply the rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData   

        self.ax.add_patch(Rectangle((x0, y0), length, height, 
                                    edgecolor='black', facecolor='none', 
                                    linestyle=linestyle, transform=trans))
        # Don't rotate text if orientation is vertical, down or up
        rotate_text = False if orientation in ['vertical', 'down', 'up', 'left'] else True
        
        # Draw text inside the block
        if text is not None:
            offset_vector = np.array([length / 2, 0])
            self.__draw_rotated_text__(initial_position, text, 
                                       angle=angle, rotate_text=rotate_text,
                                       ha='center', va='center', 
                                       fontsize=fontsize, offset=offset_vector)
            
        # Draw text above the block
        if text_above is not None:
            if orientation in ['vertical', 'down']:
                ha = 'left'
                va = 'center'
            elif orientation in ['up']:
                ha = 'right'
                va = 'center'
            else:
                ha = 'center'
                va = 'bottom'
            offset_vector = np.array([length / 2, height / 2 + text_offset])
            self.__draw_rotated_text__(initial_position, text_above, 
                                       angle=angle, rotate_text=rotate_text,
                                       ha=ha, va=va, 
                                       fontsize=fontsize, offset=offset_vector)
            
        # Draw text below the block
        if text_below is not None:
            if orientation in ['vertical', 'down']:
                ha = 'right'
                va = 'center'
            elif orientation in ['up']:
                ha = 'left'
                va = 'center'
            else:
                ha = 'center'
                va = 'top'
            offset_vector = np.array([length / 2, - height / 2 - text_offset])
            self.__draw_rotated_text__(initial_position, text_below, 
                                       angle=angle, rotate_text=rotate_text,
                                       ha=ha, va=va, 
                                       fontsize=fontsize, offset=offset_vector)

        if input_side is not None:
            if input_side == 'bottom':
                arrow_height = 0.75 * height
                y_init = y0 - arrow_height
                offset_vector = np.array([length / 2, - height /2 - arrow_height - text_offset])
                va = 'top'
                ha = 'center'
                if orientation in ['vertical', 'down']:
                    ha = 'right'
                    va = 'center'
                elif orientation in ['up']:
                    ha = 'left'
                    va = 'center'
                elif orientation in ['left']:
                    ha = 'center'
                    va = 'bottom'
            elif input_side == 'top':
                arrow_height = - 0.75 * height
                y_init = y0 + height - arrow_height
                offset_vector = np.array([length / 2, height /2 - arrow_height + text_offset])
                va = 'bottom'
                ha = 'center'
                if orientation in ['vertical', 'down']:
                    ha = 'left'
                    va = 'center'
                elif orientation in ['up']:
                    ha = 'right'
                    va = 'center'
                elif orientation in ['left']:
                    ha = 'center'
                    va = 'top'
            else:
                raise ValueError(f"Unknown input side: {input_side}. Use 'bottom' or 'top'.")   

            self.ax.add_patch(FancyArrow(cx, y_init, 0, arrow_height, width=0.01,
                                    length_includes_head=True, head_width=0.15, 
                                    color='black', transform=trans))
            if input_text is not None:

                self.__draw_rotated_text__(initial_position, input_text, 
                                           angle=angle, rotate_text=rotate_text,
                                           ha=ha, va=va, 
                                           fontsize=fontsize, offset=offset_vector)

        # Compute rotated output point
        output_pos = self.__get_rotated_pos__(initial_position, [length, 0], angle)
        # Compute feedback point
        feedback_pos = self.__get_rotated_pos__(initial_position, [length/2, -height/2], angle)
        # Add element position to the dictionary
        self.__add_element_position__(input_pos=[x_in,y_in], output_pos=output_pos,
                                      feedback_pos=feedback_pos)
        return output_pos

    def __draw_arrow__(self, initial_position, length, text=None, 
                       text_position = 'above', text_offset=0.2, arrow = True,
                       fontsize=14, orientation='horizontal'):
        """
        Inner method.
        Draws a horizontal arrow with optional label.

        Args:
            initial_position (Numpy.NDarray or list): Coordinates of the starting point of the arrow.
            length (float, optional): Horizontal length of the block. If not entered, default `block_length` is used.
            text (string, optional): Label to display in the block.
            text_position (string, optional): Position of the optional text: {'before', 'after', 'above'}
            text_offset (float, optional): Vertical offset for the text position.
            arrow (bool, optional): Indicated if an line mush finish or not in an arrow.
            fontsize (int, optional): font size of the text inside the block. If not entered, default `fontsize` is used.
            orientation (string or float, optional): Direction of the block: {'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle}.

        Returns:
            (Numpy.NDArray): Coordinates of output point of the arrow.
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

        # Apply rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData   


        self.ax.add_patch(FancyArrow(x_in, y_in, length, 0, width=0.01,
                                length_includes_head=True, head_width=head_width, 
                                color='black', transform=trans))

        # Don't rotate text if orientation is vertical, down or up
        rotate_text = False if orientation in ['vertical', 'down', 'up', 'left'] else True

        if text:
            # Calculate offset vector based on orientation in non-rotated coordinates
            if text_position == 'before':
                ha, va = 'right', 'center'
                offset_vector = np.array([-text_offset, 0])
                if orientation in ['vertical', 'down']:
                    ha = 'center'
                    va = 'bottom'
                elif orientation in ['up']:
                    ha = 'center'
                    va = 'top'
            elif text_position == 'after':
                ha, va = 'left', 'center'
                offset_vector = np.array([length + text_offset, 0])
                if orientation in ['vertical', 'down']:
                    ha = 'center'
                    va = 'top'
                elif orientation in ['up']:
                    ha = 'center'
                    va = 'bottom'
            elif text_position == 'above':
                ha, va = 'center', 'bottom'
                offset_vector = np.array([length / 2, text_offset])
                if orientation in ['vertical', 'down',]:
                    ha = 'left'
                    va = 'bottom'
                elif orientation in ['up']:
                    ha = 'right'
                    va = 'top'
            else:
                raise ValueError(f"Unknown text_position: {text_position}")

            self.__draw_rotated_text__(initial_position, text, 
                                       angle=angle, rotate_text=rotate_text,
                                       ha=ha, va=va, offset=offset_vector,
                                       fontsize=fontsize)
        
        # Compute rotated output point
        output_pos = self.__get_rotated_pos__(initial_position, [length, 0], angle)
        # Compute feedback point
        feedback_pos = self.__get_rotated_pos__(initial_position, [length/2, 0], angle)
        # Add element position to the dictionary
        self.__add_element_position__(input_pos=[x_in,y_in], output_pos=output_pos,
                                      feedback_pos=feedback_pos)
        return output_pos

    def __draw_angled_arrow__(self, initial_position, final_position, 
                            text=None, text_offset=0.2, arrow = True, fontsize=14,
                            first_segment='horizontal', orientation='horizontal'):
        """
        Inner method.
        Draws a right-angled arrow composed of two segments, with a specified first segment orientation and optional label.

        Args:
            initial_position (Numpy.NDarray or list): Coordinates of the starting point of the arrow.
            final_position (Numpy.NDarray or list): Coordinates of the ending point of the arrow.
            text (string, optional): Label to display in the block.
            text_offset (float, optional): Vertical offset for the text position.
            arrow (bool, optional): Indicates if it must finish or not in an arrow.
            fontsize (int, optional): font size of the text inside the block. If not entered, default `fontsize` is used.
            first_segment (string, optional): Drawing order: {'horizontal', 'vertical'}
            orientation (string or float, optional): Direction of the block: {'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle}.

        Returns:
            (Numpy.NDArray): Coordinates of output point of the arrow.
        """
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
        x_out, y_out = final_position
        dx = x_out - x_in
        dy = y_out - y_in

        # Apply rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData   

        if first_segment == 'horizontal':
            corner = (x_out, y_in)
        elif first_segment == 'vertical':
            corner = (x_in, y_out)
        else:
            raise ValueError("first_segment must be either 'horizontal' or 'vertical'")

        # Draw segments
        if first_segment == 'horizontal':
            if dx != 0:
                self.ax.add_patch(FancyArrow(x_in, y_in, dx, 0, width=0.01,
                        length_includes_head=True, head_width=0, 
                        color='black', transform=trans))
            if dy != 0:
                self.ax.add_patch(FancyArrow(corner[0], corner[1], 0, dy, width=0.01,
                        length_includes_head=True, head_width=head_width, 
                        color='black', transform=trans))
        else:  # first vertical
            if dy != 0:
                self.ax.add_patch(FancyArrow(x_in, y_in, 0, dy, width=0.01,
                        length_includes_head=True, head_width=0, 
                        color='black', transform=trans))
            if dx != 0:
                self.ax.add_patch(FancyArrow(corner[0], corner[1], dx, 0, width=0.01,
                        length_includes_head=True, head_width=head_width, 
                        color='black', transform=trans))

        # Don't rotate text if orientation is vertical, down or up
        rotate_text = False if orientation in ['vertical', 'down', 'up', 'left'] else True

        # Optional text near the corner
        if text:
            # Calculate offset vector based on orientation in non-rotated coordinates
            if first_segment == 'horizontal':
                offset_vector = np.array([dx/2, text_offset])    
            else: # first vertical
                offset_vector = np.array([dx/2, dy + text_offset])    

            self.__draw_rotated_text__(initial_position, text, 
                                       angle=angle, rotate_text=rotate_text,
                                       ha='center', va='bottom', offset=offset_vector,
                                       fontsize=fontsize)

        # Compute rotated output point
        output_pos = self.__get_rotated_pos__(final_position, [0, 0], angle)
        # Compute feedback point
        feedback_pos = self.__get_rotated_pos__(corner, [0, 0], angle)
        # Save element position
        self.__add_element_position__(input_pos=initial_position, output_pos=output_pos, feedback_pos=feedback_pos)

        return output_pos

    def __draw_combiner__(self, initial_position, height=1,
                        input_text=None, input_side='bottom', operation='mult', 
                        text_offset=0.1, signs=[None, None], fontsize=14, orientation='horizontal'):
        """
        Inner method.
        Draws a combiner block: a circle with a multiplication sign (×), sum sign (+) 
        or substraction sign (-) inside, with optional signs on each input.

        Args:
            initial_position (Numpy.NDarray or list): Coordinates of the starting point of the arrow.
            height (float, optional): Vertical height of the block. If not entered, default `block_height` is used.
            input_text (string, optional): Label for the input arrow (below or above the arrow).
            input_side (string, optional): Side of the lateral input: {'bottom', 'top'}.
            operation (string, optional): Operation of the combiner: {'mult', 'sum', 'dif'}.
            text_offset (float, optional): Vertical offset for the text position.
            signs (list, optional): Sign to be shown on the horizontal (signs[0]) and vertical (signs[1]) inputs.
            fontsize (int, optional): font size of the text inside the block. If not entered, default `fontsize` is used.
            orientation (string or float, optional): Direction of the block: {'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle}.

        Returns:
            (Numpy.NDArray): Coordinates of output point of the combiner.
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
                         linewidth=2, transform=trans, zorder=3)
        else:
            raise ValueError(f"Unknown operation: {operation}. 'operation' must be 'mult', 'sum' or 'dif'.")

        # Don't rotate text if orientation is vertical, down or up
        rotate_text = False if orientation in ['vertical', 'down', 'up', 'left'] else True

        # Side input
        if input_side == 'bottom':
            arrow_height = height - radius
            y_init = y_in - radius - arrow_height
            offset_vector = np.array([radius, - (height + text_offset)])
            va = 'top'
            ha = 'center'
            if orientation in ['vertical', 'down']:
                ha = 'right'
                va = 'center'
            elif orientation in ['up']:
                ha = 'left'
                va = 'center'
        elif input_side == 'top':
            arrow_height = - (height - radius)
            y_init = y_in + radius - arrow_height
            offset_vector = np.array([radius, height + text_offset])
            va = 'bottom'
            ha = 'center'
            if orientation in ['vertical', 'down']:
                ha = 'left'
                va = 'center'
            elif orientation in ['up']:
                ha = 'right'
                va = 'center'
        else:
            raise ValueError(f"Unknown input_side: {input_side}. 'input_side' must be 'bottom' or 'top'.")

        # Show signs on each input if not None
        if signs[0] is not None:
            self.__draw_rotated_text__(initial_position, signs[0], 
                                    angle=angle, rotate_text=rotate_text,
                                    ha=ha, va=va, 
                                    fontsize=fontsize, offset=[-radius, 1.5*radius])
        if signs[1] is not None:
            self.__draw_rotated_text__(initial_position, signs[1], 
                                    angle=angle, rotate_text=rotate_text,
                                    ha=ha, va=va, 
                                    fontsize=fontsize, offset=[0, -1.5*radius])

        self.ax.add_patch(FancyArrow(cx, y_init, 0, arrow_height, width=0.01,
                                length_includes_head=True, head_width=0.15, 
                                color='black', transform=trans))
        if input_text is not None:
            self.__draw_rotated_text__(initial_position, input_text, 
                                       angle=angle, rotate_text=rotate_text,
                                       ha=ha, va=va, 
                                       fontsize=fontsize, offset=offset_vector)
        
        # Compute rotated output point
        output_pos = self.__get_rotated_pos__(initial_position, [2 * radius, 0], angle)
        # Compute feedback point
        feedback_pos = self.__get_rotated_pos__(initial_position, [radius, y_init - y_in + arrow_height], angle)
        # Add element position to the dictionary
        self.__add_element_position__(input_pos=[x_in,y_in], output_pos=output_pos,
                                      feedback_pos=feedback_pos)
        return output_pos

    def __draw_mult_combiner__(self, initial_position, length, inputs, 
                               operation='sum', orientation='horizontal'):
        """
        Inner method.
        Draws a summation or multiplication block with multiple inputs distributed 
        along the left edge of a circle, from pi/2 to 3*pi/2. Inputs can have a sign.

        Args:
            initial_position (Numpy.NDarray or list): Coordinates of the starting point of the arrow.
            length (float, optional): Horizontal length of the block. If not entered, default `block_length` is used.
            inputs (list of str): Thread names to combine.
            operation (string, optional): Operation of the combiner: {'mult', 'sum'}.
            orientation (string or float, optional): Direction of the block: {'horizontal', 'vertical', 'up', 'down', 'left', 'right', angle}.

        Returns:
            (Numpy.NDArray): Coordinates of output point of the combiner.
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
            initial_position = [x_in, y_in]
        # If position is given, use it
        else:
            x_in, y_in = initial_position
        
        radius = length / 4
        cx = x_in + length - radius
        cy = y_in

        # Apply rotation around the connection point (x_ini, y_ini)
        trans = transforms.Affine2D().rotate_deg_around(x_in, y_in, angle) + self.ax.transData  

        # Circle
        circle = plt.Circle((cx, cy), radius, edgecolor='black', 
                            facecolor='white', transform=trans, zorder=2)
        self.ax.add_patch(circle)

        # Draw symbol inside circle depending on operation
        rel_size = 0.7
        if operation == 'mult':
            # "X" inside circle
            dx = radius * rel_size * np.cos(np.pi / 4)  # Escalamos un poco para que quepa dentro del círculo
            dy = radius * rel_size * np.sin(np.pi / 4)
            #  45° line
            self.ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
            # 135° line
            self.ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color='black', 
                         linewidth=2, transform=trans, zorder=3)
        elif operation == 'sum':
            dx = radius * rel_size
            dy = radius * rel_size
            # "+" inside circle
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

        # Input arrows
        for i, inp in enumerate(thread_input_pos):
            xi, yi = inp[:2]

            x_edge = cx + radius * np.cos(angles[i])
            y_edge = cy + radius * np.sin(angles[i])

            dx = x_edge - x_in
            dy = y_edge - y_in
            offset_vec = [dx, dy]

            # Rotated offset vector with respect to initial_position of element
            dx_rot, dy_rot = rot_matrix @ offset_vec
            # Rotated offset vector with respect to initial position of arrow
            dx_rot_rel = dx_rot - xi + x_in
            dy_rot_rel = dy_rot - yi + y_in

            self.ax.add_patch(FancyArrow(
                xi, yi, dx_rot_rel, dy_rot_rel,
                width=arrow_width,
                length_includes_head=True,
                head_width=arrow_head_width,
                color='black', transform=self.ax.transData, zorder=1
            ))

        # Compute rotated output point
        output_pos = self.__get_rotated_pos__(initial_position, [length, 0], angle)
        # Compute feedback point
        feedback_pos = self.__get_rotated_pos__(initial_position, [length - radius, -radius], angle)
        # Add element position to the dictionary
        self.__add_element_position__(input_pos=[x_in,y_in], output_pos=output_pos,
                                      feedback_pos=feedback_pos)
        return output_pos

    def add(self,name, kind='block', thread='main', position=None, debug=False, **kwargs):
        """
        Adds an element to the block diagram at the current or specified position of a given thread.

        This is the main interface for constructing diagrams by adding components such as blocks, arrows,
        inputs, outputs, combiners, and connectors. The `kind` parameter determines the type of element,
        and each type accepts specific keyword arguments listed below.

        Args:
            name (str): Main label or identifier for the element.
            kind (str, optional): Type of element. One of:
                - 'block': Rectangular block.
                - 'arrow': Straight line with ending arrow.
                - 'angled_arrow': Rect angle line with or without ending arrow.
                - 'input': Arrow with text before it.
                - 'output': Arrow with text after it.
                - 'line': Straight line without arrow ending.
                - 'combiner': Circle with (x), (+) or (-) and additional input.
                - 'mult_combiner': Combiner with multiple inputs.
            thread (str, optional): Thread identifier.
            position (tuple or str or None, optional): (x, y) position, 'auto' (for mult_combiner), or None to use current thread position.
            debug (bool, optional): If True, prints thread positions after placing the element.

        The `**kwargs` vary depending on the `kind`:

        - **kind = 'block'**:
            - text (str, optional): Label inside the block (defaults to `name`).
            - text_above (str, optional): Text above the block.
            - text_below (str, optional): Text below the block.
            - text_offset (float, optional): Offset for above/below text (defaults to 0.1).
            - input_text (str, optional): Label for input arrow.
            - input_side (str, optional): Side of a second optional input: {'top', 'bottom'} (defaults to `None`)
            - length (float, optional): Block length (defaults to `self.block_length`).
            - height (float, optional): Block height (defaults to `self.block_height`)..
            - linestyle (str, optional): Block border line style (defaults to `-`).
            - orientation (str or float, optional): Orientation of the block: {'horizontal', 'vertical', 'up', 'down' 'right', left' or angle in degrees} (defaults to 'horizontal').
            - fontsize (int, optional): Text font size (defaults to `self.fontsize`).

        - **kind = 'arrow'**, **'input'**, **'output'** or **'line'**:
            - text (str, optional): Text on the arrow or line (defaults to `name`).
            - text_position (str, optional): 'above', 'below', 'before', or 'after' (defaults to 'above' for 'arrow' and 'line', 'before' for 'input', and 'after' for 'output').
            - text_offset (float, optional): Offset for text (defaults to 0.1).
            - length (float, optional): Arrow or line length (defaults to `self.block_length`).
            - orientation (str or float, optional): Orientation of the block: {'horizontal', 'vertical', 'up', 'down' 'right', left' or angle in degrees} (defaults to 'horizontal').
            - fontsize (int, optional): Text font size (defaults to `self.fontsize`).

        - **kind = 'angled_arrow'**:
            - text (str, optional): Text on the arrow or line (defaults to `name`).
            - final_pos (Numpy.NDarray or list): Coordinates of the ending point of the arrow.
            - text_position (str, optional): 'above', 'below', 'before', or 'after' (defaults to 'above' for 'arrow' and 'line', 'before' for 'input', and 'after' for 'output').
            - text_offset (float, optional): Offset for text (defaults to 0.1).
            - arrow (bool, optional): Indicates if it must finish or not in an arrow.
            - first_segment (string, optional): Drawing order: {'horizontal', 'vertical'}
            - orientation (str or float, optional): Orientation of the block: {'horizontal', 'vertical', 'up', 'down' 'right', left' or angle in degrees} (defaults to 'horizontal').
            - fontsize (int, optional): Text font size (defaults to `self.fontsize`).

        - **kind = 'combiner'**:
            - operation (string, optional): Operation of the combiner: {'mult', 'sum', 'dif'} (defaultds to 'mult').
            - height (float, optional): Vertical height of the block. (defaults to `self.block_height`).
            - input_side (string, optional): Side of the lateral input: {'bottom', 'top'} (defaults to 'bottom').
            - input_text (string, optional): Label for the input arrow (below or above the arrow).
            - text_offset (float, optional): Offset for text (defaults to 0.1).
            - signs (list, optional): Sign to be shown on the horizontal (signs[0]) and vertical (signs[1]) inputs.
            - orientation (str or float, optional): Orientation of the block: {'horizontal', 'vertical', 'up', 'down' 'right', left' or angle in degrees} (defaults to 'horizontal').
            - fontsize (int, optional): Text font size (defaults to `self.fontsize`).

        - **kind = 'mult_combiner'**:
            - operation (string, optional): Operation of the combiner: {'mult', 'sum', 'dif'} (defaults to 'mult').
            - length (float, optional): Total element length (defaults to `self.block_length`).
            - inputs (list of str): Thread names to combine.
            - operation (string, optional): Operation of the combiner: {'mult', 'sum'} (defaults to 'sum').
            - orientation (str or float, optional): Orientation of the block: {'horizontal', 'vertical', 'up', 'down' 'right', left' or angle in degrees} (defaults to 'horizontal').
            - fontsize (int, optional): Text font size (defaults to `self.fontsize`).

        Examples:
            >>> db = DiagramBuilder()
            >>> db.add("x(t)", kind="input")
            >>> db.add("H(s)", kind="block")
            >>> db.add("y(t)", kind="output")
        """

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

        elif kind == 'angled_arrow':
            # Default arguments
            default_kwargs = {
                'text': name,
                'text_offset': 0.1,
                'fontsize': self.fontsize,
                'orientation': 'horizontal',
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_angled_arrow__(initial_pos, **block_args)

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
                'length': self.block_length,
                'operation': 'mult',
                'orientation': 'horizontal'
            }
            # Overrides default arguments with provided ones
            block_args = {**default_kwargs, **kwargs}
            # Function call
            final_pos = self.__draw_mult_combiner__(initial_pos, **block_args)


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

    def get_current_element(self):
        """
        Returns the current element index (last added).

        Returns:
            (int): Index of the last added element.
        """
        return self.current_element
    
    def get_position(self, element=None):
        """
        Returns the positions of the specified element index. If no element specified, last added element is used.
        The return is a dictinoary with coordinates of `input_pos`, `output_pos` and `feedback_pos` (feedback port coordinates).
        
        Args:
            element (int, optional): index of the element.
        
        Returns:
            (dict of Tuples): Dictionary with three 2-element tuples: `input_pos`, `output_pos` and `feedback_pos`.
        """
        if element is None:
            return self.element_positions[self.current_element]
        elif element <= self.current_element:
            return self.element_positions[element]
        else:
            raise ValueError(f"Element '{element}' not found.")

    def get_thread_position(self, thread='main'):
        """
        Returns the current output position of the specified thread.

        Args:
            thread (str, optional): Thread identifier.
        """
        if thread in self.thread_positions:
            return self.thread_positions[thread]
        else:
            raise ValueError(f"Thread '{thread}' not found.")

        
    def show(self, margin=0.5, scale=1.0, savepath=None):
        """
        Displays the current diagram or saves it to a file.

        Adjusts the view to fit the full diagram with an optional margin and scaling factor.
        If no elements have been drawn, simply displays an empty figure.

        Args:
            margin (float, optional): Margin to add around the diagram (in data units).
            scale (float, optional): Scaling factor for the figure size.
            savepath (str, optional): If provided, saves the figure to the specified path (e.g., 'diagram.png' or 'diagram.pdf').
                                      If None, the diagram is shown in an interactive window.

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

        fig_width = width * scale
        fig_height = height * scale
        self.fig.set_size_inches(fig_width, fig_height)

        self.ax.set_xlim(x0, x1)
        self.ax.set_ylim(y0, y1)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_position([0, 0, 1, 1])
        self.ax.axis("off")

        if savepath:
            self.fig.savefig(savepath, bbox_inches='tight', dpi=self.fig.dpi, transparent=False, facecolor='white')
            print(f"Saved in: {savepath}")
        else:
            plt.show()

