import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrow
from matplotlib.transforms import Bbox

# --- Drawing functions ---

def draw_block(ax, position, text, length=1.5, height=1, fontsize=14):
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
    x, y = position
    ax.add_patch(Rectangle((x, y - height / 2), length, height, edgecolor='black', facecolor='white'))
    ax.text(x + length / 2, y, f"${text}$", ha='center', va='center', fontsize=fontsize)

def draw_block_uparrow(ax, position, text, input_bottom_text, length=1.5, height=1, text_offset=0.1, fontsize=14):
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
    x0, y = position
    x1 = x0 + length
    cx = (x0 + x1) / 2

    ax.add_patch(Rectangle((x0, y - height / 2), length, height, edgecolor='black', facecolor='white'))
    ax.text(x0 + length / 2, y, f"${text}$", ha='center', va='center', fontsize=fontsize)

    ax.add_patch(FancyArrow(cx, y - 1.25 * length, 0, 1.25 * length - height / 2, width=0.01,
                            length_includes_head=True, head_width=0.15, color='black'))
    if input_bottom_text:
        ax.text(cx, y - 1.25 * length - text_offset, f"${input_bottom_text}$",
                ha='center', va='top', fontsize=fontsize)

def draw_arrow(ax, position, length, text=None, arrow = True, text_offset=(0, 0.2), fontsize=14):
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
    end = (position[0] + length, position[1])
    head_width = 0.15 if arrow else 0

    ax.add_patch(FancyArrow(position[0], position[1], length, 0, width=0.01,
                            length_includes_head=True, head_width=head_width, color='black'))
    if text:
        tx = position[0] + length / 2 + text_offset[0]
        ty = position[1] + text_offset[1]
        ax.text(tx, ty, f"${text}$", ha='center', fontsize=fontsize)

def draw_io_arrow(ax, position, length=1, text="", io='input', text_offset=0.3, fontsize=14):
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
    x, y = position
    absolute_text_offset = text_offset * length
    ax.add_patch(FancyArrow(x, y, length, 0, width=0.01,
                            length_includes_head=True, head_width=0.15, color='black'))
    if text:
        tx = x - absolute_text_offset if io == 'input' else x + length + absolute_text_offset
        ty = y
        ax.text(tx, ty, f"${text}$", ha='center', va='center', fontsize=fontsize)

def draw_2combiner(ax, position, length,
                    input_left_text=None, input_bottom_text=None, output_text=None,
                    operation='mult', text_offset=0.1, fontsize=14):
    """
    Draws a combiner block: a circle with a multiplication sign (×), sum sign (+) 
    or substraction sign (-) inside,
    with input arrows from the left and bottom, and an output arrow to the right.

    Parameters:
    - ax: matplotlib Axes object where the diagram is drawn.
    - position: (x, y) coordinates of the start of the left input arrow.
    - length: total horizontal length from input to output.
    - input_left_text: label for the left input arrow (above the arrow).
    - input_bottom_text: label for the bottom input arrow (below the arrow).
    - output_text: label for the right output arrow (above the arrow).
    - operation: 'mult' for multiplication sign (×), 'sum' for addition sign (+), 'dif' for substraction sign (-).
    - text_offset: vertical offset for input/output labels.
    - fontsize: font size of the labels.
    """
    x0, y = position
    x1 = x0 + length
    cx = (x0 + x1) / 2
    radius = length / 10

    circle = plt.Circle((cx, y), radius, edgecolor='black', facecolor='white', zorder=2)
    ax.add_patch(circle)

    rel_size = 0.7
    if operation == 'mult':
        # Líneas diagonales (forma de "X") dentro del círculo
        dx = radius * rel_size * np.cos(np.pi / 4)  # Escalamos un poco para que quepa dentro del círculo
        dy = radius * rel_size * np.sin(np.pi / 4)

        # Línea de 45°
        ax.plot([cx - dx, cx + dx], [y - dy, y + dy], color='black', linewidth=2, zorder=3)

        # Línea de 135°
        ax.plot([cx - dx, cx + dx], [y + dy, y - dy], color='black', linewidth=2, zorder=3)
    elif operation == 'sum':
        # Líneas horizontales y verticales (forma de "+") dentro del círculo
        ax.plot([cx - radius * rel_size, cx + radius * rel_size], [y, y], color='black', linewidth=2, zorder=3)
        ax.plot([cx, cx], [y - radius * rel_size, y + radius * rel_size], color='black', linewidth=2, zorder=3)
    elif operation == 'dif':
        # Línea horizontal (forma de "-") dentro del círculo
        ax.plot([cx - radius * rel_size, cx + radius * rel_size], [y, y], color='black', linewidth=2, zorder=3)
    else:
        raise ValueError(f"Unknown operation: {operation}. 'operation' must be 'mult', 'sum' or 'dif'.")

    # Left input
    ax.add_patch(FancyArrow(x0, y, cx - x0 - radius, 0, width=0.01,
                            length_includes_head=True, head_width=0.15, color='black'))
    if input_left_text:
        ax.text(x0 + (cx - x0 - radius) / 2, y + text_offset, f"${input_left_text}$",
                ha='center', va='bottom', fontsize=fontsize)

    # Bottom input
    ax.add_patch(FancyArrow(cx, y - (cx - x0), 0, cx - x0 - radius, width=0.01,
                            length_includes_head=True, head_width=0.15, color='black'))
    if input_bottom_text:
        ax.text(cx, y - (cx - x0) - text_offset, f"${input_bottom_text}$",
                ha='center', va='top', fontsize=fontsize)

    # Right output
    ax.add_patch(FancyArrow(cx + radius, y, x1 - cx - radius, 0, width=0.01,
                            length_includes_head=True, head_width=0.15, color='black'))
    if output_text:
        ax.text(x1 - (x1 - cx - radius) / 2, y + text_offset, f"${output_text}$",
                ha='center', va='bottom', fontsize=fontsize)

def draw_mult_combiner(ax, position, length, inputs, output_text=None, operation='sum', fontsize=14):
    """
    Dibuja un sumador o multiplicador con múltiples entradas distribuidas desde pi/2 a 3*pi/2
    a lo largo del borde izquierdo de un círculo. Las entradas pueden tener signo.

    Parámetros:
    - inputs: lista de tuplas (x, y) o (x, y, signo)
    - output_text: etiqueta en la flecha de salida (a la derecha)
    - position: centro del sumador (x0, y0)
    - operation: 'sum' para suma, 'mult' para producto

    """
    
    if position == "auto":
        inputs = np.array(inputs)
        x0 = np.max(inputs[:, 0])
        y0 = np.mean(inputs[:,1])
    else:
        x0, y0 = position
    
    x1 = x0 + length
    cx = (x0 + x1) / 2
    cy = y0
    radius = length / 10
    
    # Círculo
    circle = plt.Circle((cx, cy), radius, edgecolor='black', facecolor='white', linewidth=1.5, zorder=2)
    ax.add_patch(circle)

    # Dibujar símbolo dentro del círculo según operación
    rel_size = 0.7
    if operation == 'mult':
        dx = radius * rel_size * np.cos(np.pi / 4)
        dy = radius * rel_size * np.sin(np.pi / 4)
        ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], color='black', linewidth=2, zorder=3)
        ax.plot([cx - dx, cx + dx], [cy + dy, cy - dy], color='black', linewidth=2, zorder=3)
    elif operation == 'sum':
        ax.plot([cx - radius * rel_size, cx + radius * rel_size], [cy, cy], color='black', linewidth=2, zorder=3)
        ax.plot([cx, cx], [cy - radius * rel_size, cy + radius * rel_size], color='black', linewidth=2, zorder=3)
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'sum' or 'mult'.")

    n = len(inputs)
    angles = np.linspace(5* np.pi / 8, 11 * np.pi / 8, n)

    arrow_width = 0.01
    arrow_head_width = 0.15

    # Flechas de entrada
    for i, inp in enumerate(inputs):
        xi, yi = inp[:2]

        x_edge = cx + radius * np.cos(angles[i])
        y_edge = cy + radius * np.sin(angles[i])

        dx = x_edge - xi
        dy = y_edge - yi

        ax.add_patch(FancyArrow(
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

    ax.add_patch(FancyArrow(
        x1, cy, x2 - x1, 0,
        width=arrow_width,
        length_includes_head=True,
        head_width=arrow_head_width,
        color='black',
        zorder=1
    ))

    # Texto de salida
    if output_text:
        ax.text(
            (x1 + x2) / 2, cy + arrow_head_width / 2,
            f"${output_text}$",
            ha='center', va='bottom', fontsize=fontsize
        )

    return [x0, y0]


# --- DiagramBuilder class ---

class DiagramBuilder:
    """
    Helper class for creating signal processing diagrams step by step.
    
    Keeps track of horizontal position and allows adding components
    like blocks, arrows, multipliers, and input/output signals in order.
    """
    def __init__(self, figsize=(12, 3), block_length=1.0, fontsize=20):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        # self.ax.set_xlim(*xlim)
        # self.ax.set_ylim(*ylim)
        self.ax.axis('off')  # Hide axes
        self.fontsize = fontsize
        self.block_length = block_length
        # self.spacing = spacing
        self.current_x = {}
        self.current_x['main'] = 0
        self.y = 0
        self.positions = {}
        self.xlim = [0,0]
        self.ylim = [0,0]

    def add(self,name, kind='block', position=(0,0), **kwargs):
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
        pos = list(position)

        if kind == 'input':
            length=kwargs.get('length', self.block_length)
            draw_io_arrow(self.ax, position, length=length, text=kwargs.get('text', name),
                          io='input', fontsize=self.fontsize)

        elif kind == 'arrow':
            length=kwargs.get('length', self.block_length)
            draw_arrow(self.ax, position, length=length,
                       text=kwargs.get('text', name), arrow = True, fontsize=self.fontsize)

        elif kind == 'line':
            length=kwargs.get('length', self.block_length)
            draw_arrow(self.ax, position, length=length,
                       text=kwargs.get('text', name), arrow = False, fontsize=self.fontsize)

        elif kind == 'block':
            length=kwargs.get('length', self.block_length)
            draw_block(self.ax, position, text=kwargs.get('text', name),
                       length=length, height=height, fontsize=self.fontsize)

        elif kind == 'block_uparrow':
            length=kwargs.get('length', self.block_length)
            draw_block_uparrow(self.ax, position, text=kwargs.get('text', name),
                               input_bottom_text=kwargs.get('input_bottom_text'),
                               length=length, height=height, fontsize=self.fontsize)

        elif kind == '2combiner':
            length=kwargs.get('length', self.block_length*2.5)
            draw_2combiner(self.ax, position, length=length,
                            input_left_text=kwargs.get('input_left_text'),
                            input_bottom_text=kwargs.get('input_bottom_text'),
                            output_text=kwargs.get('output_text'), 
                            operation=kwargs.get('operation'), fontsize=self.fontsize)

        elif kind == 'mult_combiner':
            length=kwargs.get('length', self.block_length*2.5)
            pos = draw_mult_combiner(self.ax, position, length=length, inputs=kwargs.get('inputs'),
                            output_text=kwargs.get('text', name), 
                            operation=kwargs.get('operation'), fontsize=self.fontsize)

        elif kind == 'output':
            length=kwargs.get('length', self.block_length)
            draw_io_arrow(self.ax, position, length=length, text=kwargs.get('text', name),
                          io='output', fontsize=self.fontsize)

        else:
            raise ValueError(f"Unknown block type: {kind}")

        new_pos = [pos[0] + length, pos[1]]

        # Return end position
        return new_pos
    

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

