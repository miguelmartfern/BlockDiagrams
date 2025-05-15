
# blockdiag

**blockdiag** is a lightweight Python library for drawing horizontal block diagrams using Matplotlib. It simplifies the visual creation of system and signal diagrams with functions to add blocks, arrows, summation nodes, and multipliers.

---

## Features

- Draw rectangular blocks with LaTeX math text.
- Add signal arrows with descriptive labels.
- Include summation and multiplication nodes for system diagrams.
- Horizontal orientation (left to right).
- Easy to use and extend for custom diagrams.

---

## Installation

The library is not yet published on PyPI. You can clone the repository directly:

```bash
git clone https://github.com/your_username/blockdiag.git
```

Then simply import the `blockdiag.py` module into your project.

---

## Basic Usage

```python
import matplotlib.pyplot as plt
from blockdiag import DiagramBuilder

builder = DiagramBuilder()

# Add input node
builder.add("input", kind='input', position=(0, 0), text="x")

# Add block
builder.add("block1", kind='block', text="H(z)")

# Add arrow
builder.add("arrow1", kind='arrow', text=r"y[n]")

# Add output node
builder.add("output", kind='output', text="y")

# Display the diagram
builder.show()
```

---

## Main Functions

- `DiagramBuilder.add(name, kind, position=None, **kwargs)`: Adds elements to the diagram.
  - `kind`: `'block'`, `'arrow'`, `'input'`, `'output'`, `'2combiner'`, `'mult_combiner'`.
  - `position`: Coordinates `(x, y)` to place the element.
  - `text`: LaTeX-formatted text to display inside or near the element.

---

## Upcoming Improvements

- Support for vertical orientation.
- Support for multiple tracks or branches in diagrams.
- Improved automatic layout and positioning.

---

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

---

## License

[Specify the license here, e.g., MIT License]

---

## Contact

For questions or suggestions, feel free to contact me via GitHub.

---

Thank you for using **blockdiag**!
