
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
from blockdiag import DiagramBuilder

db = DiagramBuilder(block_length=1, fontsize=16)

# Dibujo del diagrama
pos = db.add("x(t)", kind="input", position=(0, 0))
pos = db.add("h_{aa}(t)", kind="block", position = pos)
pos = db.add("mult", kind="2combiner", input_left_text="x_c(t)", input_bottom_text="p(t)", output_text="x_p(t)", operation='mult', position = pos)
pos = db.add("C/D", kind="block_uparrow",input_bottom_text="T_s", position = pos)
pos = db.add("x_d[n]", kind="arrow", position = pos)
pos = db.add("h_d[n]", kind="block", position = pos)
pos = db.add("y_d[n]", kind="arrow", position = pos)
pos = db.add("D/C", kind="block", position = pos)
pos = db.add("x_p(t)", kind="arrow", position = pos)
pos = db.add("h_r(t)", kind="block", position = pos)
pos = db.add("x_r(t)", kind="output", position = pos)

db.show(savepath = "diag1.pdf")
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
