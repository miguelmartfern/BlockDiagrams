from pathlib import Path
from setuptools import setup, find_packages

# Reads README for long description
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name="blockdiagrams",
    version="1.3.1",
    description="Library to draw block diagrams with Matplotlib",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Miguel Á. Martín-Fernández",
    url="https://github.com/miguelmartfern/blockdiagrams",
    license="MIT",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "matplotlib>=3.5",
        "numpy>=1.22",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
