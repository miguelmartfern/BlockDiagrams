from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name='SignalBlocks',
    version='2.0.0',
    author='Miguel Á. Martín-Fernández',
    author_email='migmar@uva.es',
    description='A complete symbolic and graphical framework for block diagrams, signals transformations and plotting, and Z-transform ROC visualization.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/miguelmartfern/SignalBlocks',
    project_urls={
        'Documentation': 'https://miguelmartfern.github.io/SignalBlocks/',
        'Source': 'https://github.com/miguelmartfern/SignalBlocks',
        'Bug Tracker': 'https://github.com/miguelmartfern/SignalBlocks/issues',
    },
    packages=find_packages(include=['signalblocks', 'signalblocks.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib>=3.5',
        'numpy<1.25',
        'scipy>=1.7',
        'sympy>=1.10',
    ],
    license='GPL-3.0-or-later',
)