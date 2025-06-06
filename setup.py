from pathlib import Path
from setuptools import setup, find_packages

# Reads README for long description
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name='BlockDiagrams',
    version='1.4.1',
    author='Miguel Á. Martín-Fernández',
    author_email='migmar@uva.es',
    description='Lightweight library for drawing signal processing block diagrams with any orientation and several branches using Matplotlib',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/miguelmartfern/BlockDiagrams',
    packages=find_packages(include=['blockdiagrams', 'blockdiagrams.*']),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib>=3.0.0',
        'numpy>=1.0.0',
    ],
)
