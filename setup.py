from pathlib import Path
from setuptools import setup, find_packages

# Reads README for long description
this_dir = Path(__file__).parent
long_description = (this_dir / "README.md").read_text(encoding="utf-8")

setup(
    name='BlockDiagrams',
    version='1.4.1',
    author='Miguel Martín Fernández',
    author_email='miguelmartfern@gmail.com',
    description='Librería ligera para dibujar diagramas de bloques en Python utilizando Matplotlib.',
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
