#!/usr/bin/env python3

from pathlib import Path

from setuptools import setup


def version():
    """
    Read the VERSION file and return the version number of the code.

    Returns
    -------
    str
        Version number of the code.
    """
    with open('VERSION', 'r') as f:
        return f.read().rstrip()


def read(file_name):
    """
    Read a file in the current directory and return its content.

    Parameters
    ----------
    file_name : str
        Name of the file.

    Returns
    -------
    str
        Content of the file.
    """
    current_dir = Path(__file__).parent
    return open(Path(current_dir, file_name)).read()


setup(
    name='Algoz',
    version=version(),
    author='Matthieu Nugue',
    author_email='matthieu.nugue@nanoz-group.com',
    description=(
        'Nanoz tools to develop algorithms for gases sensor.'
    ),
    license='Apache 2.0',
    keywords='nanoz algorithm gas',
    url='http://nanoz-group.eu/',
    long_description=read('README.md'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: GPU :: NVIDIA CUDA :: 11.1',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development'
    ],
    packages=['nanoz', 'tests'],
    python_requires='>=3.8',
    dependency_links=[
        "https://download.pytorch.org/whl/torch_stable.html",
    ],
    install_requires=[
        'pynvml>=11.4.1',
        'numpy>=1.22.4',
        'pandas>=1.4.2',
        'librosa>=0.8.1',
        'plotly>=5.14.1',
        'scikit-learn>=1.1.1',
        'torch==1.9.0+cu111',
        'torchvision==0.10.0+cu111',
        'torchaudio==0.9.0',
        'skorch>=0.11.0',
    ]
)
