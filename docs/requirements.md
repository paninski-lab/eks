# Requirements

This document explains the purpose of each package listed in the `install_requires` and `extras_require` sections of the `setup.py` file for the `ensemble-kalman-smoother` project.

## Basic Requirements

These are the core dependencies required for the project to function properly, and can be installed via:
```
pip install -e .
```

- **ipykernel**: Provides the IPython kernel for Jupyter, allowing the execution of Python code in Jupyter notebooks.
- **matplotlib**: A comprehensive library for creating static, animated, and interactive visualizations in Python.
- **numpy**: The fundamental package for scientific computing with Python, providing support for arrays, matrices, and many mathematical functions.
- **opencv-python**: A library of programming functions mainly aimed at real-time computer vision. It allows for image and video capture and processing.
- **pandas**: A powerful data analysis and manipulation library for Python, providing data structures like DataFrames.
- **scikit-learn**: A machine learning library for Python, offering simple and efficient tools for data mining and data analysis.
- **scipy (>=1.2.0)**: A library used for scientific and technical computing, building on the capabilities of numpy and providing additional functionality.
- **tqdm**: A fast, extensible progress bar for Python and CLI, useful for tracking the progress of loops and processes.
- **typing**: Provides support for type hints, making it easier to write and maintain Python code by specifying expected types of variables.
- **sleap_io**: A library for reading and writing SLEAP (Single Leap Application Protocol) files, which are used for pose estimation in biological research.
- **jax**: A library for high-performance numerical computing, offering support for automatic differentiation and optimized operations on CPUs and GPUs.
- **jaxlib**: A companion library to jax, providing implementations of numerical operations on various hardware platforms.

## Additional Requirements (for devs)

These are optional dependencies used for development and documentation purposes, and can be installed via:
```
pip install -e ".[dev]"
```

- **flake8**: A linting tool for Python that checks the code for style and quality issues, ensuring adherence to coding standards.
- **isort**: A tool to sort imports in Python files, organizing them according to the PEP8 style guide.
- **Sphinx**: A documentation generator for Python projects, converting reStructuredText files into various output formats such as HTML and PDF.
- **sphinx_rtd_theme**: The theme for Sphinx documentation, used to create documentation that looks similar to Read the Docs.
- **sphinx-rtd-dark-mode**: An extension for Sphinx to add dark mode support to the Read the Docs theme.
- **sphinx-automodapi**: A Sphinx extension that helps generate documentation from docstrings in the code.
- **sphinx-copybutton**: A Sphinx extension that adds a copy button to code blocks in the documentation.
- **sphinx-design**: A Sphinx extension that adds design elements such as cards, grids, and buttons to the documentation.