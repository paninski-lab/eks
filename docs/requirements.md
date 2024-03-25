# Project Requirements

This document outlines the Python packages required to run the project.

## Overview

The `requirements.txt` file lists all the necessary dependencies for the project. Below is a breakdown of each package and its purpose:

1. **ipykernel**
   - Description: Provides the IPython kernel for Jupyter notebooks and interactive computing.
   - Usage: This package is used to enable the execution of Python code within Jupyter notebooks.

2. **matplotlib**
   - Description: Provides a MATLAB-like plotting interface for creating static, interactive, and animated visualizations in Python.
   - Usage: This package is used for data visualization and plotting graphs within the project.

3. **numpy**
   - Description: Provides support for numerical computations and multidimensional array operations in Python.
   - Usage: Numpy is extensively used for numerical computing tasks such as array manipulation, linear algebra, and mathematical operations.

4. **opencv-python**
   - Description: Provides the OpenCV library for computer vision and image processing tasks in Python.
   - Usage: This package is used for various computer vision tasks, including image manipulation, object detection, and feature extraction.

5. **pandas**
   - Description: Provides data structures and data analysis tools for handling structured data in Python.
   - Usage: Pandas is used for data manipulation, exploration, and analysis, especially with tabular data structures like DataFrames.

6. **scikit-learn**
   - Description: Provides a collection of machine learning algorithms and tools for data mining and data analysis tasks.
   - Usage: Scikit-learn is used for implementing machine learning models, including classification, regression, clustering, and dimensionality reduction.

7. **scipy>=1.2.0**
   - Description: Provides scientific computing tools and algorithms for numerical integration, optimization, interpolation, and more.
   - Usage: Scipy complements Numpy and provides additional mathematical functions and routines for scientific computing tasks.

8. **tqdm**
   - Description: Provides a fast, extensible progress bar for loops and tasks in Python.
   - Usage: This package is used to display progress bars and monitor the progress of iterative tasks, such as data processing or model training.

9. **typing**
   - Description: Provides support for type hints and type checking in Python.
   - Usage: Typing is used to annotate function signatures and variables with type information, improving code readability and enabling static type checking.

## Installation

To install the required packages, run the following command (copied from `README.md`):

```bash
pip install -r requirements.txt
