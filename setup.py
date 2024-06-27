from pathlib import Path

from setuptools import setup

# add the README.md file to the long_description
with open('README.md', 'r') as fh:
    long_description = fh.read()


def read(rel_path):
    here = Path(__file__).parent.absolute()
    with open(here.joinpath(rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')


# basic requirements
install_requires = [
    'ipykernel',
    'jax',
    'jaxlib',
    'matplotlib',
    'numpy',
    'opencv-python',
    'optax',
    'pandas',
    'scikit-learn',
    'scipy>=1.2.0',
    'sleap_io',
    'tqdm',
    'typing',
]

# additional requirements
extras_require = {
    'dev': {
        'flake8',
        'isort',
        'Sphinx',
        'sphinx_rtd_theme',
        'sphinx-rtd-dark-mode',
        'sphinx-automodapi',
        'sphinx-copybutton',
        'sphinx-design',
    },
}


setup(
    name='ensemble-kalman-smoother',
    version=get_version(Path('eks').joinpath('__init__.py')),
    description='Ensembling and kalman smoothing for pose estimation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Cole Hurwitz',
    author_email='',
    url='http://www.github.com/colehurwitz/eks',
    packages=['eks'],
    install_requires=install_requires,
    extras_require=extras_require,
)
