from distutils.core import setup
from Cython.Build import cythonize

setup(
    name = "iterations",
    ext_modules = cythonize("./calculate_iterations.py")
    )
