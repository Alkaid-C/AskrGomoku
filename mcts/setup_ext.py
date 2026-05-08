"""Build the mcts_ext C++ extension in-place.

    python setup_ext.py build_ext --inplace
"""

from setuptools import Extension, setup

import pybind11

ext = Extension(
    "mcts_ext",
    sources=["mcts_ext.cpp"],
    include_dirs=[pybind11.get_include()],
    extra_compile_args=["-O3", "-std=c++17", "-march=native"],
    language="c++",
)

setup(
    name="mcts_ext",
    ext_modules=[ext],
)
