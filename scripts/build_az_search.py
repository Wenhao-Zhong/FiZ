"""Build C++ tree search; neural inference stays in PyTorch."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
import pybind11
from setuptools import Extension, setup

if len(sys.argv) == 1:
    sys.argv += ["build_ext", "--build-lib", str(ROOT / "build/az_gomoku"),
                 "--build-temp", str(ROOT / "build/az_gomoku/temp")]
setup(name="az-gomoku-search", version="0.1.0", packages=[],
      ext_modules=[Extension("az_search", [str(ROOT / "src/az_gomoku/search.cpp")],
                              include_dirs=[pybind11.get_include()], language="c++",
                              extra_compile_args=["/O2", "/std:c++17", "/EHsc", "/utf-8"]
                              if os.name == "nt" else ["-O3", "-std=c++17"])])
