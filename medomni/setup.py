from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [
    Extension("models.medomni",  ["./models/medomni.py"]),
]

setup(ext_modules = ext_modules)
