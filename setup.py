from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np

ext = [
    Extension(
        "cython_dtw._dtw", ["cython_dtw/_dtw.pyx"], include_dirs=[np.get_include()]
    ),
]

setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=ext,
    packages=["cython_dtw"],
    package_dir={"cython_dtw": "cython_dtw"},
)
