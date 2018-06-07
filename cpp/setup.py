from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='BinActivateFunc_cpp',
    ext_modules=[
        CppExtension('BinActivateFunc_cpp', ['BinActivateFunc.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
