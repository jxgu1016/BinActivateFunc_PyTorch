from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='BinActivateFunc_cuda',
    ext_modules=[
        CUDAExtension('BinActivateFunc_cuda', [
            'BinActivateFunc_cuda.cpp',
            'BinActivateFunc_cuda_kernel.cu'
            ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
