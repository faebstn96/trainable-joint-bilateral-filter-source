from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='jointbilateralfilter',
    ext_modules=[
        CUDAExtension('jointbilateralfilter_gpu_lib', [
            'csrc/jointbilateralfilter_gpu.cu',
            'csrc/jbf_layer_gpu_forward.cu',
            'csrc/jbf_layer_gpu_backward.cu',
        ]),
        CppExtension('jointbilateralfilter_cpu_lib', [
            'csrc/jointbilateralfilter_cpu.cpp',
            'csrc/jbf_layer_cpu_forward.cpp',
            'csrc/jbf_layer_cpu_backward.cpp'],
            extra_compile_args=['-fopenmp'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
