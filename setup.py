from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gfmnet',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='src.ext',
            sources=[
                'src/extensions/extra/cloud/cloud.cpp',
                'src/extensions/cpu/grid_subsampling/grid_subsampling.cpp',
                'src/extensions/cpu/grid_subsampling/grid_subsampling_cpu.cpp',
                'src/extensions/cpu/radius_neighbors/radius_neighbors.cpp',
                'src/extensions/cpu/radius_neighbors/radius_neighbors_cpu.cpp',
                'src/extensions/pybind.cpp',
            ],
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)