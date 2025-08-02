import importlib


ext_module = importlib.import_module('src.ext')


def grid_subsample(points, lengths, voxel_size):
    s_points, s_lengths = ext_module.grid_subsampling(points, lengths, voxel_size)
    return s_points, s_lengths
