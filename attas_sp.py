"""Tests with the ATTAS aircraft short-period mode estimation."""


import os


import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.io
import scipy.linalg


jax.config.update("jax_enable_x64", True)


def load_data():
    # Retrieve data
    d2r = np.pi / 180
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(module_dir, 'data', 'fAttasElv1.mat')
    data = scipy.io.loadmat(data_file_path)['fAttasElv1'][30:-30]
    t = data[:, 0] - data[0, 0]
    u = data[:, [21]] * d2r
    y = data[:, [7, 12]] * d2r

    # Shift and rescale
    yshift = np.r_[-0.003, -0.04]
    yscale = np.r_[10.0, 20.0]
    ushift = np.r_[-0.04]
    uscale = np.r_[25.0]
    
    y = (y + yshift) * yscale
    u = (u + ushift) * uscale
        
    return t, u, y, yshift, yscale, ushift, uscale


if __name__ == '__main__':
    nx = 2
    nu = 1
    ny = 2

    # Load experiment data
    t, u, y, yshift, yscale, ushift, uscale = load_data()
