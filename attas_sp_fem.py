"""ATTAS aircraft short-period mode estimation --- Traditional PEM/FEM."""


import collections
import os

import attrdict
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy.io
import scipy.linalg
from scipy import optimize

import riccati


jax.config.update("jax_enable_x64", True)


class Decision:
    """Decision variable specification."""

    def __init__(self, shape, start):
        if isinstance(shape, int):
            shape = (shape,)
        
        self.shape = shape
        """Decision variable shape."""

        self.size = np.prod(shape, dtype=int)
        """Total number of elements."""
        
        self.start = start
        """Start index in parent vector."""
        
        end = start + self.size
        self.slice = np.s_[start:end]
        """Slice of variable in parent vector."""

    def unpack(self, vec):
        """Unpack variable from parent vector."""
        return vec[self.slice].reshape(self.shape)

    def pack(self, vec, value):
        """Pack variable into parent vector."""
        val_flat = np.broadcast_to(value, self.shape).ravel()
        vec[self.slice] = val_flat


class Problem:
    
    def __init__(self, nx, u, y):
        self.dec_specs = collections.OrderedDict()
        """Decision variable specifications."""

        self.ndec = 0
        """Total number of decision variables."""

        self.u = jnp.asarray(u)
        """Inputs."""
        
        self.y = jnp.asarray(y)
        """Measurements."""
        
        self.nx = nx
        """Size of state vector."""

        self.nu = np.size(u, 1)
        """Size of input vector."""

        self.ny = np.size(y, 1)
        """Size of output vector."""

        N = np.size(y, 0)
        assert N == np.size(u, 0)        
        self.N = N
        """Number of measurement instants."""

        # Register decision variables
        self.add_decision('x0', nx)
        self.add_decision('A', (nx, nx))
        self.add_decision('B', (nx, self.nu))
        self.add_decision('lsQd', nx)
        self.add_decision('lsRd', self.ny)
    
    def add_decision(self, name, shape=()):
        self.dec_specs[name] = spec = Decision(shape, self.ndec)
        self.ndec += spec.size
    
    def unpack_decision(self, dvec):
        if jnp.shape(dvec) != (self.ndec,):
            raise ValueError("invalid shape for `dvec`")
        
        dvars = attrdict.AttrDict()
        for name, spec in self.dec_specs.items():
            dvars[name] = spec.unpack(dvec)
        return dvars
    
    def pack_decision(self, dvars, dvec=None):
        if dvec is None:
            dvec = np.zeros(self.ndec)
        
        for name, value in dvars.items():
            spec = self.dec_specs.get(name)
            if spec is not None:
                spec.pack(dvec, value)
        
        return dvec

    def merit(self, dvec):
        v = self.unpack_decision(dvec)
        A = v.A
        B = v.B

        u = self.u
        y = self.y
        
        C = jnp.identity(self.nx)
        D = jnp.zeros((self.ny, self.nu))
        
        sQd = jnp.exp(v.lsQd)
        sRd = jnp.exp(v.lsRd)
        Qd = sQd ** 2
        Rd = sRd ** 2
        sQ = jnp.diag(sQd)
        sR = jnp.diag(sRd)
        Q = jnp.diag(Qd)
        R = jnp.diag(Rd)

        Pp = riccati.dare(A.T, C.T, Q, R)
        sPp = jnp.linalg.cholesky(Pp)
        
        nx = len(A)
        ny = len(C)
        N = len(y)

        # Kailath Eq. (12.3.8)
        z = jnp.zeros_like(C.T)
        corr_mat = jnp.block([[sR, C @ sPp],
                              [z,   sPp]])
        q, r = jnp.linalg.qr(corr_mat.T)
        s = jnp.sign(r.diagonal())
        sRp = (r.T * s)[:ny, :ny]
        sPc = (r.T * s)[ny:, ny:]
        Kn = (r.T * s)[ny:, :ny]
        
        x = v.x0
        loglike = -N * jnp.log(sRp.diagonal()).sum()
        for k in range(len(y)):
            e = y[k]  - (C @ x + D @ u[k])
            en = jnp.linalg.solve(sRp, e)            
            
            loglike = loglike - 0.5 * jnp.sum(en ** 2)
            
            xcorr = x + Kn @ en
            x = A @ xcorr + B @ u[k]
        
        return loglike


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
    
    # Add artificial noise
    np.random.seed(0)
    y[:, :] +=  1e-2 * np.random.randn(*y.shape)
    
    return t, u, y, yshift, yscale, ushift, uscale


if __name__ == '__main__':
    nx = 2
    nu = 1
    ny = 2

    # Load experiment data
    t, u, y, yshift, yscale, ushift, uscale = load_data()
    problem = Problem(nx, u, y)
    
    A0 = np.array([[0.92, -0.097], [0.0831, 0.977]])
    B0 = np.array([[-0.11], [0.015]])
    lsQd0 = np.array([-4.3, -4.9])
    lsRd0 = np.array([-5.4, -4.4])
    dvar0 = dict(A=A0, B=B0, lsQd=lsQd0, lsRd=lsRd0)
    dvec0 = problem.pack_decision(dvar0)

    # Define optimization functions
    obj = lambda x: -problem.merit(x)
    grad = jax.grad(obj)
    hessp = lambda x, p: jax.jvp(grad, (x,), (p,))[1]
    
    opt = {'gtol': 2e-6, 'disp': True, 'maxiter': 200}
    sol = optimize.minimize(
        obj, dvec0, method='trust-krylov', jac=grad, hessp=hessp, options=opt
    )
    varopt = problem.unpack_decision(sol.x)
    vargrad = problem.unpack_decision(sol.jac)
    
    A = varopt.A
    B = varopt.B
    lsQd = varopt.lsQd
    lsRd = varopt.lsRd
    x0 = varopt.x0
    
    sRd = np.exp(lsRd)
    sQd = np.exp(lsQd)
    
    xsim = np.zeros_like(y)
    xsim[0] = x0
    for i in range(1, len(y)):
        xsim[i] = A @ xsim[i-1] + B @ u[i - 1]
