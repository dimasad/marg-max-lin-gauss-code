"""Tests with the ATTAS aircraft short-period mode estimation."""


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
        self.add_decision('en', (N, nx))
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
        en = v.en
        A = v.A
        B = v.B
        
        u = self.u
        y = self.y
        
        C = jnp.identity(self.nx)
        D = jnp.zeros((self.ny, self.nu))
        
        e = en * jnp.exp(v.lsRd)
        x = y - e
        
        xprev = x[:-1]
        uprev = u[:-1]
        xnext = x[1:]
        w = xnext - xprev @ A.T - uprev @ B.T
        e = y - x @ C.T - u @ D.T

        lprior = normal_logpdf(w, v.lsQd)
        llike = normal_logpdf2(en, v.lsRd)
        ldmarg = logdet_marg(A, C, v.lsQd, v.lsRd, self.N)
        
        return lprior + llike + ldmarg


def normal_logpdf(x, logsigma):
    """Unnormalized normal distribution logpdf."""
    N = len(x)
    inv_sigma2 = jnp.exp(-2 * logsigma)
    sigma_factor = - N * jnp.sum(logsigma)
    return -0.5 * jnp.sum(jnp.sum(x ** 2, axis=0) * inv_sigma2) + sigma_factor


def normal_logpdf2(xn, logsigma):
    """Unnormalized normal distribution logpdf."""
    N = len(xn)
    sigma_factor = - N * jnp.sum(logsigma)
    return -0.5 * jnp.sum(xn ** 2) + sigma_factor


def logdet_marg(A, C, lsQd, lsRd, N):
    # Assemble the input matrices
    sQd = jnp.exp(lsQd)
    sRd = jnp.exp(lsRd)
    Qd = sQd ** 2
    Rd = sRd ** 2
    sQ = jnp.diag(sQd)
    sR = jnp.diag(sRd)
    Q = jnp.diag(Qd)
    R = jnp.diag(Rd)
    
    Pp = riccati.dare(A.T, C.T, Q, R)
    
    nx = len(A)
    ny = len(C)
    z = jnp.zeros_like(C.T)
    sPp = jnp.linalg.cholesky(Pp)
    corr_mat = jnp.block([[sR, C @ sPp],
                          [z,   sPp]])
    q, r = jnp.linalg.qr(corr_mat.T)
    s = jnp.sign(r.diagonal())
    sPc = (r.T * s)[ny:, ny:]
    
    z = jnp.zeros_like(A)
    pred_mat = jnp.block([[A @ sPc, sQ],
                          [sPc,     z]])
    q, r = jnp.linalg.qr(pred_mat.T)
    s = jnp.sign(r.diagonal())
    sPr = (r.T * s)[nx:, nx:]
    
    eps = 1e-40
    log_det_sPc = jnp.sum(jnp.log(jnp.abs(sPc.diagonal()) + eps))
    log_det_sPr = jnp.sum(jnp.log(jnp.abs(sPr.diagonal()) + eps))
    return (N-1) * log_det_sPr + log_det_sPc


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
    problem = Problem(nx, u, y)
    
    x0 = y
    A0 = np.diag([0.9, 0.9])
    B0 = np.zeros((2, 1))
    lsQd0 = np.array([-1, -1])
    lsRd0 = np.array([-1, -1])
    dvar0 = dict(A=A0, B=B0, lsQd=lsQd0, lsRd=lsRd0)
    dvec0 = problem.pack_decision(dvar0)

    # Define optimization functions
    obj = lambda x: -problem.merit(x)
    grad = jax.grad(obj)
    hessp = lambda x, p: jax.jvp(grad, (x,), (p,))[1]
    
    opt = {'gtol': 1e-6, 'disp': True, 'maxiter': 200}
    sol = optimize.minimize(
        obj, dvec0, method='trust-krylov', jac=grad, hessp=hessp, options=opt
    )
    varopt = problem.unpack_decision(sol.x)
    vargrad = problem.unpack_decision(sol.jac)
    
    A = varopt.A
    B = varopt.B
    lsQd = varopt.lsQd
    lsRd = varopt.lsRd
    en = varopt.en
    
    sRd = np.exp(lsRd)
    e = en * sRd
    x = y - e
    
    xsim = np.zeros_like(x)
    xsim[0] = x[0]
    for i in range(1, len(x)):
        xsim[i] = A @ xsim[i-1] + B @ u[i - 1]

    
