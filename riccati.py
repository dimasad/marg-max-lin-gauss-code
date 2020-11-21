"""JAX-traceable solutions to the Algebraic Riccati equations."""


from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy


@partial(jnp.vectorize, signature='(m,m),(m,n),(m,m),(n,n),(m,m)->(m,m)')
def dare_residue(a, b, q, r, p):
    """Residue function for implicit differentiation of `dare`."""
    aT_p_b = a.T @ p @ b
    sol = jnp.linalg.solve(r + b.T @ p @ b, aT_p_b.T)

    res = a.T @ p @ a - p - aT_p_b @ sol + q
    return res


def dare_fwd(a, b, q, r):
    """Forward pass of  `dare` for reverse differentiation."""
    p = dare(a, b, q, r)
    return p, (a, b, q, r, p)


def dare_bwd(fwd_vars, out_grad):
    """Backward pass of `dare` for reverse differentiation."""
    a, b, q, r, p = fwd_vars
    dres_dp = jax.jacobian(dare_residue, -1)(*fwd_vars)
    
    adj = jnp.linalg.tensorsolve(dres_dp.T, out_grad.T)
    
    dres_da = jax.jacobian(dare_residue, 0)(*fwd_vars)
    dres_db = jax.jacobian(dare_residue, 1)(*fwd_vars)
    dres_dq = jax.jacobian(dare_residue, 2)(*fwd_vars)
    dres_dr = jax.jacobian(dare_residue, 3)(*fwd_vars)
    
    N = adj.ndim
    a_grad = -jnp.tensordot(dres_da.T, adj, N).T
    b_grad = -jnp.tensordot(dres_db.T, adj, N).T
    q_grad = -jnp.tensordot(dres_dq.T, adj, N).T
    r_grad = -jnp.tensordot(dres_dr.T, adj, N).T

    return (a_grad, b_grad, q_grad, r_grad)


@jax.custom_jvp
@partial(jnp.vectorize, signature='(m,m),(m,n),(m,m),(n,n)->(m,m)')
def dare(a, b, q, r):
    """JAX-traceable solution to Discrete Algebraic Ricatti Equation."""
    return dare_prim.bind(a, b, q, r)


# Define reverse differentiation functions
# dare.defvjp(dare_fwd, dare_bwd)


def dare_impl(a, b, q, r):
  """Concrete implementation of the Discrete Algebraic Ricatti Equation."""
  return scipy.linalg.solve_discrete_are(a, b, q, r)


@dare.defjvp
def dare_jvp(values, tangents):
    p = dare(*values)
    residue_values = values + (p,)
    residue_tangents = tangents + (jnp.zeros_like(p),)
    residue, r_tan = jax.jvp(dare_residue, residue_values, residue_tangents)
    dr_dp = jax.jacobian(dare_residue, -1)(*residue_values)
    p_tan = jnp.linalg.tensorsolve(dr_dp, -r_tan)
    return (p, p_tan)

dare_prim = jax.core.Primitive("dare")
"""Discrete Algebraic Ricatti Equation jax primitive."""

dare_prim.def_impl(dare_impl)
#jax.interpreters.ad.primitive_jvps[dare_prim] = dare_jvp



if __name__ == '__main__':
    A = np.diag([0.9, 0.5])
    B = np.identity(2)
    Q = np.diag([0.2, 0.4])
    R = np.diag([0.5, 0.1])
    
    P = dare(A, B, Q, R)
