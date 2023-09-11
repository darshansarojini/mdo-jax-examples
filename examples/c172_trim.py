import jax
import jax.numpy as jnp
import numpy as np

import sys
sys.path.append('..')

from aircraft_data.c172.c172_mp import c172_inertial_loads, c172_mp
from aircraft_data.c172.c172_aerodynamics import c172_aerodynamics
from aircraft_data.c172.c172_propulsion import c172_propulsion

from solver.eom_6dof import eom_6dof_cg

def trim_residual(x0, Mach, h, prop_radius):
    u = Mach * a
    v = 0.
    w = 0.

    p = 0.
    q = 0.
    r = 0.

    phi = 0.
    theta = x0[0]
    psi = 0.

    x = 0.
    y = 0.
    z = h

    m, cg, I = c172_mp()

    Fi, Mi = c172_inertial_loads(h=z, th=theta)
    Fa, Ma = c172_aerodynamics(h=z, Ma=Mach, alpha=theta, delta_e=x0[1])
    Fp, Mp = c172_propulsion(h=z, Ma=Mach, omega=x0[2], prop_radius=prop_radius)

    state_vector = jnp.array([u, v, w, p, q, r, phi, theta, psi, x, y, z])

    eom_residual = eom_6dof_cg(state_vector,
                               m, cg, I,
                               Fa, Ma,
                               Fp, Mp,
                               Fi, Mi)

    # Get scalar residual
    obj_r = jnp.linalg.norm(eom_residual.at[0:6].get())
    return obj_r


if __name__ == "__main__":
    a = 330.

    h = 1000  # m
    Ma = 0.15
    prop_radius = 0.94  # m

    th = np.deg2rad(3.)
    delta_e = np.deg2rad(-2.)
    omega = 1900.

    x0 = [th, delta_e, omega]
    # obj_r = trim_residual(x0=x0, Mach=Ma, h=h, prop_radius=prop_radius)
    grad_obj_r = jax.grad(trim_residual)

    obj_r, grad_obj_r = jax.value_and_grad(fun=trim_residual)(x0, Ma, h, prop_radius)

    print('EoM trim residual: ', obj_r)
    print('EoM trim residual gradient:', grad_obj_r)

    import scipy.optimize as op

    op_outputs = op.minimize(trim_residual, x0, args=(Ma, h, prop_radius), jac=grad_obj_r, tol=1e-6, options={'maxiter':100})
    # op_outputs = op.minimize(trim_residual, x0, args=(Ma, h, prop_radius), tol=1e-6, options={'maxiter':100})

    print(op_outputs)



