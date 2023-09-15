import jax
import jax.numpy as jnp
import numpy as np
import time

jax.config.update("jax_enable_x64", True)

from c172_mp import c172_inertial_loads, c172_mp
from c172_aerodynamics import c172_aerodynamics
from c172_propulsion import c172_propulsion

from eom_6dof import eom_6dof_cg

@jax.jit
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
    # obj_r = eom_residual.at[0:6].get()
    return obj_r


if __name__ == "__main__":
    a = 336.43470050484996  # m/s

    h = 1000  # m
    Ma = 0.1
    prop_radius = 0.94  # m

    # Good guess
    th = np.deg2rad(8.739244543508379)
    delta_e = np.deg2rad(-7.815234882597328)
    omega = 1734.40209574

    # # Bad guess
    # th = np.deg2rad(5.)
    # delta_e = np.deg2rad(-5.)
    # omega = 1900.

    x0 = [th, delta_e, omega]
    # obj_r = trim_residual(x0=x0, Mach=Ma, h=h, prop_radius=prop_radius)
    obj_r, grad_obj_r = jax.value_and_grad(fun=trim_residual)(x0, Ma, h, prop_radius)

    print('EoM trim residual: ', obj_r)
    print('EoM trim residual gradient: ', grad_obj_r)

    import scipy.optimize as op
    gradFunc_obj_r = jax.grad(trim_residual)

    start = time.time()
    op_outputs = op.minimize(trim_residual, x0,
                             args=(Ma, h, prop_radius),
                             jac=gradFunc_obj_r,
                             options={'maxiter': 100},
                             method='SLSQP',
                             tol=1e-8)
    end = time.time()
    print((end-start))
    print(op_outputs)

    # jacFunc_vec_obj_r = jax.jacfwd(fun=trim_residual)
    # results = op.least_squares(trim_residual,
    #                            x0=x0,
    #                            args=(Ma, h, prop_radius),
    #                            verbose=2,
    #                            loss='linear',
    #                            jac=jacFunc_vec_obj_r,
    #                            ftol=1e-16)
    # print(results)



