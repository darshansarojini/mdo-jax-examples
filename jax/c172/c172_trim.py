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
def trim_residual(x0, Mach, prop_radius):

    num_nodes = Mach.shape[0]
    x0 = x0.reshape((num_nodes, 3))

    u = Mach * a
    v = jnp.zeros((num_nodes, ))
    w = jnp.zeros((num_nodes, ))

    p = jnp.zeros((num_nodes, ))
    q = jnp.zeros((num_nodes, ))
    r = jnp.zeros((num_nodes, ))

    phi = jnp.zeros((num_nodes, ))
    theta = x0[:, 0]
    psi = jnp.zeros((num_nodes, ))

    m, cg, I = c172_mp()

    Fi, Mi = c172_inertial_loads(th=theta)
    Fa, Ma = c172_aerodynamics(Ma=Mach, alpha=theta, delta_e=x0[:, 1])
    Fp, Mp = c172_propulsion(Ma=Mach, omega=x0[:, 2], prop_radius=prop_radius)

    eom_residual = eom_6dof_cg(u, v, w, p, q, r, phi, theta, psi,
                               m, cg, I,
                               Fa, Ma,
                               Fp, Mp,
                               Fi, Mi)

    # Get scalar residual
    obj_r = jnp.linalg.norm(eom_residual.at[:, 0:6].get())
    # obj_r = eom_residual.at[0:6].get()
    return obj_r


if __name__ == "__main__":
    a = 336.43470050484996  # m/s
    num_nodes = 100

    Ma = jnp.full(shape=(num_nodes,), fill_value=0.1)
    prop_radius = 0.94  # m

    # Good guess
    th = jnp.full(shape=(num_nodes,), fill_value=np.deg2rad(8.739244543508379))
    delta_e = jnp.full(shape=(num_nodes,), fill_value=np.deg2rad(-7.815234882597328))
    omega = jnp.full(shape=(num_nodes,), fill_value=1734.40209574)

    # # Bad guess
    # th = np.deg2rad(5.)
    # delta_e = np.deg2rad(-5.)
    # omega = 1900.

    # start = time.time()
    x0 = jnp.hstack((th, delta_e, omega))
    # x0 = jnp.array([ 1.659e-01, -1.672e-01,  1.761e+03,  1.659e-01, -1.672e-01,
    #         1.761e+03,  1.659e-01, -1.672e-01,  1.761e+03,  1.646e-01,
    #        -6.416e-01,  1.757e+03])
    obj_r = trim_residual(x0=x0, Mach=Ma, prop_radius=prop_radius)

    gradFunc_obj_r = jax.grad(trim_residual)
    grad_obj_r = gradFunc_obj_r(x0, Ma, prop_radius)
    # end = time.time()
    # print('Runtime (s):', (end - start))
    # obj_r,  = jax.value_and_grad(fun=trim_residual)(x0, Ma, prop_radius)

    print('EoM trim residual: ', obj_r)
    print('EoM trim residual gradient: ', grad_obj_r)

    import scipy.optimize as op
    # start = time.time()
    # op_outputs = op.minimize(trim_residual, x0,
    #                          args=(Ma, prop_radius),
    #                          jac=gradFunc_obj_r,
    #                          options={'maxiter': 1},
    #                          method='SLSQP',
    #                          tol=1e-8)
    # end = time.time()
    # print('Runtime (s):', (end - start))
    # print(op_outputs)

    start = time.time()
    op_outputs = op.minimize(trim_residual, x0,
                             args=(Ma, prop_radius),
                             jac=gradFunc_obj_r,
                             options={'maxiter': 2000},
                             method='SLSQP',
                             tol=1e-16)
    end = time.time()
    print('Runtime (s):', (end - start))
    print(op_outputs)



    # jacFunc_vec_obj_r = jax.jacfwd(fun=trim_residual)
    # results = op.least_squares(trim_residual,
    #                            x0=x0,
    #                            args=(Ma, prop_radius),
    #                            verbose=2,
    #                            loss='linear',
    #                            jac=jacFunc_vec_obj_r,
    #                            ftol=1e-16)
    # print(results)




