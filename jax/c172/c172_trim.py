import jax
import jax.numpy as jnp
import numpy as np
import time
import scipy.optimize as op
from numpy.random import RandomState

jax.config.update("jax_enable_x64", True)

from c172_mp import c172_inertial_loads, c172_mp
from c172_aerodynamics import c172_aerodynamics
from c172_propulsion import c172_propulsion

from eom_6dof import eom_6dof_cg


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
    num_nodes = 10
    num_timing_runs = 10

    Ma = jnp.full(shape=(num_nodes,), fill_value=0.1)
    prop_radius = 0.94  # m

    # Good guess
    th = jnp.full(shape=(num_nodes,), fill_value=np.deg2rad(8.739244543508379))
    delta_e = jnp.full(shape=(num_nodes,), fill_value=np.deg2rad(-7.815234882597328))
    omega = jnp.full(shape=(num_nodes,), fill_value=1734.40209574)

    x0 = jnp.hstack((th, delta_e, omega))

    start = time.time()  
    objFunc = jax.jit(trim_residual)
    gradFunc = jax.jit(jax.grad(trim_residual))
    obj_r = objFunc(x0=x0, Mach=Ma, prop_radius=prop_radius).block_until_ready()
    grad_obj_r = gradFunc(x0, Ma, prop_radius).block_until_ready()
    end = time.time()  
    compile_time = end - start

    # # Bad guess
    # th = np.deg2rad(5.)
    # delta_e = np.deg2rad(-5.)
    # omega = 1900.

    rng = np.random.default_rng(seed=25678)

    th_list = np.array(rng.integers(1, 10, size=num_timing_runs), dtype=float)
    delta_e_list = np.array(rng.integers(-10, 2, size=num_timing_runs), dtype=float)
    omega_list = np.array(rng.integers(1200, 2000, size=num_timing_runs), dtype=float)


    start = time.time()  
    success_check = 0
    for ii in range(num_timing_runs):

        th = np.full(shape=(num_nodes,), fill_value=np.deg2rad(th_list[ii]))
        delta_e = np.full(shape=(num_nodes,), fill_value=np.deg2rad(delta_e_list[ii]))
        omega = np.full(shape=(num_nodes,), fill_value=omega_list[ii])

        x0 = np.hstack((th, delta_e, omega))

        # print('EoM trim residual: ', obj_r)
        # print('EoM trim residual gradient: ', grad_obj_r)
    
        op_outputs = op.minimize(objFunc, x0,
                                args=(Ma, prop_radius),
                                jac=gradFunc,
                                options={'maxiter': 2000},
                                method='SLSQP',
                                tol=1e-16)
        success_check += 1
        # print(op_outputs)
    assert success_check == num_timing_runs

    end = time.time()
    runtime = (end - start)/num_timing_runs

    print('Compile time (s):', compile_time)
    print('Runtime (s):', runtime)


    # jacFunc_vec_obj_r = jax.jacfwd(fun=trim_residual)
    # results = op.least_squares(trim_residual,
    #                            x0=x0,
    #                            args=(Ma, prop_radius),
    #                            verbose=2,
    #                            loss='linear',
    #                            jac=jacFunc_vec_obj_r,
    #                            ftol=1e-16)
    # print(results)




