
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

def discipline1(x, z, y2):
    y1 = z[0] ** 2 + z[1] + x - 0.2 * y2
    return y1

def discipline2(z, y1):
    y2 = y1 ** 0.5 + z[0] + z[1]
    return y2

def res_dict(y, x=1., z=[5.,2.]):
    y1 = y["disc1"]
    y2 = y["disc2"]

    res1 = y1 - discipline1(x, z, y2)
    res2 = y2 - discipline2(z, y1)

    res = {'disc1': res1, "disc2": res2}
    return res


def res(y, x=1., z=[5.,2.]):
    '''
    returns an index pointer vector similar to csc/csr
    '''
    # Manually coded index pointer
    ind_ptr = jnp.array([0, 1, 2])
    y1 = y.at[ind_ptr[0]:ind_ptr[1]].get()
    y2 = y.at[ind_ptr[1]:ind_ptr[2]].get()

    res1 = y1 - discipline1(x, z, y2)
    res2 = y2 - discipline2(z, y1)

    res = jnp.concatenate((res1,res2))
    return res, ind_ptr


from solvers_fwd import NewtonSolver, NLBGSSolver, NLGSSolver, NLBGSSolver_dict, NLGSSolver_dict

y0_dict = {'disc1': 1., "disc2": 1.}
print(NLBGSSolver_dict(y0_dict, res_dict, tol=1e-10, maxiter=100))

y0_dict = {'disc1': 1., "disc2": 1.}
print(NLGSSolver_dict(y0_dict, res_dict, tol=1e-10, maxiter=100))

y0 = jnp.array([1., 1.])
print(NLGSSolver(y0, res, tol=1e-10, maxiter=100))

y0 = jnp.array([1., 1.])
print(NLBGSSolver(y0, res, tol=1e-10, maxiter=100))

y0 = jnp.array([1., 1.])
print(NewtonSolver(y0, res, tol=1e-10, maxiter=100))
