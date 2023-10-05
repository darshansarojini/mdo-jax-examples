
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

def res(y, x=1., z=[5.,2.]):
    y1 = y["disc1"]
    y2 = y["disc2"]

    res1 = y1 - discipline1(x, z, y2)
    res2 = y2 - discipline2(z, y1)

    res = {'disc1': res1, "disc2": res2}
    return res


from solvers import NewtonSolver, NLBGSSolver
y0 = {'disc1': 1., "disc2": 1.}
print(NLBGSSolver(y0, res, tol=1e-10, maxiter=100))
