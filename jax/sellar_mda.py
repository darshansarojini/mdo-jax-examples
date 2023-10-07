import time
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

@jax.jit
def res_dict(y, x=1., z=[5.,2.]):
    y1 = y["disc1"]
    y2 = y["disc2"]

    res1 = y1 - discipline1(x, z, y2)
    res2 = y2 - discipline2(z, y1)

    res = {'disc1': res1, "disc2": res2}
    return res

@jax.jit
def res(y, x=1., z=[5.,2.]):
    '''
    returns an index pointer vector similar to csc/csr
    '''
    # Manually coded index pointer
    ind_ptr = jnp.array([0, 1, 2])
    y1 = y[0:1]
    y2 = y[1:2]

    res1 = y1 - discipline1(x, z, y2)
    res2 = y2 - discipline2(z, y1)

    res = jnp.concatenate((res1,res2))
    # return res, ind_ptr
    return res
def ImplicitWrapper(x, y0, res, solver='Newton',  tol=1e-10, maxiter=100):

    if solver == 'Newton':
        y = NewtonSolver(x, y0, res, tol=tol, maxiter=maxiter)
    elif solver == 'NLBGS':
        y = NLBGSSolver(x, y0, res, tol=tol, maxiter=maxiter)
    else:
        raise IOError
    return y


from solvers import NewtonSolver, NLBGSSolver, NLGSSolver, NLBGSSolver_dict, NLGSSolver_dict

# time_measured = np.empty(100)
# for i in range(100):
#     y0_dict = {'disc1': 1., "disc2": 1.}
#     start = time.time()
#     print(NLBGSSolver_dict(y0_dict, res_dict, tol=1e-10, maxiter=100))
#     end = time.time()
#     time_measured[i] = end -start
# print(np.average(time_measured), time_measured[0])

# time_measured = np.empty(100)
# for i in range(100):
#     y0_dict = {'disc1': 1., "disc2": 1.}
#     start = time.time()
#     print(NLGSSolver_dict(y0_dict, res_dict, tol=1e-10, maxiter=100))
#     end = time.time()
#     time_measured[i] = end -start
# print(np.average(time_measured), time_measured[0])

# time_measured = np.empty(100)
# for i in range(100):
#     y0 = jnp.array([1., 1.])
#     start = time.time()
#     print(NLBGSSolver(y0, res, tol=1e-10, maxiter=100))
#     end = time.time()
#     time_measured[i] = end -start
# print(np.average(time_measured), time_measured[0])

# time_measured = np.empty(100)
# for i in range(100):
#     y0 = jnp.array([1., 1.])
#     start = time.time()
#     print(NLGSSolver(y0, res, tol=1e-10, maxiter=100))
#     end = time.time()
#     time_measured[i] = end - start
# print(np.average(time_measured), time_measured[0])

# time_measured = np.empty(100)
# for i in range(100):
#     y0 = jnp.array([1., 1.])
#     start = time.time()
#     print(NewtonSolver(y0, res, tol=1e-10, max_iter=100))
#     end = time.time()
#     time_measured[i] = end - start
# print(np.average(time_measured), time_measured[0])


time_measured = np.empty(100)
from jaxopt import Broyden

y0 = jnp.array([1., 1.])
@jax.jit
def root(x, z):
    broyden = Broyden(fun=res, implicit_diff=True)
    return broyden.run(y0, x=x, z=z).params

for i in range(100):
    start = time.time()
    print(root(x=1., z=[5.,2.]))
    end = time.time()
    time_measured[i] = end - start
print(np.average(time_measured), time_measured[0])
print(jax.jacrev(root)(1., [5.,2.]))

# time_measured = np.empty(100)
# from jaxopt import ScipyRootFinding
# scipyrt = ScipyRootFinding(method='krylov', optimality_fun=res, tol=1e-6)
# for i in range(100):
#     y0 = jnp.array([1., 1.])
#     start = time.time()
#     print(scipyrt.run(y0).params)
#     jax.jacrev()
#     end = time.time()
#     time_measured[i] = end - start
# print(np.average(time_measured), time_measured[0])


