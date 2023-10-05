import jax.numpy as jnp
import jax
import copy

def NewtonSolver(y0, res, tol=1e-6, maxiter=100):
    '''
    TODO: need to make the input a dictionary as in nlbgs
    '''

    y = y0 * 1.
    converged = False
    iter = 0
    while (not converged) and (iter < maxiter):
        J = jax.jacfwd(res)(y)
        r = res(y)
        res_norm = jnp.linalg.norm(r)
        if res_norm < tol:
            converged = True
            break

        # p = jax.scipy.sparse.linalg.gmres(lambda x: -J@x, r)[0]
        p = jnp.linalg.solve(-J, r)
        y += p
        iter += 1

    print("Convergence:", converged)
    print("Res norm:", res_norm)
    print("iter:", iter)

    return y


def NLBGSSolver(y0, res, tol=1e-6, maxiter=100):
    '''
    y0 : dictionary of states
    res : returns dictionary of residuals 
    res = y - r(y)
    '''
    y = copy.deepcopy(y0)
    iter = 0
    while iter < maxiter:
        converged = True
        r_dict = res(y)
        for name, res_val in r_dict.items():
            res_norm = jnp.linalg.norm(res_val)
            if res_norm > tol:
                converged = False

            y[name] = y[name] - res_val
               
        iter += 1
        if converged:
            break
        
    print("Convergence:", converged)
    print("Res norm:", res_norm)
    print("iter:", iter)

    return y