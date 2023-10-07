import jax.numpy as jnp
import jax
import copy
from jax import custom_jvp, partials

# @custom_jvp
def NewtonSolver(y0, res, tol=1e-6, maxiter=100):
    '''
    Newton solver taking a vector input 
    with or without known index pointer vector (as in csc/csr).

    Parameters
    ----------
    y0 : np.array
        initial guess vector containing all state vectors
    res : function
        function that returns a vector containing all residual vectors
        res = y - r(y)
    tol: float
        tolerance for each residual norm
    maxiter: int
        maximum number of iterations
    '''
    y = y0
    iter = 0
    while (iter < maxiter):
        r_vec, ind_ptr = res(y)
        res_norm = jnp.linalg.norm(r_vec)
        if res_norm < tol:
            converged = True
            break      

        J = jax.jacfwd(res)(y)[0]
        # print(J)
        # p = jax.scipy.sparse.linalg.gmres(lambda x: -J@x, r)[0]
        p = jnp.linalg.solve(J, r_vec)
        y -= p
        iter += 1


    # # OR checking tol for each residual separately
    # while (iter < maxiter):
    #     converged = True
    #     r_vec, ind_ptr = res(y)
    #     for i in range(len(ind_ptr)-1):
    #         res_val = r_vec.at[ind_ptr[i]:ind_ptr[i+1]].get()
    #         res_norm = jnp.linalg.norm(res_val)
    #         if res_norm > tol:
    #             converged = False  

    #     if converged:
    #         break
    #     J = jax.jacfwd(res)(y)[0]
    #     # print(J)
    #     # p = jax.scipy.sparse.linalg.gmres(lambda x: -J@x, r)[0]
    #     p = jnp.linalg.solve(J, r_vec)  
    #     y -= p
    #     iter += 1

    res_evals = iter + 1
    solver = 'Newton'
    print_solver_outputs(solver, converged, res_norm, iter, res_evals)

    return y

# @NewtonSolver.defjvp
# def NewtonSolver_jvp(primals, tangents):
#   y0, res, tol, maxiter = primals
#   x_dot, res_dot, tol_dot, maxiter_dot  = tangents
#   primal_out = NewtonSolver(y0, res, tol, maxiter)
#   J = jax.jacfwd(res)(primal_out)[0]
#   tangent_out = jnp.cos(x) * x_dot
#   return primal_out, tangent_out

def NLBGSSolver(y0, res, tol=1e-6, maxiter=100):
    '''
    Nonlinear BLOCK Gauss-Seidel solver taking a vector input 
    with known index pointer vector (as in csc/csr).

    Parameters
    ----------
    y0 : np.array
        initial guess vector containing all state vectors
    res : function
        function that returns a vector containing all residual vectors
        res = y - r(y)
    tol: float
        tolerance for each residual norm
    maxiter: int
        maximum number of iterations
    '''
    y = y0
    iter = 0
    while iter < maxiter:
        converged = True
        r_vec, ind_ptr = res(y)
        for i in range(len(ind_ptr)-1):
            res_val = r_vec.at[ind_ptr[i]:ind_ptr[i+1]].get()
            state_val =   y.at[ind_ptr[i]:ind_ptr[i+1]].get()
            res_norm = jnp.linalg.norm(res_val)
            if res_norm > tol:
                converged = False

            y = y.at[ind_ptr[i]:ind_ptr[i+1]].set(state_val-res_val)
            r_vec, ind_ptr = res(y)
            
        iter += 1
        if converged:
            break
        
    res_evals = (len(ind_ptr) - 1) * iter + 1
    solver = 'Nonlinear Block Gauss-Seidel'
    print_solver_outputs(solver, converged, res_norm, iter, res_evals)

    return y

def NLGSSolver(y0, res, tol=1e-6, maxiter=100):
    '''
    Nonlinear Gauss-Seidel solver taking a vector input 
    with known index pointer vector (as in csc/csr).

    Parameters
    ----------
    y0 : np.array
        initial guess vector containing all state vectors
    res : function
        function that returns a vector containing all residual vectors
        res = y - r(y)
    tol: float
        tolerance for each residual norm
    maxiter: int
        maximum number of iterations
    '''
    y = y0
    iter = 0
    while iter < maxiter:
        converged = True
        r_vec, ind_ptr = res(y)
        for i in range(len(ind_ptr)-1):
            res_val = r_vec.at[ind_ptr[i]:ind_ptr[i+1]].get()
            state_val =   y.at[ind_ptr[i]:ind_ptr[i+1]].get()
            res_norm = jnp.linalg.norm(res_val)
            if res_norm > tol:
                converged = False

            y = y.at[ind_ptr[i]:ind_ptr[i+1]].set(state_val-res_val)
               
        iter += 1
        if converged:
            break
        
    res_evals = iter + 1
    solver = 'Nonlinear Gauss-Seidel'
    print_solver_outputs(solver, converged, res_norm, iter, res_evals)

    return y


def NLBGSSolver_dict(y0, res, tol=1e-6, maxiter=100):
    '''
    Nonlinear BLOCK Gauss-Seidel solver taking dictionary inputs.

    Parameters
    ----------
    y0 : dict
        initial guess dict containing state vectors
    res : function
        function that returns dict containing residual vectors
        res = y - r(y)
    tol: float
        tolerance for each residual norm
    maxiter: int
        maximum number of iterations
    '''
    y = y0
    iter = 0
    r_dict = res(y)
    ct = 1
    while iter < maxiter:
        converged = True
        for name in r_dict.keys():
            res_norm = jnp.linalg.norm(r_dict[name])
            if res_norm > tol:
                converged = False

            y[name] = y[name] - r_dict[name]
            # Residuals are recomputed after states in each block residual are updated
            r_dict = res(y) 
            ct += 1
               
        iter += 1  # Minimum 1 iteration rqd.
        
        if converged:
            break  
        
    res_evals = len(r_dict) * iter + 1
    solver = 'Nonlinear Block Gauss-Seidel [dict]'
    print_solver_outputs(solver, converged, res_norm, iter, res_evals)
    print(ct)

    return y

def NLGSSolver_dict(y0, res, tol=1e-6, maxiter=100):
    '''
    Nonlinear Gauss-Seidel solver taking dictionary inputs.

    Parameters
    ----------
    y0 : dict
        initial guess dict containing state vectors
    res : function
        function that returns dict containing residual vectors
        res = y - r(y)
    tol: float
        tolerance for each residual norm
    maxiter: int
        maximum number of iterations
    '''
    y = y0
    iter = 0
    while iter < maxiter:
        converged = True
        r_dict = res(y)
        for name in r_dict.keys():
            res_norm = jnp.linalg.norm(r_dict[name])
            if res_norm > tol:
                converged = False

            y[name] = y[name] - r_dict[name]
            # Residuals are not recomputed after states in each block residual are updated
            # r_dict = res(y)
               
        if converged:
            break
        
        iter += 1


    res_evals = iter + 1
    solver = 'Nonlinear Gauss-Seidel [dict]'
    print_solver_outputs(solver, converged, res_norm, iter, res_evals)


    return y


def print_solver_outputs(solver, converged, res_norm, iter, res_evals):
    '''
    Standard utility function for printing solver outputs.
    '''
    print("\n")
    print("Solver      : ", solver)
    print("Convergence : ", converged)
    print("Res norm    : ", res_norm)
    print("iter        : ", iter)
    print("res evals   : ", res_evals)