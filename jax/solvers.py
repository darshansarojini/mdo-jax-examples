import jax.numpy as jnp
import jax


def NewtonSolver(x_0, f, tol=1e-5, max_iter=15):
    """
    A multivariate Newton root-finding routine.

    """
    x = x_0
    f_jac = jax.jacobian(f)
    @jax.jit
    def q(x):
        " Updates the current guess. "
        return x - jnp.linalg.solve(f_jac(x), f(x))
    error = tol + 1
    n = 0
    converged=True
    while error > tol:
        n += 1
        if(n > max_iter):
            converged=False
        y = q(x)
        error = jnp.linalg.norm(x - y)
        x = y
        # print(f'iteration {n}, error = {error}')
    solver = 'Newton'
    print_solver_outputs(solver, converged, error, n, n+1)
    return x


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