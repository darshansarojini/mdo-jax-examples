import numpy
import time

import casadi  as ca
import csdl
import python_csdl_backend
import jax
import sys
sys.setrecursionlimit(1000000)

rng = numpy.random.default_rng(2021)

from numpy.random import RandomState


# region Functions
def func(x,y,num_ops,var_size):
    z_new = x + y
    prng1 = RandomState(2024)
    prng2 = RandomState(123456789)

    for i in range(num_ops):
        rand_idx1 = int(prng1.randint(0, var_size, size=1)[0])
        rand_idx2 = int(prng1.randint(0, var_size, size=1)[0])

        if backend == 'csdl':
            z_new = z_new + \
                csdl.expand(csdl.reshape(x[rand_idx1, ], (1, )), (var_size, )) * \
                    csdl.expand(csdl.reshape(x[rand_idx2, ], (1, )), (var_size, )) - \
                        csdl.matvec(csdl.outer(x, y), csdl.expand(csdl.dot(x,x), (var_size,)))
        elif backend == 'casadi':
            z_new = z_new + \
                x[rand_idx1]*x[rand_idx2] - \
                    (x@ca.transpose(y))@(ca.dot(x,x)*ca.MX.ones(var_size, 1))
        elif backend == 'numpy':
            z_new = z_new + (x[rand_idx1]*x[rand_idx2])*numpy.ones(var_size) - \
                numpy.matmul(numpy.outer(x, y), numpy.inner(x, x)*numpy.ones(var_size))
        elif backend == 'jax':
            z_new = z_new + (x[rand_idx1]*x[rand_idx2])*jax.numpy.ones(var_size) - \
                jax.numpy.matmul(jax.numpy.outer(x,y), jax.numpy.inner(x,x)*jax.numpy.ones(var_size))
        else:
            raise NotImplementedError
        
        z_new = x*z_new*y**2+y
    return z_new


def build_function(num_ops, backend, size):
    build_info = {}
    start = time.time()

    if backend == 'casadi':
        x = ca.MX.sym('x', size,1)
        y = ca.MX.sym('y', size,1)
    elif backend == 'numpy':
        x = 0.01*numpy.ones(size)
        y = 0.02*numpy.ones(size)
    elif backend == 'jax':
        x = 0.01*jax.numpy.ones((size))
        y = 0.02*jax.numpy.ones((size))
    elif backend == 'csdl':
        m = csdl.Model()
        x = m.create_input('x', shape = (size,))
        y = m.create_input('y', shape = (size,))
    else:
        raise NotImplementedError

    if backend == 'casadi':
        z_new = func(x,y,num_ops,size)
        f = ca.Function('F',[x, y], [z_new])
    elif backend == 'numpy':
        def f(x,y):
            return func(x,y,num_ops,size)
    elif backend == 'jax':

        def func_no_jit(x, y):
            return func(x,y,num_ops,size)

        # # No JIT
        # func_no_jit(x, y).block_until_ready()

        # JIT
        func_jit = jax.jit(func_no_jit)
        func_jit(x, y).block_until_ready()

        def f(x,y):
            # return func_no_jit(x, y)
            return func_jit(x, y)
    elif backend == 'csdl':
        z_new = func(x,y,num_ops,size)
        m.register_output('z',z_new)
        sim = python_csdl_backend.Simulator(m, display_scripts=1)
        def f(x,y):
            sim['x'] = x
            sim['y'] = y
            sim.run()
            return sim['z']
    else:
        raise NotImplementedError

    end = time.time()
    function_compile_time = end - start

    start = time.time()
    if backend == 'casadi':
        dz_dx = ca.jacobian(z_new, x)
        dfdx = ca.Function('dFdx', [x, y], [dz_dx])
    elif backend == 'jax':
        dfdx_unjitted = jax.jacfwd(f, argnums=0)

        # # No JIT
        # dfdx_unjitted(x, y).block_until_ready()
        # dfdx = dfdx_unjitted

        # JIT
        dfdx = jax.jit(dfdx_unjitted)
        dfdx(x, y).block_until_ready()

    elif backend == 'csdl':
        # sim.compute_total_derivatives()
        sim._generate_totals(of='z', wrt=['x'])
        def dfdx(x,y):
            sim['x'] = x
            sim['y'] = y
            sim.run()
            return sim.compute_totals(of='z', wrt='x')[('z', 'x')]
    elif backend == 'numpy':
        def dfdx(x, y):
            eps = 1e-3
            return numpy.array([(f(x + eps * v, y) - f(x - eps * v, y)) / (2 * eps)
                   for v in numpy.eye(len(x))])
    else:
        raise NotImplementedError

    end = time.time()
    derivative_compile_time = end - start

    build_info['x_val'] = 0.01*numpy.ones((size))
    build_info['y_val'] = 0.02*numpy.ones((size))

    build_info['function_time'] = function_compile_time
    build_info['derivative_time'] = derivative_compile_time
    build_info['function'] = f
    build_info['dfdx'] = dfdx
    build_info['backend'] = backend

    return build_info


def run_function(build_info, numruns):
    f = build_info['function']
    dfdx = build_info['dfdx']
    x_val = build_info['x_val']
    y_val = build_info['y_val']


    f_val = f(x_val, y_val)
    dfdx_val = dfdx(x_val, y_val)
    start = time.time()
    for i in range(numruns):
        f(x_val,y_val)
    end = time.time()
    function_runtime = (end - start) / numruns
    # function_runtime = %timeit -n10 -r10 -o  f(x_val,y_val)

    start = time.time()
    for i in range(numruns):
        dfdx(x_val, y_val)
    end = time.time()
    derivative_runtime = (end - start) / numruns

    run_info = {}
    run_info['function_time'] = function_runtime
    run_info['derivative_time'] = derivative_runtime

    if build_info['backend'] == 'jax':
        run_info['value'] = jax.numpy.average(f_val)
        run_info['dfdx'] = jax.numpy.mean(dfdx_val)
        # print(dfdx_val)
        # print(dfdx_val.shape)
    elif build_info['backend'] == 'csdl':
        run_info['value'] = numpy.average(f_val)
        run_info['dfdx'] = numpy.mean(dfdx_val)
        # print(dfdx_val)
        # print(dfdx_val.shape)
    else:
        run_info['value'] = numpy.average(f_val)
        run_info['dfdx'] = numpy.mean(dfdx_val)
        # print(dfdx_val)
        # print(dfdx_val.shape)
        
    return run_info
# endregion


if __name__ == "__main__":
    num_ops = 10
    var_size = 4
    num_timing_runs = 10
    backends = ['numpy', 'casadi', 'csdl', 'jax']

    print('CASE:')
    print(f'{num_ops=}\n{var_size=}\n{num_timing_runs=}')
    for backend in backends:
        build_info = build_function(
            num_ops,
            backend = backend,
            size = var_size
        )
        run_info = run_function(
            build_info,
            num_timing_runs,
        )

        print(f'\n{backend}')
        print('FUNCTION VAL:                ', run_info['value'])
        print('DERIVATIVE VAL:              ', run_info['dfdx'])
        print('FUNCTION COMPILE TIME:       ', build_info['function_time'])
        print('DERIVATIVE COMPILE TIME:     ', build_info['derivative_time'])
        print('FUNCTION RUNTIME:            ', run_info['function_time'])
        print('DERIVATIVE RUNTIME:          ', run_info['derivative_time'])