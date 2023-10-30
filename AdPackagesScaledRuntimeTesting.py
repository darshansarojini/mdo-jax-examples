import numpy
import time

import casadi  as ca
import aesara as at
import csdl
import python_csdl_backend
import jax
import sys
sys.setrecursionlimit(1000000)


def func(x,y,num_ops):
    z_new = x + y
    for i in range(num_ops):
        # z_new = x + y + z_new
        z_new = x*z_new*y**2+y
    return z_new


def build_function(num_ops, backend, size):
    build_info = {}
    start = time.time()

    if backend == 'casadi':
        x = ca.MX.sym('x', size,1)
        y = ca.MX.sym('y', size,1)
    elif 'aesara' in backend:
        x = at.tensor.dmatrix('x')
        y = at.tensor.dmatrix('y')
    elif backend == 'numpy':
        x = numpy.ones(size)
        y = numpy.ones(size)
    elif backend == 'jax':
        x = 0.01*numpy.ones((size,1))
        y = 0.02*numpy.ones((size,1))
    elif backend == 'csdl':
        m = csdl.Model()
        x = m.create_input('x', shape = (size,1))
        y = m.create_input('y', shape = (size,1))

    z_new = func(x,y,num_ops)

    # start = time.time()
    if backend == 'casadi':
        f = ca.Function('F',[x, y], [z_new])
    elif 'aesara' in backend:

        if 'jax' in backend:
            mode = at.Mode(linker = 'jax')
        elif 'c' in backend:
            mode = at.Mode(linker = 'c')
        else:
            mode = at.Mode()
        f = at.function([x, y], z_new, mode = mode)
    elif backend == 'numpy':
        def f(x,y):
            return func(x,y,num_ops)
    elif backend == 'jax':

        def func_no(x,y):
            return func(x,y,num_ops)
        
        func_no_jit = func_no
        # func_no_jit = jax.jit(func_no)


        def f(x,y):
            # func_no_jit(x,y)
            return func_no_jit(x,y)
        
        f(x,y)
    elif backend == 'csdl':
        m.register_output('z',z_new)
        sim = python_csdl_backend.Simulator(m, display_scripts=1)
        def f(x,y):
            sim['x'] = x
            sim['y'] = y
            sim.run()
            return sim['z']

    end = time.time()
    build_info['x_val'] = 0.01*numpy.ones((size,1))
    build_info['y_val'] = 0.02*numpy.ones((size,1))

    build_info['time'] = end - start
    build_info['function'] = f
    build_info['backend'] = backend

    return build_info



def run_function(build_info, numruns):
    f = build_info['function']
    x_val = build_info['x_val']
    y_val = build_info['y_val']
    
    
    val = f(x_val,y_val)
    start = time.time()
    for i in range(numruns):
        f(x_val,y_val)
    end = time.time()

    run_info = {}
    run_info['time'] = (end - start) / numruns

    if build_info['backend'] == 'jax':
        run_info['value'] = jax.numpy.average(val)
    else:
        run_info['value'] = numpy.average(val)


    return run_info

if __name__ == "__main__":
    
    num_ops = 100000
    var_size = 10
    num_timing_runs = 10
    backends = ['numpy','csdl','casadi','jax', 'aesara', 'aesara jax', 'aesara c']
    backends = ['numpy','csdl','casadi', 'jax']

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
        print('RUN VAL:      ', run_info['value'])
        print('RUN TIME:     ', run_info['time'])
        print('COMPILE TIME: ', build_info['time'])

