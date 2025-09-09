import jax, jax.numpy as jnp
from jax import vmap, pmap
import jaxopt

def loss_and_solver(f_case, x_from_y):
    @jax.jit
    def loss_y(y, mach):
        th, de, om = x_from_y(y)
        return f_case(th, de, om, mach)

    def loss_and_grad(y, mach):
        def wrapped_loss(yy):
            return loss_y(yy, mach)
        val, grad = jax.value_and_grad(wrapped_loss)(y)
        return val, grad

    solver = jaxopt.LBFGS(
        fun=loss_y,
        value_and_grad=loss_and_grad,
        has_aux=False,
        maxiter=200,
        tol=1e-12,
    )
    return loss_y, solver

def solve_one(solver, loss_y, y_from_x, x_from_y, mach, x0):
    y0 = y_from_x(x0)
    res = solver.run(init_params=y0, mach=mach)
    y_star = res.params
    x_star = x_from_y(y_star)
    f_star = loss_y(y_star, mach)
    return x_star, f_star

def solve_batch_vmap(solver, loss_y, y_from_x, x_from_y, mach_vec, x0_batch):
    def vmapped(mach_vec, x0_batch):
        def solve_fn(m, x0):
            return solve_one(solver, loss_y, y_from_x, x_from_y, m, x0)
        return vmap(solve_fn, in_axes=(0, 0))(mach_vec, x0_batch)
    return vmapped(mach_vec, x0_batch)

def solve_batch_pmap(solver, loss_y, y_from_x, x_from_y, mach_vec, x0_batch):
    n_dev = jax.device_count()
    B = mach_vec.shape[0]
    local_B = B // n_dev
    mach_shards = mach_vec.reshape(n_dev, local_B)
    x0_shards = x0_batch.reshape(n_dev, local_B, 3)

    def pmapped(mach_shards, x0_shards):
        def vmapped_solve(mach_shard, x0_shard):
            def solve_fn(m, x0):
                return solve_one(solver, loss_y, y_from_x, x_from_y, m, x0)
            return vmap(solve_fn, in_axes=(0, 0))(mach_shard, x0_shard)
        return pmap(vmapped_solve, in_axes=(0, 0))(mach_shards, x0_shards)

    xs, fs = pmapped(mach_shards, x0_shards)
    return xs.reshape(B, 3), fs.reshape(B)
