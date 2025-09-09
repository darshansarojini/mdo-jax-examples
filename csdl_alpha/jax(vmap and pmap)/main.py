import time, math, numpy as np
import jax, jax.numpy as jnp

def main():
    rec, model = build_model(C172TrimModel)
    f_case = compile(rec, model)
    lo, hi, x_from_y, y_from_x = make_bounds()
    loss_y, solver = loss_and_solver(f_case, x_from_y)

    n_dev = jax.device_count()
    batch_sizes = tuple(n_dev * k for k in (1, 2, 4, 8, 16, 32, 64, 128))

    B_max = batch_sizes[-1]
    mach_full = jnp.linspace(0.08, 0.16, B_max)
    x0 = jnp.array([math.radians(8.7), math.radians(-7.8), 1734.0])
    x0_full = jnp.tile(x0, (B_max, 1))

    xs_v, fs_v = solve_batch_vmap(solver, loss_y, y_from_x, x_from_y, mach_full, x0_full)
    jax.block_until_ready(fs_v)
    xs_p, fs_p = solve_batch_pmap(solver, loss_y, y_from_x, x_from_y, mach_full, x0_full)
    jax.block_until_ready(fs_p)

    times_vmap = []
    times_pmap = []

    for B in batch_sizes:
        mask = jnp.arange(B_max) < B
        mach_vec = jnp.where(mask, mach_full, 0.0)
        x0_batch = jnp.where(mask[:, None], x0_full, 0.0)

        t0 = time.perf_counter()
        xs, fs = solve_batch_vmap(solver, loss_y, y_from_x, x_from_y, mach_vec, x0_batch)
        jax.block_until_ready(fs)
        t1 = time.perf_counter()
        times_vmap.append(t1 - t0)

        t0 = time.perf_counter()
        xs, fs = solve_batch_pmap(solver, loss_y, y_from_x, x_from_y, mach_vec, x0_batch)
        jax.block_until_ready(fs)
        t1 = time.perf_counter()
        times_pmap.append(t1 - t0)

    plot(batch_sizes, times_vmap, times_pmap)

    print("Batch sizes:", list(batch_sizes))
    print("vmap times (s):", [float(t) for t in times_vmap])
    print("pmap times (s):", [float(t) for t in times_pmap])

main()
