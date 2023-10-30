import jax
import numpy as np
import time
import jax.numpy as jnp


def jax_func(x, y, z):
    return x + y + z


if __name__ == "__main__":
    def f(x, y):
        z = x + y
        for i in range(1000):
            # z = jax_func(x,y,z)
            z = x + y + z
        return z


    # x, y = 3, 4
    x, y = jnp.ones(100) * 0.3, jnp.ones(100) * 0.4

    # lowered = jax.jit(f).lower(x, y)
    lowered = jax.jit(f).lower(x, y)
    # lowered = f

    import time

    s = time.time()
    compiled = lowered.compile()
    end = time.time()
    print(end - s)
    # print(compiled(x,y))
    # Print lowered HLO
    # print(lowered.as_text())
    # print(lowered.cost_analysis())