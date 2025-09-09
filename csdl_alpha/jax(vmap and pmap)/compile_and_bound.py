import math
import jax, jax.numpy as jnp, jax.nn as jnn
from jax import jit
from csdl_alpha.backends.jax.graph_to_jax import create_jax_function

def compile(rec, model):
    g = rec.get_root_graph()
    jax_fun = create_jax_function(g, outputs=[model.trim_objective], inputs=[model.theta, model.delta_e, model.omega, model.mach_number],)
    @jit
    def f_case(theta, delta_e, omega, mach):
        return jnp.asarray(jax_fun(theta, delta_e, omega, mach)[0]).reshape(())
    return f_case

def make_bounds():
    th_lo, th_hi = math.radians(-10), math.radians(10)
    de_lo, de_hi = math.radians(-15), math.radians(15)
    om_lo, om_hi = 1000.0, 2800.0
    lo = jnp.array([th_lo, de_lo, om_lo])
    hi = jnp.array([th_hi, de_hi, om_hi])

    def x_from_y(y):
        s = jnn.sigmoid(y)
        return lo + s * (hi - lo)

    def y_from_x(x):
        p = jnp.clip((x - lo) / (hi - lo), 1e-9, 1 - 1e-9)
        return jnp.log(p) - jnp.log1p(-p)

    return lo, hi, x_from_y, y_from_x
