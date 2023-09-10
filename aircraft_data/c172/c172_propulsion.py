import jax.numpy as jnp


def c172_propulsion(h, Ma, omega, prop_radius, ref_pt=jnp.zeros(3), thrust_origin=jnp.zeros(3)):

    rho = 1.225  # kg/m^3
    a = 330  # m/s

    V = Ma*a
    omega_RAD = (omega * 2 * jnp.pi) / 60.0  # rad/s

    J = (jnp.pi * V) / (omega_RAD * prop_radius)  # non-dimensional Advance ratio

    Ct_interp = -0.1692121 * J ** 2 + 0.03545196 * J + 0.10446359  # non-dimensional

    T = (2 / jnp.pi) ** 2 * rho * \
        (omega_RAD * prop_radius) ** 2 * Ct_interp

    F = jnp.zeros(3)
    F = F.at[0].set(T)

    offset = ref_pt - thrust_origin
    M = jnp.cross(F, offset)
    return F, M


if __name__ == "__main__":
    h = 1000
    Ma = 0.15
    prop_radius = 0.94  # m
    omega = 2800
    F, M = c172_propulsion(h=h, Ma=Ma, omega=omega, prop_radius=prop_radius)
    print("Forces: ", F)
    print("Moments: ", M)
