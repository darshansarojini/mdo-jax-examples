import jax
import jax.numpy as jnp
import numpy as np
from scipy import interpolate


@jax.jit
def c172_propulsion(Ma, omega, prop_radius, ref_pt=jnp.zeros(3), thrust_origin=jnp.zeros(3)):

    num_nodes = omega.shape[0]
    rho = 1.1116589850558272  # kg/m^3
    a = 336.43470050484996  # m/s

    V = Ma*a
    omega_RAD = (omega * 2 * jnp.pi) / 60.0  # rad/s

    J = (jnp.pi * V) / (omega_RAD * prop_radius)  # non-dimensional Advance ratio

    Ct_interp = -0.1692121 * J ** 2 + 0.03545196 * J + 0.10446359  # non-dimensional

    # J_data = jnp.array(
    #     [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.76, 0.77, 0.78, 0.79,
    #      0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94])
    # Ct_data = jnp.array(
    #     [0.102122, 0.11097, 0.107621, 0.105191, 0.102446, 0.09947, 0.096775, 0.094706, 0.092341, 0.088912, 0.083878,
    #      0.076336, 0.066669, 0.056342, 0.045688, 0.034716, 0.032492, 0.030253, 0.028001, 0.025735, 0.023453, 0.021159,
    #      0.018852, 0.016529, 0.014194, 0.011843, 0.009479, 0.0071, 0.004686, 0.002278, -0.0002, -0.002638, -0.005145,
    #      -0.007641, -0.010188])
    # Ct_interp = jnp.interp(J, J_data, Ct_data, left='extrapolate', right='extrapolate')

    T = (2 / jnp.pi) ** 2 * rho * \
        (omega_RAD * prop_radius) ** 2 * Ct_interp

    F = jnp.zeros((num_nodes, 3))
    F = F.at[:, 0].set(T)

    offset = ref_pt - thrust_origin
    M = jnp.cross(F, offset)
    return F, M


if __name__ == "__main__":
    Ma = np.full(shape=(3,), fill_value=0.1)
    prop_radius = 0.94  # m
    omega = np.full(shape=(3,), fill_value=2800.)
    F, M = c172_propulsion(Ma=Ma, omega=omega, prop_radius=prop_radius)
    print("Forces: ", F)
    print("Moments: ", M)
