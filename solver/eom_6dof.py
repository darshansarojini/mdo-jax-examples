import jax.numpy as jnp


def eom_6dof_cg(state_vector,
                m, cg, I,
                Fa, Ma,
                Fp, Mp,
                Fi, Mi):

    total_force = Fa + Fp + Fi
    Fx = total_force[0]
    Fy = total_force[1]
    Fz = total_force[2]
    
    total_moment = Ma + Mp + Mi
    L = total_moment[0]
    M = total_moment[1]
    N = total_moment[2]

    Ix = I[0, 0]
    Iy = I[1, 1]
    Iz = I[2, 2]
    Jxz = I[0, 2]

    u = state_vector[0]
    v = state_vector[1]
    w = state_vector[2]

    p = state_vector[3]
    q = state_vector[4]
    r = state_vector[5]

    phi = state_vector[6]
    theta = state_vector[7]
    psi = state_vector[8]

    # Linear momentum equations
    du_dt = Fx / m + r * v - q * w
    dv_dt = Fy / m - r * u + p * w
    dw_dt = Fz / m + q * u - p * v

    # Angular momentum equations
    dp_dt = (L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
             p * q * Jxz * (Ix + Iz - Iy)) / (Ix * Iz - Jxz ** 2)
    dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy
    dr_dt = (L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
             q * r * Jxz * (Iz + Ix - Iy)) / (Ix * Iz - Jxz ** 2)

    # Angular Kinematic equations
    dtheta_dt = q * jnp.cos(phi) - r * jnp.sin(phi)
    dphi_dt = p + (q * jnp.sin(phi) + r * jnp.cos(phi)) * jnp.tan(theta)
    dpsi_dt = (q * jnp.sin(phi) + r * jnp.cos(phi)) / jnp.cos(theta)

    # Linear kinematic equations
    dx_dt = (jnp.cos(theta) * jnp.cos(psi) * u +
             (jnp.sin(phi) * jnp.sin(theta) * jnp.cos(psi) - jnp.cos(phi) * jnp.sin(psi)) * v +
             (jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi) + jnp.sin(phi) * jnp.sin(psi)) * w)
    dy_dt = (jnp.cos(theta) * jnp.sin(psi) * u +
             (jnp.sin(phi) * jnp.sin(theta) * jnp.sin(psi) + jnp.cos(phi) * jnp.cos(psi)) * v +
             (jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi) - jnp.sin(phi) * jnp.cos(psi)) * w)
    dz_dt = -u * jnp.sin(theta) + v * jnp.sin(phi) * jnp.cos(theta) + w * jnp.cos(
        phi) * jnp.cos(theta)

    residual_vector = jnp.array([du_dt, dv_dt, dw_dt,
                                 dp_dt, dq_dt, dr_dt,
                                 dtheta_dt, dphi_dt, dpsi_dt,
                                 dx_dt, dy_dt, dz_dt])
    return residual_vector

