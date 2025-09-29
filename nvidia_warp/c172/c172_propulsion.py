import warp as wp
import numpy as np

@wp.func
def propulsion(omega: wp.float32, vtas: wp.float32, ref_pt: wp.vec3, thrust_origin: wp.vec3, rho: wp.float32, prop_radius: wp.float32):
    omega_RAD = (omega * 2.0 * np.pi) / 60.0
    J = (np.pi * vtas) / (omega_RAD * prop_radius + 1.0e-6)
    Ct = -0.1692121 * J * J + 0.03545196 * J + 0.10446359
    T = wp.pow((2.0 / np.pi), 2.0) * rho * wp.pow((omega_RAD * prop_radius), 2.0) * Ct
    F_p = wp.vec3(T, 0.0, 0.0)
    offset = ref_pt - thrust_origin
    M_p = wp.cross(offset, F_p)
    return Ct, T, F_p, M_p
