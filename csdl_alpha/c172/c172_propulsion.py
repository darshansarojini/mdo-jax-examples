import csdl_alpha as csdl
import numpy as np

def propulsion(omega, vtas, ref_pt, thrust_origin, rho, prop_radius):
    omega_RAD = (omega * 2 * np.pi) / 60.0
    J = (np.pi * vtas) / (omega_RAD * prop_radius)
    Ct = -0.1692121 * J ** 2 + 0.03545196 * J + 0.10446359
    T = (2 / np.pi) ** 2 * rho * (omega_RAD * prop_radius) ** 2 * Ct
    F_p = csdl.concatenate([T, 0, 0], axis=0)
    offset = ref_pt - thrust_origin
    M_p = csdl.cross(offset, F_p)
    return Ct, T, F_p, M_p
