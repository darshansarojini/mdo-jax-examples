import warp as wp
import numpy as np

@wp.func
def eom_6dof(u: wp.float32, v: wp.float32, w: wp.float32, p: wp.float32, q: wp.float32, r: wp.float32, phi: wp.float32, theta: wp.float32, psi: wp.float32, F_total: wp.vec3, M_total: wp.vec3, m_val: wp.float32, Ixx: wp.float32, Iyy: wp.float32, Izz: wp.float32, Ixz: wp.float32):
    
    Fx = F_total[0]
    Fy = F_total[1]
    Fz = F_total[2]
    
    L_mom = M_total[0]
    M_mom = M_total[1]
    N_mom = M_total[2]

    du_dt = Fx / m_val + r * v - q * w
    dv_dt = Fy / m_val - r * u + p * w
    dw_dt = Fz / m_val + q * u - p * v

    dp_dt = (L_mom * Izz + N_mom * Ixz - q * r * (Izz * Izz - Izz * Iyy + Ixz * Ixz) + p * q * Ixz * (Ixx + Izz - Iyy)) / (Ixx * Izz - Ixz * Ixz)
    dq_dt = (M_mom + (Izz - Ixx) * p * r - Ixz * (p * p - r * r)) / Iyy
    dr_dt = (L_mom * Ixz + N_mom * Ixx + p * q * (Ixx * Ixx - Ixx * Iyy + Ixz * Ixz) - q * r * Ixz * (Izz + Ixx - Iyy)) / (Ixx * Izz - Ixz * Ixz)
    
    dtheta_dt = q * wp.cos(phi) - r * wp.sin(phi)
    dphi_dt = p + (q * wp.sin(phi) + r * wp.cos(phi)) * wp.tan(theta)
    dpsi_dt = (q * wp.sin(phi) + r * wp.cos(phi)) / (wp.cos(theta) + 1.0e-6)
    
    dx_dt = u * wp.cos(theta) * wp.cos(psi) + v * (wp.sin(phi) * wp.sin(theta) * wp.cos(psi) - wp.cos(phi) * wp.sin(psi)) + w * (wp.cos(phi) * wp.sin(theta) * wp.cos(psi) + wp.sin(phi) * wp.sin(psi))
    dy_dt = u * wp.cos(theta) * wp.sin(psi) + v * (wp.sin(phi) * wp.sin(theta) * wp.sin(psi) + wp.cos(phi) * wp.cos(psi)) + w * (wp.cos(phi) * wp.sin(theta) * wp.sin(psi) - wp.sin(phi) * wp.cos(psi))
    dz_dt = -u * wp.sin(theta) + v * wp.sin(phi) * wp.cos(theta) + w * wp.cos(phi) * wp.cos(theta)

    residual_vector = wp.vec(
        du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt,
        dphi_dt, dtheta_dt, dpsi_dt, dx_dt, dy_dt, dz_dt
    )
    return residual_vector
