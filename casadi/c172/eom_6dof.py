import numpy as np
from casadi import sin, cos, tan, vertcat

def eom_6dof(u,v,w,p,q,r,phi,theta,psi,F_total,M_total,m_val,Ixx,Iyy,Izz,Ixz):
    Fx, Fy, Fz = F_total[0], F_total[1], F_total[2]
    L_mom, M_mom, N_mom = M_total[0], M_total[1], M_total[2]
    du_dt = Fx / m_val + r * v - q * w
    dv_dt = Fy / m_val - r * u + p * w
    dw_dt = Fz / m_val + q * u - p * v
    dp_dt = (L_mom * Izz + N_mom * Ixz - q * r * (Izz ** 2 - Izz * Iyy + Ixz ** 2) + p * q * Ixz * (Ixx + Izz - Iyy)) / (Ixx * Izz - Ixz ** 2)
    dq_dt = (M_mom + (Izz - Ixx) * p * r - Ixz * (p ** 2 - r ** 2)) / Iyy
    dr_dt = (L_mom * Ixz + N_mom * Ixx + p * q * (Ixx ** 2 - Ixx * Iyy + Ixz ** 2) - q * r * Ixz * (Izz + Ixx - Iyy)) / (Ixx * Izz - Ixz ** 2)
    dtheta_dt = q * cos(phi) - r * sin(phi)
    dphi_dt = p + (q * sin(phi) + r * cos(phi)) * tan(theta)
    dpsi_dt = (q * sin(phi) + r * cos(phi)) / cos(theta)
    dx_dt = u * cos(theta) * cos(psi) + v * (sin(phi) * sin(theta) * cos(psi) - cos(phi) * sin(psi)) + w * (cos(phi) * sin(theta) * cos(psi) + sin(phi) * sin(psi))
    dy_dt = u * cos(theta) * sin(psi) + v * (sin(phi) * sin(theta) * sin(psi) + cos(phi) * cos(psi)) + w * (cos(phi) * sin(theta) * sin(psi) - sin(phi) * cos(psi))
    dz_dt = -u * sin(theta) + v * sin(phi) * cos(theta) + w * cos(phi) * cos(theta)
    residual_vector = vertcat(du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dphi_dt, dtheta_dt, dpsi_dt, dx_dt, dy_dt, dz_dt)
    return residual_vector
