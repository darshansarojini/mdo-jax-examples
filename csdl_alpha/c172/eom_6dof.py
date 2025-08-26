import csdl_alpha as csdl

def eom_6dof(u,v,w,p,q,r,phi,theta,psi,F_total,M_total,m_val,Ixx,Iyy,Izz,Ixz):
    Fx, Fy, Fz = F_total[0], F_total[1], F_total[2]
    L_mom, M_mom, N_mom = M_total[0], M_total[1], M_total[2]
    du_dt = Fx / m_val + r * v - q * w
    dv_dt = Fy / m_val - r * u + p * w
    dw_dt = Fz / m_val + q * u - p * v
    dp_dt = (L_mom * Izz + N_mom * Ixz - q * r * (Izz ** 2 - Izz * Iyy + Ixz ** 2) + p * q * Ixz * (Ixx + Izz - Iyy)) / (Ixx * Izz - Ixz ** 2)
    dq_dt = (M_mom + (Izz - Ixx) * p * r - Ixz * (p ** 2 - r ** 2)) / Iyy
    dr_dt = (L_mom * Ixz + N_mom * Ixx + p * q * (Ixx ** 2 - Ixx * Iyy + Ixz ** 2) - q * r * Ixz * (Izz + Ixx - Iyy)) / (Ixx * Izz - Ixz ** 2)
    dtheta_dt = q * csdl.cos(phi) - r * csdl.sin(phi)
    dphi_dt = p + (q * csdl.sin(phi) + r * csdl.cos(phi)) * csdl.tan(theta)
    dpsi_dt = (q * csdl.sin(phi) + r * csdl.cos(phi)) / csdl.cos(theta)
    dx_dt = u * csdl.cos(theta) * csdl.cos(psi) + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.cos(psi) - csdl.cos(phi) * csdl.sin(psi)) + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.cos(psi) + csdl.sin(phi) * csdl.sin(psi))
    dy_dt = u * csdl.cos(theta) * csdl.sin(psi) + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.sin(psi) + csdl.cos(phi) * csdl.cos(psi)) + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.sin(psi) - csdl.sin(phi) * csdl.cos(psi))
    dz_dt = -u * csdl.sin(theta) + v * csdl.sin(phi) * csdl.cos(theta) + w * csdl.cos(phi) * csdl.cos(theta)
    residual_vector = csdl.concatenate([du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dphi_dt, dtheta_dt, dpsi_dt, dx_dt, dy_dt, dz_dt])
    return residual_vector
