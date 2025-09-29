import warp as wp
from typing import Tuple

@wp.func
def aerodynamics(theta: float, delta_e: float, u: float, rho: float, Sw: float, chord: float, b: float) -> Tuple[wp.vec3f, wp.vec3f]:
    cl_a1 = 0.09460627
    cl_a0 = 0.16531678
    cld_c3 = -4.64968867e-06
    cld_c2 = 0.0
    cld_c1 = 8.26663557e-03
    cld_c0 = -1.81731015e-04
    cd_q2 = 0.00033156
    cd_q1 = 0.00192141
    cd_q0 = 0.03451242
    cm_q2 = -0.00088295
    cm_q1 = -0.01230759
    cm_q0 = 0.01206867
    cmd_c3 = 1.11377133e-05
    cmd_c2 = 0.0
    cmd_c1 = -2.03797109e-02
    cmd_c0 = 1.37160466e-04
    
    alpha = theta
    beta = 0.0
    pi = 3.14159265358979323846
    alpha_deg = theta * (180.0 / pi)
    de_deg = delta_e * (180.0 / pi)
    
    CL_alpha = cl_a1 * alpha_deg + cl_a0
    CL_de = ((cld_c3 * de_deg + cld_c2) * de_deg + cld_c1) * de_deg + cld_c0
    CL = CL_alpha + CL_de
    
    CD = (cd_q2 * alpha_deg + cd_q1) * alpha_deg + cd_q0
    
    Cm_alpha = (cm_q2 * alpha_deg + cm_q1) * alpha_deg + cm_q0
    Cm_de = ((cmd_c3 * de_deg + cmd_c2) * de_deg + cmd_c1) * de_deg + cmd_c0
    Cm = Cm_alpha + Cm_de
    
    Y = 0.0
    Cl = 0.0
    Cn = 0.0
    
    qbar = 0.5 * rho * u * u
    L = qbar * Sw * CL
    D = qbar * Sw * CD
    l_m = qbar * Sw * b * Cl
    m_m = qbar * Sw * chord * Cm
    n_m = qbar * Sw * b * Cn
    
    F_wind = wp.vec3f(-D, 0.0, -L)
    M_wind = wp.vec3f(l_m, m_m, n_m)
    
    ca = wp.cos(alpha)
    sa = wp.sin(alpha)
    cb = wp.cos(beta)
    sb = wp.sin(beta)
    
    DCM_wb = wp.mat33(ca * cb, -ca * sb, -sa, sb, cb, 0.0, sa * cb, -sa * sb, ca)
    
    F_a = DCM_wb * F_wind
    M_a = DCM_wb * M_wind
    
    return F_a, M_a
