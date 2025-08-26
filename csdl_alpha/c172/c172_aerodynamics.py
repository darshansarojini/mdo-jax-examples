import csdl_alpha as csdl
import numpy as np

def aerodynamics(theta, delta_e, u, rho, Sw, chord, b):
    alpha = theta
    beta = csdl.Variable(name='beta', value=0.)
    vtas = u
    alpha_deg = alpha * (180/np.pi)
    delta_e_deg = delta_e * (180/np.pi)
    CL_alpha = 0.09460627 * alpha_deg + 0.16531678
    CL_delta_elev = -4.64968867e-06 * delta_e_deg ** 3 + 3.95734084e-06 * delta_e_deg ** 2 + 8.26663557e-03 * delta_e_deg - 1.81731015e-04
    CL = CL_alpha + CL_delta_elev
    CD_alpha = 0.00033156 * alpha_deg ** 2 + 0.00192141 * alpha_deg + 0.03451242
    CD = CD_alpha
    Cm_alpha = -0.00088295 * alpha_deg ** 2 - 0.01230759 * alpha_deg + 0.01206867
    Cm_delta_elev = 1.11377133e-05 * delta_e_deg ** 3 - 9.96895700e-06 * delta_e_deg ** 2 - 2.03797109e-02 * delta_e_deg + 1.37160466e-04
    Cm = Cm_alpha + Cm_delta_elev
    CY = csdl.Variable(name='CY', value=0.)
    Cl = csdl.Variable(name='Cl', value=0.)
    Cn = csdl.Variable(name='Cn', value=0.)
    qBar = 0.5 * rho * vtas ** 2
    L = qBar * Sw * CL
    D = qBar * Sw * CD
    Y = qBar * Sw * CY
    l = qBar * Sw * b * Cl
    m_aero = qBar * Sw * chord * Cm
    n = qBar * Sw * b * Cn
    F_wind = csdl.concatenate([-D, Y, -L], axis=0)
    M_wind = csdl.concatenate([l, m_aero, n], axis=0)
    ca, sa = csdl.cos(alpha), csdl.sin(alpha)
    cb, sb = csdl.cos(beta), csdl.sin(beta)
    DCM_bw_T = csdl.reshape(csdl.concatenate([ca*cb, -ca*sb, -sa, sb, cb, 0, sa*cb, -sa*sb, ca]), (3, 3))
    F_a = csdl.matvec(DCM_bw_T, F_wind)
    M_a = csdl.matvec(DCM_bw_T, M_wind)
    return F_a, M_a
