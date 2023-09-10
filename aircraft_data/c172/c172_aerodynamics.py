import jax.numpy as jnp
import numpy as np


def c172_aerodynamics(h, Ma, alpha, delta_e, wing_area=16.2, wing_chord=1.49352, wing_span=10.91184):

    rho = 1.225
    a = 330

    V = Ma*a

    alpha_deg = jnp.rad2deg(alpha)
    delta_e_deg = jnp.rad2deg(delta_e)

    # Drag
    CD_alpha = 0.00033156 * alpha_deg ** 2 + \
               0.00192141 * alpha_deg + \
               0.03451242
    CD_delta_elev = 0.  # todo: fit a bivariate regression

    # Lift
    CL_alpha = 0.09460627 * alpha_deg + 0.16531678
    CL_delta_elev = -4.64968867e-06 * delta_e_deg ** 3 + \
                    3.95734084e-06 * delta_e_deg ** 2 + \
                    8.26663557e-03 * delta_e_deg + \
                    -1.81731015e-04

    # Pitching moment
    Cm_alpha = -0.00088295 * alpha_deg ** 2 + \
               -0.01230759 * alpha_deg + \
               0.01206867
    Cm_delta_elev = 1.11377133e-05 * delta_e_deg ** 3 + \
                    -9.96895700e-06 * delta_e_deg ** 2 + \
                    -2.03797109e-02 * delta_e_deg + \
                    1.37160466e-04

    # Final aerodynamic coefficients
    CL = CL_alpha + CL_delta_elev
    CD = CD_alpha + CD_delta_elev
    Cm = Cm_alpha + Cm_delta_elev
    CY = 0.
    Cl = 0.
    Cn = 0.

    qBar = 0.5 * rho * V ** 2
    L = qBar * wing_area * CL
    D = qBar * wing_area * CD
    Y = qBar * wing_area * CY
    l = qBar * wing_area * wing_span * Cl
    m = qBar * wing_area * wing_chord * Cm
    n = qBar * wing_area * wing_span * Cn

    F_wind = jnp.zeros(3)
    F_wind = F_wind.at[0].set(-D)
    F_wind = F_wind.at[1].set(Y)
    F_wind = F_wind.at[2].set(-L)

    M_wind = jnp.zeros(3)
    M_wind = M_wind.at[0].set(l)
    M_wind = M_wind.at[1].set(m)
    M_wind = M_wind.at[2].set(n)
    beta = 0.

    DCM_bw = jnp.zeros((3, 3))
    DCM_bw = DCM_bw.at[0, 0].set(jnp.cos(alpha) * jnp.cos(beta))
    DCM_bw = DCM_bw.at[0, 1].set(jnp.sin(beta))
    DCM_bw = DCM_bw.at[0, 2].set(jnp.sin(alpha) * jnp.cos(beta))
    DCM_bw = DCM_bw.at[1, 0].set(-jnp.cos(alpha) * jnp.sin(beta))
    DCM_bw = DCM_bw.at[1, 1].set(jnp.cos(beta))
    DCM_bw = DCM_bw.at[1, 2].set(-jnp.sin(alpha) * jnp.sin(beta))
    DCM_bw = DCM_bw.at[2, 0].set(-jnp.sin(alpha))
    DCM_bw = DCM_bw.at[2, 1].set(0)
    DCM_bw = DCM_bw.at[2, 2].set(jnp.cos(alpha))

    F = jnp.matmul(jnp.transpose(DCM_bw), F_wind)
    M = jnp.matmul(jnp.transpose(DCM_bw), M_wind)

    return F, M


if __name__ == "__main__":
    h = 1000
    Ma = 0.15
    alpha = np.deg2rad(5.)
    delta_e = np.deg2rad(0.)
    F, M = c172_aerodynamics(h=h, Ma=Ma, alpha=alpha, delta_e=delta_e)
    print("Forces: ", F)
    print("Moments: ", M)