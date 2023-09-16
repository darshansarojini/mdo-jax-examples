import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
def c172_aerodynamics(Ma, alpha, delta_e,
                      beta=0.,
                      wing_area=16.2, wing_chord=1.49352, wing_span=10.91184):

    num_nodes = alpha.shape[0]

    # alpha_data = np.array([-7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10, 15, 17, 18, 19.5])  # degree
    # delta_aile_data = np.array([-15, -10, -5, -2.5, 0, 5, 10, 15, 20])  # degree
    # delta_elev_data = np.array([-26, -20, -10, -5, 0, 7.5, 15, 22.5, 28])  # degree

    rho = 1.1116589850558272  # kg/m^3
    a = 336.43470050484996  # m/s

    V = Ma*a

    alpha_deg = jnp.rad2deg(alpha)
    delta_e_deg = jnp.rad2deg(delta_e)

    # region CD
    CD_alpha = 0.00033156 * alpha_deg ** 2 + \
               0.00192141 * alpha_deg + \
               0.03451242
    # CD_data = np.array([0.044, 0.034, 0.03, 0.03, 0.036, 0.048, 0.067, 0.093, 0.15, 0.169, 0.177, 0.184])
    # CD_alpha = jnp.interp(alpha_deg, alpha_data, CD_data)

    CD_delta_elev = 0.  # todo: fit a bivariate regression

    CD = CD_alpha + CD_delta_elev
    # endregion

    # region CL
    # CL_alpha
    # CL_data = np.array([-0.571, -0.321, -0.083, 0.148, 0.392, 0.65, 0.918, 1.195, 1.659, 1.789, 1.84, 1.889])
    # CL_alpha = jnp.interp(alpha_deg, alpha_data, CL_data)
    CL_alpha = 0.09460627 * alpha_deg + 0.16531678

    # CL_delta_elev
    # CL_delta_elev_data = np.array([-0.132, -0.123, -0.082, -0.041, 0, 0.061, 0.116, 0.124, 0.137])
    # CL_delta_elev = jnp.interp(delta_e_deg, delta_elev_data, CL_delta_elev_data)
    CL_delta_elev = -4.64968867e-06 * delta_e_deg ** 3 + \
                    3.95734084e-06 * delta_e_deg ** 2 + \
                    8.26663557e-03 * delta_e_deg + \
                    -1.81731015e-04
    CL = CL_alpha + CL_delta_elev
    # endregion

    # region Cm
    # Cm_alpha
    # Cm_data = np.array(
    #     [0.0597, 0.0498, 0.0314, 0.0075, -0.0248, -0.068, -0.1227, -0.1927, -0.3779, -0.4605, -0.5043, -0.5496, ])
    # Cm_alpha = jnp.interp(alpha_deg, alpha_data, Cm_data)
    Cm_alpha = -0.00088295 * alpha_deg ** 2 + \
               -0.01230759 * alpha_deg + \
               0.01206867

    # Cm_delta_elev
    # Cm_delta_elev_data = np.array([0.3302, 0.3065, 0.2014, 0.1007, -0.0002, -0.1511, -0.2863, -0.3109, -0.345])
    # Cm_delta_elev = jnp.interp(delta_e_deg, delta_elev_data, Cm_delta_elev_data)
    Cm_delta_elev = 1.11377133e-05 * delta_e_deg ** 3 + \
                    -9.96895700e-06 * delta_e_deg ** 2 + \
                    -2.03797109e-02 * delta_e_deg + \
                    1.37160466e-04
    Cm = Cm_alpha + Cm_delta_elev
    # endregion

    # region Cy
    CY = 0.
    # endregion

    # region Cl
    Cl = 0.
    # endregion

    # region Cn
    Cn = 0.
    # endregion

    # region Loads in wind axis
    qBar = 0.5 * rho * V ** 2
    L = qBar * wing_area * CL
    D = qBar * wing_area * CD
    Y = qBar * wing_area * CY
    l = qBar * wing_area * wing_span * Cl
    m = qBar * wing_area * wing_chord * Cm
    n = qBar * wing_area * wing_span * Cn

    F_wind = jnp.zeros((num_nodes, 3))
    F_wind = F_wind.at[:, 0].set(-D)
    F_wind = F_wind.at[:, 1].set(Y)
    F_wind = F_wind.at[:, 2].set(-L)

    M_wind = jnp.zeros((num_nodes, 3))
    M_wind = M_wind.at[:, 0].set(l)
    M_wind = M_wind.at[:, 1].set(m)
    M_wind = M_wind.at[:, 2].set(n)
    # endregion

    DCM_bw = jnp.zeros((num_nodes, 3, 3))
    DCM_bw = DCM_bw.at[:, 0, 0].set(jnp.cos(alpha) * jnp.cos(beta))
    DCM_bw = DCM_bw.at[:, 0, 1].set(jnp.sin(beta))
    DCM_bw = DCM_bw.at[:, 0, 2].set(jnp.sin(alpha) * jnp.cos(beta))
    DCM_bw = DCM_bw.at[:, 1, 0].set(-jnp.cos(alpha) * jnp.sin(beta))
    DCM_bw = DCM_bw.at[:, 1, 1].set(jnp.cos(beta))
    DCM_bw = DCM_bw.at[:, 1, 2].set(-jnp.sin(alpha) * jnp.sin(beta))
    DCM_bw = DCM_bw.at[:, 2, 0].set(-jnp.sin(alpha))
    DCM_bw = DCM_bw.at[:, 2, 1].set(0)
    DCM_bw = DCM_bw.at[:, 2, 2].set(jnp.cos(alpha))

    # DCM_bw_T = jnp.transpose(DCM_bw)
    F = jnp.einsum('ijk,ij->ik', DCM_bw, F_wind)  # todo: maybe remove einsum
    M = jnp.einsum('ijk,ij->ik', DCM_bw, M_wind)

    return F, M


if __name__ == "__main__":
    Ma = np.full(shape=(4,), fill_value=0.1)
    alpha = np.full(shape=(4,), fill_value=np.deg2rad(0.))
    delta_e = np.full(shape=(4,), fill_value=np.deg2rad(-5.))
    F, M = c172_aerodynamics(Ma=Ma, alpha=alpha, delta_e=delta_e)
    print("Forces: ", F)
    print("Moments: ", M)