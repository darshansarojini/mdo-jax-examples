import numpy as np
import csdl_alpha as csdl

class C172TrimModel:
    def __init__(self):
        mach_number = csdl.Variable(name="mach_number", value=0.117)
        a = 336.43470050484996
        u = mach_number * a
        v = csdl.Variable(name="v", value=0.0)
        w = csdl.Variable(name="w", value=0.0)
        p = csdl.Variable(name="p", value=0.0)
        q = csdl.Variable(name="q", value=0.0)
        r = csdl.Variable(name="r", value=0.0)
        phi = csdl.Variable(name="phi", value=0.0)
        psi = csdl.Variable(name="psi", value=0.0)
        x = csdl.Variable(name="x", value=0.0)
        y = csdl.Variable(name="y", value=0.0)
        z = csdl.Variable(name="z", value=0.0)

        theta   = csdl.Variable(name="theta",   value=np.deg2rad(8.7))
        delta_e = csdl.Variable(name="delta_e", value=np.deg2rad(-7.8))
        omega   = csdl.Variable(name="omega",   value=1734.0)
        theta.set_as_design_variable  (lower=np.deg2rad(-10), upper=np.deg2rad(10))
        delta_e.set_as_design_variable(lower=np.deg2rad(-15), upper=np.deg2rad(15))
        omega.set_as_design_variable  (lower=1000.0,          upper=2800.0)

        Sw = 16.2; chord = 1.49352; b = 10.91184
        rho = 1.1116589850558272; m_val = 1043.2616
        Ixx = 1285.3154166; Iyy = 1824.9309607; Izz = 2666.89390765; Ixz = 0.0
        g = 9.803565306802405; prop_radius = 0.94

        thrust_origin = csdl.Variable(name="thrust_origin", value=np.array([0.0, 0.0, 0.0]))
        ref_pt        = csdl.Variable(name="ref_pt",        value=np.array([0.0, 0.0, 0.0]))
        cg_vector     = csdl.Variable(name="cg_vector",     value=np.array([0.0, 0.0, 0.0]))

        self.m = csdl.Variable(name="mass", value=m_val)
        inertia_matrix = np.array([[Ixx, 0.0, Ixz], [0.0, Iyy, 0.0], [Ixz, 0.0, Izz]])
        self.inertia_tensor = csdl.Variable(name="inertia_tensor", value=inertia_matrix)

        alpha = theta
        beta = csdl.Variable(name="beta", value=0.0)
        vtas = u
        alpha_deg   = alpha   * (180/np.pi)
        delta_e_deg = delta_e * (180/np.pi)

        CL_alpha      = 0.09460627 * alpha_deg + 0.16531678
        CL_delta_elev = -4.64968867e-06 * delta_e_deg**3 + 3.95734084e-06 * delta_e_deg**2 + 8.26663557e-03 * delta_e_deg - 1.81731015e-04
        CL = CL_alpha + CL_delta_elev
        CD = 0.00033156 * alpha_deg**2 + 0.00192141 * alpha_deg + 0.03451242
        Cm_alpha      = -0.00088295 * alpha_deg**2 - 0.01230759 * alpha_deg + 0.01206867
        Cm_delta_elev =  1.11377133e-05 * delta_e_deg**3 - 9.96895700e-06 * delta_e_deg**2 - 2.03797109e-02 * delta_e_deg + 1.37160466e-04
        Cm = Cm_alpha + Cm_delta_elev
        CY = 0.0; Cl = 0.0; Cn = 0.0

        qBar = 0.5 * rho * vtas**2
        L = qBar * Sw * CL
        D = qBar * Sw * CD
        Y = qBar * Sw * CY
        l = qBar * Sw * b     * Cl
        m_aero = qBar * Sw * chord * Cm
        n = qBar * Sw * b     * Cn

        F_wind = csdl.concatenate([-D, Y, -L], axis=0)
        M_wind = csdl.concatenate([l, m_aero, n], axis=0)

        ca, sa = csdl.cos(alpha), csdl.sin(alpha)
        cb, sb = csdl.cos(beta),  csdl.sin(beta)

        DCM_bw_T = csdl.reshape(
            csdl.concatenate([ca*cb, -ca*sb, -sa,
                              sb,     cb,     0.0,
                              sa*cb, -sa*sb,  ca]), (3, 3)
        )
        F_a = csdl.matvec(DCM_bw_T, F_wind)
        M_a = csdl.matvec(DCM_bw_T, M_wind)

        omega_RAD = (omega * 2 * np.pi) / 60.0
        J  = (np.pi * vtas) / (omega_RAD * prop_radius)
        Ct = -0.1692121 * J**2 + 0.03545196 * J + 0.10446359
        T  = ((2/np.pi)**2) * rho * (omega_RAD * prop_radius)**2 * Ct
        F_p = csdl.concatenate([T, 0.0, 0.0], axis=0)
        M_p = csdl.cross(ref_pt - thrust_origin, F_p)

        F_i = csdl.concatenate(
            [-self.m * g * csdl.sin(theta),
              self.m * g * csdl.cos(theta) * csdl.sin(phi),
              self.m * g * csdl.cos(theta) * csdl.cos(phi)],
            axis=0
        )
        M_i = csdl.cross(cg_vector - ref_pt, F_i)

        total_force  = F_a + F_p + F_i
        Fx, Fy, Fz   = total_force[0], total_force[1], total_force[2]
        total_moment = M_a + M_p + M_i
        L_mom, M_mom, N_mom = total_moment[0], total_moment[1], total_moment[2]

        du_dt = Fx / self.m + r * v - q * w
        dv_dt = Fy / self.m - r * u + p * w
        dw_dt = Fz / self.m + q * u - p * v

        dp_dt = (L_mom * Izz + N_mom * Ixz - q * r * (Izz**2 - Izz*Iyy + Ixz**2) + p * q * Ixz * (Ixx + Izz - Iyy)) / (Ixx * Izz - Ixz**2)
        dq_dt = (M_mom + (Izz - Ixx) * p * r - Ixz * (p**2 - r**2)) / Iyy
        dr_dt = (L_mom * Ixz + N_mom * Ixx + p * q * (Ixx**2 - Ixx*Iyy + Ixz**2) - q * r * Ixz * (Izz + Ixx - Iyy)) / (Ixx * Izz - Ixz**2)

        dtheta_dt = q * csdl.cos(phi) - r * csdl.sin(phi)
        dphi_dt   = p + (q * csdl.sin(phi) + r * csdl.cos(phi)) * csdl.tan(theta)
        dpsi_dt   = (q * csdl.sin(phi) + r * csdl.cos(phi)) / csdl.cos(theta)

        dx_dt = u * csdl.cos(theta) * csdl.cos(psi) + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.cos(psi) - csdl.cos(phi) * csdl.sin(psi)) + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.cos(psi) + csdl.sin(phi) * csdl.sin(psi))
        dy_dt = u * csdl.cos(theta) * csdl.sin(psi) + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.sin(psi) + csdl.cos(phi) * csdl.cos(psi)) + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.sin(psi) - csdl.sin(phi) * csdl.cos(psi))
        dz_dt = -u * csdl.sin(theta) + v * csdl.sin(phi) * csdl.cos(theta) + w * csdl.cos(phi) * csdl.cos(theta)

        residual_vector = csdl.concatenate(
            [du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dphi_dt, dtheta_dt, dpsi_dt, dx_dt, dy_dt, dz_dt]
        )
        xddot = residual_vector[0:6]
        trim_objective = csdl.norm(xddot)
        trim_objective.set_as_objective()

        self.theta = theta
        self.delta_e = delta_e
        self.omega = omega
        self.mach_number = mach_number
        self.trim_objective = trim_objective
