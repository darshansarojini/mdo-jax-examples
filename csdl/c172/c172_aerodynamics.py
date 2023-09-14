import numpy as np
import csdl
import python_csdl_backend as pcb


class C172Aerodynamics(csdl.Model):

    def define(self):
        # Inputs
        h = self.create_input('h', val=0.0)  # in meters
        Ma = self.create_input('Ma', val=0.0)
        alpha = self.create_input('alpha', val=0.0)  # in radians
        delta_e self.create_input('delta_e', val=0.0)  # in radians
        beta = self.create_input('beta', val=0.0)  # in radians

        rho = 1.1116589850558272  # kg/m^3
        a = 336.43470050484996  # m/s

        V = Ma * a
        alpha_deg = np.rad2deg(alpha)
        delta_e_deg = np.rad2deg(delta_e)

        CD_alpha = 0.00033156 * alpha_deg ** 2 + 0.00192141 * alpha_deg + 0.03451242
        CD_delta_elev = 0.  # placeholder for a potential future model
        CD = CD_alpha + CD_delta_elev

        CL_alpha = 0.09460627 * alpha_deg + 0.16531678
        CL_delta_elev = -4.64968867e-06 * delta_e_deg ** 3 + 3.95734084e-06 * delta_e_deg ** 2 + 8.26663557e-03 * delta_e_deg - 1.81731015e-04
        CL = CL_alpha + CL_delta_elev

        Cm_alpha = -0.00088295 * alpha_deg ** 2 - 0.01230759 * alpha_deg + 0.01206867
        Cm_delta_elev = 1.11377133e-05 * delta_e_deg ** 3 - 9.96895700e-06 * delta_e_deg ** 2 - 2.03797109e-02 * delta_e_deg + 1.37160466e-04
        Cm = Cm_alpha + Cm_delta_elev

        CY = 0.
        Cl = 0.
        Cn = 0.

        qBar = 0.5 * rho * V ** 2
        wing_area = 16.2
        wing_chord = 1.49352
        wing_span = 10.91184
        L = qBar * wing_area * CL
        D = qBar * wing_area * CD
        Y = qBar * wing_area * CY
        l = qBar * wing_area * wing_span * Cl
        m = qBar * wing_area * wing_chord * Cm
        n = qBar * wing_area * wing_span * Cn

        F_wind = np.array([-D, Y, -L])
        M_wind = np.array([l, m, n])

        DCM_bw = np.array([
            [csdl.cos(alpha) * csdl.cos(beta), csdl.sin(beta), csdl.sin(alpha) * csdl.cos(beta)],
            [-csdl.cos(alpha) * csdl.sin(beta), csdl.cos(beta), -csdl.sin(alpha) * csdl.sin(beta)],
            [-csdl.sin(alpha), 0., csdl.cos(alpha)]
        ])

        F = csdl.dot(DCM_bw.T, F_wind)
        M = csdl.dot(DCM_bw.T, M_wind)

        self.register_output('F', F)
        self.register_output('M', M)


if __name__ == "__main__":
    prob = om.Problem()
    prob.model.add_subsystem('aero', C172Aerodynamics())

    prob.setup()
    prob['aero.h'] = 1000
    prob['aero.Ma'] = 0.1
    prob['aero.alpha'] = np.deg2rad(0.)
    prob['aero.delta_e'] = np.deg2rad(-5.)

    prob.run_model()

    print("Forces:", prob['aero.F'])
    print("Moments:", prob['aero.M'])
