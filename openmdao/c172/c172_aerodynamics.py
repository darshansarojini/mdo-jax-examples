import numpy as np
import openmdao.api as om


class C172Aerodynamics(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        n = self.options['num_nodes']
        # Inputs
        self.add_input('h', shape=(n,), val=0.0)  # in meters
        self.add_input('Ma', shape=(n,), val=0.0)
        self.add_input('alpha', shape=(n,), val=0.0)  # in radians
        self.add_input('delta_e', shape=(n,), val=0.0)  # in radians
        self.add_input('beta', shape=(n,), val=0.0)  # in radians

        # Outputs
        self.add_output('F', shape=(n, 3))
        self.add_output('M', shape=(n, 3))

        # self.declare_partials('*', '*', method='fd')
        self.declare_partials('F', '*', method='fd')
        self.declare_partials('M', '*', method='fd')

    def compute(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        h, Ma, alpha, delta_e, beta = inputs['h'], inputs['Ma'], inputs['alpha'], inputs['delta_e'], inputs['beta']

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

        F_wind = np.array([-D, Y, -L]).T
        M_wind = np.array([l, m, n]).T

        DCM_bw = np.zeros((num_nodes,3,3), dtype=float)
        DCM_bw[:,0,0] = np.cos(alpha) * np.cos(beta)
        DCM_bw[:,0,1] = np.sin(beta)
        DCM_bw[:,0,2] = np.sin(alpha) * np.cos(beta)
        DCM_bw[:,1,0] = -np.cos(alpha) * np.sin(beta)
        DCM_bw[:,1,1] = np.cos(beta)
        DCM_bw[:,1,2] = -np.sin(alpha) * np.sin(beta)
        DCM_bw[:,2,0] = -np.sin(alpha)
        DCM_bw[:,2,1] = 0.
        DCM_bw[:,2,2] = np.cos(alpha)
        
        # np.array([
        #     [np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)],
        #     [-np.cos(alpha) * np.sin(beta), np.cos(beta), -np.sin(alpha) * np.sin(beta)],
        #     [-np.sin(alpha), 0, np.cos(alpha)]
        # ])

        # outputs['F'] = np.dot(DCM_bw.T, F_wind)
        # outputs['M'] = np.dot(DCM_bw.T, M_wind)
        outputs['F'] = np.einsum('ijk,ij->ik', DCM_bw, F_wind)
        outputs['M'] = np.einsum('ijk,ij->ik', DCM_bw, M_wind)


if __name__ == "__main__":
    prob = om.Problem()
    num_nodes = 4
    prob.model.add_subsystem('aero', C172Aerodynamics(num_nodes=num_nodes))

    prob.setup()
    prob['aero.h'] = 1000
    prob['aero.Ma'] = 0.1
    prob['aero.alpha'] = np.deg2rad(0.)
    prob['aero.delta_e'] = np.deg2rad(-5.)

    prob.run_model()
    # prob.check_partials(compact_print=True)

    print("Forces:", prob['aero.F'])
    print("Moments:", prob['aero.M'])
