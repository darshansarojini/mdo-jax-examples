import numpy as np
import csdl
import python_csdl_backend as pcb


class C172Aerodynamics(csdl.Model):

    def initialize(self):
        self.parameters.declare(name='name', default='aerodynamics')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        num_nodes = self.parameters['num_nodes']

        # Inputs constant across conditions (segments)
        Sw = self.declare_variable(name='wing_area', shape=(1,), val=16.2, units='m**2')
        chord = self.declare_variable(name='wing_chord', shape=(1,), val=1.49352, units='m')
        b = self.declare_variable(name='wing_span', shape=(1,), val=10.91184, units='m')
        ref_pt = self.declare_variable(name='ref_pt', shape=(3,), units='m', val=0.)
        rho = self.create_input(name='rho', val=1.1116589850558272, shape=(1,))  # kg/m^3
        a = self.create_input(name='a', val=336.43470050484996, shape=(1,))  # m/s

        # Inputs changing across conditions (segments)
        mach_number = self.declare_variable('mach_number', shape=(num_nodes, 1), units='m/s')
        alpha = self.declare_variable('alpha', shape=(num_nodes, 1), units='m/s')
        delta_e = self.declare_variable('delta_e', shape=(num_nodes, 1), units='m/s')
        beta = self.declare_variable('beta', shape=(num_nodes, 1), val=0., units='m/s')

        vtas = mach_number * csdl.expand(var=a, shape=(num_nodes, 1))
        self.register_output(name='velocity', var=vtas)

        alpha_deg = alpha * 57.2958
        delta_e_deg = delta_e * 57.2958

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

        qBar = 0.5 * csdl.expand(var=rho, shape=(num_nodes, 1)) * vtas ** 2
        wing_area = csdl.expand(var=Sw, shape=(num_nodes, 1))
        wing_span = csdl.expand(var=b, shape=(num_nodes, 1))
        wing_chord = csdl.expand(var=chord, shape=(num_nodes, 1))

        L = qBar * wing_area * CL
        D = qBar * wing_area * CD
        Y = qBar * wing_area * CY
        l = qBar * wing_area * wing_span * Cl
        m = qBar * wing_area * wing_chord * Cm
        n = qBar * wing_area * wing_span * Cn

        F_wind = self.create_output(name='F_wind', shape=(num_nodes, 3))
        F_wind[:, 0] = -D
        F_wind[:, 1] = Y
        F_wind[:, 2] = -L

        M_wind = self.create_output(name='M_wind', shape=(num_nodes, 3))
        M_wind[:, 0] = l
        M_wind[:, 1] = m
        M_wind[:, 2] = n

        F = self.create_output(name='F_aerodynamics', shape=(num_nodes, 3))
        M = self.create_output(name='M_aerodynamics', shape=(num_nodes, 3))

        for ii in range(num_nodes):
            # https://www.mathworks.com/help/aeroblks/directioncosinematrixbodytowind.html
            DCM_bw = self.create_output(name=f'DCM_body_to_wind_{ii}', shape=(3, 3), val=0)
            DCM_bw[0:1, 0:1] = csdl.cos(alpha[ii, 0]) * csdl.cos(beta[ii, 0])
            DCM_bw[0:1, 1:2] = csdl.sin(beta[ii, 0])
            DCM_bw[0:1, 2:3] = csdl.sin(alpha[ii, 0]) * csdl.cos(beta[ii, 0])
            DCM_bw[1:2, 0:1] = -csdl.cos(alpha[ii, 0]) * csdl.sin(beta[ii, 0])
            DCM_bw[1:2, 1:2] = csdl.cos(beta[ii, 0])
            DCM_bw[1:2, 2:3] = -csdl.sin(alpha[ii, 0]) * csdl.sin(beta[ii, 0])
            DCM_bw[2:3, 0:1] = -csdl.sin(alpha[ii, 0])
            DCM_bw[2:3, 1:2] = alpha[ii, 0] * 0
            DCM_bw[2:3, 2:3] = csdl.cos(alpha[ii, 0])

            F[ii, :] = csdl.reshape(csdl.matvec(csdl.transpose(DCM_bw), csdl.reshape(F_wind[ii, :], (3,))), (1, 3))
            M[ii, :] = csdl.reshape(csdl.matvec(csdl.transpose(DCM_bw), csdl.reshape(M_wind[ii, :], (3,))), (1, 3))
            # todo: optimized einsum this
        return


if __name__ == "__main__":
    sim = pcb.Simulator(C172Aerodynamics())

    sim['alpha'] = np.deg2rad(0.)
    sim['mach_number'] = 0.1
    sim['delta_e'] = np.deg2rad(-5.)
    sim.run()

    print('Forces: ', sim['F_aerodynamics'])
    print('Thrust: ', sim['M_aerodynamics'])
