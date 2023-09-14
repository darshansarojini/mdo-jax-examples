import csdl
import python_csdl_backend as pcb
import numpy as np

class C172Propulsion(csdl.Model):
    def initialize(self):
        self.parameters.declare(name='name', default='propulsion')
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        name = self.parameters['name']
        num_nodes = self.parameters['num_nodes']

        # # Inputs constant across conditions (segments)
        prop_radius = self.declare_variable(name='propeller_radius', shape=(1,), units='m')
        ref_pt = self.declare_variable(name='ref_pt', shape=(3,), units='m', val=0.)
        thrust_origin = self.declare_variable(name='thrust_origin', shape=(3,), val=0., units='m')
        rho = self.create_input(name='rho', val=1.1116589850558272, shape=(1, ))  # kg/m^3
        a = self.create_input(name='a', val=336.43470050484996, shape=(1, ))  # m/s

        # Inputs changing across conditions (segments)
        omega = self.declare_variable('omega', shape=(num_nodes, 1), units='rpm')
        mach_number = self.declare_variable('mach_number', shape=(num_nodes, 1), units='m/s')

        vtas = mach_number * csdl.expand(var=a, shape=(num_nodes, 1))
        self.register_output(name='velocity', var=vtas)

        omega_RAD = (omega * 2 * np.pi) / 60.0  # rad/s
        J = (np.pi * vtas) / (omega_RAD * csdl.expand(prop_radius, shape=(num_nodes, 1)))  # non-dimensional Advance ratio
        self.register_output(name='advance_ratio', var=J)

        Ct_interp = -0.1692121 * J ** 2 + 0.03545196 * J + 0.10446359  # non-dimensional
        self.register_output(name='Ct', var=Ct_interp)

        T = (2 / np.pi) ** 2 * csdl.expand(var=rho, shape=(num_nodes, 1)) * (omega_RAD * csdl.expand(prop_radius, shape=(num_nodes, 1))) ** 2 * Ct_interp
        self.register_output(name='T', var=T)

        F = self.create_output(name='F_propulsion', shape=(num_nodes, 3))
        F[:, 0] = T
        F[:, 1] = T * 0.
        F[:, 2] = T * 0.

        offset = ref_pt - thrust_origin
        M = self.create_output(name='M_propulsion', shape=(num_nodes, 3))
        M[:, 0] = T * 0
        for ii in range(num_nodes):
            M[ii, 1] = F[ii, 0] * csdl.reshape(offset[2], (1, 1)) + F[ii, 2] * csdl.reshape(offset[0], (1, 1))
        M[:, 2] = T * 0
        return


if __name__ == "__main__":
    sim = pcb.Simulator(C172Propulsion())

    sim['propeller_radius'] = 0.94  # m
    sim['mach_number'] = 0.1
    sim['omega'] = 2800.
    sim.run()

    print('Thrust: ', sim['T'])
    print('Forces: ', sim['F_propulsion'])
    print('Thrust: ', sim['M_propulsion'])
