import numpy as np
import openmdao.api as om


class C172Vehicle(om.ExplicitComponent):

    def setup(self):
        # Inputs
        self.add_input('x', shape=(3, ))  # design variables: AoA, elevator deflection, RPM
        self.add_input('h_input', shape=(1, ), val=1000.0, units='m')  # height in
        self.add_input('Ma_input', shape=(1, ), val=0.1)
        self.add_input('ref_pt_input', shape=(3, ), val=0., units='m')
        self.add_input('beta_input', shape=(1, ), val=0.0, units='rad')

        # Outputs
        self.add_output('alpha', shape=(1,))
        self.add_output('delta_e', shape=(1,))
        self.add_output('omega', shape=(1,))

        self.add_output('h', shape=(1,))
        self.add_output('Ma', shape=(1,))
        self.add_output('ref_pt', shape=(3,))
        self.add_output('beta', shape=(1,))

        self.add_output('state_vector', shape=(12,))

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        alpha = inputs['x'][0] * 1.
        delta_e = inputs['x'][1] * 1.
        omega = inputs['x'][2] * 1.

        outputs['alpha'] = alpha
        outputs['delta_e'] = delta_e
        outputs['omega'] = omega

        a = 336.43470050484996  # m/s

        state_vector = np.empty((12, ))
        state_vector[0] = inputs['Ma_input'] * a  # u
        state_vector[1] = 0.  # v
        state_vector[2] = 0.  # w

        state_vector[3] = 0.  # p
        state_vector[4] = 0.  # q
        state_vector[5] = 0.  # r

        state_vector[6] = 0.  # phi
        state_vector[7] = alpha  # theta
        state_vector[8] = 0.  # psi

        state_vector[9] = 0.  # x
        state_vector[10] = 0.  # y
        state_vector[11] = inputs['h_input']  # z

        outputs['h'] = inputs['h_input']
        outputs['Ma'] = inputs['Ma_input']
        outputs['ref_pt'] = inputs['ref_pt_input']
        outputs['beta'] = inputs['beta_input']
        outputs['state_vector'] = state_vector
        return