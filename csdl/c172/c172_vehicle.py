import numpy as np
import csdl
import python_csdl_backend as pcb


class C172Vehicle(csdl.Model):

    def define(self):
        # Inputs
        x = self.create_input('x', shape=(3, ))  # design variables: AoA, elevator deflection, RPM
        h_input = self.create_input('h_input', shape=(1, ), val=1000.0, units='m')  # height in
        Ma_input = self.create_input('Ma_input', shape=(1, ), val=0.1)
        ref_pt_input = self.create_input('ref_pt_input', shape=(3, ), val=0., units='m')
        beta_input = self.create_input('beta_input', shape=(1, ), val=0.0, units='rad')

        alpha = x[0] * 1.
        delta_e = x[1] * 1.
        omega = x[2] * 1.

        self.register_output('alpha', alpha)
        self.register_output('delta_e', delta_e)
        self.register_output('omega', omega)

        a = 336.43470050484996  # m/s

        state_vector = csdl.create_output('state_vector', shape=(12,), val=np.empty((12,))
        state_vector[0] = Ma_input * a  # u
        state_vector[1] = 0.  # v
        state_vector[2] = 0.  # w

        state_vector[3] = 0.  # p
        state_vector[4] = 0.  # q
        state_vector[5] = 0.  # r

        state_vector[6] = 0.  # phi
        state_vector[7] = alpha*1.  # theta
        state_vector[8] = 0.  # psi

        state_vector[9] = 0.  # x
        state_vector[10] = 0.  # y
        state_vector[11] = h_input*1.  # z

        self.register_output('h', h_input)
        self.register_output('Ma', Ma_input)
        self.register_output('ref_pt', ref_pt_input)
        self.register_output('beta', beta_input)
        return