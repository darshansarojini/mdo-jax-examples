import csdl
import python_csdl_backend as pcb
import numpy as np

class Eom6DofCg(csdl.Model):
    eom_model_name = 'EulerEoM'

    def initialize(self):
        self.parameters.declare(name='name', default=self.eom_model_name)
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        num_nodes = self.parameters['num_nodes']

        ref_pt = self.declare_variable(name='ref_pt', shape=(3,), units='m', val=0.)

        # region Inputs
        # Aerodynamic Loads
        Fa = self.declare_variable(name='F_a', shape=(num_nodes, 3), units='N')
        Ma = self.declare_variable(name='M_a', shape=(num_nodes, 3), units='N*m')
        # Propulsion Loads
        Fp = self.declare_variable(name='F_p', shape=(num_nodes, 3), units='N')
        Mp = self.declare_variable(name='M_p', shape=(num_nodes, 3), units='N*m')
        # Inertial Loads
        Fi = self.declare_variable(name='F_i', shape=(num_nodes, 3), units='N')
        Mi = self.declare_variable(name='M_i', shape=(num_nodes, 3), units='N*m')
        # State vector
        u = self.declare_variable(name='u', shape=(num_nodes, 1), units='m/s')
        v = self.declare_variable(name='v', shape=(num_nodes, 1), units='m/s', val=0.)
        w = self.declare_variable(name='w', shape=(num_nodes, 1), units='m/s', val=0.)
        p = self.declare_variable(name='p', shape=(num_nodes, 1), units='rad/s', val=0.)
        q = self.declare_variable(name='q', shape=(num_nodes, 1), units='rad/s', val=0.)
        r = self.declare_variable(name='r', shape=(num_nodes, 1), units='rad/s', val=0.)
        phi = self.declare_variable(name='phi', shape=(num_nodes, 1), units='rad', val=0.)
        theta = self.declare_variable(name='theta', shape=(num_nodes, 1), units='rad')
        psi = self.declare_variable(name='psi', shape=(num_nodes, 1), units='rad', val=0.)
        x = self.declare_variable(name='x', shape=(num_nodes, 1), units='m', val=0.)
        y = self.declare_variable(name='y', shape=(num_nodes, 1), units='m', val=0.)
        z = self.declare_variable(name='z', shape=(num_nodes, 1), units='m', val=0.)
        # Mass properties
        m = self.declare_variable('m', shape=(1, ))
        # cg = self.declare_variable('cg', shape=(3, ), val=0.)
        I = self.declare_variable('I', shape=(3, 3))
        # endregion

        # region Calculations
        total_force = Fa + Fp + Fi
        Fx = total_force[:, 0]
        Fy = total_force[:, 1]
        Fz = total_force[:, 2]

        total_moment = Ma + Mp + Mi
        L = total_moment[:, 0]
        M = total_moment[:, 1]
        N = total_moment[:, 2]

        mass = csdl.expand(var=m, shape=(num_nodes, 1))
        Ix = csdl.expand(csdl.reshape(I[0, 0], (1, )), shape=(num_nodes, 1))
        Iy = csdl.expand(csdl.reshape(I[1, 1], (1, )), shape=(num_nodes, 1))
        Iz = csdl.expand(csdl.reshape(I[2, 2], (1, )), shape=(num_nodes, 1))
        Jxz = csdl.expand(csdl.reshape(I[0, 2], (1, )), shape=(num_nodes, 1))

        # Linear momentum equations
        du_dt = Fx / mass + r * v - q * w
        dv_dt = Fy / mass - r * u + p * w
        dw_dt = Fz / mass + q * u - p * v

        # Angular momentum equations
        dp_dt = (L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
                 p * q * Jxz * (Ix + Iz - Iy)) / (Ix * Iz - Jxz ** 2)
        dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy
        dr_dt = (L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
                 q * r * Jxz * (Iz + Ix - Iy)) / (Ix * Iz - Jxz ** 2)

        # Angular Kinematic equations
        dtheta_dt = q * csdl.cos(phi) - r * csdl.sin(phi)
        dphi_dt = p + (q * csdl.sin(phi) + r * csdl.cos(phi)) * csdl.tan(theta)
        dpsi_dt = (q * csdl.sin(phi) + r * csdl.cos(phi)) / csdl.cos(theta)

        # Linear kinematic equations
        dx_dt = (csdl.cos(theta) * csdl.cos(psi) * u +
                 (csdl.sin(phi) * csdl.sin(theta) * csdl.cos(psi) - csdl.cos(phi) * csdl.sin(psi)) * v +
                 (csdl.cos(phi) * csdl.sin(theta) * csdl.cos(psi) + csdl.sin(phi) * csdl.sin(psi)) * w)
        dy_dt = (csdl.cos(theta) * csdl.sin(psi) * u +
                 (csdl.sin(phi) * csdl.sin(theta) * csdl.sin(psi) + csdl.cos(phi) * csdl.cos(psi)) * v +
                 (csdl.cos(phi) * csdl.sin(theta) * csdl.sin(psi) - csdl.sin(phi) * csdl.cos(psi)) * w)
        dz_dt = -u * csdl.sin(theta) + v * csdl.sin(phi) * csdl.cos(theta) + w * csdl.cos(
            phi) * csdl.cos(theta)
        # endregion

        # region Outputs
        res_vector = self.create_output(name='residual_vector', shape=(num_nodes, 12))
        res_vector[:, 0] = du_dt
        res_vector[:, 1] = dv_dt
        res_vector[:, 2] = dw_dt
        res_vector[:, 3] = dp_dt
        res_vector[:, 4] = dq_dt
        res_vector[:, 5] = dr_dt
        res_vector[:, 6] = dphi_dt
        res_vector[:, 7] = dtheta_dt
        res_vector[:, 8] = dpsi_dt
        res_vector[:, 9] = dx_dt
        res_vector[:, 10] = dy_dt
        res_vector[:, 11] = dz_dt
        # endregion

        return
