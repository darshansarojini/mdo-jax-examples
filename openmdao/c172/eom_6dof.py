import openmdao.api as om
import numpy as np

class Eom6DofCg(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        n = self.options['num_nodes']
        # Inputs
        self.add_input('state_vector', shape=(n, 12))
        self.add_input('m', shape=(1,))
        self.add_input('cg', shape=(3,))
        self.add_input('I', shape=(3, 3))
        self.add_input('Fa', shape=(n, 3))
        self.add_input('Ma', shape=(n, 3))
        self.add_input('Fp', shape=(n, 3))
        self.add_input('Mp', shape=(n, 3))
        self.add_input('Fi', shape=(n, 3))
        self.add_input('Mi', shape=(n, 3))

        # Outputs
        self.add_output('residual_vector', shape=(n, 12,))
        self.add_output('trim_residual', shape=(1,))

        self.declare_partials('residual_vector', ['state_vector', 'Fa', 'Ma', 'Fp', 'Mp', 'Fi', 'Mi'], method='fd')
        self.declare_partials('trim_residual', ['state_vector', 'Fa', 'Ma', 'Fp', 'Mp', 'Fi', 'Mi'], method='fd')

    def compute(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        m = inputs['m']
        cg, I, Fa, Ma, Fp, Mp, Fi, Mi = (inputs[key] for key in ('cg', 'I', 'Fa', 'Ma', 'Fp', 'Mp', 'Fi', 'Mi'))
        state_vector = inputs['state_vector']

        total_force = Fa + Fp + Fi
        Fx = total_force[:, 0]
        Fy = total_force[:, 1]
        Fz = total_force[:, 2]

        total_moment = Ma + Mp + Mi
        L = total_moment[:, 0]
        M = total_moment[:, 1]
        N = total_moment[:, 2]

        Ix = I[0, 0]
        Iy = I[1, 1]
        Iz = I[2, 2]
        Jxz = I[0, 2]

        u = state_vector[:, 0]
        v = state_vector[:, 1]
        w = state_vector[:, 2]

        p = state_vector[:, 3]
        q = state_vector[:, 4]
        r = state_vector[:, 5]

        phi = state_vector[:, 6]
        theta = state_vector[:, 7]
        psi = state_vector[:, 8]

        # Linear momentum equations
        du_dt = Fx / m + r * v - q * w
        dv_dt = Fy / m - r * u + p * w
        dw_dt = Fz / m + q * u - p * v

        # Angular momentum equations
        dp_dt = (L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
                 p * q * Jxz * (Ix + Iz - Iy)) / (Ix * Iz - Jxz ** 2)
        dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy
        dr_dt = (L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
                 q * r * Jxz * (Iz + Ix - Iy)) / (Ix * Iz - Jxz ** 2)

        # Angular Kinematic equations
        dtheta_dt = q * np.cos(phi) - r * np.sin(phi)
        dphi_dt = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
        dpsi_dt = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

        # Linear kinematic equations
        dx_dt = (np.cos(theta) * np.cos(psi) * u +
                 (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * v +
                 (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w)
        dy_dt = (np.cos(theta) * np.sin(psi) * u +
                 (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * v +
                 (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w)
        dz_dt = -u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + w * np.cos(
            phi) * np.cos(theta)

        residual_vector = np.array([du_dt, dv_dt, dw_dt,
                                    dp_dt, dq_dt, dr_dt,
                                    dtheta_dt, dphi_dt, dpsi_dt,
                                    dx_dt, dy_dt, dz_dt]).T

        outputs['residual_vector'] = residual_vector

        trim_residual = np.linalg.norm(residual_vector[:, 0:6])
        outputs['trim_residual'] = trim_residual

    # def compute_partials(self, inputs, partials):
    #     m = inputs['m'][0]
    #     cg, I, Fa, Ma, Fp, Mp, Fi, Mi = (inputs[key] for key in ('cg', 'I', 'Fa', 'Ma', 'Fp', 'Mp', 'Fi', 'Mi'))
    #     state_vector = inputs['state_vector']

    #     total_force = Fa + Fp + Fi
    #     Fx = total_force[0]
    #     Fy = total_force[1]
    #     Fz = total_force[2]

    #     total_moment = Ma + Mp + Mi
    #     L = total_moment[0]
    #     M = total_moment[1]
    #     N = total_moment[2]

    #     # dF_dFa = dF_dFi = dF_dFp = dM_dMa = dM_dMi = dM_dMp = np.identity(3)

    #     Ix = I[0, 0]
    #     Iy = I[1, 1]
    #     Iz = I[2, 2]
    #     Jxz = I[0, 2]

    #     u = state_vector[0]
    #     v = state_vector[1]
    #     w = state_vector[2]

    #     p = state_vector[3]
    #     q = state_vector[4]
    #     r = state_vector[5]

    #     phi = state_vector[6]
    #     theta = state_vector[7]
    #     psi = state_vector[8]

    #     # Linear momentum equations : a = [u v w], b = [p q r], c = [phi theta psi], d = [x y z]
    #     # du_dt = Fx / m + r * v - q * w
    #     # dv_dt = Fy / m - r * u + p * w
    #     # dw_dt = Fz / m + q * u - p * v

    #     da_dt_dx = np.array([[0., r, -q, 0., -w, v, 0., 0., 0., 0., 0., 0.,]
    #                          [-r, 0., p, w, 0., -r, 0., 0., 0., 0., 0., 0.,]
    #                          [q, -p, 0., -v, u, 0., 0., 0., 0., 0., 0., 0.]])
    #     da_dt_dF = np.diag([1/m, 1/m, 1/m])
    #     da_dt_dM = np.zeros((3,3))

    #     # Angular momentum equations
    #     dp_dt = (L * Iz + N * Jxz - q * r * (Iz ** 2 - Iz * Iy + Jxz ** 2) +
    #              p * q * Jxz * (Ix + Iz - Iy)) / (Ix * Iz - Jxz ** 2)
    #     dq_dt = (M + (Iz - Ix) * p * r - Jxz * (p ** 2 - r ** 2)) / Iy
    #     dr_dt = (L * Jxz + N * Ix + p * q * (Ix ** 2 - Ix * Iy + Jxz ** 2) -
    #              q * r * Jxz * (Iz + Ix - Iy)) / (Ix * Iz - Jxz ** 2)
        
    #     dp_dt_dx = np.array([0., 0., 0., xx, xx, xx, 0., 0., 0., 0., 0., 0.,])
    #     dq_dt_dx = np.array([0., 0., 0., xx, xx, xx, 0., 0., 0., 0., 0., 0.,])
    #     dr_dt_dx = np.array([0., 0., 0., xx, xx, xx, 0., 0., 0., 0., 0., 0.,])

    #     db_dt_dx = np.array([0., 0., 0., xx, xx, xx, 0., 0., 0., 0., 0., 0.,])
        
    #     db_dt_dF = np.zeros((3,3))
    #     db_dt_dM = np.array([Iz, 0., Jxz,
    #                          0., Iy*(Ix * Iz - Jxz ** 2), 0.,
    #                          Jxz, 0., Ix,])/ (Ix * Iz - Jxz ** 2)

    #     # Angular Kinematic equations
    #     dtheta_dt = q * np.cos(phi) - r * np.sin(phi)
    #     dphi_dt = p + (q * np.sin(phi) + r * np.cos(phi)) * np.tan(theta)
    #     dpsi_dt = (q * np.sin(phi) + r * np.cos(phi)) / np.cos(theta)

    #     # Linear kinematic equations
    #     dx_dt = (np.cos(theta) * np.cos(psi) * u +
    #              (np.sin(phi) * np.sin(theta) * np.cos(psi) - np.cos(phi) * np.sin(psi)) * v +
    #              (np.cos(phi) * np.sin(theta) * np.cos(psi) + np.sin(phi) * np.sin(psi)) * w)
    #     dy_dt = (np.cos(theta) * np.sin(psi) * u +
    #              (np.sin(phi) * np.sin(theta) * np.sin(psi) + np.cos(phi) * np.cos(psi)) * v +
    #              (np.cos(phi) * np.sin(theta) * np.sin(psi) - np.sin(phi) * np.cos(psi)) * w)
    #     dz_dt = -u * np.sin(theta) + v * np.sin(phi) * np.cos(theta) + w * np.cos(
    #         phi) * np.cos(theta)

    #     residual_vector = np.array([du_dt, dv_dt, dw_dt,
    #                                 dp_dt, dq_dt, dr_dt,
    #                                 dtheta_dt, dphi_dt, dpsi_dt,
    #                                 dx_dt, dy_dt, dz_dt])

    #     outputs['residual_vector'] = residual_vector

    #     trim_residual = np.linalg.norm(residual_vector[0:6])
    #     outputs['trim_residual'] = trim_residual

