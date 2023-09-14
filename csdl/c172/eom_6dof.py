import csdl
import python_csdl_backend as pcb
import numpy as np

class Eom6DofCg(csdl.Model):
    def define(self):
        # Inputs
        state_vector = self.create_input('state_vector', shape=(12,))
        vector = self.create_input('m', shape=(1,))
        cg = self.create_input('cg', shape=(3,))
        I = self.create_input('I', shape=(3,3))
        Fa = self.create_input('Fa', shape=(3,))
        Ma = self.create_input('Ma', shape=(3,))
        Fp = self.create_input('Fp', shape=(3,))
        Mp = self.create_input('Mp', shape=(3,))
        Fi = self.create_input('Fi', shape=(3,))
        Mi = self.create_input('Mi', shape=(3,))

        total_force = Fa + Fp + Fi
        Fx = total_force[0]
        Fy = total_force[1]
        Fz = total_force[2]

        total_moment = Ma + Mp + Mi
        L = total_moment[0]
        M = total_moment[1]
        N = total_moment[2]

        Ix = I[0, 0]
        Iy = I[1, 1]
        Iz = I[2, 2]
        Jxz = I[0, 2]

        u = state_vector[0]
        v = state_vector[1]
        w = state_vector[2]

        p = state_vector[3]
        q = state_vector[4]
        r = state_vector[5]

        phi = state_vector[6]
        theta = state_vector[7]
        psi = state_vector[8]

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

        residual_vector = np.array([du_dt, dv_dt, dw_dt,
                                               dp_dt, dq_dt, dr_dt,
                                               dtheta_dt, dphi_dt, dpsi_dt,
                                               dx_dt, dy_dt, dz_dt])

        self.register_output('residual_vector', residual_vector)

        trim_residual = csdl.linalg.norm(residual_vector[0:6])
        self.register_output('trim_residual', trim_residual)
