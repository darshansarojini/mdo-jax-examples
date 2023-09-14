import csdl
import python_csdl_backend as pcb
import numpy as np


class C172InertialLoads(csdl.Model):
    def initialize(self):
        self.parameters.declare('name', default='C172MP', types=str)
        self.parameters.declare('num_nodes', default=1)
        self.parameters.declare('load_factor', default=1.)

    def define(self):
        shape = (1,)

        # region Mass properties
        area = self.create_input('wing_area', shape=shape, units='m^2', val=210.)
        ar = self.create_input('wing_AR', shape=shape, val=13.)

        # Random junk computations. The value is specified
        m = 1043.2616 + (1.2 * area + 0.6 * ar) * 0
        Ixx = 1285.3154166 + (0.34343 * area + 2121 * ar) * 0
        Iyy = 1824.9309607 + (1.9 * area + 0.1 * ar) * 0
        Izz = 2666.89390765 + (1.7 * area + 0.8 * ar) * 0
        Ixz = 0. + (0.3 * area + 456 * ar) * 0

        cgx = (0.21 * area + 312312 * ar) * 0
        cgy = (0.2343 * area + 321 * ar) * 0
        cgz = (0.2212 * area + 454 * ar) * 0

        self.register_output(
            name='mass',
            var=m)

        inertia_tensor = self.create_output('inertia_tensor', shape=(3, 3), val=0)
        inertia_tensor[0, 0] = csdl.reshape(Ixx, (1, 1))
        inertia_tensor[0, 2] = csdl.reshape(Ixz, (1, 1))
        inertia_tensor[1, 1] = csdl.reshape(Iyy, (1, 1))
        inertia_tensor[2, 0] = csdl.reshape(Ixz, (1, 1))
        inertia_tensor[2, 2] = csdl.reshape(Izz, (1, 1))

        cg_vector = self.create_output('cg_vector', shape=(3,), val=0)
        cg_vector[0] = cgx
        cg_vector[1] = cgy
        cg_vector[2] = cgz
        # endregion

        # region Inertial loads

        num_nodes = self.parameters['num_nodes']
        load_factor = self.parameters['load_factor']

        mass = csdl.expand(var=m, shape=(num_nodes, 1))

        ref_pt = self.create_input(name='ref_pt', shape=(3,), val=np.array([0, 0, 0]), units='m')

        # Inputs changing across conditions (segments)
        th = self.declare_variable('theta', shape=(num_nodes, 1), units='rad', val=0.)
        ph = self.declare_variable('phi', shape=(num_nodes, 1), units='rad', val=0.)

        g = 9.803565306802405 * load_factor

        F = self.create_output(name='F_inertial', shape=(num_nodes, 3))

        F[:, 0] = -mass * g * csdl.sin(th)
        F[:, 1] = mass * g * csdl.cos(th) * csdl.sin(ph)
        F[:, 2] = mass * g * csdl.cos(th) * csdl.cos(ph)

        r_vec = cg_vector - ref_pt
        r_vec = csdl.reshape(r_vec, (num_nodes, 3))

        M = self.create_output(name='M_inertial', shape=(num_nodes, 3))
        for n in range(num_nodes):
            M[n, :] = csdl.cross(r_vec, F[n, :], axis=1)
        # endregion


if __name__ == "__main__":

    sim = pcb.Simulator(C172InertialLoads())

    sim['theta'] = np.deg2rad(5.)
    sim['phi'] = np.deg2rad(3.)
    sim.run()

    print('Mass: ', sim['mass'])
    print('Center of gravity: ', sim['cg_vector'])
    print('Inertia tensor: ', sim['inertia_tensor'])
    print('Reference point: ', sim['ref_pt'])
    print("Forces: ", sim['F_inertial'])
    print("Moments: ", sim['M_inertial'])
