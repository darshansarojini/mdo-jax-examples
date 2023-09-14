import csdl
import python_csdl_backend as pcb
import numpy as np


class C172InertialLoads(csdl.Model):
    def define(self):
        # Define inputs
        h = self.create_input('h', val=1000.0)  # Altitude, default value from provided code
        th = self.create_input('th', val=np.deg2rad(5.))  # theta angle in radians
        phi = self.create_input('phi', val=np.deg2rad(3.))  # phi angle in radians
        ref_pt = self.create_input('ref_pt', val=np.zeros(3))  # Reference point, default is a zero vector

        g = 9.803565306802405
        m, cg, I = self.c172_mp()

        th = th[0]
        phi = phi[0]

        F = np.array([-m * g * csdl.sin(th),
                      m * g * csdl.cos(th) * csdl.sin(phi),
                      m * g * csdl.cos(th) * csdl.cos(phi)])

        offset = cg - self.inputs['ref_pt']
        M = csdl.cross(offset, F)

        self.register_output('m', m)
        self.register_output('cg', cg)
        self.register_output('I', I)
        self.register_output('F', F)
        self.register_output('M', M)

    def c172_mp(self):
        m = 1043.2616
        Ixx = 1285.3154166
        Iyy = 1824.9309607
        Izz = 2666.89390765

        I = np.diag([Ixx, Iyy, Izz])
        cg = np.zeros(3)

        return m, cg, I


if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model

    model.add_subsystem('inertial_loads', C172InertialLoads())

    prob.setup()
    prob.run_model()

    print("Forces: ", prob['inertial_loads.F'])
    print("Moments: ", prob['inertial_loads.M'])
