import openmdao.api as om
import numpy as np


class C172InertialLoads(om.ExplicitComponent):
    def setup(self):
        # Define inputs
        self.add_input('h', val=1000.0)  # Altitude, default value from provided code
        self.add_input('th', val=np.deg2rad(5.))  # theta angle in radians
        self.add_input('phi', val=np.deg2rad(3.))  # phi angle in radians
        self.add_input('ref_pt', val=np.zeros(3))  # Reference point, default is a zero vector

        # Define outputs
        self.add_output('m', shape=(1,), units='kg')  # Mass
        self.add_output('cg', shape=(3,), units='m')  # Center of gravity
        self.add_output('I', shape=(3, 3), units='N')  # Intertia matrix
        self.add_output('F', shape=(3,), units='N')  # Force vector
        self.add_output('M', shape=(3,), units='N*m')  # Moment vector

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        g = 9.803565306802405
        m, cg, I = self.c172_mp()

        th = inputs['th'][0]
        phi = inputs['phi'][0]

        F = np.array([-m * g * np.sin(th),
                      m * g * np.cos(th) * np.sin(phi),
                      m * g * np.cos(th) * np.cos(phi)])

        offset = cg - inputs['ref_pt']
        M = np.cross(offset, F)

        outputs['m'] = m
        outputs['cg'] = cg
        outputs['I'] = I

        outputs['F'] = F
        outputs['M'] = M

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
