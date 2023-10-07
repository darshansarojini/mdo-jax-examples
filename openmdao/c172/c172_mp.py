import openmdao.api as om
import numpy as np


class C172InertialLoads(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        n = self.options['num_nodes']
        # Define inputs
        self.add_input('h', shape=(n,), val=1000.0)  # Altitude, default value from provided code
        self.add_input('th', shape=(n,), val=np.deg2rad(5.))  # theta angle in radians
        self.add_input('phi', shape=(n,), val=np.deg2rad(3.))  # phi angle in radians
        self.add_input('ref_pt', val=np.zeros(3))  # Reference point, default is a zero vector

        # Define outputs
        self.add_output('m', shape=(1,), units='kg')  # Mass
        self.add_output('cg', shape=(3,), units='m')  # Center of gravity
        self.add_output('I', shape=(3, 3), units='N')  # Intertia matrix
        self.add_output('F', shape=(n, 3), units='N')  # Force vector
        self.add_output('M', shape=(n, 3), units='N*m')  # Moment vector

        # self.declare_partials('F', ['th','phi'], method='exact')
        # self.declare_partials('M', ['th','phi'], method='exact')
        self.declare_partials('F', ['th','phi'], method='fd')
        self.declare_partials('M', ['th','phi'], method='fd')

    def c172_mp(self):
        m = 1043.2616
        Ixx = 1285.3154166
        Iyy = 1824.9309607
        Izz = 2666.89390765

        I = np.diag([Ixx, Iyy, Izz])
        cg = np.zeros(3)

        return m, cg, I

    def compute(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        g = 9.803565306802405
        m, cg, I = self.c172_mp()

        th = inputs['th'][0]
        phi = inputs['phi'][0]

        F = np.zeros((num_nodes, 3), dtype=float)

        F[:,0] = -m * g * np.sin(th)
        F[:,1] =  m * g * np.cos(th) * np.sin(phi)
        F[:,2] =  m * g * np.cos(th) * np.cos(phi)

        # F = np.array([-m * g * np.sin(th),
        #               m * g * np.cos(th) * np.sin(phi),
        #               m * g * np.cos(th) * np.cos(phi)])

        offset = cg - inputs['ref_pt']
        M = np.cross(offset, F)

        outputs['m'] = m
        outputs['cg'] = cg
        outputs['I'] = I

        outputs['F'] = F
        outputs['M'] = M

    # def compute_partials(self, inputs, partials):
    #     g = 9.803565306802405
    #     th = inputs['th'][0]
    #     phi = inputs['phi'][0]
    #     m, cg, I = self.c172_mp()

    #     offset = cg - inputs['ref_pt']

    #     dF_dth = np.array([-m * g * np.cos(th),
    #                   -m * g * np.sin(th) * np.sin(phi),
    #                   -m * g * np.sin(th) * np.cos(phi)])
        
    #     dF_dphi = np.array([0.,
    #                   m * g * np.cos(th) * np.cos(phi),
    #                   -m * g * np.cos(th) * np.sin(phi)])
        
    #     partials['F', 'th']  = dF_dth
    #     partials['F', 'phi'] = dF_dphi

    #     partials['M', 'th']  = np.cross(offset, dF_dth)
    #     partials['M', 'phi'] = np.cross(offset, dF_dphi)



if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model
    num_nodes = 4

    model.add_subsystem('inertial_loads', C172InertialLoads(num_nodes=num_nodes))

    prob.setup()
    prob.run_model()
    # prob.check_partials(compact_print=True)

    print("Forces: ", prob['inertial_loads.F'])
    print("Moments: ", prob['inertial_loads.M'])
