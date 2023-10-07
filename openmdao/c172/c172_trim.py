import openmdao.api as om
from math import radians
import numpy as np
import time

from c172_vehicle_state import C172VehicleState
from c172_propulsion import C172Propulsion
from c172_mp import C172InertialLoads
from c172_aerodynamics import C172Aerodynamics
from eom_6dof import Eom6DofCg

class TrimGroup(om.Group):

    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        n = self.options['num_nodes']
        # Independent variables
        indeps = self.add_subsystem('indeps', om.IndepVarComp())
        indeps.add_output('h', shape=(n,), val=1000.0, units='m')
        indeps.add_output('ref_pt', shape=(3,), val=0., units='m')
        indeps.add_output('Ma', shape=(n, ), val=0.1)
        indeps.add_output('beta', shape=(n, ), val=0.0)
        indeps.add_output('x', shape=(n, 3))

        # Add subcomponents
        self.add_subsystem('c172vehiclestate', C172VehicleState(num_nodes=n),
                           # promotes=['h', 'Ma', 'ref_pt', 'beta']
                           )
        self.add_subsystem('c172aero', C172Aerodynamics(num_nodes=n),
                           # promotes=['h', 'Ma', 'beta']
                           )
        self.add_subsystem('c172prop', C172Propulsion(num_nodes=n),
                           # promotes=['h', 'Ma', 'ref_pt']
                           )
        self.add_subsystem('c172inertial', C172InertialLoads(num_nodes=n),
                           # promotes=['h', 'ref_pt']
                           )
        self.add_subsystem('eom_6dof', Eom6DofCg(num_nodes=n))

        self.connect('indeps.h', ['c172prop.h', 'c172aero.h', 'c172inertial.h', 'c172vehiclestate.h'])
        self.connect('indeps.Ma', ['c172prop.Ma', 'c172aero.Ma', 'c172vehiclestate.Ma'])
        self.connect('indeps.ref_pt', ['c172prop.ref_pt', 'c172inertial.ref_pt'])
        self.connect('indeps.beta', ['c172aero.beta', 'c172inertial.phi'])

        self.connect('indeps.x', ['c172aero.alpha', 'c172inertial.th', 'c172vehiclestate.alpha'], src_indices=om.slicer[:, 0])
        self.connect('indeps.x', 'c172aero.delta_e', src_indices=om.slicer[:, 1])
        self.connect('indeps.x', 'c172prop.omega', src_indices=om.slicer[:, 2])

        self.connect('c172vehiclestate.state_vector', 'eom_6dof.state_vector')

        # Connect loads
        self.connect('c172inertial.F', 'eom_6dof.Fi')
        self.connect('c172inertial.M', 'eom_6dof.Mi')
        self.connect('c172aero.F', 'eom_6dof.Fa')
        self.connect('c172aero.M', 'eom_6dof.Ma')
        self.connect('c172prop.F', 'eom_6dof.Fp')
        self.connect('c172prop.M', 'eom_6dof.Mp')
        # Connect mass properties
        self.connect('c172inertial.m', 'eom_6dof.m')
        self.connect('c172inertial.cg', 'eom_6dof.cg')
        self.connect('c172inertial.I', 'eom_6dof.I')


if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model
    num_nodes = 4

    # Add the trim group
    model.add_subsystem('trim_group', TrimGroup(num_nodes=num_nodes))

    # Define design variables and objective
    # model.add_design_var('trim_group.theta', lower=radians(-20), upper=radians(20))
    # model.add_design_var('trim_group.delta_e', lower=radians(-20), upper=radians(20))
    # model.add_design_var('trim_group.omega', lower=1000, upper=2500)
    model.add_design_var('trim_group.indeps.x',
                         lower=np.tile(np.array([np.deg2rad(-5), np.deg2rad(-15), 1000.]), (num_nodes,1)),
                         upper=np.tile(np.array([np.deg2rad(15), np.deg2rad(15), 2800.]), (num_nodes,1)))

    # Assuming eom_6dof_cg provides a norm of the residuals as output
    model.add_objective('trim_group.eom_6dof.trim_residual')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-16

    prob.setup()
    # om.n2(prob)

    # prob['trim_group.c172vehicle.h_input'] = 1000.  # m
    # prob['trim_group.c172vehicle.Ma_input'] = 0.1

    # Good guess
    th = np.deg2rad(8.739244543508379)
    delta_e = np.deg2rad(-7.815234882597328)
    omega = 1734.40209574
    prob['trim_group.indeps.x'] = np.tile(np.array([th, delta_e, omega]), (num_nodes, 1))

    # # Bad guess
    # th = np.deg2rad(5.)
    # delta_e = np.deg2rad(-5.)
    # omega = 1900.
    # prob['trim_group.c172vehicle.x'] = np.array([th, delta_e, omega])

    # prob.run_model()
    # prob.compute_totals(of='trim_group.eom_6dof.trim_residual', wrt='trim_group.indeps.x')
    # prob.check_totals(of='trim_group.eom_6dof.trim_residual', wrt='trim_group.indeps.x')
    # exit()
    # start = time.time()
    # prob.run_driver()
    # end = time.time()
    # print((end - start)/62)

    def obj(x):
        prob['trim_group.indeps.x'] = x
        prob.run_model()
        return prob['trim_group.eom_6dof.trim_residual']

    def jac(x):
        prob['trim_group.indeps.x'] = x
        totals = prob.compute_totals(of='trim_group.eom_6dof.trim_residual', wrt='trim_group.indeps.x')
        return totals['trim_group.eom_6dof.trim_residual', 'trim_group.indeps.x'].flatten()
    
    # exit()


    import scipy.optimize as op
    start = time.time()
    op_outputs = op.minimize(obj, np.tile(np.array([th, delta_e, omega]), (num_nodes, 1)),
                             jac=jac,
                             options={'maxiter': 100},
                             method='SLSQP',
                             tol = 1e-8)
    end = time.time()
    print((end - start))
    print(op_outputs)

    # start = time.time()
    # op_outputs = op.minimize(obj, np.array([th, delta_e, omega]),
    #                          # args=(Ma, h, prop_radius),
    #                          jac=jac,
    #                          options={'maxiter': 100},
    #                          method='SLSQP',
    #                          tol=1e-8)
    # end = time.time()
    # print((end - start))
    # print(op_outputs)


    # start = time.time()
    # prob['trim_group.c172vehicle.x'] = np.array([th, delta_e, omega])
    # prob.run_driver()
    # end = time.time()
    # print((end - start) / 62)
    #
    # # Results
    # om.n2(prob, show_browser=False)
    print('EoM trim residual: ', prob['trim_group.eom_6dof.trim_residual'])
