import openmdao.api as om
from math import radians
import numpy as np
import time

from c172_vehicle import C172Vehicle
from c172_propulsion import C172Propulsion
from c172_mp import C172InertialLoads
from c172_aerodynamics import C172Aerodynamics
from eom_6dof import Eom6DofCg

class TrimGroup(om.Group):
    def setup(self):

        # Add subcomponents
        self.add_subsystem('c172vehicle', C172Vehicle(),
                           # promotes=['h', 'Ma', 'ref_pt', 'beta']
                           )
        self.add_subsystem('c172aero', C172Aerodynamics(),
                           # promotes=['h', 'Ma', 'beta']
                           )
        self.add_subsystem('c172prop', C172Propulsion(),
                           # promotes=['h', 'Ma', 'ref_pt']
                           )
        self.add_subsystem('c172inertial', C172InertialLoads(),
                           # promotes=['h', 'ref_pt']
                           )
        self.add_subsystem('eom_6dof', Eom6DofCg())

        self.connect('c172vehicle.h', 'c172prop.h')
        self.connect('c172vehicle.h', 'c172aero.h')
        self.connect('c172vehicle.h', 'c172inertial.h')
        self.connect('c172vehicle.Ma', 'c172prop.Ma')
        self.connect('c172vehicle.Ma', 'c172aero.Ma')
        self.connect('c172vehicle.ref_pt', 'c172prop.ref_pt')
        self.connect('c172vehicle.ref_pt', 'c172inertial.ref_pt')

        self.connect('c172vehicle.alpha', 'c172aero.alpha')
        self.connect('c172vehicle.delta_e', 'c172aero.delta_e')
        self.connect('c172vehicle.alpha', 'c172inertial.th')
        self.connect('c172vehicle.beta', 'c172aero.beta')
        self.connect('c172vehicle.beta', 'c172inertial.phi')
        self.connect('c172vehicle.omega', 'c172prop.omega')
        self.connect('c172vehicle.state_vector', 'eom_6dof.state_vector')

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

    # Add the trim group
    model.add_subsystem('trim_group', TrimGroup())

    # Define design variables and objective
    # model.add_design_var('trim_group.theta', lower=radians(-20), upper=radians(20))
    # model.add_design_var('trim_group.delta_e', lower=radians(-20), upper=radians(20))
    # model.add_design_var('trim_group.omega', lower=1000, upper=2500)
    model.add_design_var('trim_group.c172vehicle.x',
                         lower=np.array([np.deg2rad(-5), np.deg2rad(-15), 1000.]),
                         upper=np.array([np.deg2rad(15), np.deg2rad(15), 2800.]))

    # Assuming eom_6dof_cg provides a norm of the residuals as output
    model.add_objective('trim_group.eom_6dof.trim_residual')

    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options['optimizer'] = 'SLSQP'
    prob.driver.options['tol'] = 1e-16

    prob.setup()

    # prob['trim_group.c172vehicle.h_input'] = 1000.  # m
    # prob['trim_group.c172vehicle.Ma_input'] = 0.1

    # Good guess
    th = np.deg2rad(8.739244543508379)
    delta_e = np.deg2rad(-7.815234882597328)
    omega = 1734.40209574
    prob['trim_group.c172vehicle.x'] = np.array([th, delta_e, omega])

    # # Bad guess
    # th = np.deg2rad(5.)
    # delta_e = np.deg2rad(-5.)
    # omega = 1900.
    # prob['trim_group.c172vehicle.x'] = np.array([th, delta_e, omega])

    # prob.run_model()
    # prob.compute_totals(of='trim_group.eom_6dof.trim_residual', wrt='trim_group.c172vehicle.x')
    # start = time.time()
    # prob.run_driver()
    # end = time.time()
    # print((end - start)/62)

    def obj(x):
        prob['trim_group.c172vehicle.x'] = x
        prob.run_model()
        return prob['trim_group.eom_6dof.trim_residual']

    def jac(x):
        prob['trim_group.c172vehicle.x'] = x
        totals = prob.compute_totals(of='trim_group.eom_6dof.trim_residual', wrt='trim_group.c172vehicle.x')
        return totals['trim_group.eom_6dof.trim_residual', 'trim_group.c172vehicle.x']


    import scipy.optimize as op
    start = time.time()
    op_outputs = op.minimize(obj, np.array([th, delta_e, omega]),
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
