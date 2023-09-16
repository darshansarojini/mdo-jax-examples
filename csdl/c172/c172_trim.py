import csdl
import python_csdl_backend as pcb
from math import radians
import numpy as np
import time

from c172_propulsion import C172Propulsion
from c172_mp import C172InertialLoads
from c172_aerodynamics import C172Aerodynamics
from eom_6dof import Eom6DofCg


class C172Trim(csdl.Model):

    def initialize(self):
        self.parameters.declare('num_nodes', default=1)
        return

    def define(self):
        num_nodes = self.parameters['num_nodes']
        a = 336.43470050484996  # m/s
        mach_number = self.create_input('mach_number', shape=(num_nodes, 1))
        self.create_input('theta', shape=(num_nodes, 1))
        self.create_input('delta_e', shape=(num_nodes, 1))
        self.create_input('omega', shape=(num_nodes, 1))
        self.add_design_variable(dv_name='theta',
                                 lower=np.deg2rad(-10)*np.ones(num_nodes, ),
                                 upper=np.deg2rad(10)*np.ones(num_nodes, ))
        self.add_design_variable(dv_name='delta_e',
                                 lower=np.deg2rad(-15) * np.ones(num_nodes, ),
                                 upper=np.deg2rad(15) * np.ones(num_nodes, ))
        self.add_design_variable(dv_name='omega',
                                 lower=1000. * np.ones(num_nodes, ),
                                 upper=2800. * np.ones(num_nodes, ))

        self.create_input('prop_radius', shape=(1, ), units='m')

        u = mach_number * a
        self.register_output(name='u', var=u)

        self.add(submodel=C172InertialLoads(num_nodes=num_nodes),
                 name='inertial_loads', promotes=[])
        self.connect('theta', 'inertial_loads.theta')

        self.add(submodel=C172Aerodynamics(num_nodes=num_nodes),
                 name='aerodynamic_loads', promotes=[])
        self.connect('mach_number', 'aerodynamic_loads.mach_number')
        self.connect('theta', 'aerodynamic_loads.alpha')
        self.connect('delta_e', 'aerodynamic_loads.delta_e')

        self.add(submodel=C172Propulsion(num_nodes=num_nodes),
                 name='propulsion_loads', promotes=[])
        self.connect('prop_radius', 'propulsion_loads.propeller_radius')
        self.connect('mach_number', 'propulsion_loads.mach_number')
        self.connect('omega', 'propulsion_loads.omega')

        self.add(submodel=Eom6DofCg(num_nodes=num_nodes),
                 name='EoM', promotes=[])
        self.connect('inertial_loads.F_inertial', 'EoM.F_i')
        self.connect('inertial_loads.M_inertial', 'EoM.M_i')
        self.connect('aerodynamic_loads.F_aerodynamics', 'EoM.F_a')
        self.connect('aerodynamic_loads.M_aerodynamics', 'EoM.M_a')
        self.connect('propulsion_loads.F_propulsion', 'EoM.F_p')
        self.connect('propulsion_loads.M_propulsion', 'EoM.M_p')
        self.connect('u', 'EoM.u')
        self.connect('theta', 'EoM.theta')
        self.connect('inertial_loads.mass', 'EoM.m')
        # self.connect('inertial_loads.cg_vector', 'EoM.cg')
        self.connect('inertial_loads.inertia_tensor', 'EoM.I')

        res_vec = self.declare_variable(name='residual_vector', shape=(num_nodes, 12))
        self.connect('EoM.residual_vector', 'residual_vector')

        xddot = self.create_output(name='xddot', shape=(num_nodes, 6))
        xddot[:, 0] = res_vec[:, 0]
        xddot[:, 1] = res_vec[:, 1]
        xddot[:, 2] = res_vec[:, 2]
        xddot[:, 3] = res_vec[:, 3]
        xddot[:, 4] = res_vec[:, 4]
        xddot[:, 5] = res_vec[:, 5]

        obj_r = csdl.pnorm(csdl.pnorm(var=xddot, axis=1))
        self.register_output(name='trim_residual', var=obj_r)
        self.add_objective(name='trim_residual')
        return


if __name__ == "__main__":
    num_nodes = 100

    sim = pcb.Simulator(C172Trim(num_nodes=num_nodes))

    sim['mach_number'] = 0.1
    sim['prop_radius'] = 0.94  # m

    # Good guess
    th = np.full(shape=(num_nodes,), fill_value=np.deg2rad(8.739244543508379))
    delta_e = np.full(shape=(num_nodes,), fill_value=np.deg2rad(-7.815234882597328))
    omega = np.full(shape=(num_nodes,), fill_value=1734.40209574)
    # sim['theta'] = th
    # sim['delta_e'] = delta_e
    # sim['omega'] = omega
    #
    # sim.run()
    # sim.compute_total_derivatives()
    # print('EoM trim residual: ', sim['trim_residual'])
    # print('EoM trim residual gradient: ', sim.objective_gradient())

    def obj(x):
        sim.update_design_variables(x)
        sim.run()
        return sim['trim_residual']


    def jac(x):
        sim.update_design_variables(x)
        sim.compute_total_derivatives()
        return sim.objective_gradient()


    import scipy.optimize as op

    start = time.time()
    op_outputs = op.minimize(obj, np.concatenate([th, delta_e, omega]),
                             jac=jac,
                             options={'maxiter': 2000},
                             method='SLSQP',
                             tol=1e-16)
    end = time.time()
    print('Runtime (s): ', (end - start))
    print(op_outputs)
