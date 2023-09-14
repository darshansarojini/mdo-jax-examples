import openmdao.api as om
import numpy as np

class C172Propulsion(om.ExplicitComponent):
    def setup(self):
        # Define inputs
        self.add_input('h', val=1000.0)  # Altitude
        self.add_input('Ma', val=0.1)  # Mach number
        self.add_input('omega', val=2800.0)  # RPM
        self.add_input('prop_radius', val=0.94)  # Propeller radius
        self.add_input('ref_pt', val=np.zeros(3))  # Reference point
        self.add_input('thrust_origin', val=np.zeros(3))  # Thrust origin

        # Define outputs
        self.add_output('F', shape=(3,), units='N')  # Force vector
        self.add_output('M', shape=(3,), units='N*m')  # Moment vector

        self.declare_partials('F', 'omega', method='exact')
        self.declare_partials('M', 'omega', method='exact')

    def compute(self, inputs, outputs):
        rho = 1.1116589850558272  # kg/m^3
        a = 336.43470050484996  # m/s
        prop_radius = inputs['prop_radius']

        V = inputs['Ma'][0] * a
        omega_RAD = (inputs['omega'][0] * 2 * np.pi) / 60.0  # rad/s

        J = (np.pi * V) / (omega_RAD * prop_radius)  # non-dimensional Advance ratio
        Ct_interp = -0.1692121 * J ** 2 + 0.03545196 * J + 0.10446359  # non-dimensional

        T = (2 / np.pi) ** 2 * rho * (omega_RAD * prop_radius) ** 2 * Ct_interp

        F = np.zeros(3)
        F[0] = T

        offset = inputs['ref_pt'] - inputs['thrust_origin']
        M = np.cross(F, offset)

        outputs['F'] = F
        outputs['M'] = M

    def compute_partials(self, inputs, partials):
        rho = 1.1116589850558272  # kg/m^3
        a = 336.43470050484996  # m/s
        prop_radius = inputs['prop_radius']

        V = inputs['Ma'][0] * a
        omega = inputs['omega'][0]

        J = (30 * V) / (omega * prop_radius)  # non-dimensional Advance ratio
        dJ_domega = -30 * V /(omega ** 2 * prop_radius)

        Ct_interp = -0.1692121 * J ** 2 + 0.03545196 * J + 0.10446359  # non-dimensional
        dCt_domega = (-2 * 0.1692121 * J + 0.03545196) * dJ_domega

        dT_domega = rho * prop_radius ** 2 / 225 * (2 * omega * Ct_interp + omega ** 2 * dCt_domega)
        
        partials['F', 'omega'][0]  = dT_domega

        offset = inputs['ref_pt'] - inputs['thrust_origin']
        partials['M', 'omega'] = np.cross(offset, partials['F', 'omega'][:,0])


if __name__ == "__main__":
    prob = om.Problem()
    model = prob.model

    model.add_subsystem('propulsion', C172Propulsion())

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)

    print("Forces: ", prob['propulsion.F'])
    print("Moments: ", prob['propulsion.M'])
