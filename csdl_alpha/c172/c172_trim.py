import csdl_alpha as csdl
import numpy as np
from modopt import CSDLAlphaProblem, SLSQP
from c172_aerodynamics import aerodynamics
from c172_propulsion import propulsion
from c172_mp import mp
from eom_6dof import eom_6dof

class C172TrimModel:
    def __init__(self):
        mach_number = csdl.Variable(name='mach_number', value=0.117)
        a = 336.43470050484996
        u = mach_number * a
        
        v = csdl.Variable(name='v', value=0.)
        w = csdl.Variable(name='w', value=0.)
        p = csdl.Variable(name='p', value=0.)
        q = csdl.Variable(name='q', value=0.)
        r = csdl.Variable(name='r', value=0.)
        
        phi = csdl.Variable(name='phi', value=0.)
        psi = csdl.Variable(name='psi', value=0.)
        x = csdl.Variable(name='x', value=0.)
        y = csdl.Variable(name='y', value=0.)
        z = csdl.Variable(name='z', value=0.)
        
        theta = csdl.Variable(name='theta', value=np.deg2rad(8.7))
        delta_e = csdl.Variable(name='delta_e', value=np.deg2rad(-7.8))
        omega = csdl.Variable(name='omega', value=1734.0)

        theta.set_as_design_variable(lower=np.deg2rad(-10), upper=np.deg2rad(10))
        delta_e.set_as_design_variable(lower=np.deg2rad(-15), upper=np.deg2rad(15))
        omega.set_as_design_variable(lower=1000., upper=2800.)

        Sw = 16.2
        chord = 1.49352
        b = 10.91184
        rho = 1.1116589850558272
        m_val = 1043.2616
        
        Ixx = 1285.3154166
        Iyy = 1824.9309607
        Izz = 2666.89390765
        Ixz = 0.
        g = 9.803565306802405
        prop_radius = 0.94
        
        thrust_origin = csdl.Variable(name='thrust_origin', value=np.array([0, 0, 0]))
        ref_pt = csdl.Variable(name='ref_pt', value=np.array([0, 0, 0]))
        cg_vector = csdl.Variable(name='cg_vector', value=np.array([0, 0, 0]))
        
        self.m = csdl.Variable(name='mass', value=m_val)
        inertia_matrix = np.array([[Ixx, 0, Ixz], [0, Iyy, 0], [Ixz, 0, Izz]])
        self.inertia_tensor = csdl.Variable(name='inertia_tensor', value=inertia_matrix)

        F_a, M_a = aerodynamics(theta, delta_e, u, rho, Sw, chord, b)
        Ct, T, F_p, M_p = propulsion(omega, u, ref_pt, thrust_origin, rho, prop_radius)
        F_i, M_i = mp(self.m, g, theta, phi, cg_vector, ref_pt)

        total_force = F_a + F_p + F_i
        Fx, Fy, Fz = total_force[0], total_force[1], total_force[2]
        
        total_moment = M_a + M_p + M_i
        L_mom, M_mom, N_mom = total_moment[0], total_moment[1], total_moment[2]

        du_dt = Fx / self.m + r * v - q * w
        dv_dt = Fy / self.m - r * u + p * w
        dw_dt = Fz / self.m + q * u - p * v
        
        dp_dt = (L_mom * Izz + N_mom * Ixz - q * r * (Izz ** 2 - Izz * Iyy + Ixz ** 2) + p * q * Ixz * (Ixx + Izz - Iyy)) / (Ixx * Izz - Ixz ** 2)
        dq_dt = (M_mom + (Izz - Ixx) * p * r - Ixz * (p ** 2 - r ** 2)) / Iyy
        dr_dt = (L_mom * Ixz + N_mom * Ixx + p * q * (Ixx ** 2 - Ixx * Iyy + Ixz ** 2) - q * r * Ixz * (Izz + Ixx - Iyy)) / (Ixx * Izz - Ixz ** 2)

        dtheta_dt = q * csdl.cos(phi) - r * csdl.sin(phi)
        dphi_dt = p + (q * csdl.sin(phi) + r * csdl.cos(phi)) * csdl.tan(theta)
        dpsi_dt = (q * csdl.sin(phi) + r * csdl.cos(phi)) / csdl.cos(theta)

        dx_dt = u * csdl.cos(theta) * csdl.cos(psi) + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.cos(psi) - csdl.cos(phi) * csdl.sin(psi)) + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.cos(psi) + csdl.sin(phi) * csdl.sin(psi))
        dy_dt = u * csdl.cos(theta) * csdl.sin(psi) + v * (csdl.sin(phi) * csdl.sin(theta) * csdl.sin(psi) + csdl.cos(phi) * csdl.cos(psi)) + w * (csdl.cos(phi) * csdl.sin(theta) * csdl.sin(psi) - csdl.sin(phi) * csdl.cos(psi))
        dz_dt = -u * csdl.sin(theta) + v * csdl.sin(phi) * csdl.cos(theta) + w * csdl.cos(phi) * csdl.cos(theta)

        residual_vector = csdl.concatenate([du_dt, dv_dt, dw_dt, dp_dt, dq_dt, dr_dt, dphi_dt, dtheta_dt, dpsi_dt, dx_dt, dy_dt, dz_dt])
        xddot = residual_vector[0:6]
        trim_objective = csdl.norm(xddot)
        trim_objective.set_as_objective()

        self.theta = theta
        self.delta_e = delta_e
        self.omega = omega
        self.F_a = F_a
        self.M_a = M_a
        self.T = T
        self.F_p = F_p
        self.M_p = M_p
        self.F_i = F_i
        self.M_i = M_i
        self.cg_vector = cg_vector
        self.ref_pt = ref_pt
        self.Ct = Ct
        self.trim_objective = trim_objective
        self.residual_vector = residual_vector

if __name__ == "__main__":
    rec = csdl.Recorder(inline=True)

    rec.start()
    model = C172TrimModel()
    rec.stop()
    
    sim = csdl.experimental.PySimulator(rec)
    prob = CSDLAlphaProblem(simulator=sim, problem_name='c172_trim')
    opt = SLSQP(prob, solver_options={'ftol':1e-12, 'maxiter':200})
    opt.solve()
    opt.print_results()

    print("\nFinal Results:")
    print("\nmp.py:")
    print("Mass (kg): ", sim[model.m])
    print("Inertia Tensor (kg*m^2):\n", sim[model.inertia_tensor])
    print("Center of Gravity (m): ", sim[model.cg_vector])
    print("Reference Point (m): ", sim[model.ref_pt])
    print("Inertial Forces (N): ", sim[model.F_i])
    print("Inertial Moments (N*m): ", sim[model.M_i])
    print("\naerodynamics.py")
    print("Aerodynamic Forces (N): ", sim[model.F_a])
    print("Aerodynamic Moments (N*m): ", sim[model.M_a])
    print("\npropulsion.py")
    print("Thrust Coefficient (Ct): ", sim[model.Ct])
    print("Thrust (N): ", sim[model.T])
    print("Propulsion Forces (N): ", sim[model.F_p])
    print("Propulsion Moments (N*m): ", sim[model.M_p])
    print("\ntrim.py")
    print(f"Theta (deg): {np.rad2deg(sim[model.theta])[0]}")
    print(f"Delta_e (deg): {np.rad2deg(sim[model.delta_e])[0]}")
    print(f"Omega (rpm): {sim[model.omega][0]}")
    print(f"Trim Residual (Objective): {sim[model.trim_objective]}")
    print("\neom_6dof.py")
    print("Full Residual Vector:", sim[model.residual_vector])
