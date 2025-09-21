from aerodynamics import aerodynamics
from propulsion import propulsion
from mp import mp
from eom_6dof import eom_6dof
from casadi import MX, vertcat, cos, sin, reshape, mtimes, norm_2, Function, jacobian
from scipy.optimize import minimize
import numpy as np


def trim():
    mach_number = MX(0.117)
    a = 336.43470050484996
    u = mach_number * a
    
    v = MX(0.0)
    w = MX(0.0)
    p = MX(0.0)
    q = MX(0.0)
    r = MX(0.0)
    
    phi = MX(0.0)
    psi = MX(0.0)
    x =  MX(0.0)
    y = MX(0.0)
    z = MX(0.0)
    
    theta = MX.sym("theta")
    delta_e = MX.sym("delta_e")
    omega = MX.sym("omega")

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
    
    thrust_origin = vertcat(0.0,0.0,0.0) 
    ref_pt = vertcat(0.0,0.0,0.0)
    cg_vector = vertcat(0.0,0.0,0.0)
    
    F_a, M_a = aerodynamics(theta, delta_e, u, rho, Sw, chord, b)
    Ct, T, F_p, M_p = propulsion(omega, u, ref_pt, thrust_origin, rho, prop_radius)
    m = m_val
    F_i, M_i = mp(m, g, theta, phi, cg_vector, ref_pt)

    total_force = F_a + F_p + F_i
    total_moment = M_a + M_p + M_i

    residual = eom_6dof(u, v, w, p, q, r, phi, theta, psi, total_force, total_moment, m_val, Ixx, Iyy, Izz, Ixz)
    xddot = residual[0:6]
    trim_objective = norm_2(xddot)
    
    
    f_analysis = Function("analysis", [theta, delta_e, omega], [trim_objective, F_a, M_a, Ct, T, F_p, M_p, F_i, M_i, residual])
    
    opt_vars = vertcat(theta, delta_e, omega)
    obj_and_grad = Function("obj_and_grad", [opt_vars], [trim_objective, jacobian(trim_objective, opt_vars)])  
                            
    return f_analysis, obj_and_grad

def main():
    analysis, obj_and_grad = trim()

    x0 = np.array([np.deg2rad(8.7), np.deg2rad(-7.8), 1734.0])
    
    bounds = [(np.deg2rad(-10), np.deg2rad(10)), (np.deg2rad(-15), np.deg2rad(15)), (1000.0, 3500.0)] 

 
    def obj_grad(x):
        obj, grad = obj_and_grad(x)
        return float(obj), np.array(grad).ravel()


    res = minimize(obj_grad, x0, bounds=bounds, method="SLSQP", jac=True,  options={"ftol": 1e-12, "maxiter": 200, "disp": True})

    th_opt, de_opt, om_opt = res.x
    outs = analysis(th_opt, de_opt, om_opt)

    print("\nFinal Results:")
    print(f"Theta (deg): {np.rad2deg(th_opt)}")
    print(f"Delta_e (deg): {np.rad2deg(de_opt)}")
    print(f"Omega (rpm): {om_opt}")
    print(f"Trim Residual (Objective): {float(outs[0])}")
    print(f"Aerodynamic Forces (N): {np.array(outs[1]).flatten()}")
    print(f"Propulsion Forces (N): {np.array(outs[5]).flatten()}")
    print(f"Inertial Forces (N): {np.array(outs[7]).flatten()}")
    print(f"Residual Vector: {np.array(outs[9]).flatten()}")

if __name__ == "__main__":
    main()


'''
Optimization terminated successfully    (Exit mode 0)
            Current function value: 1.6275948663406873e-12
            Iterations: 68
            Function evaluations: 134
            Gradient evaluations: 68

Final Results:
Theta (deg): 6.530817793145723
Delta_e (deg): -5.287471193294246
Omega (rpm): 2883.273023197867
Trim Residual (Objective): 1.6275948663406873e-12
Aerodynamic Forces (N): [ -2022.73180284      0.         -10161.31398216]
Propulsion Forces (N): [3186.00406747    0.            0.        ]
Inertial Forces (N): [-1163.27226463     0.         10161.31398216]
Residual Vector: [-2.05740104e-13  0.00000000e+00  2.30149946e-13  0.00000000e+00
 -1.59805102e-12  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00  3.91074274e+01  0.00000000e+00 -4.47703769e+00]
'''
