from aerodynamics import aerodynamics
from propulsion import propulsion
from mp import mp
from eom_6dof import eom_6dof
import warp as wp
from scipy.optimize import minimize
import numpy as np
import time

wp.init()

def trim():
    mach_number = 0.117
    a = 336.43470050484996
    u = mach_number * a

    v = 0.0
    w = 0.0
    p = 0.0
    q = 0.0
    r = 0.0

    phi = 0.0
    psi = 0.0
    x = 0.0
    y = 0.0
    z = 0.0

    Sw = 16.2
    chord = 1.49352
    b = 10.91184
    rho = 1.1116589850558272
    m_val = 1043.2616

    Ixx = 1285.3154166
    Iyy = 1824.9309607
    Izz = 2666.89390765
    Ixz = 0.0
    g = 9.803565306802405
    prop_radius = 0.94

    thrust_origin = wp.vec3f(0.0, 0.0, 0.0)
    ref_pt        = wp.vec3f(0.0, 0.0, 0.0)
    cg_vector     = wp.vec3f(0.0, 0.0, 0.0)

    @wp.kernel
    def obj_kernel(theta_arr: wp.array(dtype=wp.float32), delta_e_arr: wp.array(dtype=wp.float32), omega_arr: wp.array(dtype=wp.float32), u_: wp.float32, rho_: wp.float32, Sw_: wp.float32, chord_: wp.float32, b_: wp.float32, m_: wp.float32, Ixx_: wp.float32, Iyy_: wp.float32, Izz_: wp.float32, Ixz_: wp.float32, g_: wp.float32, prop_radius_: wp.float32, ref_pt_: wp.vec3f, thrust_origin_: wp.vec3f, cg_vector_: wp.vec3f, out_obj: wp.array(dtype=wp.float32)):
        th = theta_arr[0]
        de = delta_e_arr[0]
        om = omega_arr[0]
        v_ = 0.0; w_ = 0.0
        p_ = 0.0; q_ = 0.0; r_ = 0.0
        phi_ = 0.0; psi_ = 0.0

        F_a, M_a = aerodynamics(th, de, u_, rho_, Sw_, chord_, b_)
        Ct, T, F_p, M_p = propulsion(om, u_, ref_pt_, thrust_origin_, rho_, prop_radius_)
        F_i, M_i = mp(m_, g_, th, phi_, cg_vector_, ref_pt_)

        total_force  = F_a + F_p + F_i
        total_moment = M_a + M_p + M_i
        residual = eom_6dof(u_, v_, w_, p_, q_, r_, phi_, th, psi_, total_force, total_moment, m_, Ixx_, Iyy_, Izz_, Ixz_)
        s = 0.0
        for k in range(6):
            s += residual[k] * residual[k]
        out_obj[0] = wp.sqrt(s)

    @wp.kernel
    def analysis_kernel(theta_arr: wp.array(dtype=wp.float32), delta_e_arr: wp.array(dtype=wp.float32), omega_arr: wp.array(dtype=wp.float32), u_: wp.float32, rho_: wp.float32, Sw_: wp.float32, chord_: wp.float32, b_: wp.float32, m_: wp.float32, Ixx_: wp.float32, Iyy_: wp.float32, Izz_: wp.float32, Ixz_: wp.float32, g_: wp.float32, prop_radius_: wp.float32, ref_pt_: wp.vec3f, thrust_origin_: wp.vec3f, cg_vector_: wp.vec3f, out: wp.array(dtype=wp.float32)):
        th = theta_arr[0]
        de = delta_e_arr[0]
        om = omega_arr[0]
        v_ = 0.0; w_ = 0.0
        p_ = 0.0; q_ = 0.0; r_ = 0.0
        phi_ = 0.0; psi_ = 0.0

        F_a, M_a = aerodynamics(th, de, u_, rho_, Sw_, chord_, b_)
        Ct, T, F_p, M_p = propulsion(om, u_, ref_pt_, thrust_origin_, rho_, prop_radius_)
        F_i, M_i = mp(m_, g_, th, phi_, cg_vector_, ref_pt_)

        total_force  = F_a + F_p + F_i
        total_moment = M_a + M_p + M_i
        residual = eom_6dof(u_, v_, w_, p_, q_, r_, phi_, th, psi_, total_force, total_moment, m_, Ixx_, Iyy_, Izz_, Ixz_)

        s = 0.0
        for k in range(6):
            s += residual[k] * residual[k]
        trim_objective = wp.sqrt(s)

        out[0] = trim_objective
        out[1] = F_a[0]; out[2] = F_a[1]; out[3] = F_a[2]
        out[4] = M_a[0]; out[5] = M_a[1]; out[6] = M_a[2]
        out[7] = Ct;     out[8] = T
        out[9]  = F_p[0]; out[10] = F_p[1]; out[11] = F_p[2]
        out[12] = M_p[0]; out[13] = M_p[1]; out[14] = M_p[2]
        out[15] = F_i[0]; out[16] = F_i[1]; out[17] = F_i[2]
        out[18] = M_i[0]; out[19] = M_i[1]; out[20] = M_i[2]
        for i in range(12):
            out[21 + i] = residual[i]

    device = "cpu"

    def f_analysis(theta, delta_e, omega):
        th_arr = wp.array([np.float32(theta)], dtype=wp.float32, device=device)
        de_arr = wp.array([np.float32(delta_e)], dtype=wp.float32, device=device)
        om_arr = wp.array([np.float32(omega)], dtype=wp.float32, device=device)
        out = wp.zeros(33, dtype=wp.float32, device=device)
        wp.launch(analysis_kernel, dim=1, device=device, inputs=[th_arr, de_arr, om_arr, np.float32(u), np.float32(rho), np.float32(Sw), np.float32(chord), np.float32(b), np.float32(m_val), np.float32(Ixx), np.float32(Iyy), np.float32(Izz), np.float32(Ixz), np.float32(g), np.float32(prop_radius), ref_pt, thrust_origin, cg_vector, out]); a = out.numpy()
        return (float(a[0]), np.array(a[1:4], dtype=float), np.array(a[4:7], dtype=float), float(a[7]), float(a[8]), np.array(a[9:12], dtype=float), np.array(a[12:15], dtype=float), np.array(a[15:18], dtype=float), np.array(a[18:21], dtype=float), np.array(a[21:33], dtype=float))

    def obj_and_grad(opt_vars):
        theta,delta_e,omega=opt_vars; theta_arr=wp.array([np.float32(theta)],dtype=wp.float32,device=device,requires_grad=True); delta_e_arr=wp.array([np.float32(delta_e)],dtype=wp.float32,device=device,requires_grad=True); omega_arr=wp.array([np.float32(omega)],dtype=wp.float32,device=device,requires_grad=True); out_obj=wp.zeros(1,dtype=wp.float32,device=device,requires_grad=True); tape=wp.Tape()
        with tape: wp.launch(obj_kernel,dim=1,device=device,inputs=[theta_arr,delta_e_arr,omega_arr,np.float32(u),np.float32(rho),np.float32(Sw),np.float32(chord),np.float32(b),np.float32(m_val),np.float32(Ixx),np.float32(Iyy),np.float32(Izz),np.float32(Ixz),np.float32(g),np.float32(prop_radius),ref_pt,thrust_origin,cg_vector,out_obj])
        tape.backward(loss=out_obj); obj=float(out_obj.numpy()[0]); grad=np.array([float(tape.gradients[theta_arr].numpy()[0]),float(tape.gradients[delta_e_arr].numpy()[0]),float(tape.gradients[omega_arr].numpy()[0])],dtype=float); return obj,grad


    return f_analysis, obj_and_grad


def main():
    analysis, obj_and_grad = trim()

    x0 = np.array([np.deg2rad(8.7), np.deg2rad(-7.8), 1734.0])
    bounds = [(np.deg2rad(-10), np.deg2rad(10)), (np.deg2rad(-15), np.deg2rad(15)), (1000.0, 3500.0)]

    def obj_grad(x):
        obj, grad = obj_and_grad(x)
        return float(obj), np.array(grad).ravel()

    t0 = time.time()
    res = minimize(obj_grad, x0, bounds=bounds, method="SLSQP", jac=True, options={"ftol": 1e-12, "maxiter": 200, "disp": True})
    t1 = time.time()

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
