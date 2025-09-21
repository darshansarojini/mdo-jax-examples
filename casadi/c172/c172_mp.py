import numpy as np
from casadi import MX, vertcat, cos, sin, cross

def mp(m, g, theta, phi, cg_vector, ref_pt):
    F_i = vertcat(-m * g * sin(theta), m * g * cos(theta) * sin(phi), m * g * cos(theta) * cos(phi))
    r_vec_inertial = cg_vector - ref_pt
    M_i = cross(r_vec_inertial, F_i)
    return F_i, M_i
