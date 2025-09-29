import warp as wp
import numpy as np

@wp.func
def mp(m: wp.float32, g: wp.float32, theta: wp.float32, phi: wp.float32, cg_vector: wp.vec3, ref_pt: wp.vec3):
    F_i = wp.vec3(
        -m * g * wp.sin(theta),
         m * g * wp.cos(theta) * wp.sin(phi),
         m * g * wp.cos(theta) * wp.cos(phi)
    )
    
    r_vec_inertial = cg_vector - ref_pt
    M_i = wp.cross(r_vec_inertial, F_i)
    
    return F_i, M_i
