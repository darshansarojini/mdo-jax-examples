import csdl_alpha as csdl

def mp(m, g, theta, phi, cg_vector, ref_pt):
    F_i = csdl.concatenate(
        [-m * g * csdl.sin(theta),
         m * g * csdl.cos(theta) * csdl.sin(phi),
         m * g * csdl.cos(theta) * csdl.cos(phi)],
        axis=0
    )
    r_vec_inertial = cg_vector - ref_pt
    M_i = csdl.cross(r_vec_inertial, F_i)
    return F_i, M_i
