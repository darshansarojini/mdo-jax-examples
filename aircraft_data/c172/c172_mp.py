import jax.numpy as jnp
import numpy as np


def c172_mp():
    m = 1043.2616

    Ixx = 1285.3154166
    Iyy = 1824.9309607
    Izz = 2666.89390765

    I = jnp.zeros((3, 3))
    I = I.at[0, 0].set(Ixx)
    I = I.at[1, 1].set(Iyy)
    I = I.at[2, 2].set(Izz)

    cg = jnp.zeros(3)
    return m, cg, I

def c172_inertial_loads(h, th, phi=jnp.deg2rad(0), ref_pt=jnp.zeros(3)):

    g = 9.81
    m, cg, I = c172_mp()

    F = jnp.zeros(3)
    F = F.at[0].set(-m * g * jnp.sin(th))
    F = F.at[1].set(m * g * jnp.cos(th) * jnp.sin(phi))
    F = F.at[2].set(m * g * jnp.cos(th) * jnp.cos(phi))

    offset = cg - ref_pt
    M = jnp.cross(offset, F)

    return F, M


if __name__ == "__main__":
    h = 1000
    th = np.deg2rad(5.)
    F, M = c172_inertial_loads(h=h, th=th)
    print("Forces: ", F)
    print("Moments: ", M)