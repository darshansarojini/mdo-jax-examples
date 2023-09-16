import jax
import jax.numpy as jnp
import numpy as np

@jax.jit
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

@jax.jit
def c172_inertial_loads(th, phi=jnp.deg2rad(0), ref_pt=jnp.zeros(3)):

    num_nodes  = th.shape[0]
    g = 9.803565306802405
    m, cg, I = c172_mp()

    F = jnp.zeros((num_nodes, 3))
    F = F.at[:, 0].set(-m * g * jnp.sin(th))
    F = F.at[:, 1].set(m * g * jnp.cos(th) * jnp.sin(phi))
    F = F.at[:, 2].set(m * g * jnp.cos(th) * jnp.cos(phi))

    offset = cg - ref_pt
    M = jnp.cross(offset, F)

    return F, M


if __name__ == "__main__":
    th = np.full(shape=(3, ), fill_value=np.deg2rad(5.))
    phi = np.full(shape=(3, ), fill_value=np.deg2rad(3.))
    F, M = c172_inertial_loads(th=th, phi=phi)
    print("Forces: ", F)
    print("Moments: ", M)