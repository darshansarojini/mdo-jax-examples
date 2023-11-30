import jax
import jax.numpy as jnp
from scipy.optimize import minimize


jax.config.update("jax_enable_x64", True)


k = 1.2  # form factor
e = 0.95  # Oswald efficiency factor
mu = 1.78e-5  # kg m-1 s-1, sea-level dynamic viscosity of air
rho = 1.23  # kg/m^3, sea-level density of air
tau = 0.12  # airfoil thickness-to-chord ratio
N = 3.8  # ultimate load factor
Vmin = 22  # m/s takeoff airspeed
CLmax = 1.5  # takeoff lift coefficient
SwetbyS = 2.05  # wetted area ratio
CDA0 = 0.031  # m^2, fuselage drag area
W0 = 4940  # N, aircraft weight excluding the wing
Ww_c1 = 8.71e-5  # m-1, a coefficient used to calculate wing weight
Ww_c2 = 45.24  # Pa, another wing weight coefficient
Wfuselage = 4940.0


def compute_drag(x):

    AR = x[0]
    S = x[1]
    V = x[2]
    W = x[3]
    CL = x[4]

    c = jnp.sqrt(S/AR)

    Re = rho * V * c / mu
    Cf = 0.074 * Re ** (-0.2)

    CD = CDA0/S + k*Cf*SwetbyS + CL**2/(jnp.pi*AR*e)

    D = 0.5*rho*V**2*CD*S
    return D

def compute_ineq_constraints(x):
    AR = x[0]
    S = x[1]
    V = x[2]
    W = x[3]
    CL = x[4]

    Lcruise = 0.5*rho*V**2*CL*S
    Ltakeoff = 0.5*rho*Vmin**2*CLmax*S

    g = jnp.zeros((2,))
    g = g.at[0].set(W - Lcruise)
    g = g.at[1].set(W - Ltakeoff)
    return -g  # negative because scipy inequality constraints are of the form >=

def compute_eq_constraints(x):
    AR = x[0]
    S = x[1]
    V = x[2]
    W = x[3]
    CL = x[4]

    Ww_struct = Ww_c1 * (N * AR ** 1.5 * jnp.sqrt(W0 * W * S)) / (tau)
    Ww_surf = Ww_c2 * S
    Wwing = Ww_struct + Ww_surf

    h = W - (Wwing + Wfuselage)
    return h



if __name__ == "__main__":
    AR = 8.46
    S = 16.44
    V = 38.15
    W = 7341
    CL = 0.4988

    x = jnp.array([AR, S, V, W, CL])
    D = compute_drag(x)
    g = compute_ineq_constraints(x)
    h = compute_eq_constraints(x)

    cons = ({'type': 'ineq', 'fun': compute_ineq_constraints},
            {'type': 'eq', 'fun': compute_eq_constraints})

    bnds = ((8, 10), (15, 20), (30, 50), (6000, 8000), (0.25, 1))

    res = minimize(fun=compute_drag,
                   x0=x,
                   method='SLSQP',
                   constraints=cons,
                   bounds=bnds)
    print('Optimal solution: ', res.x)
    pass