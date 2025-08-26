import csdl_alpha as csdl
import numpy as np
from modopt import CSDLAlphaProblem, SLSQP

class SellarMDA:
    def __init__(self):
        z = csdl.Variable(name='z', value=np.array([5.0, 2.0]))
        x = csdl.Variable(name='x', value=1.0)
        y1_state = csdl.Variable(name='y1_state', value=1.0)
        y2_state = csdl.Variable(name='y2_state', value=1.0)

        y1_state.set_as_design_variable(lower=-1e3, upper=1e3)
        y2_state.set_as_design_variable(lower=-1e3, upper=1e3)

        y1 = z[0]**2 + z[1] + x - 0.2 * y2_state
        y2 = csdl.sqrt(y1_state) + z[0] + z[1]

        y1_res = y1_state - y1
        y2_res = y2_state - y2

        obj = csdl.norm(csdl.concatenate([y1_res, y2_res]))
        obj.set_as_objective()

        self.x = x
        self.z = z
        self.y1 = y1
        self.y2 = y2
        self.y1_state = y1_state
        self.y2_state = y2_state
        self.y1_res = y1_res
        self.y2_res = y2_res
        self.obj = obj

if __name__ == "__main__":
    rec = csdl.Recorder(inline=True)
    rec.start()
    model = SellarMDA()
    rec.stop()

    sim = csdl.experimental.PySimulator(rec)
    prob = CSDLAlphaProblem(simulator=sim, problem_name="sellar_mda_alpha")
    opt = SLSQP(prob, solver_options={"ftol":1e-12, "maxiter":200})
    opt.solve()
    opt.print_results()

    print(sim[model.y1_state])
    print(sim[model.y2_state])
